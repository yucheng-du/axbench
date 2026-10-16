"""Offline SAE checkpoint loading with a real tiny LM and PyVene wrapper."""

import importlib
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


class _UnusedPipeline:
    def __call__(self, *args, **kwargs):
        raise AssertionError("SAE loading must not use Stanza")

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected Stanza access: {name}")


@pytest.fixture(scope="session")
def runtime():
    """Import the real checkout while denying remote connections and downloads."""
    denied = []
    original_lookup = socket.getaddrinfo
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def reject_remote(host):
        if host not in {None, "localhost", "127.0.0.1", "::1"}:
            denied.append(repr(host))
            raise AssertionError(f"Unexpected network destination: {host!r}")

    def lookup(host, *args, **kwargs):
        reject_remote(host)
        return original_lookup(host, *args, **kwargs)

    def connect(sock, address):
        if isinstance(address, tuple):
            reject_remote(address[0])
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if isinstance(address, tuple):
            reject_remote(address[0])
        return original_connect_ex(sock, address)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(ROOT / "axbench" / "scripts"))
        monkeypatch.syspath_prepend(str(ROOT))
        for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
            monkeypatch.setenv(variable, "1")
        monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
        monkeypatch.setattr(socket, "getaddrinfo", lookup)
        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)

        import stanza
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        # Package import constructs an unrelated evaluator's Stanza pipeline.
        # Keep all modules real and fail if that pipeline is actually used.
        with patch.object(stanza, "Pipeline", return_value=_UnusedPipeline()):
            sae_module = importlib.import_module("axbench.models.sae")
            assert Path(sae_module.__file__).resolve() == ROOT / "axbench/models/sae.py"
            vocabulary = {
                "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
                "a": 4, "local": 5, "tiny": 6, "model": 7,
            }
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                bos_token="[BOS]", eos_token="[EOS]",
            )
            yield SimpleNamespace(
                torch=torch, sae=sae_module, tokenizer=tokenizer,
                config=LlamaConfig, model=LlamaForCausalLM,
            )
            assert denied == [], f"Unexpected network attempts: {denied}"


@pytest.fixture
def checkpoint(runtime, tmp_path):
    torch = runtime.torch
    encoder = torch.zeros(16, 3)
    encoder[0, 0], encoder[1, 1], encoder[2, 2] = 1, 2, -1
    params = {
        "W_enc": encoder,
        "W_dec": torch.arange(48, dtype=torch.float32).reshape(3, 16) / 16,
        "b_enc": torch.tensor([0.25, -0.5, 0.75]),
        "threshold": torch.tensor([0.5, 0.1, 1.0]),
        "b_dec": torch.arange(16, dtype=torch.float32) / 100,
    }
    torch.save(params, tmp_path / "GemmaScopeSAE.pt")
    return tmp_path, params


@pytest.mark.parametrize("mode", ["latent", "steering"])
@pytest.mark.parametrize("caller_rank", [None, 1], ids=["rank-omitted", "rank-one"])
def test_load_uses_checkpoint_width(runtime, checkpoint, mode, caller_rank):
    torch = runtime.torch
    torch.manual_seed(42)
    model = runtime.model(runtime.config(
        vocab_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, pad_token_id=0, bos_token_id=2, eos_token_id=3,
    ))
    original_weights = {key: value.clone() for key, value in model.state_dict().items()}
    sae = runtime.sae.GemmaScopeSAE(model, runtime.tokenizer, layer=0, device="cpu", seed=42)
    dump_dir, params = checkpoint
    kwargs = {
        "mode": mode,
        "sae_path": "https://www.neuronpedia.org/local/sae/2",
        "concept_id": 2,
    }
    if mode == "steering":
        kwargs.update(intervention_type="addition", priority_mode="compute_priority")
    if caller_rank is not None:
        # The official steering caller supplies 1 even for wider checkpoints.
        kwargs["low_rank_dimension"] = caller_rank
    original_kwargs = kwargs.copy()

    sae.load(dump_dir=dump_dir, **kwargs)

    assert kwargs == original_kwargs
    assert isinstance(sae.ax_model, runtime.sae.IntervenableModel)
    assert sae.ax_model.model is model
    assert list(sae.ax_model.interventions.values()) == [sae.ax]
    (representation,) = sae.ax_model.config.representations
    assert representation.component == "model.layers[0].output"
    assert representation.low_rank_dimension == 3
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original_weights[key], rtol=0, atol=0)
    for value in sae.ax.state_dict().values():
        assert value.device.type == "cpu"

    if mode == "latent":
        assert isinstance(sae.ax, runtime.sae.JumpReLUSAECollectIntervention)
        for key, value in params.items():
            torch.testing.assert_close(getattr(sae.ax, key), value, rtol=0, atol=0)
        base = torch.zeros(1, 2, 16)
        base[0, 0, :3] = torch.tensor([1.0, 1.0, -1.0])
        base[0, 1, :3] = torch.tensor([0.0, 0.0, -0.25])
        # The final feature at token 2 equals its threshold and must be zero.
        expected = torch.tensor([[[1.25, 1.5, 1.75], [0.0, 0.0, 0.0]]])
        torch.testing.assert_close(sae.ax(base), expected, rtol=0, atol=0)
    else:
        assert isinstance(sae.ax, runtime.sae.AdditionIntervention)
        torch.testing.assert_close(sae.ax.proj.weight, params["W_dec"], rtol=0, atol=0)
        # Assigning .weight.data alone can hide construction at the wrong width.
        assert sae.ax.proj.out_features == 3
        assert sae.ax.proj.bias.shape == (3,)
        base = torch.arange(64, dtype=torch.float32).reshape(2, 2, 16) / 8
        actual = sae.ax(base, subspaces={
            "idx": torch.tensor([2, 0]),
            "max_act": torch.tensor([4.0, 2.0]),
            "mag": torch.tensor([0.5, -1.0]),
        })
        expected = base.clone()
        expected[0] += 2 * params["W_dec"][2]
        expected[1] -= 2 * params["W_dec"][0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
