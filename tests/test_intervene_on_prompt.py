"""Exercise prompt intervention control through real offline SAE generation."""

import importlib
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


class _UnusedPipeline:
    def __call__(self, *args, **kwargs):
        raise AssertionError("SAE steering must not use Stanza")

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected Stanza access: {name}")


@pytest.fixture(scope="session")
def runtime(tmp_path_factory):
    """Import real modules, keep unused NLP setup inert, and reject remote IO."""
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
        monkeypatch.delenv("NP_API_KEY", raising=False)
        monkeypatch.setattr(socket, "getaddrinfo", lookup)
        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)

        import pandas as pd
        import stanza
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        with patch.object(stanza, "Pipeline", return_value=_UnusedPipeline()):
            sae_module = importlib.import_module("axbench.models.sae")
            inference = importlib.import_module("axbench.scripts.inference")
            for module, relative_path in (
                (sae_module, "axbench/models/sae.py"),
                (inference, "axbench/scripts/inference.py"),
            ):
                assert Path(module.__file__).resolve() == ROOT / relative_path
            vocabulary = {
                "[PAD]": 0, "[UNK]": 1, "alpha": 2, "beta": 3,
                "gamma": 4, "delta": 5, "epsilon": 6, "zeta": 7,
            }
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                model_input_names=["input_ids", "attention_mask"],
            )
            assert tokenizer("alpha beta gamma").input_ids == [2, 3, 4]
            rendezvous = tmp_path_factory.mktemp("gloo") / "rendezvous"
            torch.distributed.init_process_group(
                "gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1,
            )
            try:
                yield SimpleNamespace(
                    torch=torch, pd=pd, sae=sae_module, inference=inference,
                    tokenizer=tokenizer, config=LlamaConfig, model=LlamaForCausalLM,
                )
            finally:
                torch.distributed.destroy_process_group()
            assert denied == [], f"Unexpected network attempts: {denied}"


@pytest.fixture
def loaded_sae(runtime, tmp_path):
    torch, pd = runtime.torch, runtime.pd
    torch.manual_seed(42)
    model = runtime.model(runtime.config(
        vocab_size=8, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=32, pad_token_id=0, bos_token_id=None,
        eos_token_id=None, use_cache=True,
    ))
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    model.generation_config.use_cache = True
    params = {
        "W_enc": torch.zeros(16, 2),
        "W_dec": torch.stack((torch.arange(1, 17) / 16, -torch.arange(1, 17) / 8)),
        "b_enc": torch.zeros(2),
        "threshold": torch.zeros(2),
        "b_dec": torch.zeros(16),
    }
    torch.save(params, tmp_path / "GemmaScopeSAE.pt")
    sae = runtime.sae.GemmaScopeSAE(model, runtime.tokenizer, layer=0, device="cpu")
    # Omitted optional width uses the existing public loading path on main.
    sae.load(dump_dir=tmp_path, mode="steering", intervention_type="addition")
    assert sae.ax_model.model is model
    assert sae.ax.proj.out_features == 2
    torch.testing.assert_close(sae.ax.proj.weight, params["W_dec"], rtol=0, atol=0)

    # Existing latent output + cache exercise the original preparation path.
    metadata_rows = [
        {"concept_id": i, "concept": concept,
         "concept_genres_map": {concept: ["text"]},
         "ref": f"https://www.neuronpedia.org/local/sae/{feature}"}
        for i, (concept, feature) in enumerate((("first", 17), ("second", 42)))
    ]
    (tmp_path / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metadata_rows)
    )
    metadata = runtime.inference.load_metadata_flatten(tmp_path)
    latent_dir = tmp_path / "latent"
    latent_dir.mkdir()
    pd.DataFrame({
        "concept_id": [row["concept_id"] for row in metadata],
        "sae_link": [row["ref"] for row in metadata],
        "GemmaScopeSAE_max_act": [3.0, 8.0],
    }).to_parquet(latent_dir / "latent_data.parquet", engine="pyarrow")
    cache = tmp_path / "local_sae_max_activations.json"
    cache.write_text(json.dumps({17: 3.0, 42: 8.0}))
    original_cache = cache.read_bytes()
    assert sae.pre_compute_mean_activations(
        latent_dir, master_data_dir=tmp_path, metadata=metadata,
    ) == {17: 3.0, 42: 8.0}
    assert cache.read_bytes() == original_cache
    assert sae.max_activations == {17: 3.0, 42: 8.0}
    examples = pd.DataFrame([{
        "input": "alpha beta gamma", "input_id": 0, "concept_id": 0,
        "sae_id": 17, "sae_link": metadata[0]["ref"], "factor": 2.0,
    }])
    return sae, examples, params


@pytest.mark.parametrize(
    "prompt_flag,expected_lengths",
    [(False, [1, 1]), (True, [3, 1, 1]), (None, [3, 1, 1])],
    ids=["false", "true", "omitted"],
)
def test_predict_steer_respects_prompt_flag(runtime, loaded_sae, tmp_path, prompt_flag, expected_lengths):
    torch = runtime.torch
    sae, examples, params = loaded_sae
    original_weights = {key: value.clone() for key, value in sae.model.state_dict().items()}
    lengths = []

    def observe(module, args, output):
        base = args[0]
        lengths.append(base.shape[1])
        expected_delta = (6 * params["W_dec"][0]).expand_as(base)
        torch.testing.assert_close(output - base, expected_delta, rtol=1e-6, atol=1e-6)

    handle = sae.ax.register_forward_hook(observe)
    kwargs = {} if prompt_flag is None else {"intervene_on_prompt": prompt_flag}
    try:
        torch.manual_seed(123)
        result = sae.predict_steer(
            examples, batch_size=1, eval_output_length=3, temperature=1.0,
            prefix_length=0, **kwargs,
        )
    finally:
        handle.remove()

    assert lengths == expected_lengths
    assert result["strength"] == [6.0]
    assert len(result["steered_generation"]) == 1
    for key, value in sae.model.state_dict().items():
        torch.testing.assert_close(value, original_weights[key], rtol=0, atol=0)

    # Independently decode a real PyVene generation using the requested flag.
    # Do not infer correctness from sampled text changing between flag values.
    inputs = runtime.tokenizer(["alpha beta gamma"], return_tensors="pt", padding=True, truncation=True)
    torch.manual_seed(123)
    with torch.no_grad():
        _, reference_ids = sae.ax_model.generate(
            inputs, unit_locations=None,
            intervene_on_prompt=True if prompt_flag is None else prompt_flag,
            subspaces=[{
                "idx": torch.tensor([0]), "mag": torch.tensor([2.0]),
                "max_act": torch.tensor([3.0]), "prefix_length": 0,
            }],
            max_new_tokens=3, do_sample=True, temperature=1.0,
        )
    assert reference_ids.shape == (1, 6)
    assert result["steered_generation"] == [
        runtime.tokenizer.decode(reference_ids[0, 3:], skip_special_tokens=True)
    ]

    saved = examples.copy()
    for key, value in result.items():
        saved[f"GemmaScopeSAE_{key}"] = value
    output_dir = tmp_path / "output"
    runtime.inference.save(output_dir, "steering", saved, rank=0)
    roundtrip = runtime.pd.read_parquet(output_dir / "rank_0_steering_data.parquet")
    runtime.pd.testing.assert_frame_equal(roundtrip, saved)
