"""Offline integration tests for SAE activation preparation, not steering main.

Use the real package, SAE class, metadata reader, parquet reader and cache IO.
Only the unused eager Stanza constructor and external HTTP/connection
boundaries are adapted. The local random LM is never downloaded or trained.
"""

import errno
import importlib
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "test-model/test-sae"
CACHE_NAME = "test-model_test-sae_max_activations.json"


def _ref(feature_id):
    return f"https://www.neuronpedia.org/{SOURCE}/{feature_id}"


class _UnusedPipeline:
    def __call__(self, *args, **kwargs):
        raise AssertionError("SAE activation preparation must not use Stanza")

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected Stanza access: {name}")


@pytest.fixture(scope="session")
def runtime():
    """Import normally from this checkout, despite another editable install."""
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

        import pandas as pd
        import pyarrow as pa
        import stanza
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        with patch.object(stanza, "Pipeline", return_value=_UnusedPipeline()) as pipeline:
            sae = importlib.import_module("axbench.models.sae")
            inference = importlib.import_module("axbench.scripts.inference")
            for module, relative_path in (
                (sae, "axbench/models/sae.py"),
                (inference, "axbench/scripts/inference.py"),
            ):
                assert Path(module.__file__).resolve() == ROOT / relative_path, (
                    f"Loaded {module.__file__}, expected this checkout: {ROOT}"
                )
            assert pipeline.call_count == 1
            vocabulary = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "local": 4}
            backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
                bos_token="[BOS]", eos_token="[EOS]",
            )
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=32, bos_token_id=2, eos_token_id=3, pad_token_id=0,
            ))
            yield SimpleNamespace(sae=sae, inference=inference, pd=pd, pa=pa,
                                  model=model, tokenizer=tokenizer)
            assert pipeline.call_count == 1
        assert not denied, denied


@pytest.fixture
def data(tmp_path, runtime):
    source = tmp_path / "generate"
    source.mkdir()
    records = [
        {"concept_id": concept_id, "concept": f"concept {concept_id}",
         "ref": _ref(feature_id),
         "concept_genres_map": {f"concept {concept_id}": ["text"]}}
        for concept_id, feature_id in enumerate((7, 42, 75))
    ]
    metadata_path = source / "metadata.jsonl"
    metadata_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    original_metadata = metadata_path.read_bytes()
    metadata = runtime.inference.load_metadata_flatten(source)
    assert metadata == records
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / CACHE_NAME
    cache_path.write_text(json.dumps({"7": 2.5, "99": 9.0}, indent=2) + "\n")
    instance = runtime.sae.GemmaScopeSAE(
        runtime.model, runtime.tokenizer, layer=0, device="cpu", seed=42,
    )
    assert isinstance(instance, runtime.sae.GemmaScopeSAE)
    yield SimpleNamespace(instance=instance, metadata=metadata, cache_dir=cache_dir,
                          cache_path=cache_path, dump_dir=tmp_path / "inference")
    assert metadata_path.read_bytes() == original_metadata


@pytest.fixture
def http_boundary(monkeypatch, runtime):
    """Only this module's exact Neuronpedia calls may receive a response."""
    calls = []
    responses = {}

    class Requests:
        def get(self, url, *, headers):
            assert url in responses, f"Unexpected HTTP request: {url}"
            assert headers == {"X-Api-Key": "offline-test-key"}
            calls.append(url)
            return SimpleNamespace(json=lambda: {"activations": [{"maxValue": responses[url]}]})

        def __getattr__(self, name):
            raise AssertionError(f"Unexpected HTTP operation: {name}")

    monkeypatch.setattr(runtime.sae, "requests", Requests())
    monkeypatch.delenv("NP_API_KEY", raising=False)
    return SimpleNamespace(calls=calls, responses=responses)


@pytest.mark.parametrize("directory_exists", [True, False], ids=["empty-directory", "missing-directory"])
def test_metadata_fallback_preserves_feature_ids_and_cache(data, http_boundary, monkeypatch,
                                                          directory_exists):
    if directory_exists:
        data.dump_dir.mkdir()
    monkeypatch.setenv("NP_API_KEY", "offline-test-key")
    expected_urls = [f"https://www.neuronpedia.org/api/feature/{SOURCE}/{feature_id}"
                     for feature_id in (42, 75)]
    http_boundary.responses.update(zip(expected_urls, (4.25, 0)))

    result = data.instance.pre_compute_mean_activations(
        data.dump_dir, metadata=data.metadata, master_data_dir=data.cache_dir,
    )

    assert result == data.instance.max_activations == {7: 2.5, 99: 9.0, 42: 4.25, 75: 50}
    assert http_boundary.calls == expected_urls
    assert json.loads(data.cache_path.read_text()) == {"7": 2.5, "99": 9.0, "42": 4.25, "75": 50}


def test_existing_latent_refs_take_priority_over_metadata(data, runtime, http_boundary):
    data.dump_dir.mkdir()
    runtime.pd.DataFrame({"sae_link": [_ref(7)]}).to_parquet(
        data.dump_dir / "latent_data.parquet", index=False,
    )
    original_cache = data.cache_path.read_bytes()

    result = data.instance.pre_compute_mean_activations(
        data.dump_dir, metadata=data.metadata[1:], master_data_dir=data.cache_dir,
    )

    assert result == data.instance.max_activations == {7: 2.5, 99: 9.0}
    assert http_boundary.calls == []
    assert data.cache_path.read_bytes() == original_cache


def test_metadata_cache_hit_needs_no_api_key_or_http(data, http_boundary):
    data.dump_dir.mkdir()
    original_cache = data.cache_path.read_bytes()

    result = data.instance.pre_compute_mean_activations(
        data.dump_dir, metadata=data.metadata[:1], master_data_dir=data.cache_dir,
    )

    assert result == data.instance.max_activations == {7: 2.5, 99: 9.0}
    assert http_boundary.calls == []
    assert data.cache_path.read_bytes() == original_cache


@pytest.mark.parametrize("metadata_kwargs", [{}, {"metadata": []}], ids=["omitted", "empty"])
def test_missing_refs_fail_explicitly_without_http_or_cache_changes(data, http_boundary,
                                                                  metadata_kwargs):
    data.dump_dir.mkdir()
    original_cache = data.cache_path.read_bytes()

    with pytest.raises(ValueError, match="(?i)(sae|ref|metadata|latent)"):
        data.instance.pre_compute_mean_activations(
            data.dump_dir, master_data_dir=data.cache_dir, **metadata_kwargs,
        )

    assert http_boundary.calls == []
    assert data.cache_path.read_bytes() == original_cache
    assert list(data.cache_dir.iterdir()) == [data.cache_path]


def test_disabled_neuronpedia_uses_real_parent_calculation(data, runtime, http_boundary):
    data.dump_dir.mkdir()
    runtime.pd.DataFrame({
        "concept_id": [0, 0, 1], "GemmaScopeSAE_max_act": [2.0, 3.5, 0.0],
    }).to_parquet(data.dump_dir / "latent_data.parquet", index=False)
    original_cache = data.cache_path.read_bytes()

    result = data.instance.pre_compute_mean_activations(
        data.dump_dir, metadata=data.metadata, master_data_dir=data.cache_dir,
        disable_neuronpedia_max_act=True,
    )

    assert result == data.instance.max_activations == {0: 3.5, 1: 50}
    assert http_boundary.calls == []
    assert data.cache_path.read_bytes() == original_cache


def test_metadata_fallback_does_not_hide_directory_permission_errors(data, runtime,
                                                                   http_boundary, monkeypatch):
    data.dump_dir.mkdir()
    original_cache = data.cache_path.read_bytes()
    original_listdir = runtime.sae.os.listdir

    def permission_error(path):
        # Inject only this OS operation's failure; chmod cannot establish a
        # permission error in privileged test runners. Other listdir calls work.
        if Path(path) == data.dump_dir:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return original_listdir(path)

    monkeypatch.setattr(runtime.sae.os, "listdir", permission_error)
    with pytest.raises(PermissionError):
        data.instance.pre_compute_mean_activations(
            data.dump_dir, metadata=data.metadata, master_data_dir=data.cache_dir,
        )
    assert http_boundary.calls == []
    assert data.cache_path.read_bytes() == original_cache


def test_metadata_fallback_does_not_hide_non_directory_path(data, http_boundary):
    data.dump_dir.write_text("This is a file, not an inference directory.")
    original_cache = data.cache_path.read_bytes()

    with pytest.raises(NotADirectoryError):
        data.instance.pre_compute_mean_activations(
            data.dump_dir, metadata=data.metadata, master_data_dir=data.cache_dir,
        )
    assert http_boundary.calls == []
    assert data.cache_path.read_bytes() == original_cache


def test_metadata_fallback_does_not_hide_invalid_parquet(data, runtime, http_boundary):
    data.dump_dir.mkdir()
    (data.dump_dir / "latent_data.parquet").write_bytes(b"This is not a parquet file.")
    original_cache = data.cache_path.read_bytes()

    with pytest.raises(runtime.pa.ArrowInvalid):
        data.instance.pre_compute_mean_activations(
            data.dump_dir, metadata=data.metadata, master_data_dir=data.cache_dir,
        )
    assert http_boundary.calls == []
    assert data.cache_path.read_bytes() == original_cache
