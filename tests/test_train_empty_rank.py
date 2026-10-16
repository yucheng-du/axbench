"""Exercise the real training entry point with local fixtures and CPU Gloo.

These are CPU adapters for the script's CUDA/NCCL boundary, not CPU support
for the production launcher. No AxBench module is stubbed. The unused Stanza
evaluation pipeline is a fail-on-use sentinel, and the SAE HTTP boundary has
one fixed Neuronpedia response; all other outbound connections are rejected.
"""

import contextlib
import datetime
import importlib
import json
import os
import pickle
import resource
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()


def _environment(cache_dir):
    env = os.environ.copy()
    for variable, directory in {
        "MPLCONFIGDIR": "matplotlib",
        "NUMBA_CACHE_DIR": "numba",
        "HF_HOME": "huggingface",
        "STANZA_RESOURCES_DIR": "stanza",
        "PYTHONPYCACHEPREFIX": "pycache",
        "TMPDIR": "tmp",
    }.items():
        env.setdefault(variable, str(cache_dir / directory))
        Path(env[variable]).mkdir(parents=True, exist_ok=True)
    env.update(
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    return env


@contextlib.contextmanager
def _offline_connections():
    """Keep the Python HTTP stack offline while allowing local rendezvous."""
    denied = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def check(address):
        if isinstance(address, tuple) and address[0] not in {
            "127.0.0.1", "::1", "localhost"
        }:
            denied.append(repr(address))
            raise AssertionError(f"Unexpected outbound connection: {address!r}")

    def connect(sock, address):
        check(address)
        return original_connect(sock, address)

    def connect_ex(sock, address):
        check(address)
        return original_connect_ex(sock, address)

    with patch.object(socket.socket, "connect", connect), patch.object(
        socket.socket, "connect_ex", connect_ex
    ):
        yield denied


def _build_fixtures(destination):
    with _offline_connections() as denied:
        import pandas as pd
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        vocabulary = [
            "[PAD]", "[UNK]", "[BOS]", "[EOS]", "1", "2", "user",
            "assistant", "bright", "sunshine", "warm", "summer", "cold",
            "rain", "dark", "winter", "green", "forest", "trees", "leaves",
        ]
        backend = Tokenizer(WordLevel(dict(zip(vocabulary, range(len(vocabulary)))),
                                      unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="[PAD]",
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            chat_template=(
                "{{ bos_token }} {% for message in messages %}"
                "{{ message['content'] }} {% endfor %}{{ eos_token }}"
            ),
        )
        model_dir = destination / "tiny-model"
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tokenizer), n_layer=1, n_head=2, n_embd=16,
            n_positions=32, n_ctx=32, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        ))
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        for concept_count in (1, 2):
            generate = destination / f"generate-{concept_count}"
            generate.mkdir()
            metadata = [
                {
                    "concept": concept,
                    "concept_id": concept_id,
                    "ref": f"https://www.neuronpedia.org/test/source/{concept_id}",
                    "concept_genres_map": {concept: ["text"]},
                }
                for concept_id, concept in enumerate(["sunshine", "forest"][:concept_count])
            ]
            _write_jsonl(generate / "metadata.jsonl", metadata)
            rows = []
            for entry in metadata:
                outputs = ["bright sunshine", "warm summer"] if entry["concept_id"] == 0 else [
                    "green forest", "trees leaves"
                ]
                for output in outputs:
                    rows.append({
                        "concept_id": entry["concept_id"], "input": "user ",
                        "output": output, "output_concept": entry["concept"],
                        "category": "positive", "concept_genre": "text",
                    })
            rows.extend({
                "concept_id": -1, "input": "user ", "output": output,
                "output_concept": "EEEEE", "category": "negative", "concept_genre": "text",
            } for output in ["cold rain", "dark winter"])
            pd.DataFrame(rows).to_parquet(generate / "train_data.parquet", index=False)
        assert not denied, denied


class _UnusedPipeline:
    def __call__(self, *args, **kwargs):
        raise AssertionError("The training test must not use the Stanza evaluator")

    def __getattr__(self, name):
        raise AssertionError(f"Unexpected Stanza evaluator access: {name}")


class _TorchCPU:
    """Forward torch operations, adapting only train.main's device boundary."""

    def __init__(self, torch):
        self._torch = torch
        self.cuda = self._CUDA(torch.cuda)

    class _CUDA:
        def __init__(self, cuda):
            self._cuda = cuda

        def set_device(self, device):
            assert device.type == "cpu"

        def __getattr__(self, name):
            return getattr(self._cuda, name)

    def device(self, device):
        assert str(device).startswith("cuda:"), device
        return self._torch.device("cpu")

    def __getattr__(self, name):
        return getattr(self._torch, name)


class _Gloo:
    def __init__(self, dist, result, remove_metadata):
        self._dist = dist
        self._result = result
        self._remove_metadata = remove_metadata

    def init_process_group(self, *, backend, init_method, timeout):
        assert backend == "nccl"
        self._dist.init_process_group(
            backend="gloo", init_method=init_method,
            timeout=datetime.timedelta(seconds=60),
        )
        self._result.update(
            backend=self._dist.get_backend(), rank=self._dist.get_rank(),
            world_size=self._dist.get_world_size(), barriers=0,
        )

    def barrier(self, *args, **kwargs):
        self._result["barriers"] += 1
        # Delete an actually written file after all save_state calls, before
        # rank zero can leave the pre-merge barrier. Real barriers remain in use.
        if self._remove_metadata and self._result["rank"] == 1:
            assert self._remove_metadata.exists()
            self._remove_metadata.unlink()
            self._result["deleted_metadata"] = str(self._remove_metadata)
            self._remove_metadata = None
        return self._dist.barrier(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._dist, name)


class _SAERequests:
    def __init__(self, requests, result):
        self._requests = requests
        self._result = result

    def get(self, url, **kwargs):
        assert url == "https://www.neuronpedia.org/api/feature/test/source/0", url
        self._result["neuronpedia_requests"].append(url)

        class Response:
            def json(self):
                return {"source": {"hfRepoId": None, "hfFolderId": None}}

        return Response()

    def __getattr__(self, name):
        return getattr(self._requests, name)


def _worker(config_path, dump_dir, result_path, remove_metadata):
    result = {"pid": os.getpid(), "neuronpedia_requests": []}
    start = time.monotonic()
    dist = None
    exit_code = 0
    try:
        with _offline_connections() as denied:
            import stanza
            import torch
            import torch.distributed as dist

            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            result["torch_threads"] = torch.get_num_threads()
            result["torch_interop_threads"] = torch.get_num_interop_threads()
            sys.path[:0] = [str(ROOT), str(ROOT / "axbench" / "scripts")]
            with patch.object(stanza, "Pipeline", return_value=_UnusedPipeline()) as pipeline:
                train = importlib.import_module("axbench.scripts.train")
                sae = importlib.import_module("axbench.models.sae")
                result["stanza_constructions"] = pipeline.call_count
                result["train_module"] = train.__file__
                with patch.object(train, "torch", _TorchCPU(torch)), patch.object(
                    train, "dist", _Gloo(dist, result, remove_metadata)
                ), patch.object(sae, "requests", _SAERequests(sae.requests, result)), patch.object(
                    sys, "argv", [str(ROOT / "axbench" / "scripts" / "train.py"),
                                  "--config", str(config_path), "--dump_dir", str(dump_dir)]
                ):
                    train.main()
                assert not denied, denied
                assert pipeline.call_count == 1
            if result["rank"] == 0 and (dump_dir / "train" / "bow").exists():
                import joblib

                bow = dump_dir / "train" / "bow" / "0"
                classifier = joblib.load(bow / "classifier.joblib")
                vectorizer = joblib.load(bow / "vectorizer.joblib")
                result["bow_classes"] = classifier.classes_.tolist()
                result["bow_predictions"] = classifier.predict(vectorizer.transform([
                    "user bright sunshine", "user cold rain"
                ])).tolist()
                assert classifier.coef_.shape[1] == len(vectorizer.vocabulary_)
            result["ok"] = True
    except BaseException as error:  # noqa: BLE001 - report every worker exit to the parent
        result.update(ok=False, error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        if isinstance(error, FileNotFoundError):
            result["missing_file"] = error.filename
        traceback.print_exc()
        exit_code = 1
    finally:
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()
        result["elapsed_seconds"] = time.monotonic() - start
        result["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        result_path.write_text(json.dumps(result, indent=2) + "\n")
    return exit_code


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="session")
def local_fixtures(tmp_path_factory):
    destination = tmp_path_factory.mktemp("train-local-fixtures")
    command = [sys.executable, str(THIS_FILE), "--build-fixtures", str(destination)]
    completed = subprocess.run(command, check=False, env=_environment(destination / "cache"), cwd=ROOT,
                               capture_output=True, text=True, timeout=120)
    (destination / "fixture-build.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return destination


def _run_training(tmp_path, local_fixtures, *, world_size=2, concept_count=2,
                  max_concepts=1, bow=False, resume=False, remove_metadata=False):
    import yaml

    dump_dir = tmp_path / "run"
    shutil.copytree(local_fixtures / f"generate-{concept_count}", dump_dir / "generate")
    models = {"BoW": {"binarize_dataset": True, "bow_penalty": "l2", "bow_C": 1.0}} if bow else {}
    config = {
        "train": {
            "model_name": str(local_fixtures / "tiny-model"), "layer": 0,
            "component": "res", "seed": 42, "use_bf16": False,
            "use_wandb": False, "use_dpo_loss": False, "output_length": 8,
            "max_concepts": max_concepts, "models": models,
        },
        "generate": {"keep_orig_axbench_format": False},
    }
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config))
    existing = {}
    if resume:
        training = dump_dir / "train"
        training.mkdir()
        # Rank 1 is empty in the current partition but has valid prior output.
        # Preserve both files, including the populated rank's skipped state.
        for rank, record in enumerate(_read_jsonl(dump_dir / "generate" / "metadata.jsonl")):
            metadata = training / f"rank_{rank}_metadata.jsonl"
            _write_jsonl(metadata, [record])
            existing[metadata] = metadata.read_bytes()
            with (training / f"train_state.pkl_rank_{rank}").open("wb") as stream:
                pickle.dump({"last_concept_id": record["concept_id"]}, stream)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    processes = []
    log_paths = []
    result_paths = []
    try:
        for rank in range(world_size):
            log_path = tmp_path / f"worker-{rank}.log"
            result_path = tmp_path / f"worker-{rank}.json"
            log_paths.append(log_path)
            result_paths.append(result_path)
            env = _environment(local_fixtures / "cache")
            env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                       LOCAL_RANK=str(rank), WORLD_SIZE=str(world_size))
            command = [sys.executable, str(THIS_FILE), "--worker", str(config_path),
                       str(dump_dir), str(result_path)]
            if remove_metadata:
                command.append(str(dump_dir / "train" / "rank_1_metadata.jsonl"))
            with log_path.open("w") as stream:
                processes.append(subprocess.Popen(command, cwd=ROOT, env=env,
                                                  stdout=stream, stderr=subprocess.STDOUT,
                                                  start_new_session=True))
        deadline = time.monotonic() + 180
        for process in processes:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        pytest.fail("Training workers timed out\n" + "\n".join(
            path.read_text() for path in log_paths
        ))
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
    diagnostics = "\n".join(f"{path}:\n{path.read_text()}" for path in log_paths)
    assert all(path.exists() for path in result_paths), diagnostics
    results = [json.loads(path.read_text()) for path in result_paths]
    for rank, result in enumerate(results):
        assert result.get("backend") == "gloo", diagnostics
        assert result["rank"] == rank and result["world_size"] == world_size, diagnostics
        assert result["barriers"] == 2, diagnostics
        assert result["torch_threads"] == result["torch_interop_threads"] == 1
        assert Path(result["train_module"]).resolve() == ROOT / "axbench" / "scripts" / "train.py"
        assert result["stanza_constructions"] == 1, diagnostics
        assert len(result["neuronpedia_requests"]) == (1 if rank == 0 else 0), diagnostics
        assert processes[rank].returncode == (0 if result["ok"] else 1), diagnostics
    for path, original in existing.items():
        if remove_metadata and path.name == "rank_1_metadata.jsonl":
            continue
        assert path.read_bytes() == original
    return dump_dir / "train", results, diagnostics


def _assert_success(training, results, diagnostics, expected_ids):
    assert all(result["ok"] for result in results), diagnostics
    source = {entry["concept_id"]: entry for entry in _read_jsonl(
        training.parent / "generate" / "metadata.jsonl"
    )}
    assert _read_jsonl(training / "metadata.jsonl") == [source[concept_id] for concept_id in expected_ids]
    config = json.loads((training / "config.json").read_text())
    assert config["layer"] == 0 and config["component"] == "res"


@pytest.mark.parametrize("world_size", [1, 2], ids=["single-rank", "all-ranks-populated"])
def test_populated_ranks_keep_existing_behavior(tmp_path, local_fixtures, world_size):
    training, results, diagnostics = _run_training(
        tmp_path, local_fixtures, world_size=world_size, concept_count=world_size,
        max_concepts=None,
    )
    _assert_success(training, results, diagnostics, list(range(world_size)))


@pytest.mark.parametrize("concept_count,max_concepts", [(2, 1), (1, None)],
                         ids=["selected-concepts", "small-dataset"])
def test_empty_rank_can_finish_training(tmp_path, local_fixtures, concept_count, max_concepts):
    training, results, diagnostics = _run_training(
        tmp_path, local_fixtures, concept_count=concept_count, max_concepts=max_concepts,
    )
    assert [entry["concept_id"] for entry in _read_jsonl(training / "rank_0_metadata.jsonl")] == [0]
    _assert_success(training, results, diagnostics, [0])


def test_resume_preserves_metadata_even_for_currently_empty_rank(tmp_path, local_fixtures):
    training, results, diagnostics = _run_training(tmp_path, local_fixtures, resume=True)
    _assert_success(training, results, diagnostics, [0, 1])


@pytest.mark.parametrize("resume", [False, True], ids=["fresh", "resumed"])
def test_missing_metadata_for_populated_rank_still_fails(tmp_path, local_fixtures, resume):
    training, results, diagnostics = _run_training(
        tmp_path, local_fixtures, max_concepts=None, remove_metadata=True, resume=resume,
    )
    assert results[1]["ok"], diagnostics
    assert not results[0]["ok"], diagnostics
    assert results[0]["error_type"] == "FileNotFoundError", diagnostics
    assert Path(results[0]["missing_file"]) == training / "rank_1_metadata.jsonl"
    assert results[1]["deleted_metadata"] == str(training / "rank_1_metadata.jsonl")
    assert not (training / "metadata.jsonl").exists()
    assert not (training / "config.json").exists()


def test_real_bow_fit_survives_an_empty_rank(tmp_path, local_fixtures):
    training, results, diagnostics = _run_training(tmp_path, local_fixtures, bow=True)
    assert (training / "bow" / "0" / "classifier.joblib").is_file(), diagnostics
    assert (training / "bow" / "0" / "vectorizer.joblib").is_file(), diagnostics
    _assert_success(training, results, diagnostics, [0])
    assert results[0]["bow_classes"] == [0, 1]
    assert results[0]["bow_predictions"] == [1, 0]
    assert not (training / "bow" / "1").exists()


if __name__ == "__main__":
    if sys.argv[1] == "--build-fixtures":
        _build_fixtures(Path(sys.argv[2]))
    elif sys.argv[1] == "--worker":
        config, dump, result = map(Path, sys.argv[2:5])
        remove = Path(sys.argv[5]) if len(sys.argv) > 5 else None
        raise SystemExit(_worker(config, dump, result, remove))
    else:
        raise SystemExit(f"Unknown helper mode: {sys.argv[1]}")
