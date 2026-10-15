"""Regression tests for `train.output_length` being optional (issues #142 / #153).

The demo config (axbench/demo/sweep/simple.yaml) has no `train.output_length`,
which used to crash axbench/scripts/train.py with `int(None)` before any data
was prepared. These tests run the real `train.main()` with the real argument
parsers and the real `prepare_df` / `prepare_df_combined`. Only the outer
boundaries are replaced: process groups / CUDA, model loading, the pruned-SAE
download and the benchmark models' training step. A tiny local tokenizer with a
chat template stands in for the HF tokenizer, so nothing is downloaded.
"""
import copy
import importlib.util
import json
import os
import shutil
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = REPO_ROOT / "axbench"
SCRIPTS_DIR = PACKAGE_DIR / "scripts"
SIMPLE_YAML = PACKAGE_DIR / "demo" / "sweep" / "simple.yaml"

CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}{% set role = 'model' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "<start_of_turn>{{ role }}\n{{ message['content'] }}<end_of_turn>\n"
    "{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)
SUFFIX = "<end_of_turn>"

CONCEPT = "cats"
POSITIVE_ROWS = [
    ("tell me about cats", "cats purr softly"),
    ("describe a kitten", "kitten naps"),
    ("name a pet", "cats"),
]
NEGATIVE_ROWS = [  # (input, output, genre)
    ("write a poem", "roses bloom", "text"),
    ("say hello", "hello there", "text"),
    ("count to three", "one two three", "math"),
]


def _make_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<bos>", "<eos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
    words = {"user", "model", "system", "You", "are", "a", "helpful", "assistant.", "1", "2"}
    for row in POSITIVE_ROWS + NEGATIVE_ROWS:
        for text in row[:2]:
            words.update(text.split())
    vocab = {tok: i for i, tok in enumerate(specials + sorted(words))}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
        pad_token="<pad>", unk_token="<unk>",
        additional_special_tokens=["<start_of_turn>", "<end_of_turn>"],
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def _torch_stand_in():
    """Import-time stand-in for torch, used only when torch is not installed.

    train.main() never touches tensors on the paths exercised here; it only needs
    these names to exist (and CUDA / distributed calls are patched out anyway).
    """
    torch = types.ModuleType("torch")
    torch.no_grad = lambda: (lambda fn: fn)
    torch.device = lambda name: name
    torch.cuda = SimpleNamespace(set_device=lambda device: None, empty_cache=lambda: None)
    torch.bfloat16 = "bfloat16"
    torch.nn = types.ModuleType("torch.nn")
    torch.distributed = types.ModuleType("torch.distributed")
    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.DataLoader = object
    return {
        "torch": torch, "torch.nn": torch.nn, "torch.distributed": torch.distributed,
        "torch.utils": torch.utils, "torch.utils.data": torch.utils.data,
    }


def _load_train_module():
    """Import axbench/scripts/train.py without running axbench/__init__.py.

    The package __init__ imports every model and scripts/evaluate.py, which builds
    a stanza pipeline at import time. A bare package object pointing at the real
    directory lets train.py import the real submodules it uses instead.
    """
    package = types.ModuleType("axbench")
    package.__path__ = [str(PACKAGE_DIR)]
    sae_stub = types.ModuleType("axbench.models.sae")

    def save_pruned_sae(*args, **kwargs):
        raise RuntimeError("pruned SAE download is disabled in tests")

    sae_stub.save_pruned_sae = save_pruned_sae
    # Resolve transformers' names before any torch stand-in is installed, so its
    # torch availability check sees the real environment.
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed  # noqa: F401
    with mock.patch.dict(sys.modules), mock.patch.object(sys, "path", [str(SCRIPTS_DIR)] + sys.path):
        for name in list(sys.modules):
            if name == "axbench" or name.startswith(("axbench.", "args")):
                del sys.modules[name]
        if importlib.util.find_spec("torch") is None:
            sys.modules.update(_torch_stand_in())
        sys.modules["axbench"] = package
        sys.modules["axbench.models.sae"] = sae_stub
        spec = importlib.util.spec_from_file_location("axbench_train_under_test", SCRIPTS_DIR / "train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, package


class _FakeLM:
    config = SimpleNamespace(hidden_size=8)

    def eval(self):
        return self

    def to(self, device):
        return self


class _FakeBenchmarkModel:
    """Records the DataFrame handed to `train()` instead of training."""

    records = None

    def __init__(self, model, tokenizer, **kwargs):
        self.ax = SimpleNamespace(to=lambda dtype: None)

    def make_model(self, **kwargs):
        pass

    def train(self, df, **kwargs):
        self.records[type(self).__name__] = df.copy()

    def save(self, dump_dir, model_name=None):
        pass


def _refuse_network(*args, **kwargs):
    raise RuntimeError("network access is disabled in tests")


class TestTrainOutputLength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.package = _load_train_module()
        cls.tokenizer = _make_tokenizer()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        data_dir = self.tmp / "dump" / "generate"
        data_dir.mkdir(parents=True)
        rows = [
            {"input": i, "output": o, "output_concept": CONCEPT, "concept_genre": "text",
             "category": "positive", "concept_id": 0}
            for i, o in POSITIVE_ROWS
        ] + [
            {"input": i, "output": o, "output_concept": "EEEEE", "concept_genre": g,
             "category": "negative", "concept_id": -1}
            for i, o, g in NEGATIVE_ROWS
        ]
        pd.DataFrame(rows).to_parquet(data_dir / "train_data.parquet")
        with open(data_dir / "metadata.jsonl", "w") as f:
            f.write(json.dumps({"concept_id": 0, "concept": CONCEPT,
                                "concept_genres_map": {CONCEPT: ["text"]}}) + "\n")
        with open(SIMPLE_YAML) as f:
            self.simple_config = yaml.safe_load(f)
        self.records = {}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_config(self, config):
        path = self.tmp / "config.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(config, f)
        return path

    def _run_main(self, config_path, *extra_argv):
        train = self.train
        fake_dist = SimpleNamespace(
            init_process_group=lambda **kwargs: None, get_rank=lambda: 0,
            get_world_size=lambda: 1, barrier=lambda: None, destroy_process_group=lambda: None,
        )
        recorder = type("Recorder", (_FakeBenchmarkModel,), {"records": self.records})
        for name in ("LsReFT", "LinearProbe", "HyperSteer"):
            setattr(self.package, name, type(name, (recorder,), {}))
        argv = ["train.py", "--config", str(config_path), "--dump_dir", str(self.tmp / "dump"), *extra_argv]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "LOCAL_RANK": "0"}), \
                mock.patch.object(socket.socket, "connect", _refuse_network), \
                mock.patch.object(train, "dist", fake_dist), \
                mock.patch.object(train.torch.cuda, "set_device", lambda device: None), \
                mock.patch.object(train.torch.cuda, "empty_cache", lambda: None), \
                mock.patch.object(train, "AutoTokenizer",
                                  SimpleNamespace(from_pretrained=lambda *a, **k: self.tokenizer)), \
                mock.patch.object(train, "AutoModelForCausalLM",
                                  SimpleNamespace(from_pretrained=lambda *a, **k: _FakeLM())), \
                mock.patch.object(train, "prepare_df", wraps=train.prepare_df) as prepare_df_spy:
            train.main()
        return prepare_df_spy

    def _passed_output_lengths(self, spy):
        return [call.kwargs["output_length"] for call in spy.call_args_list]

    def test_demo_config_without_train_output_length(self):
        self.assertNotIn("output_length", self.simple_config["train"])

        spy = self._run_main(SIMPLE_YAML)

        self.assertEqual(self._passed_output_lengths(spy), [None, None])
        self.assertEqual(set(self.records), {"LinearProbe", "LsReFT"})

        # LinearProbe binarizes: positives of the concept vs. negatives of its genre.
        probe_df = self.records["LinearProbe"]
        self.assertEqual(list(probe_df["labels"]), [1, 1, 1, 0, 0])
        examples = POSITIVE_ROWS + [row[:2] for row in NEGATIVE_ROWS if row[2] == "text"]
        for combined, (inp, out) in zip(probe_df["input"], examples):
            self.assertTrue(combined.startswith(f"<start_of_turn> user {inp} <end_of_turn>"))
            self.assertIn(f"<start_of_turn> model {out} <end_of_turn>", combined)

        # LsReFT trains on positives plus same-genre negatives with chat-formatted inputs.
        reft_df = self.records["LsReFT"]
        self.assertEqual(list(reft_df["category"]), ["positive"] * 3 + ["negative"] * 2)
        self.assertEqual(
            list(reft_df["input"]),
            [f"<start_of_turn> user {inp} <end_of_turn> <start_of_turn> model" for inp, _ in examples],
        )
        # simple.yaml keeps the original AxBench format: outputs get no suffix.
        self.assertEqual(list(reft_df["output"]), [out for _, out in examples])

    def test_output_suffix_without_orig_axbench_format(self):
        config = copy.deepcopy(self.simple_config)
        config["generate"]["keep_orig_axbench_format"] = False

        self._run_main(self._write_config(config))

        reft_df = self.records["LsReFT"]
        expected = [out for _, out in POSITIVE_ROWS] + [o for _, o, g in NEGATIVE_ROWS if g == "text"]
        self.assertEqual(list(reft_df["output"]), [out + SUFFIX for out in expected])

    def test_explicit_output_length_from_yaml_and_cli(self):
        config = copy.deepcopy(self.simple_config)
        config["train"]["output_length"] = 64
        config_path = self._write_config(config)

        spy = self._run_main(config_path)
        self.assertEqual(self._passed_output_lengths(spy), [64, 64])

        # Drop the resume state of the first run so the concept is trained again.
        shutil.rmtree(self.tmp / "dump" / "train")
        self.records.clear()
        spy = self._run_main(config_path, "--output_length", "128")
        self.assertEqual(self._passed_output_lengths(spy), [128, 128])
        self.assertEqual(set(self.records), {"LinearProbe", "LsReFT"})

    def test_hypersteer_uses_generate_output_length(self):
        config = copy.deepcopy(self.simple_config)
        config["generate"]["output_length"] = 3
        config["train"] = {
            "model_name": "meta-llama/Llama-3.1-8B-Instruct", "layer": 20, "component": "res",
            "seed": 42, "use_bf16": False, "output_length": 99,
            "models": {"HyperSteer": {"binarize_dataset": False, "train_on_negative": False,
                                      "intervention_type": "addition"}},
        }

        spy = self._run_main(self._write_config(config))

        spy.assert_not_called()
        # The suffix is only appended when the output is shorter than
        # generate.output_length (3 tokens), not train.output_length (99).
        self.assertEqual(
            list(self.records["HyperSteer"]["output"]),
            ["cats purr softly", "kitten naps" + SUFFIX, "cats" + SUFFIX],
        )


if __name__ == "__main__":
    unittest.main()
