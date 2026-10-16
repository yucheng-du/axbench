## How to run our script-based tests for released datasets

This script will do basic checks on Concept10, Concept500 and Concept16K datasets.

```bash
python axbench/tests/test_released_datasets.py
```


## How to run our unit tests

Once the dataset test passes, you can run the unit tests for functonal modules.

```bash
python -m unittest discover -s axbench/tests/unit_tests
```

## Offline training regression tests

The standalone training tests require the project dependencies plus `pytest`,
`stanza`, and `plotnine`. They generate a tiny random language model, tokenizer,
and parquet fixtures locally; no released dataset or model download is needed.

```bash
python -m pytest -q tests/test_train_empty_rank.py
```

These tests call the real training entry point with CPU device adapters and
real Gloo processes. The unused Stanza evaluation pipeline fails if invoked,
and the Neuronpedia HTTP boundary supplies a fixed empty SAE source. They test
training metadata behavior, including a real BoW fit, but do not validate
CUDA/NCCL execution. The test file lives outside the `axbench` package so pytest
does not initialize the evaluation pipeline before the worker sets up these
boundaries.
