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

## Offline SAE activation metadata tests

With the project dependencies plus `pytest`, `stanza`, and `plotnine` installed:

```bash
python -m pytest -q tests/test_sae_activation_metadata.py
```

These tests use the real SAE activation helper, a local random tiny model and
tokenizer, and temporary metadata, parquet, and cache files. External network
access is blocked; the unused Stanza evaluation pipeline fails if called, and
Neuronpedia responses are fixed fixtures. They do not run the full steering
pipeline or require downloaded models or datasets. The test lives outside the
`axbench` package to avoid initializing evaluation during pytest collection.
