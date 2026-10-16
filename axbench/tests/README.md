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

## Offline SAE checkpoint loading regression

This test uses a random tiny Llama and a local SAE checkpoint, without downloading
model weights or datasets. Run it from the repository root with pytest installed:

```bash
python -m pytest -q tests/test_sae_load.py
```
