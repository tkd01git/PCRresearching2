# results directory

This directory stores all outputs produced by the PCR group-testing pipeline.

The canonical input data for all methods is:

```text
data/samples/sample_001/
...
data/samples/sample_050/
```

No method should read `data/raw/` directly during comparison. `data/raw/` is only a source archive for dataset generation.

## Current status

- `results/dataset_validation/` is already populated from the current 50 generated samples.
- `results/pooling_designs/` will be populated by `scripts/03_run_all_methods.py` when pooling-design saving is enabled.
- `results/method_outputs/` will be populated by `scripts/03_run_all_methods.py`.
- `results/comparison/` and `results/figures/` will be populated by `scripts/04_compare_results.py`.

## Execution order

```bash
python scripts/01_generate_datasets.py
python scripts/02_validate_datasets.py
python scripts/03_run_all_methods.py
python scripts/04_compare_results.py
```

Or run everything through:

```bash
python scripts/run_pipeline.py
```
