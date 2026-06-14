# scripts directory

This directory contains executable entry points for the PCR group-testing pipeline.

## Files

| file | role |
|---|---|
| `_path_setup.py` | Adds the project root to `sys.path` so scripts can import `src` reliably. |
| `01_generate_datasets.py` | Copies existing generated samples into canonical `data/samples/sample_001` style folders. It does not normalize or overwrite symptom columns. |
| `02_validate_datasets.py` | Validates `data/samples` and writes dataset summaries. |
| `03_run_all_methods.py` | Runs Baseline 1, Baseline 2, Baseline 3, and Proposed Method on the same samples. |
| `04_compare_results.py` | Summarizes method outputs and writes comparison reports and figures. |
| `run_pipeline.py` | Runs the whole pipeline and creates `outputs/final_package.zip`. |

## Standard run

```bash
python scripts/run_pipeline.py
```

## Smoke test

```bash
python scripts/run_pipeline.py --max-samples 1 --sparse-max-iterations 2 --no-save-pooling-design
```

The smoke test is only for checking whether files and imports are consistent. For final results, omit `--max-samples` and `--sparse-max-iterations`.

## Canonical data policy

The current canonical data is the existing generated sample set copied without symptom normalization. Therefore, the expected asymptomatic-positive rate is approximately 0.60, not forcibly 2/3.
