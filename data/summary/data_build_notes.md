# Data build notes

This data folder is intentionally based on the existing generated data, not on a newly normalized 2/3 asymptomatic-positive dataset.

Decision:

- Use `openabm_sample_outputs/seed_1/samples_asymptomatic_2of3/` as the source.
- Copy each generated sample into `data/samples/sample_001` through `sample_050`.
- Do not rewrite `population.csv` symptom columns.
- Do not force the asymptomatic-positive rate to exactly 2/3.
- Keep source copies under `data/raw/openabm_seed_1/` for provenance.

The measured average asymptomatic-positive rate is documented in `data/summary/dataset_validation_report.md` and `data/summary/dataset_summary.csv`.
