# Dataset validation report

## Source

This `data/` folder uses the existing generated samples from:

```text
openabm_sample_outputs/seed_1/samples_asymptomatic_2of3/
```

The samples were copied into:

```text
data/samples/sample_001/ ... data/samples/sample_050/
```

`population.csv` and `contacts.csv` were copied as-is. Reported symptom columns were **not normalized or overwritten**.

## Summary

| item | value |
|---|---:|
| sample_count | 50 |
| mean_total_n | 3000.0 |
| mean_positive_count | 57.68 |
| mean_positive_rate | 0.019227 |
| mean_symptomatic_positive_rate | 0.404670 |
| mean_asymptomatic_positive_rate | 0.595330 |
| min_asymptomatic_positive_rate | 0.492308 |
| max_asymptomatic_positive_rate | 0.764706 |
| mean_symptomatic_negative_rate | 0.127521 |

## Interpretation

The existing generated dataset has an average asymptomatic-positive rate of approximately **59.53%**.

This is closer to the previously generated around-60% condition, not a newly normalized 66.7% condition.

## Files

- `data/summary/dataset_summary.csv`: per-sample validation table
- `data/summary/source_existing_generated_validation.csv`: same per-sample validation table for source-traceability
- `data/summary/generation_summary.csv`: aggregate summary
- `data/summary/sample_manifest.csv`: mapping from standardized sample IDs to source directories
