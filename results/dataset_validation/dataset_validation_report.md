# Dataset validation report

## Summary

- sample_count: 50
- ok_count: 49
- warning_count: 1
- mean_positive_count: 57.680
- mean_asymptomatic_positive_rate: 0.595330
- mean_symptomatic_positive_rate: 0.404670
- min_asymptomatic_positive_rate: 0.492308
- max_asymptomatic_positive_rate: 0.764706

## Policy

This validation assumes the current canonical dataset is the existing generated dataset copied without symptom normalization.
The reference asymptomatic-positive rate is 0.60 with tolerance 0.15.

## Warning samples

- sample_009: asymptomatic_positive_rate=0.764706, positive_count=51
