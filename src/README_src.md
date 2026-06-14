# src folder overview

This folder contains reusable functions for the PCR group-testing research pipeline.

## Design rule

- `baseline1.py`, `baseline2.py`, `baseline3.py`, and `proposed.py` contain the four method bodies.
- `methods.py` is only an integration layer.
- `pooling_design.py` creates pooling matrices `A`.
- `sparse_reconstruction.py` never creates `A`; it only solves reconstruction and sequential testing from given `A, s, x_true, mu`.
- The current canonical samples are expected at `data/samples/sample_001` through `data/samples/sample_050`.
- The current data policy is to use the existing generated samples without symptom normalization. The expected asymptomatic-positive rate is about 0.60.

## Main imports

```python
from src.methods import run_all_methods_for_sample
from src.dataset_validation import validate_all_samples
```

## Four method files

- `baseline1.py`: multi-stage hierarchical pooling.
- `baseline2.py`: risk-ranking-only individual testing.
- `baseline3.py`: random pooling + sparse reconstruction.
- `proposed.py`: 4-group + snake pooling + sparse reconstruction.
