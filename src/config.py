"""Shared configuration for the PCR group-testing pipeline.

The current canonical data folder is expected to be created from the existing
``samples_asymptomatic_2of3`` generated data and copied without symptom
normalization. In that data, the observed mean asymptomatic-positive rate is
about 0.60, not exactly 2/3.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SAMPLES_DIR = DATA_DIR / "samples"
DATA_SUMMARY_DIR = DATA_DIR / "summary"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_N = 3000
DEFAULT_SAMPLE_COUNT = 50
DEFAULT_POOL_SIZE = 300
DEFAULT_RANDOM_SEED = 1
DEFAULT_BETA = 1.0
DEFAULT_GRAPH_WEIGHT = 1.0
DEFAULT_X_MAX = 1.0e8
DEFAULT_QPCR_THRESHOLD = 0.0
DEFAULT_SAMPLE_FRACTION = 1.0

# Existing generated data policy. This is intentionally not forced to 2/3.
EXPECTED_ASYMPTOMATIC_POSITIVE_RATE = 0.60
EXPECTED_SYMPTOMATIC_POSITIVE_RATE = 0.40
ASYMPTOMATIC_RATE_TOLERANCE = 0.15

# If datasets are regenerated later, these can be changed in one place.
TARGET_POSITIVE_SYMPTOMATIC_RATE = EXPECTED_SYMPTOMATIC_POSITIVE_RATE
TARGET_POSITIVE_ASYMPTOMATIC_RATE = EXPECTED_ASYMPTOMATIC_POSITIVE_RATE

DEFAULT_SCHEDULES = [
    [25, 5, 3],
    [25, 8, 3],
    [30, 6, 3],
    [20, 5, 2],
]

METHOD_NAMES = {
    "baseline1": "baseline1_multistage",
    "baseline2": "baseline2_risk_ranking_only",
    "baseline3": "baseline3_random_sparse",
    "proposed": "proposed_4group_snake_sparse",
}
