"""Metrics and result-row helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.data_generation import SYMPTOM_TOTAL_COL, TRUE_COL, RNA_COL


def get_x_true(population: pd.DataFrame) -> np.ndarray:
    if RNA_COL in population.columns:
        return population[RNA_COL].fillna(0).astype(float).to_numpy()
    if TRUE_COL not in population.columns:
        raise ValueError("population must have either viral_rna_load or y_true")
    return population[TRUE_COL].fillna(0).astype(float).to_numpy()


def compute_sample_positive_counts(population: pd.DataFrame) -> dict[str, int]:
    y = population[TRUE_COL].astype(int).to_numpy()
    symptom = population[SYMPTOM_TOTAL_COL].fillna(0).astype(float).to_numpy()
    positive = y == 1
    symptomatic = symptom > 0
    return {
        "true_positive_count": int(positive.sum()),
        "asymptomatic_positive_count": int((positive & ~symptomatic).sum()),
        "symptomatic_positive_count": int((positive & symptomatic).sum()),
    }


def build_method_result_row(
    *,
    sample_id: str,
    method: str,
    population: pd.DataFrame,
    pool_size: int | None = None,
    initial_pool_count: int | None = None,
    positive_pool_count: int | None = None,
    candidate_count: int | None = None,
    individual_tests: int | None = None,
    total_tests: int | None = None,
    detected_count: int | None = None,
    recall: float | None = None,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "sample_id": sample_id,
        "method": method,
        "pool_size": pool_size,
        **compute_sample_positive_counts(population),
        "initial_pool_count": initial_pool_count,
        "positive_pool_count": positive_pool_count,
        "candidate_count": candidate_count,
        "individual_tests": individual_tests,
        "total_tests": total_tests,
        "detected_count": detected_count,
        "recall": recall,
        "status": status,
    }
    if extra:
        row.update(extra)
    return row


def summarize_methods(all_methods_df: pd.DataFrame) -> pd.DataFrame:
    if all_methods_df.empty:
        return pd.DataFrame()
    grouped = all_methods_df.groupby("method", dropna=False)
    summary = grouped.agg(
        mean_total_tests=("total_tests", "mean"),
        median_total_tests=("total_tests", "median"),
        std_total_tests=("total_tests", "std"),
        min_total_tests=("total_tests", "min"),
        max_total_tests=("total_tests", "max"),
        mean_individual_tests=("individual_tests", "mean"),
        mean_initial_pool_count=("initial_pool_count", "mean"),
        mean_positive_pool_count=("positive_pool_count", "mean"),
        mean_candidate_count=("candidate_count", "mean"),
        mean_recall=("recall", "mean"),
        failure_count=("status", lambda s: int((s.astype(str).str.startswith("ok") == False).sum())),
        sample_count=("sample_id", "nunique"),
    ).reset_index()
    return summary.sort_values("mean_total_tests", kind="mergesort")
