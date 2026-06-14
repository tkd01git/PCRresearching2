"""Baseline 2: Risk Ranking Only."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.metrics import build_method_result_row, get_x_true
from src.priors import compute_combined_risk_score, rank_by_risk


def run_risk_ranking_only(
    x_true: np.ndarray,
    risk_scores: np.ndarray,
    *,
    threshold: float = 0.0,
) -> dict[str, Any]:
    x_true = np.asarray(x_true, dtype=float)
    risk_scores = np.asarray(risk_scores, dtype=float)
    positive_mask = x_true > threshold
    true_positive_count = int(positive_mask.sum())
    if true_positive_count == 0:
        return {
            "individual_tests": 0,
            "total_tests": 0,
            "detected_count": 0,
            "true_positive_count": 0,
            "recall": 1.0,
            "tested_indices": [],
            "status": "ok_no_positives",
        }
    order = rank_by_risk(risk_scores)
    detected = 0
    tested_indices: list[int] = []
    for idx in order:
        idx = int(idx)
        tested_indices.append(idx)
        if positive_mask[idx]:
            detected += 1
            if detected >= true_positive_count:
                break
    individual_tests = len(tested_indices)
    recall = detected / true_positive_count
    return {
        "individual_tests": int(individual_tests),
        "total_tests": int(individual_tests),
        "detected_count": int(detected),
        "true_positive_count": true_positive_count,
        "recall": float(recall),
        "tested_indices": tested_indices,
        "status": "ok" if recall >= 1.0 else "incomplete_recall",
    }


def run_baseline2(
    population: pd.DataFrame,
    contacts: pd.DataFrame,
    sample_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    graph_weight = float(params.get("graph_weight", config.DEFAULT_GRAPH_WEIGHT))
    normalize_neighbor_by_degree = bool(params.get("normalize_neighbor_by_degree", False))
    threshold = float(params.get("qpcr_threshold", config.DEFAULT_QPCR_THRESHOLD))
    combined_score, symptom_score, neighbor_score = compute_combined_risk_score(
        population,
        contacts,
        graph_weight=graph_weight,
        normalize_neighbor_by_degree=normalize_neighbor_by_degree,
    )
    result = run_risk_ranking_only(get_x_true(population), combined_score, threshold=threshold)
    row = build_method_result_row(
        sample_id=sample_id,
        method=config.METHOD_NAMES["baseline2"],
        population=population,
        pool_size=None,
        initial_pool_count=0,
        positive_pool_count=0,
        candidate_count=len(population),
        individual_tests=result["individual_tests"],
        total_tests=result["total_tests"],
        detected_count=result["detected_count"],
        recall=result["recall"],
        status=result["status"],
        extra={
            "risk_score_mean": float(np.mean(combined_score)),
            "risk_score_max": float(np.max(combined_score)) if len(combined_score) else 0.0,
        },
    )
    return row
