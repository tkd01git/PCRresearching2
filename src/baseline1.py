"""Baseline 1: multi-stage hierarchical pooling."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.metrics import build_method_result_row, get_x_true


def _partition(indices: list[int] | np.ndarray, group_size: int) -> list[list[int]]:
    indices = [int(i) for i in indices]
    return [indices[i : i + group_size] for i in range(0, len(indices), group_size)]


def run_multistage_pooling(
    x_true: np.ndarray,
    schedule: list[int],
    *,
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Run one multi-stage pooling schedule.

    At each stage, only members of positive pools proceed. After the final stage,
    all remaining candidates are individually tested.
    """
    if not schedule:
        raise ValueError("schedule must not be empty")
    if any(int(s) <= 0 for s in schedule):
        raise ValueError(f"invalid schedule: {schedule}")

    x_true = np.asarray(x_true, dtype=float)
    true_positive_mask = x_true > threshold
    true_positive_count = int(true_positive_mask.sum())

    current = np.arange(len(x_true), dtype=int).tolist()
    total_pool_tests = 0
    initial_pool_count = 0
    positive_pool_count_total = 0
    stage_records: list[dict[str, Any]] = []

    for stage_idx, group_size in enumerate(schedule, start=1):
        pools = _partition(current, int(group_size))
        positive_pools: list[list[int]] = []
        for pool in pools:
            if len(pool) == 0:
                continue
            is_positive = bool(np.any(x_true[pool] > threshold))
            if is_positive:
                positive_pools.append(pool)
        if stage_idx == 1:
            initial_pool_count = len(pools)
        total_pool_tests += len(pools)
        positive_pool_count_total += len(positive_pools)
        stage_records.append(
            {
                "stage": stage_idx,
                "pool_size": int(group_size),
                "pool_count": int(len(pools)),
                "positive_pool_count": int(len(positive_pools)),
                "candidate_count_after_stage": int(sum(len(p) for p in positive_pools)),
            }
        )
        current = [idx for pool in positive_pools for idx in pool]
        if not current:
            break

    candidates = sorted(set(current))
    individual_tests = int(len(candidates))
    detected_count = int(np.sum(x_true[candidates] > threshold)) if candidates else 0
    total_tests = int(total_pool_tests + individual_tests)
    recall = detected_count / true_positive_count if true_positive_count else 1.0

    return {
        "schedule": [int(s) for s in schedule],
        "initial_pool_count": int(initial_pool_count),
        "positive_pool_count": int(positive_pool_count_total),
        "candidate_count": int(len(candidates)),
        "pool_tests": int(total_pool_tests),
        "individual_tests": individual_tests,
        "total_tests": total_tests,
        "detected_count": detected_count,
        "true_positive_count": true_positive_count,
        "recall": float(recall),
        "stage_records": stage_records,
        "status": "ok" if recall >= 1.0 else "incomplete_recall",
    }


def search_best_multistage_schedule(
    x_true: np.ndarray,
    schedules: list[list[int]],
    *,
    threshold: float = 0.0,
) -> dict[str, Any]:
    if not schedules:
        raise ValueError("schedules must not be empty")
    results = [run_multistage_pooling(x_true, schedule, threshold=threshold) for schedule in schedules]
    # Prefer full recall, then smaller total tests.
    results_sorted = sorted(results, key=lambda r: (r["recall"] < 1.0, r["total_tests"]))
    best = results_sorted[0]
    best["all_schedule_results"] = [
        {
            "schedule": r["schedule"],
            "total_tests": r["total_tests"],
            "individual_tests": r["individual_tests"],
            "pool_tests": r["pool_tests"],
            "recall": r["recall"],
        }
        for r in results_sorted
    ]
    return best


def run_baseline1(
    population: pd.DataFrame,
    contacts: pd.DataFrame | None,
    sample_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    x_true = get_x_true(population)
    schedules = params.get("schedules", config.DEFAULT_SCHEDULES)
    threshold = float(params.get("qpcr_threshold", config.DEFAULT_QPCR_THRESHOLD))
    best = search_best_multistage_schedule(x_true, schedules, threshold=threshold)
    row = build_method_result_row(
        sample_id=sample_id,
        method=config.METHOD_NAMES["baseline1"],
        population=population,
        pool_size=best["schedule"][0] if best["schedule"] else None,
        initial_pool_count=best["initial_pool_count"],
        positive_pool_count=best["positive_pool_count"],
        candidate_count=best["candidate_count"],
        individual_tests=best["individual_tests"],
        total_tests=best["total_tests"],
        detected_count=best["detected_count"],
        recall=best["recall"],
        status=best["status"],
        extra={
            "best_schedule": "-".join(map(str, best["schedule"])),
            "pool_tests": best["pool_tests"],
        },
    )
    return row
