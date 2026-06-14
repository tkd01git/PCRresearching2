"""Baseline 3: Random Pooling + Sparse Reconstruction."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.metrics import build_method_result_row, get_x_true
from src.pooling_design import (
    build_random_pooling_matrix,
    candidate_mask_from_positive_pools,
    compute_pool_measurements,
    positive_pool_mask,
    save_pooling_design,
)
from src.priors import compute_combined_risk_score, compute_mu
from src.sparse_reconstruction import run_sequential_sparse_reconstruction


def run_baseline3(
    population: pd.DataFrame,
    contacts: pd.DataFrame,
    sample_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or {}
    n = len(population)
    pool_size = int(params.get("pool_size", config.DEFAULT_POOL_SIZE))
    seed = params.get("seed", config.DEFAULT_RANDOM_SEED)
    if seed is not None:
        # Make per-sample random pooling reproducible but different.
        try:
            sample_num = int(str(sample_id).split("_")[-1])
            seed = int(seed) + sample_num
        except Exception:
            seed = int(seed)
    sample_fraction = float(params.get("sample_fraction", config.DEFAULT_SAMPLE_FRACTION))
    threshold = float(params.get("qpcr_threshold", config.DEFAULT_QPCR_THRESHOLD))
    beta = float(params.get("beta", config.DEFAULT_BETA))
    graph_weight = float(params.get("graph_weight", config.DEFAULT_GRAPH_WEIGHT))
    save_design = bool(params.get("save_pooling_design", True))
    design_root = Path(params.get("pooling_design_dir", config.RESULTS_DIR / "pooling_designs"))

    x_true = get_x_true(population)
    combined_score, _, _ = compute_combined_risk_score(population, contacts, graph_weight=graph_weight)
    mu = compute_mu(combined_score, beta=beta)

    A, pools, info = build_random_pooling_matrix(n=n, pool_size=pool_size, seed=seed)
    s = compute_pool_measurements(A, x_true, sample_fraction=sample_fraction)
    positive_pools = positive_pool_mask(s, threshold=threshold)
    candidate_mask = candidate_mask_from_positive_pools(A, positive_pools)

    if save_design:
        save_pooling_design(A, pools, design_root / sample_id, "baseline3_random", info)

    # Restricting only by candidate mask in the sequential function keeps the
    # original A dimensions and avoids index remapping bugs.
    sparse_result = run_sequential_sparse_reconstruction(
        A,
        s,
        x_true,
        mu,
        positive_candidate_mask=candidate_mask,
        params=params.get("sparse_params"),
    )

    total_tests = int(A.shape[0] + sparse_result["individual_tests"])
    row = build_method_result_row(
        sample_id=sample_id,
        method=config.METHOD_NAMES["baseline3"],
        population=population,
        pool_size=pool_size,
        initial_pool_count=int(A.shape[0]),
        positive_pool_count=int(positive_pools.sum()),
        candidate_count=int(candidate_mask.sum()),
        individual_tests=sparse_result["individual_tests"],
        total_tests=total_tests,
        detected_count=sparse_result["detected_count"],
        recall=sparse_result["recall"],
        status=sparse_result["status"],
        extra={
            "lp_failures": sparse_result.get("lp_failures", 0),
            "pool_tests": int(A.shape[0]),
        },
    )
    return row
