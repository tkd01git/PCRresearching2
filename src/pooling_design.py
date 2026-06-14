"""Pooling matrix designs for baseline and proposed methods."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.io_utils import ensure_dir
from src.priors import get_symptom_count


def pools_to_matrix(pools: list[list[int]], n: int) -> np.ndarray:
    A = np.zeros((len(pools), n), dtype=float)
    for p, members in enumerate(pools):
        if not members:
            continue
        A[p, np.asarray(members, dtype=int)] = 1.0
    return A


def compute_pool_measurements(
    A: np.ndarray,
    x_true: np.ndarray,
    sample_fraction: float = 1.0,
) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    x_true = np.asarray(x_true, dtype=float)
    return float(sample_fraction) * (A @ x_true)


def build_random_pooling_matrix(
    n: int,
    pool_size: int,
    seed: int | None = None,
) -> tuple[np.ndarray, list[list[int]], dict[str, Any]]:
    """Build a random one-membership pooling matrix for Baseline 3."""
    if n <= 0:
        raise ValueError("n must be positive")
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    pools = [order[i : i + pool_size].astype(int).tolist() for i in range(0, n, pool_size)]
    A = pools_to_matrix(pools, n)
    info = {
        "method": "baseline3_random_sparse",
        "n": int(n),
        "pool_size": int(pool_size),
        "num_pools": int(len(pools)),
        "seed": seed,
        "membership_per_person": 1,
    }
    return A, pools, info


def snake_pool_index(rank: int, num_pools: int) -> int:
    if num_pools <= 0:
        raise ValueError("num_pools must be positive")
    cycle = rank // num_pools
    q = rank % num_pools
    if cycle % 2 == 0:
        return q
    return num_pools - 1 - q


def _sort_group_by_risk(indices: np.ndarray, risk_scores: np.ndarray) -> list[int]:
    if len(indices) == 0:
        return []
    local_order = np.argsort(-risk_scores[indices], kind="mergesort")
    return indices[local_order].astype(int).tolist()


def build_proposed_pooling_matrix(
    patient_data: pd.DataFrame,
    risk_scores: np.ndarray,
    neighbor_symptom_scores: np.ndarray,
    pool_size: int,
) -> tuple[np.ndarray, list[list[int]], dict[str, Any]]:
    """Build proposed 4-group + snake pooling matrix.

    Groups:
        A: self symptomatic, neighbor symptomatic
        B: self symptomatic, neighbor not symptomatic
        C: self not symptomatic, neighbor symptomatic
        D: self not symptomatic, neighbor not symptomatic
    """
    n = len(patient_data)
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    risk_scores = np.asarray(risk_scores, dtype=float)
    neighbor_symptom_scores = np.asarray(neighbor_symptom_scores, dtype=float)
    if len(risk_scores) != n or len(neighbor_symptom_scores) != n:
        raise ValueError("risk_scores and neighbor_symptom_scores must have length n")

    num_pools = int(np.ceil(n / pool_size))
    symptom_count = get_symptom_count(patient_data)
    self_symptomatic = symptom_count > 0
    neighbor_symptomatic = neighbor_symptom_scores > 0

    group_A = np.where(self_symptomatic & neighbor_symptomatic)[0]
    group_B = np.where(self_symptomatic & ~neighbor_symptomatic)[0]
    group_C = np.where(~self_symptomatic & neighbor_symptomatic)[0]
    group_D = np.where(~self_symptomatic & ~neighbor_symptomatic)[0]

    ordered_groups = {
        "A": _sort_group_by_risk(group_A, risk_scores),
        "B": _sort_group_by_risk(group_B, risk_scores),
        "C": _sort_group_by_risk(group_C, risk_scores),
        "D": _sort_group_by_risk(group_D, risk_scores),
    }

    pools: list[list[int]] = [[] for _ in range(num_pools)]
    # Reset snake rank for each group so each group is independently spread.
    for group_members in ordered_groups.values():
        for rank, person_idx in enumerate(group_members):
            p = snake_pool_index(rank, num_pools)
            pools[p].append(int(person_idx))

    A = pools_to_matrix(pools, n)
    info = {
        "method": "proposed_4group_snake_sparse",
        "n": int(n),
        "pool_size": int(pool_size),
        "num_pools": int(num_pools),
        "group_A_count": int(len(group_A)),
        "group_B_count": int(len(group_B)),
        "group_C_count": int(len(group_C)),
        "group_D_count": int(len(group_D)),
        "membership_per_person": 1,
    }
    return A, pools, info


def build_multi_membership_pooling_matrix(
    n: int,
    pool_size: int,
    memberships_per_person: int = 2,
    seed: int | None = None,
) -> tuple[np.ndarray, list[list[int]], dict[str, Any]]:
    """Future extension: put each person in multiple randomized pools.

    This is not used by the current baseline/proposed definitions, but is kept
    because one-membership pooling can provide too few equations for sparse
    reconstruction.
    """
    if memberships_per_person < 1:
        raise ValueError("memberships_per_person must be >= 1")
    rng = np.random.default_rng(seed)
    num_pools = int(np.ceil(n * memberships_per_person / pool_size))
    pools: list[list[int]] = [[] for _ in range(num_pools)]
    for rep in range(memberships_per_person):
        order = rng.permutation(n)
        for rank, person_idx in enumerate(order):
            p = (rank + rep) % num_pools
            pools[p].append(int(person_idx))
    A = pools_to_matrix(pools, n)
    info = {
        "method": "multi_membership_random_sparse_extension",
        "n": int(n),
        "pool_size": int(pool_size),
        "num_pools": int(num_pools),
        "seed": seed,
        "membership_per_person": int(memberships_per_person),
        "currently_used": False,
    }
    return A, pools, info


def save_pooling_design(
    A: np.ndarray,
    pools: list[list[int]],
    output_dir: str | Path,
    prefix: str,
    info: dict[str, Any] | None = None,
) -> None:
    output_dir = ensure_dir(output_dir)
    np.save(output_dir / f"{prefix}_A.npy", A)
    pool_rows = [
        {"pool_id": p, "person_index": int(person_idx)}
        for p, members in enumerate(pools)
        for person_idx in members
    ]
    pd.DataFrame(pool_rows).to_csv(output_dir / f"{prefix}_pools.csv", index=False)
    if info is None:
        info = {}
    (output_dir / f"{prefix}_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def positive_pool_mask(s: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.asarray(s, dtype=float) > float(threshold)


def candidate_mask_from_positive_pools(A: np.ndarray, positive_pools: np.ndarray) -> np.ndarray:
    if len(positive_pools) == 0:
        return np.zeros(A.shape[1], dtype=bool)
    candidate = (A[np.asarray(positive_pools, dtype=bool)].sum(axis=0) > 0)
    return np.asarray(candidate, dtype=bool)
