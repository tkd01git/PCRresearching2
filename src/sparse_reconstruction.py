"""Sparse reconstruction and sequential individual testing.

This module never builds pooling matrices. It only receives A, s, x_true, mu,
and optional masks/parameters.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linprog


@dataclass
class SparseReconstructionParams:
    x_max: float = 1.0e8
    tolerance: float = 1.0e-7
    max_iterations: int | None = None
    fallback_to_prior_ranking: bool = True


def _as_params(params: SparseReconstructionParams | dict[str, Any] | None) -> SparseReconstructionParams:
    if params is None:
        return SparseReconstructionParams()
    if isinstance(params, SparseReconstructionParams):
        return params
    return SparseReconstructionParams(**params)


def solve_weighted_sparse_reconstruction(
    A: np.ndarray,
    s: np.ndarray,
    mu: np.ndarray,
    *,
    fixed_values: dict[int, float] | None = None,
    x_max: float = 1.0e8,
    tolerance: float = 1.0e-7,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve min sum_i mu_i x_i subject to Ax=s and 0<=x_i<=x_max.

    Fixed values are added as equality constraints. If the LP is infeasible or
    fails numerically, the returned info has ``success=False``.
    """
    A = np.asarray(A, dtype=float)
    s = np.asarray(s, dtype=float)
    mu = np.asarray(mu, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape
    if s.shape[0] != m:
        raise ValueError("s length must match A rows")
    if mu.shape[0] != n:
        raise ValueError("mu length must match A columns")

    A_eq_parts = [A]
    b_eq_parts = [s]
    if fixed_values:
        fixed_A = np.zeros((len(fixed_values), n), dtype=float)
        fixed_b = np.zeros(len(fixed_values), dtype=float)
        for row, (idx, val) in enumerate(fixed_values.items()):
            fixed_A[row, int(idx)] = 1.0
            fixed_b[row] = float(val)
        A_eq_parts.append(fixed_A)
        b_eq_parts.append(fixed_b)
    A_eq = np.vstack(A_eq_parts) if A_eq_parts else None
    b_eq = np.concatenate(b_eq_parts) if b_eq_parts else None

    bounds = [(0.0, float(x_max)) for _ in range(n)]
    result = linprog(
        c=mu,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"primal_feasibility_tolerance": tolerance, "dual_feasibility_tolerance": tolerance},
    )
    if result.success and result.x is not None:
        return np.asarray(result.x, dtype=float), {
            "success": True,
            "status": int(result.status),
            "message": str(result.message),
            "objective": float(result.fun),
        }
    return np.zeros(n, dtype=float), {
        "success": False,
        "status": int(result.status),
        "message": str(result.message),
        "objective": None,
    }


def _fallback_scores(mu: np.ndarray, tested: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    # Lower mu = higher risk. Use negative mu as score.
    score = -np.asarray(mu, dtype=float).copy()
    score[tested] = -np.inf
    score[~candidate_mask] = -np.inf
    return score


def run_sequential_sparse_reconstruction(
    A: np.ndarray,
    s: np.ndarray,
    x_true: np.ndarray,
    mu: np.ndarray,
    *,
    positive_candidate_mask: np.ndarray | None = None,
    params: SparseReconstructionParams | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sequentially reconstruct and individually test until all positives found.

    The next individual is selected by the largest reconstructed x_hat among
    untested candidates. If LP fails, the fallback ranking uses lower mu first.
    """
    params_obj = _as_params(params)
    A = np.asarray(A, dtype=float)
    s = np.asarray(s, dtype=float)
    x_true = np.asarray(x_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = len(x_true)
    if positive_candidate_mask is None:
        candidate_mask = np.ones(n, dtype=bool)
    else:
        candidate_mask = np.asarray(positive_candidate_mask, dtype=bool)
        if len(candidate_mask) != n:
            raise ValueError("positive_candidate_mask length must match x_true")

    true_positive_mask = x_true > 0
    true_positive_count = int(true_positive_mask.sum())
    detectable_positive_count = int((true_positive_mask & candidate_mask).sum())

    if true_positive_count == 0:
        return {
            "individual_tests": 0,
            "detected_count": 0,
            "true_positive_count": 0,
            "recall": 1.0,
            "tested_indices": [],
            "detected_indices": [],
            "status": "ok_no_positives",
            "lp_failures": 0,
        }
    if detectable_positive_count < true_positive_count:
        # This should not happen for noiseless pooling if negative-pool exclusion
        # is correct. Continue, but recall can never reach 1.
        status_prefix = "warning_candidate_mask_missing_positives"
    else:
        status_prefix = "ok"

    tested = np.zeros(n, dtype=bool)
    detected = np.zeros(n, dtype=bool)
    fixed_values: dict[int, float] = {}
    tested_indices: list[int] = []
    lp_failures = 0
    max_iter = params_obj.max_iterations or int(candidate_mask.sum())

    for _ in range(max_iter):
        if int((detected & true_positive_mask).sum()) >= true_positive_count:
            break
        available = candidate_mask & ~tested
        if not available.any():
            break

        x_hat, info = solve_weighted_sparse_reconstruction(
            A,
            s,
            mu,
            fixed_values=fixed_values,
            x_max=params_obj.x_max,
            tolerance=params_obj.tolerance,
        )
        if info["success"]:
            scores = np.asarray(x_hat, dtype=float).copy()
            scores[tested] = -np.inf
            scores[~candidate_mask] = -np.inf
        else:
            lp_failures += 1
            if not params_obj.fallback_to_prior_ranking:
                return {
                    "individual_tests": int(tested.sum()),
                    "detected_count": int((detected & true_positive_mask).sum()),
                    "true_positive_count": true_positive_count,
                    "recall": int((detected & true_positive_mask).sum()) / true_positive_count,
                    "tested_indices": tested_indices,
                    "detected_indices": np.where(detected & true_positive_mask)[0].astype(int).tolist(),
                    "status": "failed_lp",
                    "lp_failures": lp_failures,
                }
            scores = _fallback_scores(mu, tested, candidate_mask)

        next_idx = int(np.argmax(scores))
        if not np.isfinite(scores[next_idx]):
            break
        tested[next_idx] = True
        tested_indices.append(next_idx)
        true_val = float(x_true[next_idx])
        fixed_values[next_idx] = true_val
        if true_val > 0:
            detected[next_idx] = True

    detected_count = int((detected & true_positive_mask).sum())
    recall = detected_count / true_positive_count if true_positive_count else 1.0
    status = status_prefix if recall >= 1.0 else ("incomplete_recall" if status_prefix == "ok" else f"{status_prefix}_incomplete_recall")
    return {
        "individual_tests": int(tested.sum()),
        "detected_count": detected_count,
        "true_positive_count": true_positive_count,
        "recall": float(recall),
        "tested_indices": tested_indices,
        "detected_indices": np.where(detected & true_positive_mask)[0].astype(int).tolist(),
        "status": status,
        "lp_failures": int(lp_failures),
    }
