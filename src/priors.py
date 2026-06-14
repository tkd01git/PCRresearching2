"""Risk scores and priors for PCR group-testing methods."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.data_generation import SYMPTOM_TOTAL_COL


def get_person_id_array(population: pd.DataFrame) -> np.ndarray:
    if "person_id" in population.columns:
        return population["person_id"].to_numpy()
    return np.arange(len(population))


def build_person_id_to_index(population: pd.DataFrame) -> dict[Any, int]:
    return {pid: i for i, pid in enumerate(get_person_id_array(population))}


def _detect_contact_columns(contacts: pd.DataFrame) -> tuple[str, str, str | None]:
    possible_i = ["person_i", "i", "source", "person_id_i", "id_i"]
    possible_j = ["person_j", "j", "target", "person_id_j", "id_j"]
    col_i = next((c for c in possible_i if c in contacts.columns), None)
    col_j = next((c for c in possible_j if c in contacts.columns), None)
    if col_i is None or col_j is None:
        raise ValueError(f"Could not detect contact endpoint columns. columns={contacts.columns.tolist()}")
    weight_col = "weight" if "weight" in contacts.columns else None
    return col_i, col_j, weight_col


def build_contact_weight_matrix(
    population: pd.DataFrame,
    contacts: pd.DataFrame,
    *,
    dense: bool = False,
    symmetric: bool = True,
) -> sparse.csr_matrix | np.ndarray:
    """Build an n x n contact weight matrix aligned to population row order.

    Contacts may use original ``person_id`` values rather than row indices. Rows
    with endpoints not present in the population are ignored.
    """
    n = len(population)
    if contacts.empty:
        W = sparse.csr_matrix((n, n), dtype=float)
        return W.toarray() if dense else W

    id_to_idx = build_person_id_to_index(population)
    col_i, col_j, weight_col = _detect_contact_columns(contacts)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for rec in contacts[[col_i, col_j] + ([weight_col] if weight_col else [])].itertuples(index=False, name=None):
        pi, pj = rec[0], rec[1]
        weight = float(rec[2]) if weight_col else 1.0
        if pi not in id_to_idx or pj not in id_to_idx:
            continue
        ii = id_to_idx[pi]
        jj = id_to_idx[pj]
        if ii == jj:
            continue
        rows.append(ii)
        cols.append(jj)
        vals.append(weight)
        if symmetric:
            rows.append(jj)
            cols.append(ii)
            vals.append(weight)

    W = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=float).tocsr()
    W.sum_duplicates()
    return W.toarray() if dense else W


def get_symptom_count(population: pd.DataFrame) -> np.ndarray:
    if SYMPTOM_TOTAL_COL in population.columns:
        return population[SYMPTOM_TOTAL_COL].fillna(0).astype(float).to_numpy()
    symptom_cols = [c for c in population.columns if c.startswith("reported_symptom_")]
    if symptom_cols:
        return population[symptom_cols].fillna(0).astype(float).sum(axis=1).to_numpy()
    return np.zeros(len(population), dtype=float)


def compute_symptom_score(population: pd.DataFrame, *, max_symptoms: float = 3.0) -> np.ndarray:
    symptom_count = get_symptom_count(population)
    denom = max(max_symptoms, 1.0)
    return symptom_count / denom


def compute_neighbor_symptom_score(
    population: pd.DataFrame,
    contacts: pd.DataFrame,
    *,
    max_symptoms: float = 3.0,
    normalize_by_degree: bool = False,
) -> np.ndarray:
    W = build_contact_weight_matrix(population, contacts, dense=False)
    symptom_count = get_symptom_count(population)
    raw = W @ symptom_count
    if normalize_by_degree:
        degree_weight = np.asarray(W.sum(axis=1)).ravel()
        raw = np.divide(raw, degree_weight, out=np.zeros_like(raw, dtype=float), where=degree_weight > 0)
    return np.asarray(raw, dtype=float) / max(max_symptoms, 1.0)


def compute_combined_risk_score(
    population: pd.DataFrame,
    contacts: pd.DataFrame,
    *,
    graph_weight: float = 1.0,
    max_symptoms: float = 3.0,
    normalize_neighbor_by_degree: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute combined risk score.

    Returns:
        combined_score, symptom_score, neighbor_symptom_score
    """
    symptom_score = compute_symptom_score(population, max_symptoms=max_symptoms)
    neighbor_score = compute_neighbor_symptom_score(
        population,
        contacts,
        max_symptoms=max_symptoms,
        normalize_by_degree=normalize_neighbor_by_degree,
    )
    combined = symptom_score + float(graph_weight) * neighbor_score
    return combined, symptom_score, neighbor_score


def compute_mu(combined_score: np.ndarray, *, beta: float = 1.0, clip_min: float = 1e-12) -> np.ndarray:
    """Compute prior weights for sparse reconstruction.

    Larger combined risk score means higher risk. ``mu`` is therefore smaller
    for higher-risk people.
    """
    combined_score = np.asarray(combined_score, dtype=float)
    mu = np.exp(-float(beta) * combined_score)
    return np.clip(mu, clip_min, None)


def rank_by_risk(combined_score: np.ndarray) -> np.ndarray:
    """Return indices sorted from high risk to low risk."""
    return np.argsort(-np.asarray(combined_score, dtype=float), kind="mergesort")


def rank_by_mu(mu: np.ndarray) -> np.ndarray:
    """Return indices sorted from high risk to low risk when lower mu means higher risk."""
    return np.argsort(np.asarray(mu, dtype=float), kind="mergesort")
