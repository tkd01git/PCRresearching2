"""Unified interface for all four methods."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.baseline1 import run_baseline1
from src.baseline2 import run_baseline2
from src.baseline3 import run_baseline3
from src.proposed import run_proposed_method
from src.io_utils import read_contacts, read_population


METHOD_RUNNERS = {
    "baseline1": run_baseline1,
    "baseline2": run_baseline2,
    "baseline3": run_baseline3,
    "proposed": run_proposed_method,
}


def run_selected_method(
    method_name: str,
    population: pd.DataFrame,
    contacts: pd.DataFrame,
    sample_id: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if method_name not in METHOD_RUNNERS:
        raise ValueError(f"Unknown method_name={method_name!r}. Available: {list(METHOD_RUNNERS)}")
    return METHOD_RUNNERS[method_name](population, contacts, sample_id, params or {})


def run_all_methods_for_sample(
    sample_dir: str | Path,
    params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sample_dir = Path(sample_dir)
    params = params or {}
    population = read_population(sample_dir)
    contacts = read_contacts(sample_dir)
    sample_id = sample_dir.name
    method_order = params.get("method_order", ["baseline1", "baseline2", "baseline3", "proposed"])
    rows: list[dict[str, Any]] = []
    for method_name in method_order:
        method_params = dict(params)
        method_params.update(params.get(method_name, {}))
        rows.append(run_selected_method(method_name, population, contacts, sample_id, method_params))
    return rows


def run_all_methods_for_samples(
    sample_dirs: list[str | Path],
    params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        rows.extend(run_all_methods_for_sample(sample_dir, params=params))
    return pd.DataFrame(rows)
