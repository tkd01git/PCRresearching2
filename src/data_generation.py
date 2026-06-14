"""Dataset construction helpers.

This module is intentionally conservative: when existing generated samples are
already available, it copies them without changing the symptom columns. Any
normalization or regeneration should be an explicit separate action.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.io_utils import ensure_dir, sample_id_from_index, save_json, write_csv


SYMPTOM_TOTAL_COL = "reported_total_symptom_count"
RNA_COL = "viral_rna_load"
TRUE_COL = "y_true"


def get_reported_symptom_columns(population: pd.DataFrame) -> list[str]:
    """Return reported symptom indicator columns, excluding the total column."""
    return [
        c
        for c in population.columns
        if c.startswith("reported_symptom_") and c != SYMPTOM_TOTAL_COL
    ]


def ensure_analysis_columns(population: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with required analysis columns.

    The existing data has six named symptom columns such as
    ``reported_symptom_0_fever``. This function keeps them and only creates the
    total column if it is absent. It also creates ``viral_rna_load`` if absent.
    """
    df = population.copy()
    if TRUE_COL not in df.columns:
        raise ValueError("population.csv must contain y_true")

    symptom_cols = get_reported_symptom_columns(df)
    if SYMPTOM_TOTAL_COL not in df.columns:
        if symptom_cols:
            df[SYMPTOM_TOTAL_COL] = df[symptom_cols].sum(axis=1).astype(int)
        else:
            df[SYMPTOM_TOTAL_COL] = 0

    if RNA_COL not in df.columns:
        df[RNA_COL] = 0.0
        positive = df[TRUE_COL].astype(int).to_numpy() == 1
        # Deterministic fallback. Prefer preserving existing RNA when present.
        ranks = np.arange(positive.sum())
        loads = 10.0 ** (3.0 + 5.0 * ((ranks % 1000) / max(999, len(ranks) - 1)))
        df.loc[positive, RNA_COL] = loads

    if "log10_viral_rna_load" not in df.columns:
        vals = df[RNA_COL].astype(float).to_numpy()
        df["log10_viral_rna_load"] = np.where(vals > 0, np.log10(vals), 0.0)

    return df


def copy_existing_generated_sample(
    source_sample_dir: str | Path,
    output_sample_dir: str | Path,
    *,
    sample_id: str,
    source_label: str = "existing_generated",
) -> dict[str, Any]:
    """Copy one existing generated sample into the canonical data/samples format.

    ``population.csv`` is not symptom-normalized. Required analysis columns are
    checked/filled only when absent.
    """
    source_sample_dir = Path(source_sample_dir)
    output_sample_dir = Path(output_sample_dir)
    ensure_dir(output_sample_dir)

    population_path = source_sample_dir / "population.csv"
    contacts_path = source_sample_dir / "contacts.csv"
    if not population_path.exists():
        raise FileNotFoundError(f"Missing population.csv: {population_path}")
    if not contacts_path.exists():
        raise FileNotFoundError(f"Missing contacts.csv: {contacts_path}")

    population = pd.read_csv(population_path)
    population = ensure_analysis_columns(population)
    contacts = pd.read_csv(contacts_path)

    population.to_csv(output_sample_dir / "population.csv", index=False)
    contacts.to_csv(output_sample_dir / "contacts.csv", index=False)

    positive = population[TRUE_COL].astype(int) == 1
    symptomatic = population[SYMPTOM_TOTAL_COL].astype(float) > 0
    positive_count = int(positive.sum())
    symptomatic_positive_count = int((positive & symptomatic).sum())
    asymptomatic_positive_count = int((positive & ~symptomatic).sum())
    metadata = {
        "sample_id": sample_id,
        "source": source_label,
        "source_sample_name": source_sample_dir.name,
        "copied_without_symptom_normalization": True,
        "population_file": "population.csv",
        "contacts_file": "contacts.csv",
        "total_n": int(len(population)),
        "positive_count": positive_count,
        "symptomatic_positive_count": symptomatic_positive_count,
        "asymptomatic_positive_count": asymptomatic_positive_count,
        "symptomatic_positive_rate": symptomatic_positive_count / positive_count if positive_count else 0.0,
        "asymptomatic_positive_rate": asymptomatic_positive_count / positive_count if positive_count else 0.0,
        "note": "Existing generated sample copied as-is. Reported symptom columns were not normalized or overwritten.",
    }
    save_json(metadata, output_sample_dir / "metadata.json")
    return metadata


def copy_existing_generated_samples(
    source_root: str | Path,
    output_root: str | Path = config.SAMPLES_DIR,
    *,
    max_samples: int | None = config.DEFAULT_SAMPLE_COUNT,
    source_label: str = "existing_generated",
) -> pd.DataFrame:
    """Copy existing generated sample folders to ``sample_001`` style folders."""
    source_root = Path(source_root)
    output_root = Path(output_root)
    ensure_dir(output_root)
    source_dirs = sorted([p for p in source_root.iterdir() if p.is_dir()])
    if max_samples is not None:
        source_dirs = source_dirs[:max_samples]

    rows: list[dict[str, Any]] = []
    for i, src in enumerate(source_dirs, start=1):
        sample_id = sample_id_from_index(i)
        out = output_root / sample_id
        if out.exists():
            shutil.rmtree(out)
        row = copy_existing_generated_sample(src, out, sample_id=sample_id, source_label=source_label)
        rows.append(row)

    manifest = pd.DataFrame(rows)
    return manifest


def write_generation_summary(manifest: pd.DataFrame, output_dir: str | Path = config.DATA_SUMMARY_DIR) -> None:
    """Write manifest and aggregate generation summary."""
    output_dir = ensure_dir(output_dir)
    write_csv(manifest, output_dir / "sample_manifest.csv")
    if manifest.empty:
        summary = pd.DataFrame([{"sample_count": 0}])
    else:
        summary = pd.DataFrame([
            {
                "sample_count": int(len(manifest)),
                "mean_total_n": float(manifest["total_n"].mean()),
                "mean_positive_count": float(manifest["positive_count"].mean()),
                "mean_symptomatic_positive_rate": float(manifest["symptomatic_positive_rate"].mean()),
                "mean_asymptomatic_positive_rate": float(manifest["asymptomatic_positive_rate"].mean()),
                "copied_without_symptom_normalization": bool(manifest["copied_without_symptom_normalization"].all()),
            }
        ])
    write_csv(summary, output_dir / "generation_summary.csv")
