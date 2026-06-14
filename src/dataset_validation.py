"""Dataset validation for PCR group-testing samples."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.data_generation import SYMPTOM_TOTAL_COL, TRUE_COL, ensure_analysis_columns
from src.io_utils import list_sample_dirs, read_contacts, read_metadata, read_population, write_csv
from src.priors import build_contact_weight_matrix


def _safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def validate_one_sample(
    sample_dir: str | Path,
    *,
    expected_asymptomatic_positive_rate: float = config.EXPECTED_ASYMPTOMATIC_POSITIVE_RATE,
    tolerance: float = config.ASYMPTOMATIC_RATE_TOLERANCE,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    population = ensure_analysis_columns(read_population(sample_dir))
    contacts = read_contacts(sample_dir)
    metadata = read_metadata(sample_dir)

    total_n = int(len(population))
    y_true = population[TRUE_COL].astype(int).to_numpy()
    symptom_count = population[SYMPTOM_TOTAL_COL].fillna(0).astype(float).to_numpy()
    positive = y_true == 1
    negative = ~positive
    symptomatic = symptom_count > 0

    positive_count = int(positive.sum())
    negative_count = int(negative.sum())
    symptomatic_positive_count = int((positive & symptomatic).sum())
    asymptomatic_positive_count = int((positive & ~symptomatic).sum())
    symptomatic_negative_count = int((negative & symptomatic).sum())
    asymptomatic_negative_count = int((negative & ~symptomatic).sum())

    asym_rate = _safe_rate(asymptomatic_positive_count, positive_count)
    sym_rate = _safe_rate(symptomatic_positive_count, positive_count)
    diff = abs(asym_rate - expected_asymptomatic_positive_rate)
    status = "ok" if diff <= tolerance else "warning"

    try:
        W = build_contact_weight_matrix(population, contacts, dense=False)
        contact_edges = int(W.nnz // 2)
        average_degree = float(W.getnnz(axis=1).mean())
    except Exception:
        contact_edges = int(len(contacts))
        average_degree = float("nan")
        status = "warning"

    return {
        "sample_id": sample_dir.name,
        "source_sample_name": metadata.get("source_sample_name", ""),
        "total_n": total_n,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "symptomatic_positive_count": symptomatic_positive_count,
        "asymptomatic_positive_count": asymptomatic_positive_count,
        "symptomatic_positive_rate": sym_rate,
        "asymptomatic_positive_rate": asym_rate,
        "symptomatic_negative_count": symptomatic_negative_count,
        "asymptomatic_negative_count": asymptomatic_negative_count,
        "positive_rate": _safe_rate(positive_count, total_n),
        "contact_edges": contact_edges,
        "average_degree": average_degree,
        "expected_asymptomatic_positive_rate": expected_asymptomatic_positive_rate,
        "asymptomatic_rate_abs_diff": diff,
        "status": status,
    }


def validate_all_samples(
    samples_dir: str | Path = config.SAMPLES_DIR,
    *,
    expected_asymptomatic_positive_rate: float = config.EXPECTED_ASYMPTOMATIC_POSITIVE_RATE,
    tolerance: float = config.ASYMPTOMATIC_RATE_TOLERANCE,
) -> pd.DataFrame:
    rows = [
        validate_one_sample(
            sample_dir,
            expected_asymptomatic_positive_rate=expected_asymptomatic_positive_rate,
            tolerance=tolerance,
        )
        for sample_dir in list_sample_dirs(samples_dir)
    ]
    return pd.DataFrame(rows)


def write_validation_report(summary_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        text = "# Dataset validation report\n\nNo samples were found.\n"
        output_path.write_text(text, encoding="utf-8")
        return

    ok_count = int((summary_df["status"] == "ok").sum())
    warning_count = int((summary_df["status"] != "ok").sum())
    mean_asym = float(summary_df["asymptomatic_positive_rate"].mean())
    mean_sym = float(summary_df["symptomatic_positive_rate"].mean())
    mean_pos = float(summary_df["positive_count"].mean())
    min_asym = float(summary_df["asymptomatic_positive_rate"].min())
    max_asym = float(summary_df["asymptomatic_positive_rate"].max())

    lines = [
        "# Dataset validation report",
        "",
        "## Summary",
        "",
        f"- sample_count: {len(summary_df)}",
        f"- ok_count: {ok_count}",
        f"- warning_count: {warning_count}",
        f"- mean_positive_count: {mean_pos:.3f}",
        f"- mean_asymptomatic_positive_rate: {mean_asym:.6f}",
        f"- mean_symptomatic_positive_rate: {mean_sym:.6f}",
        f"- min_asymptomatic_positive_rate: {min_asym:.6f}",
        f"- max_asymptomatic_positive_rate: {max_asym:.6f}",
        "",
        "## Policy",
        "",
        "This validation assumes the current canonical dataset is the existing generated dataset copied without symptom normalization.",
        f"The reference asymptomatic-positive rate is {config.EXPECTED_ASYMPTOMATIC_POSITIVE_RATE:.2f} with tolerance {config.ASYMPTOMATIC_RATE_TOLERANCE:.2f}.",
        "",
    ]

    warnings = summary_df[summary_df["status"] != "ok"]
    if not warnings.empty:
        lines += ["## Warning samples", ""]
        for _, row in warnings.iterrows():
            lines.append(
                f"- {row['sample_id']}: asymptomatic_positive_rate={row['asymptomatic_positive_rate']:.6f}, positive_count={int(row['positive_count'])}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_validation_outputs(
    summary_df: pd.DataFrame,
    output_dir: str | Path = config.RESULTS_DIR / "dataset_validation",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_df, output_dir / "dataset_summary.csv")
    write_validation_report(summary_df, output_dir / "dataset_validation_report.md")
