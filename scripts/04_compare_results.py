#!/usr/bin/env python3
"""Compare four PCR group-testing methods using generated CSV outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _path_setup import add_project_root_to_path

add_project_root_to_path()

from src import config  # noqa: E402
from src.io_utils import ensure_dir, write_csv  # noqa: E402
from src.metrics import summarize_methods  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create method comparison CSV, report, and figures.")
    parser.add_argument(
        "--method-results",
        type=Path,
        default=config.RESULTS_DIR / "method_outputs" / "all_methods_raw.csv",
        help="Default: results/method_outputs/all_methods_raw.csv",
    )
    parser.add_argument(
        "--validation-summary",
        type=Path,
        default=config.RESULTS_DIR / "dataset_validation" / "dataset_summary.csv",
        help="Default: results/dataset_validation/dataset_summary.csv",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=config.RESULTS_DIR / "comparison",
        help="Default: results/comparison",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=config.RESULTS_DIR / "figures",
        help="Default: results/figures",
    )
    return parser.parse_args()


def _fmt_num(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_bar_figure(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["method"].astype(str), summary["mean_total_tests"].astype(float))
    ax.set_ylabel("Mean total tests")
    ax.set_xlabel("Method")
    ax.set_title("Mean total tests by method")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_distribution_figure(raw: pd.DataFrame, output_path: Path) -> None:
    methods = raw["method"].dropna().unique().tolist()
    data = [raw.loc[raw["method"] == method, "total_tests"].dropna().astype(float).to_numpy() for method in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=methods, showmeans=True)
    ax.set_ylabel("Total tests")
    ax.set_xlabel("Method")
    ax.set_title("Total tests distribution by method")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def validation_summary_text(validation_path: Path) -> list[str]:
    if not validation_path.exists():
        return ["- dataset_validation: not found"]
    df = pd.read_csv(validation_path)
    if df.empty:
        return ["- dataset_validation: empty"]
    return [
        f"- validation_sample_count: {df['sample_id'].nunique() if 'sample_id' in df.columns else len(df)}",
        f"- mean_asymptomatic_positive_rate: {df['asymptomatic_positive_rate'].mean():.6f}",
        f"- mean_symptomatic_positive_rate: {df['symptomatic_positive_rate'].mean():.6f}",
        f"- validation_warning_count: {(df['status'] != 'ok').sum() if 'status' in df.columns else 'NA'}",
    ]


def write_report(raw: pd.DataFrame, summary: pd.DataFrame, validation_path: Path, output_path: Path) -> None:
    lines: list[str] = [
        "# Method comparison report",
        "",
        "## Dataset consistency",
        "",
        f"- result_sample_count: {raw['sample_id'].nunique() if 'sample_id' in raw.columns else 'NA'}",
        f"- method_count: {raw['method'].nunique() if 'method' in raw.columns else 'NA'}",
        *validation_summary_text(validation_path),
        "",
        "All method conclusions below are computed from `results/method_outputs/all_methods_raw.csv`.",
        "",
        "## Summary table",
        "",
    ]

    if summary.empty:
        lines.append("No method results were available.")
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return

    cols = [
        "method",
        "sample_count",
        "mean_total_tests",
        "median_total_tests",
        "std_total_tests",
        "min_total_tests",
        "max_total_tests",
        "mean_recall",
        "failure_count",
    ]
    present_cols = [c for c in cols if c in summary.columns]
    lines.append("| " + " | ".join(present_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(present_cols)) + " |")
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(_fmt_num(row[c]) for c in present_cols) + " |")

    best = summary.dropna(subset=["mean_total_tests"]).sort_values("mean_total_tests", kind="mergesort").head(1)
    lines += ["", "## Findings", ""]
    if not best.empty:
        best_row = best.iloc[0]
        lines.append(
            f"- Lowest mean_total_tests: {best_row['method']} ({best_row['mean_total_tests']:.3f})"
        )
    else:
        lines.append("- Lowest mean_total_tests: not available")

    method_means = dict(zip(summary["method"], summary["mean_total_tests"]))
    proposed_name = config.METHOD_NAMES["proposed"]
    if proposed_name in method_means:
        proposed_mean = method_means[proposed_name]
        for key in ["baseline1", "baseline2", "baseline3"]:
            name = config.METHOD_NAMES[key]
            if name in method_means:
                diff = proposed_mean - method_means[name]
                relation = "lower" if diff < 0 else "higher" if diff > 0 else "equal"
                lines.append(
                    f"- Proposed vs {name}: proposed is {relation} by {abs(diff):.3f} mean total tests."
                )
    else:
        lines.append("- Proposed method result was not found.")

    if "status" in raw.columns:
        status_counts = raw["status"].value_counts(dropna=False)
        lines += ["", "## Status counts", ""]
        for status, count in status_counts.items():
            lines.append(f"- {status}: {count}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.method_results.exists():
        raise FileNotFoundError(f"method-results not found: {args.method_results}")
    ensure_dir(args.comparison_dir)
    ensure_dir(args.figures_dir)

    raw = pd.read_csv(args.method_results)
    summary = summarize_methods(raw)
    write_csv(summary, args.comparison_dir / "method_comparison_summary.csv")
    write_report(raw, summary, args.validation_summary, args.comparison_dir / "method_comparison_report.md")

    if not summary.empty:
        write_bar_figure(summary, args.figures_dir / "mean_total_tests_by_method.png")
    if not raw.empty and "method" in raw.columns and "total_tests" in raw.columns:
        write_distribution_figure(raw, args.figures_dir / "total_tests_distribution_by_method.png")

    print(f"comparison_summary={args.comparison_dir / 'method_comparison_summary.csv'}")
    print(f"comparison_report={args.comparison_dir / 'method_comparison_report.md'}")
    print(f"figures_dir={args.figures_dir}")


if __name__ == "__main__":
    main()
