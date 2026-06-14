#!/usr/bin/env python3
"""Validate canonical PCR datasets in data/samples."""
from __future__ import annotations

import argparse
from pathlib import Path

from _path_setup import add_project_root_to_path

add_project_root_to_path()

from src import config  # noqa: E402
from src.dataset_validation import save_validation_outputs, validate_all_samples  # noqa: E402
from src.io_utils import ensure_dir, write_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical PCR sample datasets.")
    parser.add_argument("--samples-dir", type=Path, default=config.SAMPLES_DIR, help="Default: data/samples")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.RESULTS_DIR / "dataset_validation",
        help="Default: results/dataset_validation",
    )
    parser.add_argument(
        "--data-summary-dir",
        type=Path,
        default=config.DATA_SUMMARY_DIR,
        help="Also mirror dataset_summary.csv into this directory. Default: data/summary",
    )
    parser.add_argument(
        "--expected-asymptomatic-positive-rate",
        type=float,
        default=config.EXPECTED_ASYMPTOMATIC_POSITIVE_RATE,
        help="Reference rate for validation. Current default is 0.60 because existing generated data is used as-is.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=config.ASYMPTOMATIC_RATE_TOLERANCE,
        help="Allowed absolute difference from expected rate. Default: 0.15",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.samples_dir.exists():
        raise FileNotFoundError(f"samples-dir not found: {args.samples_dir}")

    summary = validate_all_samples(
        args.samples_dir,
        expected_asymptomatic_positive_rate=args.expected_asymptomatic_positive_rate,
        tolerance=args.tolerance,
    )
    save_validation_outputs(summary, args.output_dir)

    ensure_dir(args.data_summary_dir)
    write_csv(summary, args.data_summary_dir / "dataset_summary.csv")

    print(f"validated_samples={len(summary)}")
    print(f"output_dir={args.output_dir}")
    if not summary.empty:
        print(f"mean_asymptomatic_positive_rate={summary['asymptomatic_positive_rate'].mean():.6f}")
        print(f"mean_symptomatic_positive_rate={summary['symptomatic_positive_rate'].mean():.6f}")
        print(f"warning_count={(summary['status'] != 'ok').sum()}")


if __name__ == "__main__":
    main()
