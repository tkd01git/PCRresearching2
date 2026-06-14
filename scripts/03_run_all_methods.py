#!/usr/bin/env python3
"""Run Baseline 1, Baseline 2, Baseline 3, and Proposed Method on canonical samples."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from _path_setup import add_project_root_to_path

add_project_root_to_path()

from src import config  # noqa: E402
from src.io_utils import ensure_dir, list_sample_dirs, write_csv  # noqa: E402
from src.methods import run_all_methods_for_sample, run_selected_method  # noqa: E402


METHOD_TO_FILENAME = {
    config.METHOD_NAMES["baseline1"]: "baseline1_multistage.csv",
    config.METHOD_NAMES["baseline2"]: "baseline2_risk_ranking_only.csv",
    config.METHOD_NAMES["baseline3"]: "baseline3_random_sparse.csv",
    config.METHOD_NAMES["proposed"]: "proposed_method.csv",
}


def parse_schedule(text: str) -> list[int]:
    vals = [v.strip() for v in text.replace("-", ",").split(",") if v.strip()]
    if not vals:
        raise argparse.ArgumentTypeError("schedule must contain at least one integer")
    try:
        schedule = [int(v) for v in vals]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid schedule: {text}") from exc
    if any(v <= 0 for v in schedule):
        raise argparse.ArgumentTypeError(f"schedule values must be positive: {text}")
    return schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all PCR group-testing methods.")
    parser.add_argument("--samples-dir", type=Path, default=config.SAMPLES_DIR, help="Default: data/samples")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.RESULTS_DIR / "method_outputs",
        help="Default: results/method_outputs",
    )
    parser.add_argument(
        "--pooling-design-dir",
        type=Path,
        default=config.RESULTS_DIR / "pooling_designs",
        help="Default: results/pooling_designs",
    )
    parser.add_argument("--pool-size", type=int, default=config.DEFAULT_POOL_SIZE, help="Pool size for Baseline 3 and Proposed. Default: 300")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_RANDOM_SEED, help="Base random seed. Default: 1")
    parser.add_argument("--beta", type=float, default=config.DEFAULT_BETA, help="Prior mu beta. Default: 1.0")
    parser.add_argument("--graph-weight", type=float, default=config.DEFAULT_GRAPH_WEIGHT, help="Neighbor symptom weight. Default: 1.0")
    parser.add_argument("--qpcr-threshold", type=float, default=config.DEFAULT_QPCR_THRESHOLD, help="Pool/individual positivity threshold. Default: 0.0")
    parser.add_argument("--sample-fraction", type=float, default=config.DEFAULT_SAMPLE_FRACTION, help="Pool measurement multiplier. Default: 1.0")
    parser.add_argument(
        "--schedule",
        dest="schedules",
        action="append",
        type=parse_schedule,
        help="Candidate multistage schedule. Can be repeated. Example: --schedule 25,5,3",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline1", "baseline2", "baseline3", "proposed"],
        default=["baseline1", "baseline2", "baseline3", "proposed"],
        help="Methods to run. Default: all four methods.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit sample count for smoke tests.")
    parser.add_argument(
        "--sparse-max-iterations",
        type=int,
        default=None,
        help="Limit sequential sparse individual tests. Use only for smoke tests; omit for final runs.",
    )
    parser.add_argument(
        "--no-save-pooling-design",
        action="store_true",
        help="Do not save A matrices and pool CSVs. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed method rows instead of stopping at the first error.",
    )
    return parser.parse_args()


def build_params(args: argparse.Namespace) -> dict[str, Any]:
    sparse_params: dict[str, Any] = {}
    if args.sparse_max_iterations is not None:
        sparse_params["max_iterations"] = int(args.sparse_max_iterations)

    params: dict[str, Any] = {
        "method_order": args.methods,
        "pool_size": int(args.pool_size),
        "seed": int(args.seed),
        "beta": float(args.beta),
        "graph_weight": float(args.graph_weight),
        "qpcr_threshold": float(args.qpcr_threshold),
        "sample_fraction": float(args.sample_fraction),
        "save_pooling_design": not args.no_save_pooling_design,
        "pooling_design_dir": args.pooling_design_dir,
    }
    if args.schedules:
        params["schedules"] = args.schedules
    if sparse_params:
        params["sparse_params"] = sparse_params
    return params


def failed_row(sample_id: str, method_key: str, exc: Exception) -> dict[str, Any]:
    method_name = config.METHOD_NAMES.get(method_key, method_key)
    return {
        "sample_id": sample_id,
        "method": method_name,
        "pool_size": None,
        "true_positive_count": None,
        "asymptomatic_positive_count": None,
        "symptomatic_positive_count": None,
        "initial_pool_count": None,
        "positive_pool_count": None,
        "candidate_count": None,
        "individual_tests": None,
        "total_tests": None,
        "detected_count": None,
        "recall": None,
        "status": "failed",
        "error": repr(exc),
    }


def main() -> None:
    args = parse_args()
    if not args.samples_dir.exists():
        raise FileNotFoundError(f"samples-dir not found: {args.samples_dir}")
    ensure_dir(args.output_dir)
    ensure_dir(args.pooling_design_dir)

    sample_dirs = list_sample_dirs(args.samples_dir)
    if args.max_samples is not None:
        sample_dirs = sample_dirs[: args.max_samples]
    if not sample_dirs:
        raise RuntimeError(f"No sample_XXX directories found in {args.samples_dir}")

    params = build_params(args)
    rows: list[dict[str, Any]] = []

    for sample_dir in tqdm(sample_dirs, desc="Running methods", unit="sample"):
        if args.continue_on_error:
            from src.io_utils import read_contacts, read_population  # local import to keep path setup explicit

            population = read_population(sample_dir)
            contacts = read_contacts(sample_dir)
            for method_key in args.methods:
                try:
                    rows.append(run_selected_method(method_key, population, contacts, sample_dir.name, params))
                except Exception as exc:  # noqa: BLE001 - deliberate pipeline logging
                    rows.append(failed_row(sample_dir.name, method_key, exc))
        else:
            rows.extend(run_all_methods_for_sample(sample_dir, params=params))

    all_df = pd.DataFrame(rows)
    write_csv(all_df, args.output_dir / "all_methods_raw.csv")

    for method_name, filename in METHOD_TO_FILENAME.items():
        method_df = all_df[all_df["method"] == method_name].copy()
        write_csv(method_df, args.output_dir / filename)

    print(f"samples_processed={len(sample_dirs)}")
    print(f"rows_written={len(all_df)}")
    print(f"output_dir={args.output_dir}")
    if "status" in all_df.columns:
        print("status_counts=")
        print(all_df["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
