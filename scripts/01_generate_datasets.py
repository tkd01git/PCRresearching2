#!/usr/bin/env python3
"""Create canonical data/samples from existing generated OpenABM samples.

This script intentionally copies the existing generated samples without symptom
normalization. The current research policy is to use the already-generated
approximately 60% asymptomatic-positive dataset as the canonical dataset.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from _path_setup import add_project_root_to_path

add_project_root_to_path()

from src import config  # noqa: E402
from src.data_generation import copy_existing_generated_samples, write_generation_summary  # noqa: E402
from src.io_utils import ensure_dir, write_csv  # noqa: E402


DEFAULT_SOURCE_ROOT = config.RAW_DATA_DIR / "openabm_seed_1" / "samples_asymptomatic_2of3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy existing generated PCR samples into canonical data/samples format."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Existing generated sample root. Default: data/raw/openabm_seed_1/samples_asymptomatic_2of3",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config.SAMPLES_DIR,
        help="Canonical output sample root. Default: data/samples",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=config.DATA_SUMMARY_DIR,
        help="Directory for sample_manifest.csv and generation_summary.csv. Default: data/summary",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=config.DEFAULT_SAMPLE_COUNT,
        help="Number of samples to copy. Default: 50",
    )
    parser.add_argument(
        "--source-label",
        type=str,
        default="existing_generated_samples_asymptomatic_around_60pct",
        help="Label recorded in metadata.json.",
    )
    parser.add_argument(
        "--no-clear-output",
        action="store_true",
        help="Do not remove the existing output-root before copying.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_root.exists():
        raise FileNotFoundError(
            f"source-root not found: {args.source_root}\n"
            "Expected the data folder created in the previous step, including "
            "data/raw/openabm_seed_1/samples_asymptomatic_2of3."
        )
    if not args.no_clear_output and args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root)
    ensure_dir(args.summary_dir)

    manifest = copy_existing_generated_samples(
        source_root=args.source_root,
        output_root=args.output_root,
        max_samples=args.max_samples,
        source_label=args.source_label,
    )
    write_generation_summary(manifest, args.summary_dir)

    # Explicitly save the same manifest path expected by later scripts.
    write_csv(manifest, args.summary_dir / "sample_manifest.csv")

    print(f"Copied {len(manifest)} samples")
    print(f"source_root={args.source_root}")
    print(f"output_root={args.output_root}")
    if not manifest.empty:
        print(f"mean_asymptomatic_positive_rate={manifest['asymptomatic_positive_rate'].mean():.6f}")
        print(f"mean_symptomatic_positive_rate={manifest['symptomatic_positive_rate'].mean():.6f}")


if __name__ == "__main__":
    main()
