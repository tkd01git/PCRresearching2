#!/usr/bin/env python3
"""Run the full PCR group-testing pipeline and create outputs/final_package.zip."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from _path_setup import add_project_root_to_path

PROJECT_ROOT = add_project_root_to_path()

from src import config  # noqa: E402
from src.io_utils import ensure_dir  # noqa: E402


SCRIPT_DIR = PROJECT_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data generation, validation, methods, comparison, and ZIP packaging.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for smoke tests. Omit for full 50-sample run.")
    parser.add_argument("--pool-size", type=int, default=config.DEFAULT_POOL_SIZE, help="Pool size for sparse methods. Default: 300")
    parser.add_argument("--seed", type=int, default=config.DEFAULT_RANDOM_SEED, help="Base random seed. Default: 1")
    parser.add_argument("--sparse-max-iterations", type=int, default=None, help="Sparse iteration cap for smoke tests. Omit for final run.")
    parser.add_argument("--skip-generate", action="store_true", help="Skip scripts/01_generate_datasets.py")
    parser.add_argument("--skip-validate", action="store_true", help="Skip scripts/02_validate_datasets.py")
    parser.add_argument("--skip-methods", action="store_true", help="Skip scripts/03_run_all_methods.py")
    parser.add_argument("--skip-compare", action="store_true", help="Skip scripts/04_compare_results.py")
    parser.add_argument("--no-save-pooling-design", action="store_true", help="Forwarded to 03_run_all_methods.py")
    parser.add_argument("--continue-on-error", action="store_true", help="Forwarded to 03_run_all_methods.py")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=config.OUTPUTS_DIR / "final_package.zip",
        help="Final ZIP path. Default: outputs/final_package.zip",
    )
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    print("$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)


def create_final_zip(zip_path: Path) -> None:
    ensure_dir(zip_path.parent)
    if zip_path.exists():
        zip_path.unlink()

    include_paths = [
        "README.md",
        "requirements.txt",
        "src",
        "scripts",
        "data/samples",
        "data/summary",
        "results",
    ]

    staging = PROJECT_ROOT / "outputs" / "_final_package_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    for rel in include_paths:
        src = PROJECT_ROOT / rel
        if not src.exists():
            continue
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        else:
            shutil.copy2(src, dst)

    archive_base = zip_path.with_suffix("")
    shutil.make_archive(str(archive_base), "zip", root_dir=staging)
    shutil.rmtree(staging)
    print(f"created_zip={zip_path}")


def main() -> None:
    args = parse_args()

    if not args.skip_generate:
        cmd = [sys.executable, str(SCRIPT_DIR / "01_generate_datasets.py")]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(args.max_samples)]
        run_command(cmd)

    if not args.skip_validate:
        run_command([sys.executable, str(SCRIPT_DIR / "02_validate_datasets.py")])

    if not args.skip_methods:
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "03_run_all_methods.py"),
            "--pool-size",
            str(args.pool_size),
            "--seed",
            str(args.seed),
        ]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(args.max_samples)]
        if args.sparse_max_iterations is not None:
            cmd += ["--sparse-max-iterations", str(args.sparse_max_iterations)]
        if args.no_save_pooling_design:
            cmd += ["--no-save-pooling-design"]
        if args.continue_on_error:
            cmd += ["--continue-on-error"]
        run_command(cmd)

    if not args.skip_compare:
        run_command([sys.executable, str(SCRIPT_DIR / "04_compare_results.py")])

    create_final_zip(args.zip_path)


if __name__ == "__main__":
    main()
