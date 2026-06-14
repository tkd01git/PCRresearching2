"""I/O helpers for the PCR group-testing pipeline."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SAMPLE_DIR_PATTERN = re.compile(r"sample_(\d{3})$")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict[str, Any], path: str | Path, *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_id_from_index(index: int) -> str:
    return f"sample_{index:03d}"


def sample_index_from_id(sample_id: str) -> int:
    m = SAMPLE_DIR_PATTERN.match(sample_id)
    if not m:
        raise ValueError(f"Invalid sample_id format: {sample_id!r}")
    return int(m.group(1))


def list_sample_dirs(samples_dir: str | Path) -> list[Path]:
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        return []
    sample_dirs = [p for p in samples_dir.iterdir() if p.is_dir() and SAMPLE_DIR_PATTERN.match(p.name)]
    return sorted(sample_dirs, key=lambda p: sample_index_from_id(p.name))


def read_population(sample_dir: str | Path) -> pd.DataFrame:
    path = Path(sample_dir) / "population.csv"
    if not path.exists():
        raise FileNotFoundError(f"population.csv not found: {path}")
    return pd.read_csv(path)


def read_contacts(sample_dir: str | Path) -> pd.DataFrame:
    path = Path(sample_dir) / "contacts.csv"
    if not path.exists():
        raise FileNotFoundError(f"contacts.csv not found: {path}")
    return pd.read_csv(path)


def read_metadata(sample_dir: str | Path) -> dict[str, Any]:
    path = Path(sample_dir) / "metadata.json"
    if not path.exists():
        return {}
    return load_json(path)


def write_metadata(sample_dir: str | Path, metadata: dict[str, Any]) -> None:
    save_json(metadata, Path(sample_dir) / "metadata.json")


def require_columns(df: pd.DataFrame, columns: Iterable[str], *, df_name: str = "dataframe") -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def write_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, **kwargs)
