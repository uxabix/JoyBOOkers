"""Shared I/O helpers for pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: Path, **read_kw: Any) -> pd.DataFrame:
    """Load CSV or Parquet with encoding fallback for CSV."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, **read_kw)
    kw = dict(low_memory=False, encoding="utf-8", on_bad_lines="skip")
    kw.update(read_kw)
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        return pd.read_csv(path, **kw)


def write_table(df: pd.DataFrame, base_path: Path) -> Path:
    """Write parquet; fall back to CSV if pyarrow unavailable."""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    pq = base_path.with_suffix(".parquet")
    try:
        df.to_parquet(pq, index=False)
        return pq
    except Exception:
        csv = base_path.with_suffix(".csv")
        df.to_csv(csv, index=False)
        return csv


def write_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def discover_files(directory: Path, patterns: tuple[str, ...]) -> list[Path]:
    """Glob multiple patterns; return sorted unique paths."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found: dict[str, Path] = {}
    for pattern in patterns:
        for p in directory.glob(pattern):
            if p.is_file():
                found[str(p.resolve())] = p
    return sorted(found.values(), key=lambda p: p.name.lower())
