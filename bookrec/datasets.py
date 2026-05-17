"""Discover and load multi-file Goodreads Kaggle shards."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from bookrec.cleaning import load_books_csv, load_ratings_csv

_BOOK_SHARD_RE = re.compile(r"^book(\d+)(k)?-(\d+)(k)?$", re.IGNORECASE)
_RATING_SHARD_RE = re.compile(r"^user_rating_(\d+)_to_(\d+)$", re.IGNORECASE)


def _range_token_to_int(num: str, has_k: str | None) -> int:
    value = int(num)
    if has_k:
        value *= 1000
    return value


def book_shard_sort_key(path: Path) -> tuple[int, str]:
    """Sort shards by starting id (book1-100k, book100k-200k, book2000k-3000k, …)."""
    match = _BOOK_SHARD_RE.match(path.stem)
    if not match:
        return (10**12, path.name)
    start = _range_token_to_int(match.group(1), match.group(2))
    return (start, path.name)


def rating_shard_sort_key(path: Path) -> tuple[int, str]:
    """Sort shards by user id range start (user_rating_0_to_1000, …)."""
    match = _RATING_SHARD_RE.match(path.stem)
    if not match:
        return (10**12, path.name)
    return (int(match.group(1)), path.name)


def discover_book_shards(directory: Path) -> list[Path]:
    """Return sorted book*.csv shard paths (excludes legacy book.csv unless it matches)."""
    directory = Path(directory)
    paths = [p for p in directory.glob("book*.csv") if p.is_file()]
    # Prefer numbered shards; keep legacy single file only when it is the sole match.
    numbered = [p for p in paths if _BOOK_SHARD_RE.match(p.stem)]
    if numbered:
        return sorted(numbered, key=book_shard_sort_key)
    legacy = directory / "book.csv"
    if legacy.is_file():
        return [legacy]
    return sorted(paths, key=book_shard_sort_key)


def discover_rating_shards(directory: Path) -> list[Path]:
    """Return sorted user_rating_*.csv shard paths."""
    directory = Path(directory)
    paths = [p for p in directory.glob("user_rating_*.csv") if p.is_file()]
    if paths:
        return sorted(paths, key=rating_shard_sort_key)
    for name in ("user-rating.csv", "user_rating.csv"):
        legacy = directory / name
        if legacy.is_file():
            return [legacy]
    return []


def _load_shards(
    paths: list[Path],
    loader,
    label: str,
    **read_csv_kw: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not paths:
        raise FileNotFoundError(f"No {label} files found.")

    frames: list[pd.DataFrame] = []
    shard_rows: dict[str, int] = {}
    for i, path in enumerate(paths, start=1):
        print(f"  [{i}/{len(paths)}] {path.name} ...", flush=True)
        part = loader(path, **read_csv_kw)
        shard_rows[path.name] = int(len(part))
        frames.append(part)

    if len(frames) == 1:
        combined = frames[0]
    else:
        combined = pd.concat(frames, ignore_index=True)

    report: dict[str, Any] = {
        "source": "shards" if len(paths) > 1 else "single_file",
        "n_shards": len(paths),
        "shard_files": [p.name for p in paths],
        "rows_per_shard": shard_rows,
        "rows_combined": int(len(combined)),
    }
    return combined, report


def load_books_dataset(
    paths: list[Path] | None = None,
    raw_dir: Path | None = None,
    **read_csv_kw: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one or more book CSV shards into a single DataFrame."""
    if paths is None:
        if raw_dir is None:
            raise ValueError("Provide paths or raw_dir.")
        paths = discover_book_shards(raw_dir)
    return _load_shards(paths, load_books_csv, "books", **read_csv_kw)


def load_ratings_dataset(
    paths: list[Path] | None = None,
    raw_dir: Path | None = None,
    **read_csv_kw: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one or more rating CSV shards into a single DataFrame."""
    if paths is None:
        if raw_dir is None:
            raise ValueError("Provide paths or raw_dir.")
        paths = discover_rating_shards(raw_dir)
    return _load_shards(paths, load_ratings_csv, "ratings", **read_csv_kw)
