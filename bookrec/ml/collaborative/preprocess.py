"""Prepare DS1 interaction splits for Surprise SVD training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from surprise import Dataset, Reader

from bookrec.io_utils import read_table, write_json
from bookrec.paths import MODEL_CF_DIR, PROC_SPLITS


def _resolve_split(path_stem: Path) -> Path:
    for suffix in (".parquet", ".csv"):
        candidate = path_stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return path_stem.with_suffix(".parquet")


def load_cf_splits(
    splits_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CF train/test tables written by bookrec.splits.save_all_splits."""
    base = splits_dir or PROC_SPLITS
    train_path = _resolve_split(base / "cf_train")
    test_path = _resolve_split(base / "cf_test")
    if not train_path.exists():
        raise FileNotFoundError(
            f"CF train split not found at {train_path}. Run data pipeline with splits stage first."
        )
    train = read_table(train_path)
    test = read_table(test_path) if test_path.exists() else train.iloc[0:0].copy()
    return train, test


def to_surprise_dataset(
    interactions: pd.DataFrame,
    *,
    rating_scale: tuple[float, float] = (1.0, 5.0),
) -> tuple[Dataset, Reader]:
    """Convert interactions DataFrame to Surprise Dataset."""
    df = interactions[["user_id", "book_id", "rating"]].copy()
    df["user_id"] = df["user_id"].astype(str)
    df["book_id"] = df["book_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df, reader)
    return data, reader


def prepare_cf_training_data(
    splits_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate splits and write preprocessing report."""
    out = out_dir or MODEL_CF_DIR
    out.mkdir(parents=True, exist_ok=True)
    train, test = load_cf_splits(splits_dir)
    report = {
        "dataset": "ds1_goodreads_2m",
        "task": "collaborative_filtering",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "n_users_train": int(train["user_id"].nunique()),
        "n_books_train": int(train["book_id"].nunique()),
        "rating_scale": [1.0, 5.0],
    }
    write_json(report, out / "preprocess_report.json")
    return report
