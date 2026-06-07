"""Feature engineering for collaborative filtering (Dataset A / DS1 only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bookrec.io_utils import write_json, write_table


def build_interaction_features(
    interactions: pd.DataFrame,
    books: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    """User/book stats, implicit confidence, dense index maps for SVD."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inter = interactions.copy()
    user_idx = {uid: i for i, uid in enumerate(sorted(inter["user_id"].unique()))}
    book_idx = {bid: i for i, bid in enumerate(sorted(inter["book_id"].unique()))}
    inter["user_idx"] = inter["user_id"].map(user_idx)
    inter["book_idx"] = inter["book_id"].map(book_idx)

    user_stats = (
        inter.groupby("user_id")
        .agg(n_ratings=("rating", "count"), mean_rating=("rating", "mean"), std_rating=("rating", "std"))
        .reset_index()
    )
    book_stats = (
        inter.groupby("book_id")
        .agg(n_ratings=("rating", "count"), mean_rating=("rating", "mean"))
        .reset_index()
    )
    if "name" in books.columns and "id" in books.columns:
        rated_ids = set(book_stats["book_id"].unique())
        titles = books[books["id"].isin(rated_ids)][["id", "name"]]
        book_stats = book_stats.merge(titles, left_on="book_id", right_on="id", how="left")

    # Confidence weight for implicit feedback extensions
    inter["confidence"] = 1.0 + 0.1 * inter.groupby("user_id")["rating"].transform("count")

    paths: dict[str, str] = {}
    paths["interactions_indexed"] = str(write_table(inter, out_dir / "interactions_indexed"))
    paths["user_index"] = str(write_table(
        pd.DataFrame({"user_id": list(user_idx.keys()), "user_idx": list(user_idx.values())}),
        out_dir / "user_index",
    ))
    paths["book_index"] = str(write_table(
        pd.DataFrame({"book_id": list(book_idx.keys()), "book_idx": list(book_idx.values())}),
        out_dir / "book_index",
    ))
    paths["user_stats"] = str(write_table(user_stats, out_dir / "user_stats"))
    paths["book_stats"] = str(write_table(book_stats, out_dir / "book_stats"))

    report = {
        "n_interactions": int(len(inter)),
        "n_users": len(user_idx),
        "n_books": len(book_idx),
        "density": float(len(inter) / max(len(user_idx) * len(book_idx), 1)),
        "rating_mean": float(inter["rating"].mean()),
        "paths": paths,
    }
    write_json(report, out_dir / "interactions_features_report.json")
    return report
