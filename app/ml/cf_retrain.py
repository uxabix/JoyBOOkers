"""Merge app ratings into CF train split and retrain Surprise SVD."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models.book import Book
from app.db.models.rating import Rating
from app.db.models.user import User
from app.logging_config import get_logger
from bookrec.io_utils import read_table, write_table
from bookrec.ml.collaborative.train import train_svd

logger = get_logger(__name__)

_CF_COLUMNS = ("user_id", "book_id", "rating")


def export_app_ratings(session: Session) -> pd.DataFrame:
    """Export in-app ratings using CF ids (external user id, source book id)."""
    rows = session.execute(
        select(User.external_id, Book.source_book_id, Rating.score)
        .join(User, Rating.user_id == User.id)
        .join(Book, Rating.book_id == Book.id)
        .where(Rating.source == "app")
    ).all()
    if not rows:
        return pd.DataFrame(columns=list(_CF_COLUMNS))

    df = pd.DataFrame(rows, columns=list(_CF_COLUMNS))
    df["user_id"] = df["user_id"].astype(str)
    df["book_id"] = df["book_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    return df.dropna(subset=["rating"])


def merge_app_ratings_into_cf_train(
    base_df: pd.DataFrame,
    overlay_df: pd.DataFrame,
) -> pd.DataFrame:
    """Append app ratings; overlay wins on duplicate user/book pairs."""
    base = base_df[list(_CF_COLUMNS)].copy()
    base["user_id"] = base["user_id"].astype(str)
    base["book_id"] = base["book_id"].astype(str)
    base["rating"] = pd.to_numeric(base["rating"], errors="coerce")

    if overlay_df.empty:
        return base.dropna(subset=["rating"])

    overlay = overlay_df[list(_CF_COLUMNS)].copy()
    overlay["user_id"] = overlay["user_id"].astype(str)
    overlay["book_id"] = overlay["book_id"].astype(str)
    overlay["rating"] = pd.to_numeric(overlay["rating"], errors="coerce")

    combined = pd.concat([base, overlay], ignore_index=True)
    combined = combined.dropna(subset=["rating"])
    return combined.drop_duplicates(subset=["user_id", "book_id"], keep="last")


def run_cf_retrain(settings: Settings, session: Session) -> dict[str, Any]:
    """Export app ratings, merge into cf_train, retrain SVD, persist artifacts."""
    train_path = settings.cf_train_path
    if not train_path.is_file():
        raise FileNotFoundError(f"CF train split missing: {train_path}")

    base_df = read_table(train_path)
    overlay_df = export_app_ratings(session)
    merged = merge_app_ratings_into_cf_train(base_df, overlay_df)
    write_table(merged, train_path.with_suffix(""))

    logger.info(
        "CF retrain: merged %s app ratings from %s users into %s total rows",
        len(overlay_df),
        int(overlay_df["user_id"].nunique()) if not overlay_df.empty else 0,
        len(merged),
    )

    train_report = train_svd(
        splits_dir=train_path.parent,
        out_dir=settings.cf_model_path.parent,
    )

    return {
        "app_ratings_exported": int(len(overlay_df)),
        "app_users": int(overlay_df["user_id"].nunique()) if not overlay_df.empty else 0,
        "train_rows": int(len(merged)),
        "train_users": int(merged["user_id"].nunique()),
        "train_books": int(merged["book_id"].nunique()),
        "validation_rmse": train_report["validation_metrics"]["rmse"],
        "model_path": str(settings.cf_model_path),
        "train_path": str(train_path),
    }
