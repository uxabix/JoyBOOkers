"""Train / validation / test split utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bookrec.io_utils import write_json, write_table


def _stratified_split_indices(
    labels: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class random split indices."""
    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_test - n_val
        if n_train < 1:
            n_train = max(1, n - n_test)
            n_val = 0
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def split_interactions_per_user(
    interactions: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
    min_user_ratings: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Hold out a fraction of each user's ratings for CF evaluation."""
    rng = np.random.default_rng(random_state)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    skipped_users = 0
    for user_id, grp in interactions.groupby("user_id"):
        if len(grp) < min_user_ratings:
            train_parts.append(grp)
            skipped_users += 1
            continue
        n_test = max(1, int(len(grp) * test_ratio))
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        test_parts.append(interactions.loc[test_idx])
        train_parts.append(interactions.loc[train_idx])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else interactions.iloc[0:0]
    test = pd.concat(test_parts, ignore_index=True) if test_parts else interactions.iloc[0:0]
    report = {
        "split": "per_user_holdout",
        "test_ratio": test_ratio,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "users_skipped_single_rating": skipped_users,
    }
    return train, test, report


def split_interactions_temporal(
    interactions: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """If timestamps exist, hold out most recent interactions per user."""
    if timestamp_col not in interactions.columns:
        return split_interactions_per_user(interactions, test_ratio=test_ratio)

    work = interactions.copy()
    work["_ts"] = pd.to_numeric(work[timestamp_col], errors="coerce")
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, grp in work.groupby("user_id"):
        grp = grp.sort_values("_ts", na_position="first")
        n_test = max(1, int(len(grp) * test_ratio))
        test_parts.append(grp.iloc[-n_test:])
        train_parts.append(grp.iloc[:-n_test] if len(grp) > n_test else grp.iloc[0:0])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test, {"split": "temporal_per_user", "test_ratio": test_ratio, "train_rows": len(train), "test_rows": len(test)}


def split_nlp_reviews(
    reviews: pd.DataFrame,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    random_state: int = 42,
    label_col: str = "sentiment_label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stratified split for sentiment classification (DS4 only)."""
    df = reviews.reset_index(drop=True)
    labels = df[label_col].to_numpy()
    train_i, val_i, test_i = _stratified_split_indices(labels, test_ratio, val_ratio, random_state)
    train = df.iloc[train_i].copy()
    val = df.iloc[val_i].copy()
    test = df.iloc[test_i].copy()
    report = {
        "split": "stratified_by_sentiment",
        "test_ratio": test_ratio,
        "val_ratio": val_ratio,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "independent_nlp_corpus": True,
    }
    return train, val, test, report


def split_clustering_holdout(
    book_ids: pd.Series,
    holdout_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Simple book holdout for cluster evaluation."""
    rng = np.random.default_rng(random_state)
    ids = book_ids.drop_duplicates().to_numpy()
    rng.shuffle(ids)
    n_hold = max(1, int(len(ids) * holdout_ratio))
    holdout = set(ids[:n_hold])
    mask = book_ids.isin(holdout)
    return book_ids[~mask], book_ids[mask], {"holdout_ratio": holdout_ratio, "n_holdout": len(holdout)}


def save_all_splits(
    interactions: pd.DataFrame | None,
    reviews: pd.DataFrame | None,
    out_dir: Path,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Write CF and NLP splits to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"paths": {}}

    if interactions is not None and len(interactions):
        train, test, cf_report = split_interactions_per_user(
            interactions, test_ratio=0.2, random_state=random_state
        )
        report["cf"] = cf_report
        report["paths"]["cf_train"] = str(write_table(train, out_dir / "cf_train"))
        report["paths"]["cf_test"] = str(write_table(test, out_dir / "cf_test"))

    if reviews is not None and len(reviews):
        train, val, test, nlp_report = split_nlp_reviews(reviews, random_state=random_state)
        report["nlp"] = nlp_report
        report["paths"]["nlp_train"] = str(write_table(train, out_dir / "nlp_train"))
        report["paths"]["nlp_val"] = str(write_table(val, out_dir / "nlp_val"))
        report["paths"]["nlp_test"] = str(write_table(test, out_dir / "nlp_test"))

    write_json(report, out_dir / "splits_report.json")
    return report
