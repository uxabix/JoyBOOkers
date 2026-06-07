"""User segmentation features from DS1 interactions (K-Means input)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bookrec.io_utils import write_json, write_table


def build_user_clustering_features(
    interactions: pd.DataFrame,
    out_dir: Path,
    *,
    min_ratings: int = 3,
) -> dict[str, Any]:
    """Aggregate per-user behaviour for K-Means (Dataset A only).

    Features: n_ratings, mean_rating, std_rating, rating_range, activity_level.
    Books are NOT clustered here — only users.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inter = interactions.copy()
    grouped = inter.groupby("user_id")["rating"]
    user_features = grouped.agg(
        n_ratings="count",
        mean_rating="mean",
        std_rating="std",
        min_rating="min",
        max_rating="max",
    ).reset_index()
    user_features["rating_range"] = user_features["max_rating"] - user_features["min_rating"]
    user_features["std_rating"] = user_features["std_rating"].fillna(0.0)

    # Activity tiers for interpretability on defence
    q66 = user_features["n_ratings"].quantile(0.66)
    q33 = user_features["n_ratings"].quantile(0.33)

    def _activity(n: float) -> str:
        if n >= q66:
            return "high"
        if n >= q33:
            return "medium"
        return "low"

    user_features["activity_level"] = user_features["n_ratings"].map(_activity)

    # One-hot activity for clustering (optional numeric encoding)
    for level in ("low", "medium", "high"):
        user_features[f"activity_{level}"] = (user_features["activity_level"] == level).astype(np.float32)

    before = len(user_features)
    user_features = user_features[user_features["n_ratings"] >= min_ratings].copy()
    dropped = before - len(user_features)

    feature_cols = [
        "n_ratings",
        "mean_rating",
        "std_rating",
        "rating_range",
        "activity_low",
        "activity_medium",
        "activity_high",
    ]
    # Simple standardization for K-Means
    matrix = user_features[feature_cols].astype(np.float32)
    mu = matrix.mean()
    sigma = matrix.std().replace(0, 1.0)
    scaled = ((matrix - mu) / sigma).astype(np.float32)
    user_features_scaled = user_features[["user_id"]].join(scaled)

    paths: dict[str, str] = {}
    paths["user_features"] = str(write_table(user_features, out_dir / "user_features"))
    paths["user_features_scaled"] = str(write_table(user_features_scaled, out_dir / "user_features_scaled"))

    report = {
        "entity": "users",
        "source_dataset": "ds1_goodreads_2m",
        "n_users": int(len(user_features)),
        "users_dropped_below_min_ratings": int(dropped),
        "min_ratings": min_ratings,
        "feature_columns": feature_cols,
        "activity_quantiles": {"q33": float(q33), "q66": float(q66)},
        "paths": paths,
    }
    write_json(report, out_dir / "clustering_features_report.json")
    return report
