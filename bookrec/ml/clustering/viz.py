"""PCA and activity charts for K-Means cluster defence dashboards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from bookrec.ml.io import save_joblib

# Bins for «how many ratings per user» (inclusive ranges).
N_RATINGS_BIN_RANGES: list[tuple[float, float]] = [
    (3, 5),
    (6, 10),
    (11, 20),
    (21, 50),
    (51, 100),
    (101, float("inf")),
]
N_RATINGS_BIN_LABELS = ["3-5", "6-10", "11-20", "21-50", "51-100", "100+"]


def _bin_n_ratings(value: float) -> int:
    for idx, (lo, hi) in enumerate(N_RATINGS_BIN_RANGES):
        if lo <= value <= hi:
            return idx
    return len(N_RATINGS_BIN_LABELS) - 1


def build_n_ratings_histogram(
    user_features: pd.DataFrame,
    cluster_ids: pd.Series,
) -> dict[str, Any]:
    """Per-cluster histogram of how many ratings each user gave."""
    merged = user_features.copy()
    merged["cluster_id"] = cluster_ids.astype(int).values
    n_bins = len(N_RATINGS_BIN_LABELS)
    by_cluster: dict[str, list[int]] = {}

    for cluster_id, group in merged.groupby("cluster_id"):
        counts = [0] * n_bins
        for n in group["n_ratings"].astype(float):
            counts[_bin_n_ratings(float(n))] += 1
        by_cluster[str(int(cluster_id))] = counts

    return {
        "bin_labels": list(N_RATINGS_BIN_LABELS),
        "by_cluster": by_cluster,
        "n_users": {cid: int(sum(vals)) for cid, vals in by_cluster.items()},
    }


def build_pca_scatter(
    scaled_features: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    feature_columns: list[str] | None = None,
    max_points: int = 2500,
    random_state: int = 42,
    pca_model: PCA | None = None,
) -> tuple[dict[str, Any], PCA | None]:
    """2D PCA projection of scaled K-Means input features, grouped by cluster."""
    if scaled_features.shape[0] < 2:
        return {"explained_variance_ratio": [], "explained_variance_pct": [], "points_by_cluster": {}}, None

    pca = pca_model or PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(scaled_features) if pca_model is None else pca.transform(scaled_features)
    evr = pca.explained_variance_ratio_.tolist()

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(coords))
    if len(indices) > max_points:
        indices = rng.choice(indices, size=max_points, replace=False)

    points_by_cluster: dict[str, list[dict[str, float]]] = {}
    for idx in indices:
        cid = str(int(cluster_ids[idx]))
        points_by_cluster.setdefault(cid, []).append(
            {"x": round(float(coords[idx, 0]), 4), "y": round(float(coords[idx, 1]), 4)}
        )

    return (
        {
            "method": "PCA",
            "n_components": 2,
            "n_points": int(len(indices)),
            "n_total": int(len(coords)),
            "feature_columns": feature_columns or [],
            "explained_variance_ratio": [round(v, 4) for v in evr],
            "explained_variance_pct": [round(v * 100, 1) for v in evr],
            "points_by_cluster": points_by_cluster,
        },
        pca,
    )


def build_clustering_visualizations(
    *,
    scaled_df: pd.DataFrame,
    feature_cols: list[str],
    cluster_ids: np.ndarray,
    user_features: pd.DataFrame,
    max_pca_points: int = 2500,
    pca_model_path: Path | None = None,
) -> dict[str, Any]:
    """Bundle PCA scatter + n_ratings histogram for evaluate_report.json."""
    features = scaled_df[feature_cols].to_numpy(dtype=np.float32)
    pca_scatter, pca_model = build_pca_scatter(
        features,
        cluster_ids,
        feature_columns=feature_cols,
        max_points=max_pca_points,
    )
    if pca_model is not None and pca_model_path is not None:
        save_joblib(pca_model, pca_model_path)
    return {
        "pca_scatter": pca_scatter,
        "n_ratings_histogram": build_n_ratings_histogram(user_features, pd.Series(cluster_ids)),
        "pca_model_path": str(pca_model_path) if pca_model_path else None,
    }
