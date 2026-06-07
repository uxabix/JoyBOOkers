"""Train K-Means on DS1 user behaviour features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from bookrec.io_utils import write_json, write_table
from bookrec.ml.clustering.preprocess import load_user_feature_matrix
from bookrec.ml.io import save_joblib
from bookrec.ml.metrics import safe_silhouette
from bookrec.paths import MODEL_CLUSTERING_DIR, PROC_FEATURES


def _pick_k(
    features: np.ndarray,
    k_range: range,
    random_state: int,
) -> tuple[int, dict[int, float]]:
    scores: dict[int, float] = {}
    best_k = k_range.start
    best_score = -1.0
    for k in k_range:
        if k < 2 or k >= len(features):
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(features)
        sil = safe_silhouette(features, labels)
        if sil is not None:
            scores[k] = sil
            if sil > best_score:
                best_score = sil
                best_k = k
    return best_k, scores


def train_user_clusters(
    *,
    features_dir: Path | None = None,
    out_dir: Path | None = None,
    n_clusters: int | None = None,
    k_min: int = 3,
    k_max: int = 12,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit K-Means; auto-select k via silhouette when n_clusters is None."""
    out = Path(out_dir or MODEL_CLUSTERING_DIR)
    out.mkdir(parents=True, exist_ok=True)

    scaled_df, feature_cols = load_user_feature_matrix(features_dir or (PROC_FEATURES / "clustering"))
    features = scaled_df[feature_cols].to_numpy(dtype=np.float32)

    silhouette_by_k: dict[int, float] = {}
    if n_clusters is None:
        n_clusters, silhouette_by_k = _pick_k(features, range(k_min, k_max + 1), random_state)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(features)
    inertia = float(model.inertia_)
    sil = safe_silhouette(features, labels)

    assignments = pd.DataFrame(
        {
            "user_id": scaled_df["user_id"],
            "cluster_id": labels,
        }
    )
    assignments_path = write_table(assignments, out / "user_cluster_assignments")

    model_path = save_joblib(
        {
            "model": model,
            "feature_columns": feature_cols,
            "n_clusters": n_clusters,
        },
        out / "kmeans_model.joblib",
    )

    cluster_sizes = {
        str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True), strict=False)
    }

    report: dict[str, Any] = {
        "algorithm": "KMeans",
        "dataset": "ds1_goodreads_2m",
        "entity": "users",
        "n_clusters": int(n_clusters),
        "n_users": int(len(scaled_df)),
        "inertia": inertia,
        "silhouette": sil,
        "silhouette_by_k": silhouette_by_k,
        "cluster_sizes": cluster_sizes,
        "paths": {
            "model": str(model_path),
            "assignments": str(assignments_path),
        },
    }
    write_json(report, out / "train_report.json")
    return report
