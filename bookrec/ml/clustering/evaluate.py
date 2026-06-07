"""Evaluate user K-Means clusters — silhouette, inertia, cluster profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bookrec.io_utils import read_table, write_json
from bookrec.ml.clustering.preprocess import load_user_feature_matrix
from bookrec.ml.clustering.profiles import build_cluster_profiles_detail
from bookrec.ml.clustering.viz import build_clustering_visualizations
from bookrec.ml.io import load_joblib
from bookrec.ml.metrics import safe_silhouette
from bookrec.paths import MODEL_CLUSTERING_DIR, MODEL_EVAL_DIR, PROC_FEATURES


def _load_interactions() -> pd.DataFrame | None:
    for path in (
        PROC_FEATURES / "interactions" / "interactions_indexed.parquet",
        PROC_FEATURES / "interactions" / "interactions_indexed.csv",
    ):
        if path.is_file():
            df = read_table(path)
            if "user_id" in df.columns and "rating" in df.columns:
                return df[["user_id", "rating"]].copy()
    return None


def evaluate_user_clusters(
    *,
    features_dir: Path | None = None,
    model_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    model_dir = Path(model_dir or MODEL_CLUSTERING_DIR)
    out = Path(out_dir or MODEL_EVAL_DIR / "clustering")
    out.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "kmeans_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"K-Means model not found at {model_path}")

    bundle = load_joblib(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_columns"]

    scaled_df, _ = load_user_feature_matrix(features_dir or (PROC_FEATURES / "clustering"))
    features = scaled_df[feature_cols].to_numpy(dtype=np.float32)
    labels = model.predict(features)
    sil = safe_silhouette(features, labels)

    raw_path = PROC_FEATURES / "clustering" / "user_features.parquet"
    if not raw_path.exists():
        raw_path = PROC_FEATURES / "clustering" / "user_features.csv"

    profiles: dict[str, Any] = {}
    profile_detail: dict[str, Any] = {}
    if raw_path.exists():
        raw = read_table(raw_path)
        assignments = pd.DataFrame({"user_id": scaled_df["user_id"], "cluster_id": labels})
        raw = raw.merge(assignments, on="user_id", how="inner")

        cluster_sizes = {
            str(int(k)): int(v) for k, v in zip(*np.unique(labels, return_counts=True), strict=False)
        }
        interactions = _load_interactions()
        detail = build_cluster_profiles_detail(
            user_features=raw.drop(columns=["cluster_id"]),
            cluster_ids=raw["cluster_id"],
            interactions=interactions,
            cluster_sizes=cluster_sizes,
        )
        profile_detail = detail
        profiles = detail["cluster_profiles_mean"]
        visualizations = build_clustering_visualizations(
            scaled_df=scaled_df,
            feature_cols=feature_cols,
            cluster_ids=labels,
            user_features=raw.drop(columns=["cluster_id"]),
            pca_model_path=model_dir / "pca_viz.joblib",
        )
    else:
        visualizations = {}

    report: dict[str, Any] = {
        "algorithm": "KMeans",
        "n_users": int(len(scaled_df)),
        "n_clusters": int(bundle.get("n_clusters", model.n_clusters)),
        "inertia": float(model.inertia_),
        "silhouette": sil,
        "cluster_profiles_mean": profiles,
        "cluster_profiles_detail": profile_detail.get("clusters", {}),
        "cluster_descriptions": profile_detail.get("cluster_descriptions", {}),
        "cluster_baseline": profile_detail.get("baseline", {}),
        "cluster_visualizations": visualizations,
        "model_path": str(model_path),
    }
    write_json(report, out / "evaluate_report.json")
    return report
