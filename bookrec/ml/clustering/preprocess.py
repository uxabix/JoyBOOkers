"""Load scaled user behaviour features for K-Means."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bookrec.io_utils import read_table, write_json
from bookrec.paths import MODEL_CLUSTERING_DIR, PROC_FEATURES


def _resolve_table(path_stem: Path) -> Path:
    for suffix in (".parquet", ".csv"):
        candidate = path_stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return path_stem.with_suffix(".parquet")


def load_user_feature_matrix(features_dir: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load user_features_scaled from clustering feature stage."""
    base = features_dir or (PROC_FEATURES / "clustering")
    scaled_path = _resolve_table(base / "user_features_scaled")
    raw_path = _resolve_table(base / "user_features")

    if not scaled_path.exists():
        raise FileNotFoundError(
            f"Scaled user features not found at {scaled_path}. "
            "Run data pipeline features stage on DS1 interactions first."
        )

    scaled = read_table(scaled_path)
    raw = read_table(raw_path) if raw_path.exists() else scaled

    feature_cols = [c for c in scaled.columns if c != "user_id"]
    if not feature_cols:
        raise ValueError("No feature columns found in user_features_scaled")

    return scaled, feature_cols


def prepare_clustering_training_data(
    features_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    out = Path(out_dir or MODEL_CLUSTERING_DIR)
    out.mkdir(parents=True, exist_ok=True)
    scaled, feature_cols = load_user_feature_matrix(features_dir)
    report = {
        "dataset": "ds1_goodreads_2m",
        "entity": "users",
        "n_users": int(len(scaled)),
        "feature_columns": feature_cols,
    }
    write_json(report, out / "preprocess_report.json")
    return report
