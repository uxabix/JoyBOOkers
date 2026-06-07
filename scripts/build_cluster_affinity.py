#!/usr/bin/env python3
"""Build cluster → book affinity JSON from DS1 interactions and K-Means assignments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bookrec.io_utils import read_table, write_json  # noqa: E402
from bookrec.ml.io import load_joblib  # noqa: E402
from bookrec.paths import MODEL_CLUSTERING_DIR, PROC_FEATURES  # noqa: E402


def _load_assignments() -> pd.DataFrame:
    for path in (
        MODEL_CLUSTERING_DIR / "user_cluster_assignments.parquet",
        MODEL_CLUSTERING_DIR / "user_cluster_assignments.csv",
        PROC_FEATURES / "clustering" / "user_cluster_assignments.parquet",
    ):
        if path.is_file():
            return read_table(path)[["user_id", "cluster_id"]].astype({"user_id": str, "cluster_id": int})

    model_path = MODEL_CLUSTERING_DIR / "kmeans_model.joblib"
    features_path = PROC_FEATURES / "clustering" / "user_features_scaled.parquet"
    if not model_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(
            "No user_cluster_assignments or kmeans_model + user_features_scaled found. "
            "Run the bookrec ML pipeline first."
        )

    bundle = load_joblib(model_path)
    model = bundle["model"] if isinstance(bundle, dict) else bundle
    cols = bundle.get("feature_columns") if isinstance(bundle, dict) else None
    scaled = read_table(features_path)
    feature_cols = [c for c in (cols or []) if c in scaled.columns]
    if not feature_cols:
        feature_cols = [c for c in scaled.columns if c != "user_id"]
    labels = model.predict(scaled[feature_cols].to_numpy(dtype=np.float32))
    return pd.DataFrame({"user_id": scaled["user_id"].astype(str), "cluster_id": labels.astype(int)})


def build_cluster_affinity(*, top_n: int = 200) -> dict[str, dict[str, float]]:
    interactions_path = PROC_FEATURES / "interactions" / "interactions_indexed.parquet"
    if not interactions_path.is_file():
        raise FileNotFoundError(f"Missing interactions: {interactions_path}")

    interactions = read_table(interactions_path)
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["book_id"] = interactions["book_id"].astype(str)

    assignments = _load_assignments()
    merged = interactions.merge(assignments, on="user_id", how="inner")
    if merged.empty:
        raise RuntimeError("No rows after merging interactions with cluster assignments.")

    agg = (
        merged.groupby(["cluster_id", "book_id"], as_index=False)
        .agg(n=("rating", "count"), avg_rating=("rating", "mean"))
    )
    agg["raw_score"] = agg["n"] * agg["avg_rating"]

    result: dict[str, dict[str, float]] = {}
    for cluster_id, group in agg.groupby("cluster_id"):
        top = group.nlargest(top_n, "raw_score")
        max_score = float(top["raw_score"].max()) if not top.empty else 1.0
        books = {
            str(row.book_id): float(row.raw_score / max_score if max_score > 0 else 0.0)
            for row in top.itertuples()
        }
        result[str(int(cluster_id))] = books

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cluster book affinity for hybrid recommender.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC_FEATURES / "clustering" / "cluster_affinity.json",
        help="Output JSON path",
    )
    parser.add_argument("--top-n", type=int, default=200, help="Top books per cluster")
    args = parser.parse_args()

    affinity = build_cluster_affinity(top_n=args.top_n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(affinity, args.out)
    total = sum(len(v) for v in affinity.values())
    print(f"Wrote {total} book scores across {len(affinity)} clusters -> {args.out}")


if __name__ == "__main__":
    main()
