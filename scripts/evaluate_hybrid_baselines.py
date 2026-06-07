#!/usr/bin/env python3
"""Compare CF-only, content-only, and hybrid Ridge on held-out ratings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.collaborative import CollaborativeFilteringEngine
from app.ml.content_based import ContentRecommendationEngine
from app.ml.hybrid_training import build_training_frame
from app.ml.hybrid_weights import HybridWeightModel
from app.ml.signals import FEATURE_ORDER
from bookrec.paths import MODEL_CF_DIR, MODEL_CONTENT_DIR, MODEL_EVAL_DIR, PROC_MODELS, PROC_SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate hybrid vs single-signal baselines.")
    parser.add_argument("--sample-size", type=int, default=8000)
    args = parser.parse_args()

    cf_engine = CollaborativeFilteringEngine(
        MODEL_CF_DIR / "svd_model.pkl",
        train_items_path=PROC_SPLITS / "cf_train.parquet",
    )
    content_engine = ContentRecommendationEngine(MODEL_CONTENT_DIR / "tfidf_combined.npz")
    cf_engine.load()
    content_engine.load()

    x, y = build_training_frame(
        sample_size=args.sample_size,
        cf_engine=cf_engine,
        content_engine=content_engine,
    )
    split = int(len(x) * 0.8)
    x_test, y_test = x[split:], y[split:]

    def _metrics(pred: np.ndarray) -> dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
        }

    report = {
        "sample_size": args.sample_size,
        "test_rows": int(len(x_test)),
        "baselines": {
            "cf_only": _metrics(x_test[:, 0]),
            "content_only": _metrics(x_test[:, 1]),
            "cluster_only": _metrics(x_test[:, 2]),
            "popularity_only": _metrics(x_test[:, 3]),
            "manual_hybrid_blend": _metrics(np.clip(x_test @ np.array([0.25, 0.35, 0.15, 0.15, 0.10]), 0, 1)),
        },
    }

    model_path = PROC_MODELS / "hybrid" / "ridge_weights.joblib"
    if model_path.is_file():
        wm = HybridWeightModel(model_path)
        wm.load()
        pred = np.clip(
            np.array([wm.score(dict(zip(FEATURE_ORDER, row, strict=False))) for row in x_test]),
            0,
            1,
        )
        report["baselines"]["learned_ridge_hybrid"] = _metrics(pred)

    out_dir = MODEL_EVAL_DIR / "hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baseline_comparison.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
