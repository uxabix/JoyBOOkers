#!/usr/bin/env python3
"""Train Ridge regression on hybrid signal features vs user ratings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.collaborative import CollaborativeFilteringEngine
from app.ml.content_based import ContentRecommendationEngine
from app.ml.hybrid_training import build_training_frame
from app.ml.hybrid_weights import HybridWeightModel
from app.ml.signals import FEATURE_ORDER
from bookrec.paths import MODEL_CF_DIR, MODEL_CONTENT_DIR, PROC_MODELS, PROC_SPLITS


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Ridge hybrid weight model.")
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC_MODELS / "hybrid" / "ridge_weights.joblib",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    cf_engine = CollaborativeFilteringEngine(
        MODEL_CF_DIR / "svd_model.pkl",
        train_items_path=PROC_SPLITS / "cf_train.parquet",
    )
    if not cf_engine.load():
        raise RuntimeError("CF model not found — run ML pipeline first.")

    content_engine = ContentRecommendationEngine(MODEL_CONTENT_DIR / "tfidf_combined.npz")
    if not content_engine.load():
        raise RuntimeError("TF-IDF matrix not found.")

    x, y = build_training_frame(
        sample_size=args.sample_size,
        cf_engine=cf_engine,
        content_engine=content_engine,
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=args.alpha)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    HybridWeightModel.save_bundle(args.out, model=model, metrics=metrics)
    coefs = dict(zip(FEATURE_ORDER, model.coef_.ravel(), strict=False))
    print(f"Trained Ridge hybrid weights -> {args.out}")
    print(f"Metrics: RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f}")
    print("Coefficients:", {k: round(float(v), 4) for k, v in coefs.items()})


if __name__ == "__main__":
    main()
