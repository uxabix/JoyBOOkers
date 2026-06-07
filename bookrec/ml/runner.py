"""Orchestrate ML preprocessing, training, and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from bookrec.io_utils import write_json
from bookrec.ml.clustering.evaluate import evaluate_user_clusters
from bookrec.ml.clustering.preprocess import prepare_clustering_training_data
from bookrec.ml.clustering.train import train_user_clusters
from bookrec.ml.collaborative.evaluate import evaluate_svd
from bookrec.ml.collaborative.preprocess import prepare_cf_training_data
from bookrec.ml.collaborative.train import train_svd
from bookrec.ml.content.evaluate import evaluate_content_vectors
from bookrec.ml.content.preprocess import prepare_content_training_data
from bookrec.ml.content.train import train_content_vectors
from bookrec.ml.sentiment.evaluate import evaluate_sentiment_model
from bookrec.ml.sentiment.preprocess import prepare_sentiment_training_data
from bookrec.ml.sentiment.train import train_sentiment_model
from bookrec.paths import MODEL_EVAL_DIR, PROC_MODELS
from bookrec.reports_export import export_reports


def preprocess_all() -> dict[str, Any]:
    return {
        "collaborative": _safe(prepare_cf_training_data),
        "content": _safe(prepare_content_training_data),
        "sentiment": _safe(prepare_sentiment_training_data),
        "clustering": _safe(prepare_clustering_training_data),
    }


def train_all(**train_kw: Any) -> dict[str, Any]:
    return {
        "collaborative": _safe(train_svd, **train_kw.get("collaborative", {})),
        "content": _safe(train_content_vectors, **train_kw.get("content", {})),
        "sentiment": _safe(train_sentiment_model, **train_kw.get("sentiment", {})),
        "clustering": _safe(train_user_clusters, **train_kw.get("clustering", {})),
    }


def evaluate_all(**eval_kw: Any) -> dict[str, Any]:
    return {
        "collaborative": _safe(evaluate_svd, **eval_kw.get("collaborative", {})),
        "content": _safe(evaluate_content_vectors, **eval_kw.get("content", {})),
        "sentiment": _safe(evaluate_sentiment_model, **eval_kw.get("sentiment", {})),
        "clustering": _safe(evaluate_user_clusters, **eval_kw.get("clustering", {})),
    }


def _safe(fn, **kwargs) -> dict[str, Any]:
    try:
        return fn(**kwargs)
    except FileNotFoundError as exc:
        return {"skipped": True, "error": str(exc)}
    except Exception as exc:
        return {"error": str(exc)}


def run_ml_pipeline(
    stages: list[str],
    *,
    module: str | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"stages": stages}
    if module:
        summary["module"] = module

    if "preprocess" in stages:
        summary["preprocess"] = (
            {module: _safe(_MODULE_PREPROCESS[module])}
            if module
            else preprocess_all()
        )

    if "train" in stages:
        summary["train"] = (
            {module: _safe(_MODULE_TRAIN[module])}
            if module
            else train_all()
        )

    if "evaluate" in stages:
        summary["evaluate"] = (
            {module: _safe(_MODULE_EVALUATE[module])}
            if module
            else evaluate_all()
        )
        # Always refresh merged evaluate_all.json from per-module reports on disk.
        from bookrec.reports_export import build_evaluate_all_summary

        merged = build_evaluate_all_summary()
        if merged:
            write_json(merged, MODEL_EVAL_DIR / "evaluate_all.json")

    PROC_MODELS.mkdir(parents=True, exist_ok=True)
    write_json(summary, MODEL_EVAL_DIR / "ml_pipeline_summary.json")
    export_reports()
    return summary


_MODULE_PREPROCESS = {
    "collaborative": prepare_cf_training_data,
    "content": prepare_content_training_data,
    "sentiment": prepare_sentiment_training_data,
    "clustering": prepare_clustering_training_data,
}
_MODULE_TRAIN = {
    "collaborative": train_svd,
    "content": train_content_vectors,
    "sentiment": train_sentiment_model,
    "clustering": train_user_clusters,
}
_MODULE_EVALUATE = {
    "collaborative": evaluate_svd,
    "content": evaluate_content_vectors,
    "sentiment": evaluate_sentiment_model,
    "clustering": evaluate_user_clusters,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="JoyBookers ML training pipeline")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["preprocess", "train", "evaluate", "all"],
        default=["all"],
    )
    parser.add_argument(
        "--module",
        choices=["collaborative", "content", "sentiment", "clustering"],
        default=None,
        help="Run a single module only",
    )
    args = parser.parse_args(argv)

    if "all" in args.stages:
        stages = ["preprocess", "train", "evaluate"]
    else:
        stages = args.stages

    run_ml_pipeline(stages, module=args.module)
    print("ML pipeline complete. See data/processed/models/evaluation/ml_pipeline_summary.json")
    return 0
