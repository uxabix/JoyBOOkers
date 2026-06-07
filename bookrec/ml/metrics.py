"""Shared evaluation metrics for all ML modules."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    silhouette_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def precision_recall_at_k(
    ground_truth: dict[str, set[str]],
    recommendations: dict[str, list[str]],
    k: int = 10,
) -> dict[str, float]:
    """Top-K ranking metrics for CF evaluation (user -> relevant book ids)."""
    precisions: list[float] = []
    recalls: list[float] = []
    for user_id, rel in ground_truth.items():
        if not rel:
            continue
        recs = recommendations.get(user_id, [])[:k]
        if not recs:
            continue
        hits = len(set(recs) & rel)
        precisions.append(hits / len(recs))
        recalls.append(hits / len(rel))
    if not precisions:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "k": float(k)}
    return {
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "k": float(k),
    }


def safe_silhouette(features: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette score; None when undefined (<2 clusters or single cluster)."""
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or len(labels) < n_clusters + 1:
        return None
    try:
        return float(silhouette_score(features, labels))
    except ValueError:
        return None
