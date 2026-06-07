"""Online K-Means cluster assignment for app users from rating behaviour."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.logging_config import get_logger
from bookrec.io_utils import read_table
from bookrec.ml.io import load_joblib

logger = get_logger(__name__)

FEATURE_COLS = [
    "n_ratings",
    "mean_rating",
    "std_rating",
    "rating_range",
    "activity_low",
    "activity_medium",
    "activity_high",
]

CLUSTER_LABELS = {
    0: "Power users — high activity",
    1: "Lenient occasional raters",
    2: "Moderate activity",
}


class UserClusteringEngine:
    def __init__(
        self,
        model_path: Path,
        features_dir: Path,
        report_path: Path,
    ) -> None:
        self.model_path = model_path
        self.features_dir = features_dir
        self.report_path = report_path
        self._model = None
        self._feature_cols: list[str] = list(FEATURE_COLS)
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._q33 = 8.0
        self._q66 = 39.0

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        if not self.model_path.is_file():
            logger.warning("K-Means model not found: %s", self.model_path)
            return False
        raw = load_joblib(self.model_path)
        if isinstance(raw, dict) and "model" in raw:
            self._model = raw["model"]
            self._feature_cols = list(raw.get("feature_columns") or FEATURE_COLS)
        else:
            self._model = raw
        self._load_scaler_stats()
        self._load_quantiles()
        logger.info("Loaded user clustering model from %s", self.model_path)
        return True

    def _load_scaler_stats(self) -> None:
        raw_path = self.features_dir / "user_features.parquet"
        if not raw_path.is_file():
            raw_path = self.features_dir / "user_features.csv"
        if not raw_path.is_file():
            self._mu = np.zeros(len(FEATURE_COLS), dtype=np.float32)
            self._sigma = np.ones(len(FEATURE_COLS), dtype=np.float32)
            return
        df = read_table(raw_path)
        cols = [c for c in self._feature_cols if c in df.columns]
        if not cols:
            cols = [c for c in FEATURE_COLS if c in df.columns]
        matrix = df[cols].astype(np.float32)
        self._mu = matrix.mean().to_numpy(dtype=np.float32)
        self._sigma = matrix.std().replace(0, 1.0).to_numpy(dtype=np.float32)

    def _load_quantiles(self) -> None:
        if not self.report_path.is_file():
            return
        with self.report_path.open(encoding="utf-8") as fh:
            report = json.load(fh)
        q = report.get("activity_quantiles") or {}
        self._q33 = float(q.get("q33", self._q33))
        self._q66 = float(q.get("q66", self._q66))

    def _activity_one_hot(self, n_ratings: int) -> tuple[float, float, float]:
        if n_ratings >= self._q66:
            return 0.0, 0.0, 1.0
        if n_ratings >= self._q33:
            return 0.0, 1.0, 0.0
        return 1.0, 0.0, 0.0

    def features_from_scores(self, scores: list[float]) -> np.ndarray:
        if not scores:
            low, med, high = 1.0, 0.0, 0.0
            raw = np.array([0.0, 3.0, 0.0, 0.0, low, med, high], dtype=np.float32)
        else:
            arr = np.array(scores, dtype=np.float32)
            low, med, high = self._activity_one_hot(len(scores))
            raw = np.array(
                [
                    float(len(scores)),
                    float(arr.mean()),
                    float(arr.std()) if len(arr) > 1 else 0.0,
                    float(arr.max() - arr.min()),
                    low,
                    med,
                    high,
                ],
                dtype=np.float32,
            )
        assert self._mu is not None and self._sigma is not None
        return ((raw - self._mu) / self._sigma).astype(np.float32)

    def predict_cluster(self, scores: list[float]) -> int:
        if self._model is None:
            return 1
        vec = self.features_from_scores(scores).reshape(1, -1)
        return int(self._model.predict(vec)[0])

    def cluster_label(self, cluster_id: int | None) -> str | None:
        if cluster_id is None:
            return None
        return CLUSTER_LABELS.get(cluster_id, f"Cluster {cluster_id}")
