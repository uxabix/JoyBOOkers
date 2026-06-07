"""Online K-Means cluster assignment for app users from rating behaviour."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

# Fallback when evaluate_report.json is missing (dev / tests).
_DEFAULT_LABELS = {
    0: "Aktywni czytelnicy",
    1: "Okazjonalni oceniający",
    2: "Umiarkowana aktywność",
}

# Backward-compatible alias (prefer data-driven titles from evaluate_report.json).
CLUSTER_LABELS = _DEFAULT_LABELS


class UserClusteringEngine:
    def __init__(
        self,
        model_path: Path,
        features_dir: Path,
        report_path: Path,
        *,
        eval_report_path: Path | None = None,
    ) -> None:
        self.model_path = model_path
        self.features_dir = features_dir
        self.report_path = report_path
        self.eval_report_path = eval_report_path
        self._model = None
        self._feature_cols: list[str] = list(FEATURE_COLS)
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._q33 = 8.0
        self._q66 = 39.0
        self._cluster_titles: dict[int, str] = dict(_DEFAULT_LABELS)
        self._cluster_descriptions: dict[int, str] = {}

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
        self._load_cluster_descriptions()
        logger.info("Loaded user clustering model from %s", self.model_path)
        return True

    def _load_cluster_descriptions(self) -> None:
        for path in (
            self.eval_report_path,
            self.report_path.parent.parent / "ml" / "evaluation" / "clustering" / "evaluate_report.json",
        ):
            if not path or not Path(path).is_file():
                continue
            try:
                with Path(path).open(encoding="utf-8") as fh:
                    report = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            desc = report.get("cluster_descriptions") or {}
            detail = report.get("cluster_profiles_detail") or {}
            titles: dict[int, str] = {}
            descriptions: dict[int, str] = {}
            for cid_str, payload in desc.items():
                cid = int(cid_str)
                titles[cid] = str(payload.get("title", _DEFAULT_LABELS.get(cid, f"Klaster {cid}")))
                descriptions[cid] = str(payload.get("description", ""))
            for cid_str, payload in detail.items():
                cid = int(cid_str)
                if cid not in titles and payload.get("title"):
                    titles[cid] = str(payload["title"])
                if cid not in descriptions and payload.get("description"):
                    descriptions[cid] = str(payload["description"])
            if titles:
                self._cluster_titles = titles
            if descriptions:
                self._cluster_descriptions = descriptions
            return

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
        return self._cluster_titles.get(cluster_id, f"Klaster {cluster_id}")

    def cluster_description(self, cluster_id: int | None) -> str | None:
        if cluster_id is None:
            return None
        return self._cluster_descriptions.get(cluster_id)

    def all_cluster_labels(self) -> dict[int, str]:
        return dict(self._cluster_titles)

    def all_cluster_profiles(self) -> dict[int, dict[str, Any]]:
        """Profiles loaded at runtime are titles/descriptions only; full detail comes from reports."""
        return {
            cid: {"title": title, "description": self._cluster_descriptions.get(cid, "")}
            for cid, title in self._cluster_titles.items()
        }
