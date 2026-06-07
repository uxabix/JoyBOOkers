"""Project live user profiles onto the clustering PCA plane."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from app.logging_config import get_logger
from bookrec.ml.io import load_joblib

if TYPE_CHECKING:
    from app.ml.user_clustering import UserClusteringEngine

logger = get_logger(__name__)


class ClusterPcaProjector:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self._pca = None

    @property
    def is_loaded(self) -> bool:
        return self._pca is not None

    def load(self) -> bool:
        if not self.model_path.is_file():
            logger.warning("Cluster PCA model not found: %s", self.model_path)
            return False
        self._pca = load_joblib(self.model_path)
        logger.info("Loaded cluster PCA projector from %s", self.model_path)
        return True

    def project_features(self, feature_vec: np.ndarray) -> tuple[float, float] | None:
        if self._pca is None:
            return None
        row = np.asarray(feature_vec, dtype=np.float32).reshape(1, -1)
        xy = self._pca.transform(row)[0]
        return float(xy[0]), float(xy[1])

    def project_scores(
        self,
        clustering: UserClusteringEngine,
        scores: list[float],
    ) -> tuple[float, float] | None:
        if not clustering.is_loaded:
            return None
        vec = clustering.features_from_scores(scores)
        return self.project_features(vec)

    def highlight_point(
        self,
        clustering: UserClusteringEngine,
        scores: list[float],
        *,
        label: str,
        kind: str,
        cluster_id: int | None = None,
    ) -> dict[str, float | str | int] | None:
        if not scores:
            return None
        coords = self.project_scores(clustering, scores)
        if coords is None:
            return None
        point: dict[str, float | str | int] = {
            "x": round(coords[0], 4),
            "y": round(coords[1], 4),
            "label": label,
            "kind": kind,
        }
        if cluster_id is not None:
            point["cluster_id"] = int(cluster_id)
        return point
