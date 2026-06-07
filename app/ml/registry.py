"""Central registry for ML engines loaded at application startup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.logging_config import get_logger
from app.ml.cluster_affinity import ClusterAffinityStore
from app.ml.collaborative import CollaborativeFilteringEngine
from app.ml.content_based import ContentRecommendationEngine
from app.ml.genre_priors import GenrePriorStore
from app.ml.hybrid_weights import HybridWeightModel
from app.ml.sentiment import SentimentEngine
from app.ml.user_clustering import UserClusteringEngine

logger = get_logger(__name__)


@dataclass
class ModelStatus:
    name: str
    path: str
    loaded: bool
    detail: str = ""


@dataclass
class MLModelRegistry:
    settings: Settings
    cf_engine: CollaborativeFilteringEngine = field(init=False)
    content_engine: ContentRecommendationEngine = field(init=False)
    sentiment_engine: SentimentEngine = field(init=False)
    clustering_engine: UserClusteringEngine = field(init=False)
    cluster_affinity: ClusterAffinityStore = field(init=False)
    genre_priors: GenrePriorStore = field(init=False)
    hybrid_weights: HybridWeightModel = field(init=False)
    statuses: list[ModelStatus] = field(default_factory=list)
    _loaded: bool = False

    def __post_init__(self) -> None:
        content_path = self._content_matrix_path()
        self.cf_engine = CollaborativeFilteringEngine(
            self.settings.cf_model_path,
            train_items_path=self.settings.cf_train_path,
        )
        self.content_engine = ContentRecommendationEngine(content_path)
        self.sentiment_engine = SentimentEngine(self.settings.sentiment_model_path)
        self.clustering_engine = UserClusteringEngine(
            self.settings.clustering_model_path,
            self.settings.clustering_features_dir,
            self.settings.clustering_report_path,
        )
        self.cluster_affinity = ClusterAffinityStore(self.settings.cluster_affinity_path)
        self.genre_priors = GenrePriorStore(self.settings.genre_priors_path)
        self.hybrid_weights = HybridWeightModel(self.settings.hybrid_weights_path)

    def _content_matrix_path(self) -> Path:
        if self.settings.content_tfidf_path.is_file():
            return self.settings.content_tfidf_path
        return self.settings.content_bow_path

    def register_status_only(self) -> None:
        self.statuses = [
            ModelStatus("collaborative_filtering", str(self.settings.cf_model_path), False, "lazy load"),
            ModelStatus("content_tfidf", str(self._content_matrix_path()), False, "lazy load"),
            ModelStatus("sentiment", str(self.settings.sentiment_model_path), False, "lazy load"),
            self._load_one("clustering", self.settings.clustering_model_path, self.clustering_engine.load),
            self._load_one("cluster_affinity", self.settings.cluster_affinity_path, self.cluster_affinity.load),
            self._load_one("genre_priors", self.settings.genre_priors_path, self.genre_priors.load),
            self._load_one("hybrid_weights", self.settings.hybrid_weights_path, self.hybrid_weights.load),
        ]

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_all()

    def load_all(self) -> None:
        self.statuses = [
            self._load_one("collaborative_filtering", self.settings.cf_model_path, self.cf_engine.load),
            self._load_one("content_tfidf", self._content_matrix_path(), self.content_engine.load),
            self._load_one("sentiment", self.settings.sentiment_model_path, self.sentiment_engine.load),
            self._load_one("clustering", self.settings.clustering_model_path, self.clustering_engine.load),
            self._load_one("cluster_affinity", self.settings.cluster_affinity_path, self.cluster_affinity.load),
            self._load_one("genre_priors", self.settings.genre_priors_path, self.genre_priors.load),
            self._load_one("hybrid_weights", self.settings.hybrid_weights_path, self.hybrid_weights.load),
        ]
        self._loaded = True
        loaded = sum(1 for s in self.statuses if s.loaded)
        logger.info("ML registry: %s/%s models loaded", loaded, len(self.statuses))

    def _load_one(self, name: str, path: Path, loader) -> ModelStatus:
        if not path.is_file():
            msg = f"artifact missing: {path}"
            logger.warning("%s — %s", name, msg)
            return ModelStatus(name=name, path=str(path), loaded=False, detail=msg)
        ok = bool(loader())
        detail = "loaded" if ok else "load() returned false"
        return ModelStatus(name=name, path=str(path), loaded=ok, detail=detail)

    def _artifact_status(self, name: str, path: Path) -> ModelStatus:
        exists = path.is_file()
        detail = "reports dashboard only" if exists else f"artifact missing: {path}"
        return ModelStatus(name=name, path=str(path), loaded=exists, detail=detail)

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": [
                {"name": s.name, "path": s.path, "loaded": s.loaded, "detail": s.detail}
                for s in self.statuses
            ],
            "all_required_loaded": all(
                s.loaded
                for s in self.statuses
                if s.name in {"collaborative_filtering", "content_tfidf", "sentiment"}
            ),
        }
