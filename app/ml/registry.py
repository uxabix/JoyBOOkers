"""Central registry for ML engines loaded at application startup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings
from app.logging_config import get_logger
from app.ml.collaborative import CollaborativeFilteringEngine
from app.ml.content_based import ContentRecommendationEngine
from app.ml.sentiment import SentimentEngine

logger = get_logger(__name__)


@dataclass
class ModelStatus:
    name: str
    path: str
    loaded: bool
    detail: str = ""


@dataclass
class MLModelRegistry:
    """Loads and tracks Surprise SVD, TF-IDF content, and sentiment pipelines."""

    settings: Settings
    cf_engine: CollaborativeFilteringEngine = field(init=False)
    content_engine: ContentRecommendationEngine = field(init=False)
    sentiment_engine: SentimentEngine = field(init=False)
    statuses: list[ModelStatus] = field(default_factory=list)

    def __post_init__(self) -> None:
        content_path = (
            self.settings.content_tfidf_path
            if self.settings.content_tfidf_path.is_file()
            else self.settings.content_bow_path
        )
        self.cf_engine = CollaborativeFilteringEngine(self.settings.cf_model_path)
        self.content_engine = ContentRecommendationEngine(content_path)
        self.sentiment_engine = SentimentEngine(self.settings.sentiment_model_path)

    def load_all(self) -> None:
        self.statuses = [
            self._load_one("collaborative_filtering", self.settings.cf_model_path, self.cf_engine.load),
            self._load_one(
                "content_tfidf",
                self._content_matrix_path(),
                self.content_engine.load,
            ),
            self._load_one("sentiment", self.settings.sentiment_model_path, self.sentiment_engine.load),
            self._artifact_status("clustering", self.settings.clustering_model_path),
        ]
        loaded = sum(1 for s in self.statuses if s.loaded)
        logger.info("ML registry: %s/%s models loaded", loaded, len(self.statuses))

    def _content_matrix_path(self) -> Path:
        if self.settings.content_tfidf_path.is_file():
            return self.settings.content_tfidf_path
        return self.settings.content_bow_path

    def _load_one(self, name: str, path: Path, loader) -> ModelStatus:
        if not path.is_file():
            msg = f"artifact missing: {path}"
            logger.warning("%s — %s", name, msg)
            return ModelStatus(name=name, path=str(path), loaded=False, detail=msg)
        ok = bool(loader())
        detail = "loaded" if ok else "load() returned false"
        if ok:
            logger.info("%s loaded from %s", name, path)
        else:
            logger.warning("%s failed to load from %s", name, path)
        return ModelStatus(name=name, path=str(path), loaded=ok, detail=detail)

    def _artifact_status(self, name: str, path: Path) -> ModelStatus:
        exists = path.is_file()
        detail = "artifact present (not wired to API yet)" if exists else f"artifact missing: {path}"
        if exists:
            logger.info("%s artifact found at %s", name, path)
        else:
            logger.warning("%s — %s", name, detail)
        return ModelStatus(name=name, path=str(path), loaded=exists, detail=detail)

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "name": s.name,
                    "path": s.path,
                    "loaded": s.loaded,
                    "detail": s.detail,
                }
                for s in self.statuses
            ],
            "all_required_loaded": all(
                s.loaded for s in self.statuses if s.name in {"collaborative_filtering", "content_tfidf", "sentiment"}
            ),
        }
