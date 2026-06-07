"""FastAPI dependency injection — DB session, services, ML engines."""

from __future__ import annotations

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_db
from app.ml.cluster_affinity import ClusterAffinityStore
from app.ml.genre_priors import GenrePriorStore
from app.ml.hybrid_weights import HybridWeightModel
from app.ml.collaborative import CollaborativeFilteringEngine
from app.ml.content_based import ContentRecommendationEngine
from app.ml.registry import MLModelRegistry
from app.ml.sentiment import SentimentEngine
from app.ml.user_clustering import UserClusteringEngine
from app.services.author_service import AuthorService
from app.services.book_service import BookService
from app.services.clustering_service import ClusteringService
from app.services.cold_start_service import ColdStartService
from app.services.collaborative_filtering_service import CollaborativeFilteringService
from app.services.content_recommendation_service import ContentRecommendationService
from app.services.rating_service import RatingService
from app.services.recommendation_service import RecommendationService
from app.services.reports_service import ReportsService
from app.services.sentiment_service import SentimentService
from app.services.user_service import UserService
from app.templates_env import get_templates_engine


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def _ensure_ml_loaded(request: Request) -> MLModelRegistry:
    registry: MLModelRegistry = request.app.state.ml_registry
    registry.ensure_loaded()
    return registry


def get_ml_registry(request: Request) -> MLModelRegistry:
    return request.app.state.ml_registry


def get_cf_engine(request: Request) -> CollaborativeFilteringEngine:
    return _ensure_ml_loaded(request).cf_engine


def get_content_engine(request: Request) -> ContentRecommendationEngine:
    return _ensure_ml_loaded(request).content_engine


def get_sentiment_engine(request: Request) -> SentimentEngine:
    return _ensure_ml_loaded(request).sentiment_engine


def get_clustering_engine(request: Request) -> UserClusteringEngine:
    registry = _ensure_ml_loaded(request)
    if not registry.clustering_engine.is_loaded:
        registry.clustering_engine.load()
    return registry.clustering_engine


def get_cluster_affinity_store(request: Request) -> ClusterAffinityStore:
    registry = _ensure_ml_loaded(request)
    if not registry.cluster_affinity.is_loaded:
        registry.cluster_affinity.load()
    return registry.cluster_affinity


def get_genre_prior_store(request: Request) -> GenrePriorStore:
    registry = _ensure_ml_loaded(request)
    if not registry.genre_priors.is_loaded:
        registry.genre_priors.load()
    return registry.genre_priors


def get_hybrid_weight_model(request: Request) -> HybridWeightModel:
    registry = _ensure_ml_loaded(request)
    if not registry.hybrid_weights.is_loaded:
        registry.hybrid_weights.load()
    return registry.hybrid_weights


def get_book_service(db: Session = Depends(get_db)) -> BookService:
    return BookService(db)


def get_author_service(db: Session = Depends(get_db)) -> AuthorService:
    return AuthorService(db)


def get_user_service(
    db: Session = Depends(get_db),
    clustering: UserClusteringEngine = Depends(get_clustering_engine),
) -> UserService:
    return UserService(db, clustering)


def get_clustering_service(
    db: Session = Depends(get_db),
    engine: UserClusteringEngine = Depends(get_clustering_engine),
    settings: Settings = Depends(get_settings_dep),
) -> ClusteringService:
    return ClusteringService(db, engine, settings)


def get_rating_service(
    db: Session = Depends(get_db),
    clustering_service: ClusteringService = Depends(get_clustering_service),
) -> RatingService:
    return RatingService(db, clustering_service)


def get_cf_service(
    db: Session = Depends(get_db),
    engine: CollaborativeFilteringEngine = Depends(get_cf_engine),
    settings: Settings = Depends(get_settings_dep),
) -> CollaborativeFilteringService:
    return CollaborativeFilteringService(db, engine, settings)


def get_content_service(
    db: Session = Depends(get_db),
    engine: ContentRecommendationEngine = Depends(get_content_engine),
    settings: Settings = Depends(get_settings_dep),
) -> ContentRecommendationService:
    return ContentRecommendationService(db, engine, settings)


def get_cold_start_service(
    db: Session = Depends(get_db),
    engine: ContentRecommendationEngine = Depends(get_content_engine),
    settings: Settings = Depends(get_settings_dep),
) -> ColdStartService:
    return ColdStartService(db, engine, settings)


def get_recommendation_service(
    db: Session = Depends(get_db),
    cf_service: CollaborativeFilteringService = Depends(get_cf_service),
    cf_engine: CollaborativeFilteringEngine = Depends(get_cf_engine),
    content_engine: ContentRecommendationEngine = Depends(get_content_engine),
    clustering_engine: UserClusteringEngine = Depends(get_clustering_engine),
    cluster_affinity: ClusterAffinityStore = Depends(get_cluster_affinity_store),
    genre_priors: GenrePriorStore = Depends(get_genre_prior_store),
    weight_model: HybridWeightModel = Depends(get_hybrid_weight_model),
    settings: Settings = Depends(get_settings_dep),
) -> RecommendationService:
    return RecommendationService(
        db,
        cf_service,
        cf_engine,
        content_engine,
        clustering_engine,
        cluster_affinity,
        genre_priors,
        weight_model,
        settings,
    )


def get_sentiment_service(
    db: Session = Depends(get_db),
    engine: SentimentEngine = Depends(get_sentiment_engine),
) -> SentimentService:
    return SentimentService(db, engine)


def get_reports_service(request: Request) -> ReportsService:
    settings: Settings = request.app.state.settings
    return ReportsService(settings.reports_dir)


def get_templates(request: Request):
    if hasattr(request.app.state, "templates"):
        return request.app.state.templates
    return get_templates_engine()
