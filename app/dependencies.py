"""FastAPI dependency injection — DB session, services, ML engines."""

from __future__ import annotations

from collections.abc import Generator
from functools import lru_cache

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.session import get_db
from app.ml.collaborative import CollaborativeFilteringEngine
from app.ml.content_based import ContentRecommendationEngine
from app.ml.sentiment import SentimentEngine
from app.services.book_service import BookService
from app.services.collaborative_filtering_service import CollaborativeFilteringService
from app.services.content_recommendation_service import ContentRecommendationService
from app.services.rating_service import RatingService
from app.services.recommendation_service import RecommendationService
from app.services.sentiment_service import SentimentService
from app.services.reports_service import ReportsService
from app.ml.registry import MLModelRegistry
from app.services.user_service import UserService
from app.templates_env import get_templates_engine


def get_settings_dep() -> Settings:
    return get_settings()


def get_ml_registry(request: Request) -> MLModelRegistry:
    return request.app.state.ml_registry


def get_cf_engine(request: Request) -> CollaborativeFilteringEngine:
    return request.app.state.cf_engine


def get_content_engine(request: Request) -> ContentRecommendationEngine:
    return request.app.state.content_engine


def get_sentiment_engine(request: Request) -> SentimentEngine:
    return request.app.state.sentiment_engine


def get_book_service(db: Session = Depends(get_db)) -> BookService:
    return BookService(db)


def get_user_service(db: Session = Depends(get_db)) -> UserService:
    return UserService(db)


def get_rating_service(db: Session = Depends(get_db)) -> RatingService:
    return RatingService(db)


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


def get_recommendation_service(
    db: Session = Depends(get_db),
    cf_service: CollaborativeFilteringService = Depends(get_cf_service),
    content_service: ContentRecommendationService = Depends(get_content_service),
    settings: Settings = Depends(get_settings_dep),
) -> RecommendationService:
    return RecommendationService(db, cf_service, content_service, settings)


def get_sentiment_service(
    db: Session = Depends(get_db),
    engine: SentimentEngine = Depends(get_sentiment_engine),
) -> SentimentService:
    return SentimentService(db, engine)


@lru_cache
def get_reports_service() -> ReportsService:
    settings = get_settings()
    return ReportsService(settings.reports_dir)


def get_templates(request: Request):
    if hasattr(request.app.state, "templates"):
        return request.app.state.templates
    return get_templates_engine()
