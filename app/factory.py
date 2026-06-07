"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from bookrec.paths import PROJECT_ROOT

from app.config import Settings, get_settings
from app.db.session import engine, init_db
from app.logging_config import get_logger, setup_logging
from app.ml.collaborative import CollaborativeFilteringEngine
from app.ml.content_based import ContentRecommendationEngine
from app.ml.sentiment import SentimentEngine
from app.routers.api import api_router
from app.routers.web import web_router

logger = get_logger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(level=settings.log_level, log_dir=settings.log_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting %s v%s", settings.app_name, settings.app_version)
        init_db()

        app.state.settings = settings
        app.state.cf_engine = CollaborativeFilteringEngine(settings.cf_model_path)
        app.state.content_engine = ContentRecommendationEngine(
            settings.content_tfidf_path
            if settings.content_tfidf_path.exists()
            else settings.content_bow_path,
        )
        app.state.sentiment_engine = SentimentEngine(settings.sentiment_model_path)

        app.state.cf_engine.load()
        app.state.content_engine.load()
        app.state.sentiment_engine.load()

        yield

        engine.dispose()
        logger.info("Shutdown complete")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    if settings.static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

    reports_eda = PROJECT_ROOT / "reports" / "eda"
    if reports_eda.is_dir():
        app.mount("/reports-assets/eda", StaticFiles(directory=str(reports_eda)), name="reports-eda")

    app.include_router(api_router, prefix="/api/v1")
    app.include_router(web_router)

    return app
