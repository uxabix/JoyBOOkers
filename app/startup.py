"""Application startup and shutdown sequence."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from sqlalchemy.engine.url import make_url

from app.config import Settings
from app.db.migrate import migrate_schema
from app.db.session import configure_engine, dispose_engine, init_db
from app.logging_config import get_logger, setup_logging
from app.ml.registry import MLModelRegistry
from app.templates_env import build_templates_engine, clear_templates_cache

logger = get_logger(__name__)


def _ensure_sqlite_parent(settings: Settings) -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    url = make_url(settings.database_url)
    if not url.database or url.database == ":memory:":
        return
    Path(url.database).parent.mkdir(parents=True, exist_ok=True)


def on_startup(app: FastAPI, settings: Settings) -> None:
    setup_logging(level=settings.log_level, log_dir=settings.log_dir)
    logger.info("=== %s v%s — startup ===", settings.app_name, settings.app_version)

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.features_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    _ensure_sqlite_parent(settings)
    db_engine = configure_engine(settings.database_url, echo=settings.debug)
    init_db(db_engine)
    migrate_schema(db_engine)
    app.state.db_engine = db_engine
    logger.info("Database ready: %s", settings.database_url)

    clear_templates_cache()
    app.state.templates = build_templates_engine(settings)
    logger.info("Templates registered: %s", settings.templates_dir)

    registry = MLModelRegistry(settings)
    if settings.ml_eager_load:
        registry.load_all()
    else:
        registry.register_status_only()

    app.state.ml_registry = registry
    app.state.cf_engine = registry.cf_engine
    app.state.content_engine = registry.content_engine
    app.state.sentiment_engine = registry.sentiment_engine
    app.state.clustering_engine = registry.clustering_engine
    app.state.settings = settings

    logger.info("=== startup complete ===")


def on_shutdown(app: FastAPI) -> None:
    logger.info("=== shutdown ===")
    dispose_engine()
    logger.info("=== shutdown complete ===")
