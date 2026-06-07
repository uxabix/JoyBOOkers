"""Application startup and shutdown sequence."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from sqlalchemy.engine.url import make_url

from app.config import Settings
from app.db.session import engine, init_db
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
    db_path = Path(url.database)
    db_path.parent.mkdir(parents=True, exist_ok=True)


def on_startup(app: FastAPI, settings: Settings) -> None:
    """Ordered boot: logging → directories → database → templates → ML models."""
    setup_logging(level=settings.log_level, log_dir=settings.log_dir)
    logger.info("=== %s v%s — startup ===", settings.app_name, settings.app_version)

    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.features_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    _ensure_sqlite_parent(settings)
    init_db()
    logger.info("Database ready: %s", settings.database_url)

    clear_templates_cache()
    templates = build_templates_engine(settings)
    app.state.templates = templates
    logger.info("Templates registered: %s", settings.templates_dir)

    registry = MLModelRegistry(settings)
    registry.load_all()
    app.state.ml_registry = registry
    app.state.cf_engine = registry.cf_engine
    app.state.content_engine = registry.content_engine
    app.state.sentiment_engine = registry.sentiment_engine
    app.state.settings = settings

    logger.info("=== startup complete ===")


def on_shutdown(app: FastAPI) -> None:
    logger.info("=== shutdown — disposing database engine ===")
    engine.dispose()
    logger.info("=== shutdown complete ===")
