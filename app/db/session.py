"""SQLAlchemy engine and session factory (SQLite by default)."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db.base import Base
from app.logging_config import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None
_database_url: str | None = None


def configure_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create or replace the global engine for the given URL."""
    global _engine, _SessionLocal, _database_url

    if _engine is not None and _database_url == database_url:
        return _engine

    if _engine is not None:
        _engine.dispose()

    connect_args: dict = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        database_url,
        connect_args=connect_args,
        echo=echo,
        pool_pre_ping=not database_url.startswith("sqlite"),
    )

    if database_url.startswith("sqlite"):

        @event.listens_for(_engine, "connect")
        def _sqlite_pragma(dbapi_connection, connection_record) -> None:  # noqa: ARG001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    _database_url = database_url
    logger.debug("SQLAlchemy engine configured for %s", database_url)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        settings = get_settings()
        configure_engine(settings.database_url, echo=settings.debug)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        get_engine()
    assert _SessionLocal is not None
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def init_db(engine: Engine | None = None) -> None:
    from app.db import models  # noqa: F401

    eng = engine or get_engine()
    Base.metadata.create_all(bind=eng)
    logger.info("Database tables ensured")


def dispose_engine() -> None:
    global _engine, _SessionLocal, _database_url
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
    _database_url = None


class _EngineModuleProxy:
    def __getattr__(self, name: str):
        return getattr(get_engine(), name)

    def dispose(self) -> None:
        dispose_engine()


engine = _EngineModuleProxy()


def SessionLocal() -> Session:
    return get_session_factory()()
