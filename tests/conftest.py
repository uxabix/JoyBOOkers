"""Shared pytest fixtures for FastAPI integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings
from app.db.base import Base
from app.db import models  # noqa: F401
from app.db.session import dispose_engine, get_db
from app.factory import create_app


@pytest.fixture
def client() -> TestClient:
    dispose_engine()
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    settings = Settings(
        database_url="sqlite://",
        log_dir=Path("logs"),
        ml_eager_load=False,
    )
    import app.db.session as db_session

    db_session._engine = engine
    db_session._SessionLocal = TestingSessionLocal
    db_session._database_url = settings.database_url
    app = create_app(settings)
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    Base.metadata.drop_all(bind=engine)
    dispose_engine()
