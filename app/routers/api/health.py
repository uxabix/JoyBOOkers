"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.session import get_db
from app.dependencies import get_settings_dep
from app.schemas.common import HealthResponse, MessageResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings_dep)) -> HealthResponse:
    return HealthResponse(
        app=settings.app_name,
        version=settings.app_version,
        database="configured",
    )


@router.get("/health/db", response_model=MessageResponse)
def health_db(db: Session = Depends(get_db)) -> MessageResponse:
    db.execute(text("SELECT 1"))
    return MessageResponse(message="database ok")
