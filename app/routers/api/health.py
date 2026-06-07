"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.session import get_db
from app.dependencies import get_settings_dep
from app.schemas.common import HealthResponse, MessageResponse, ModelStatusItem, ReadinessResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
) -> HealthResponse:
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"
    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        app=settings.app_name,
        version=settings.app_version,
        database=db_status,
    )


@router.get("/health/db", response_model=MessageResponse)
def health_db(db: Session = Depends(get_db)) -> MessageResponse:
    db.execute(text("SELECT 1"))
    return MessageResponse(message="database ok")


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
) -> ReadinessResponse:
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    registry = getattr(request.app.state, "ml_registry", None)
    models: list[ModelStatusItem] = []
    all_loaded = False
    if registry is not None:
        payload = registry.as_dict()
        models = [ModelStatusItem(**m) for m in payload["models"]]
        all_loaded = bool(payload["all_required_loaded"])

    overall = "ok" if db_status == "ok" and all_loaded else "degraded"
    return ReadinessResponse(
        status=overall,
        app=settings.app_name,
        version=settings.app_version,
        database=db_status,
        models=models,
        all_required_models_loaded=all_loaded,
    )
