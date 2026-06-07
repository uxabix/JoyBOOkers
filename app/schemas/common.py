"""Shared Pydantic types."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class MessageResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    database: str


class ModelStatusItem(BaseModel):
    name: str
    path: str
    loaded: bool
    detail: str = ""


class ReadinessResponse(BaseModel):
    status: str
    app: str
    version: str
    database: str
    models: list[ModelStatusItem]
    all_required_models_loaded: bool


class PaginatedResponse(ORMModel, Generic[T]):
    items: list[T]
    total: int
    page: int = Field(ge=1)
    page_size: int = Field(ge=1, le=100)
    pages: int
