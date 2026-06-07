"""Rating API schemas — DS1 collaborative filtering."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class RatingRead(ORMModel):
    id: int
    user_id: int
    book_id: int
    score: float
    source: str
    rated_at: datetime | None = None


class RatingCreate(BaseModel):
    user_id: int
    book_id: int
    score: float = Field(ge=1.0, le=5.0)
    rated_at: datetime | None = None
