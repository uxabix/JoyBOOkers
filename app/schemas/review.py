"""Amazon review schemas — independent DS4 pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class ReviewRead(ORMModel):
    id: int
    asin: str
    reviewer_id: str | None = None
    review_text: str
    star_rating: float | None = None
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    source: str


class ReviewCreate(BaseModel):
    asin: str = Field(min_length=1, max_length=32)
    review_text: str = Field(min_length=1)
    reviewer_id: str | None = None
    star_rating: float | None = Field(default=None, ge=1.0, le=5.0)
