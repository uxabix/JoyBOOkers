"""Rating API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.book import BookRead
from app.schemas.common import ORMModel


class RatingRead(ORMModel):
    id: int
    user_id: int
    book_id: int
    score: float
    source: str
    rated_at: datetime | None = None


class RatingWithBook(RatingRead):
    book: BookRead


class RatingCreate(BaseModel):
    user_id: int | None = None
    book_id: int
    score: float = Field(ge=1.0, le=5.0)
    rated_at: datetime | None = None


class UserRatingStats(BaseModel):
    n_ratings: int = 0
    mean_rating: float | None = None
    std_rating: float | None = None
    min_score: float | None = None
    max_score: float | None = None
    rating_range: float | None = None


class RatingBrowseResult(BaseModel):
    items: list[RatingWithBook]
    total: int
    page: int
    page_size: int
    pages: int
