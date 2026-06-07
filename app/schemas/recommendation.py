"""Recommendation API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.book import BookRead


class RecommendationRequest(BaseModel):
    user_id: int
    limit: int = Field(default=10, ge=1, le=50)
    algorithm: Literal["collaborative", "hybrid", "auto"] = "auto"


class SimilarBooksRequest(BaseModel):
    book_id: int
    limit: int = Field(default=10, ge=1, le=50)


class RecommendationItem(BaseModel):
    book: BookRead
    score: float
    algorithm: str
    rank: int


class RecommendationResponse(BaseModel):
    user_id: int | None = None
    seed_book_id: int | None = None
    algorithm: str
    items: list[RecommendationItem]
