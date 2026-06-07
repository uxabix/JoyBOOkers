"""Recommendation API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.book import BookRead


class RecommendationRequest(BaseModel):
    user_id: int
    limit: int = Field(default=10, ge=1, le=50)
    algorithm: Literal["collaborative", "content", "hybrid", "auto"] = "auto"


class SimilarBooksRequest(BaseModel):
    book_id: int
    limit: int = Field(default=10, ge=1, le=50)


class ScoreBreakdown(BaseModel):
    cf: float = 0.0
    content: float = 0.0
    cluster: float = 0.0
    popularity: float = 0.0
    genre: float = 0.0


class UserProfileSummary(BaseModel):
    cluster_id: int
    cluster_label: str | None = None
    rating_count: int = 0
    profile_strength: float = 0.0
    top_genres: list[str] = Field(default_factory=list)
    cf_available: bool = False
    weights_used: dict[str, float] = Field(default_factory=dict)


class RecommendationItem(BaseModel):
    book: BookRead
    score: float
    algorithm: str
    rank: int
    score_breakdown: ScoreBreakdown | None = None


class RecommendationResponse(BaseModel):
    user_id: int | None = None
    seed_book_id: int | None = None
    algorithm: str
    items: list[RecommendationItem]
    profile: UserProfileSummary | None = None
