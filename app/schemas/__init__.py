"""Pydantic request/response schemas."""

from app.schemas.book import BookCreate, BookEnrichmentRead, BookRead, BookSearchParams
from app.schemas.common import HealthResponse, MessageResponse, PaginatedResponse
from app.schemas.rating import RatingCreate, RatingRead
from app.schemas.recommendation import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    SimilarBooksRequest,
)
from app.schemas.review import ReviewCreate, ReviewRead
from app.schemas.sentiment import SentimentPredictRequest, SentimentPredictResponse
from app.schemas.user import UserCreate, UserRead

__all__ = [
    "BookCreate",
    "BookEnrichmentRead",
    "BookRead",
    "BookSearchParams",
    "HealthResponse",
    "MessageResponse",
    "PaginatedResponse",
    "RatingCreate",
    "RatingRead",
    "RecommendationItem",
    "RecommendationRequest",
    "RecommendationResponse",
    "ReviewCreate",
    "ReviewRead",
    "SentimentPredictRequest",
    "SentimentPredictResponse",
    "SimilarBooksRequest",
    "UserCreate",
    "UserRead",
]
