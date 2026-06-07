"""Data access layer."""

from app.repositories.book_repository import BookRepository
from app.repositories.rating_repository import RatingRepository
from app.repositories.recommendation_repository import RecommendationRepository
from app.repositories.review_repository import ReviewRepository
from app.repositories.user_repository import UserRepository

__all__ = [
    "BookRepository",
    "RatingRepository",
    "RecommendationRepository",
    "ReviewRepository",
    "UserRepository",
]
