"""Collaborative filtering service — DS1 Surprise SVD."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import Settings
from app.ml.collaborative import CollaborativeFilteringEngine
from app.repositories.book_repository import BookRepository
from app.repositories.rating_repository import RatingRepository
from app.repositories.user_repository import UserRepository


class CollaborativeFilteringService:
    def __init__(
        self,
        session: Session,
        engine: CollaborativeFilteringEngine,
        settings: Settings,
    ) -> None:
        self.session = session
        self.engine = engine
        self.settings = settings
        self.users = UserRepository(session)
        self.books = BookRepository(session)
        self.ratings = RatingRepository(session)

    def recommend_for_user(
        self,
        user_id: int,
        *,
        limit: int | None = None,
    ) -> list[tuple[int, float]]:
        limit = limit or self.settings.default_recommendation_limit
        user = self.users.get(user_id)
        if user is None:
            return []

        if self.ratings.count_for_user(user_id) < self.settings.min_cf_ratings_per_user:
            return []

        if not self.engine.is_loaded and not self.engine.load():
            return []

        catalog, _ = self.books.search(limit=5000)
        candidate_ids = [b.source_book_id for b in catalog]
        raw = self.engine.recommend(user.external_id, candidate_ids, limit=limit)

        results: list[tuple[int, float]] = []
        for source_id, score in raw:
            book = self.books.get_by_source_id(source_id)
            if book:
                results.append((book.id, score))
        return results
