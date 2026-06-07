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

        rated_source_ids = set(self.ratings.rated_source_book_ids(user_id))
        train_items = self.engine.train_item_ids()
        if not train_items:
            return []

        candidate_source_ids = sorted(bid for bid in train_items if bid not in rated_source_ids)
        if not candidate_source_ids:
            return []

        cap = self.settings.cf_candidate_limit
        if len(candidate_source_ids) > cap:
            candidate_source_ids = candidate_source_ids[:cap]

        raw = self.engine.recommend(user.external_id, candidate_source_ids, limit=limit * 3)

        results: list[tuple[int, float]] = []
        for source_id, score in raw:
            book = self.books.get_by_source_id(source_id)
            if book:
                results.append((book.id, score))
            if len(results) >= limit:
                break
        return results
