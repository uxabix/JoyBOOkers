"""Starter books and content-based recommendations for new / sparse users."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import Settings
from app.ml.content_based import ContentRecommendationEngine
from app.repositories.book_repository import BookRepository
from app.repositories.rating_repository import RatingRepository
from app.schemas.book import BookRead


class ColdStartService:
    def __init__(
        self,
        session: Session,
        content_engine: ContentRecommendationEngine,
        settings: Settings,
    ) -> None:
        self.session = session
        self.content = content_engine
        self.settings = settings
        self.books = BookRepository(session)
        self.ratings = RatingRepository(session)

    def starter_books(self, user_id: int, *, limit: int | None = None) -> list[BookRead]:
        limit = limit or self.settings.cold_start_book_count
        rows = self.books.list_starter_books(user_id, limit=limit)
        return [BookRead.model_validate(b) for b in rows]

    def recommend_for_user(
        self,
        user_id: int,
        *,
        limit: int | None = None,
    ) -> list[tuple[int, float]]:
        limit = limit or self.settings.default_recommendation_limit
        rated_ids = self.ratings.book_ids_for_user(user_id)
        if not rated_ids:
            starters = self.books.list_starter_books(user_id, limit=limit)
            return [(b.id, float(b.avg_rating or 4.0)) for b in starters[:limit]]

        if not self.content.is_loaded and not self.content.load():
            starters = self.books.list_starter_books(user_id, limit=limit)
            return [(b.id, 3.5) for b in starters[:limit]]

        scores: dict[int, float] = {}
        rated_books = [self.books.get(bid) for bid in rated_ids[:5]]
        for book in rated_books:
            if book is None:
                continue
            for source_id, sim in self.content.similar_books(book.source_book_id, limit=limit):
                match = self.books.get_by_source_id(source_id)
                if match and match.id not in rated_ids:
                    scores[match.id] = max(scores.get(match.id, 0.0), sim * (book.avg_rating or 4.0) / 5.0)

        if not scores:
            starters = self.books.list_starter_books(user_id, limit=limit)
            return [(b.id, 3.5) for b in starters[:limit]]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return ranked
