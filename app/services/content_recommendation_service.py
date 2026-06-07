"""Content-based recommendation — sparse matrices from DS2+DS3 only."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import Settings
from app.ml.content_based import ContentRecommendationEngine
from app.repositories.book_repository import BookRepository


class ContentRecommendationService:
    def __init__(
        self,
        session: Session,
        engine: ContentRecommendationEngine,
        settings: Settings,
    ) -> None:
        self.session = session
        self.engine = engine
        self.settings = settings
        self.books = BookRepository(session)

    def similar_books(
        self,
        book_id: int,
        *,
        limit: int | None = None,
    ) -> list[tuple[int, float]]:
        limit = limit or self.settings.default_recommendation_limit
        book = self.books.get(book_id)
        if book is None:
            return []

        if not self.engine.is_loaded and not self.engine.load():
            return []

        raw = self.engine.similar_books(book.source_book_id, limit=limit)
        results: list[tuple[int, float]] = []
        for source_id, score in raw:
            match = self.books.get_by_source_id(source_id)
            if match:
                results.append((match.id, score))
        return results
