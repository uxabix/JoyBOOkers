"""Orchestrates CF (DS1) and content-based (DS2+DS3) recommenders."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models.recommendation import Recommendation
from app.repositories.book_repository import BookRepository
from app.repositories.recommendation_repository import RecommendationRepository
from app.schemas.book import BookRead
from app.schemas.recommendation import RecommendationItem, RecommendationResponse
from app.services.collaborative_filtering_service import CollaborativeFilteringService
from app.services.content_recommendation_service import ContentRecommendationService


class RecommendationService:
    def __init__(
        self,
        session: Session,
        cf_service: CollaborativeFilteringService,
        content_service: ContentRecommendationService,
        settings: Settings,
    ) -> None:
        self.session = session
        self.cf = cf_service
        self.content = content_service
        self.settings = settings
        self.books = BookRepository(session)
        self.history = RecommendationRepository(session)

    def recommend_for_user(
        self,
        user_id: int,
        *,
        limit: int | None = None,
        algorithm: str = "auto",
    ) -> RecommendationResponse:
        limit = limit or self.settings.default_recommendation_limit
        algo = algorithm
        pairs: list[tuple[int, float]] = []

        if algo in ("collaborative", "auto"):
            pairs = self.cf.recommend_for_user(user_id, limit=limit)
            algo = "collaborative" if pairs else algo

        if not pairs and algo in ("hybrid", "auto"):
            rated = self.cf.ratings.list_for_user(user_id, limit=1)
            if rated:
                pairs = self.content.similar_books(rated[0].book_id, limit=limit)
                algo = "content_fallback"

        items = self._to_items(pairs, algorithm=algo)
        self._persist(user_id=user_id, items=items)
        return RecommendationResponse(user_id=user_id, algorithm=algo, items=items)

    def similar_books(self, book_id: int, *, limit: int | None = None) -> RecommendationResponse:
        limit = limit or self.settings.default_recommendation_limit
        pairs = self.content.similar_books(book_id, limit=limit)
        items = self._to_items(pairs, algorithm="content")
        return RecommendationResponse(seed_book_id=book_id, algorithm="content", items=items)

    def _to_items(self, pairs: list[tuple[int, float]], *, algorithm: str) -> list[RecommendationItem]:
        items: list[RecommendationItem] = []
        for rank, (book_id, score) in enumerate(pairs, start=1):
            book = self.books.get_with_enrichment(book_id)
            if book is None:
                continue
            items.append(
                RecommendationItem(
                    book=BookRead.model_validate(book),
                    score=score,
                    algorithm=algorithm,
                    rank=rank,
                )
            )
        return items

    def _persist(self, *, user_id: int, items: list[RecommendationItem]) -> None:
        for item in items:
            rec = Recommendation(
                user_id=user_id,
                book_id=item.book.id,
                algorithm=item.algorithm,
                score=item.score,
                rank=item.rank,
            )
            self.history.add(rec)
        self.session.commit()
