"""Orchestrates CF (DS1) and content-based (DS2+DS3) recommenders."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models.recommendation import Recommendation
from app.repositories.book_repository import BookRepository
from app.repositories.recommendation_repository import RecommendationRepository
from app.repositories.user_repository import UserRepository
from app.schemas.book import BookRead
from app.schemas.recommendation import RecommendationItem, RecommendationResponse
from app.services.cold_start_service import ColdStartService
from app.services.collaborative_filtering_service import CollaborativeFilteringService
from app.services.content_recommendation_service import ContentRecommendationService


class RecommendationService:
    def __init__(
        self,
        session: Session,
        cf_service: CollaborativeFilteringService,
        content_service: ContentRecommendationService,
        cold_start_service: ColdStartService,
        settings: Settings,
    ) -> None:
        self.session = session
        self.cf = cf_service
        self.content = content_service
        self.cold_start = cold_start_service
        self.settings = settings
        self.books = BookRepository(session)
        self.users = UserRepository(session)
        self.history = RecommendationRepository(session)

    def recommend_for_user(
        self,
        user_id: int,
        *,
        limit: int | None = None,
        algorithm: str = "auto",
    ) -> RecommendationResponse:
        limit = limit or self.settings.default_recommendation_limit
        user = self.users.get(user_id)
        algo = algorithm
        pairs: list[tuple[int, float]] = []

        if user and user.is_registered:
            pairs = self.cold_start.recommend_for_user(user_id, limit=limit)
            n = self.cf.ratings.count_for_user(user_id)
            algo = "cold_start" if n == 0 else "content_profile"
        else:
            if algo in ("collaborative", "auto"):
                pairs = self.cf.recommend_for_user(user_id, limit=limit)
                algo = "collaborative" if pairs else algo

            if not pairs and algo in ("hybrid", "auto", "content"):
                rated = self.cf.ratings.list_for_user(user_id, limit=1)
                if rated:
                    pairs = self.content.similar_books(rated[0].book_id, limit=limit)
                    algo = "content_fallback"

            if not pairs and algo in ("hybrid", "auto"):
                pairs = self.cold_start.recommend_for_user(user_id, limit=limit)
                algo = "cold_start"

        items = self._to_items(pairs, algorithm=algo)
        if items:
            self._persist(user_id=user_id, items=items)
        return RecommendationResponse(user_id=user_id, algorithm=algo, items=items)

    def explain_empty(self, user_id: int) -> str:
        user = self.users.get(user_id)
        if user is None:
            return (
                f"User #{user_id} not found. Use the internal database ID from the table below "
                "(not the DS1 external id)."
            )
        if user.is_registered:
            return (
                "No recommendations could be generated. Rate a few books from your profile "
                "or browse the catalog — suggestions will improve as you rate more titles."
            )
        n = self.cf.ratings.count_for_user(user_id)
        if n < self.settings.min_cf_ratings_per_user:
            return (
                f"User #{user_id} has only {n} rating(s) in the database; "
                f"at least {self.settings.min_cf_ratings_per_user} are required. "
                "Try a user from the list below or run "
                "`python scripts/load_db.py` with a higher `--ratings-limit`."
            )
        if not self.cf.engine.is_loaded:
            return "Collaborative model is not loaded. Check GET /api/v1/health/ready."
        return (
            "Model ran but returned no catalog matches. Try user #1 or another ID from the "
            "table below (users sorted by rating count)."
        )

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
