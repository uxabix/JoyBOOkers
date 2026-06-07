"""Rating service."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.book_stats import refresh_book_rating_stats
from app.db.models.rating import Rating
from app.repositories.book_repository import BookRepository
from app.repositories.rating_repository import RatingRepository
from app.repositories.user_repository import UserRepository
from app.schemas.book import BookRead
from app.schemas.rating import RatingBrowseResult, RatingCreate, RatingRead, RatingWithBook, UserRatingStats
from app.services.clustering_service import ClusteringService


class RatingService:
    def __init__(
        self,
        session: Session,
        clustering_service: ClusteringService | None = None,
    ) -> None:
        self.repo = RatingRepository(session)
        self.books = BookRepository(session)
        self.users = UserRepository(session)
        self.session = session
        self.clustering = clustering_service

    def create(self, payload: RatingCreate) -> RatingRead:
        if payload.user_id is None:
            raise ValueError("user_id is required")
        existing = self.repo.get_user_book_rating(payload.user_id, payload.book_id)
        if existing:
            existing.score = payload.score
            existing.source = "app"
            if payload.rated_at:
                existing.rated_at = payload.rated_at
            self.session.commit()
            self.session.refresh(existing)
            self._after_rating(payload.user_id, payload.book_id)
            return RatingRead.model_validate(existing)

        rating = Rating(
            user_id=payload.user_id,
            book_id=payload.book_id,
            score=payload.score,
            source="app",
            rated_at=payload.rated_at,
        )
        self.repo.add(rating)
        self.session.commit()
        self.session.refresh(rating)
        self._after_rating(payload.user_id, payload.book_id)
        return RatingRead.model_validate(rating)

    def _after_rating(self, user_id: int, book_id: int) -> None:
        refresh_book_rating_stats(self.session, book_id)
        self.session.commit()
        if self.clustering is not None:
            self.clustering.update_user_cluster(user_id)

    def list_for_user(self, user_id: int) -> list[RatingRead]:
        rows = self.repo.list_for_user(user_id)
        return [RatingRead.model_validate(r) for r in rows]

    def _to_with_books(self, rows: list[Rating]) -> list[RatingWithBook]:
        out: list[RatingWithBook] = []
        for r in rows:
            book = self.books.get_with_enrichment(r.book_id)
            if book is None:
                continue
            out.append(
                RatingWithBook(
                    id=r.id,
                    user_id=r.user_id,
                    book_id=r.book_id,
                    score=r.score,
                    source=r.source,
                    rated_at=r.rated_at,
                    book=BookRead.model_validate(book),
                )
            )
        return out

    def list_with_books(self, user_id: int, *, limit: int = 100) -> list[RatingWithBook]:
        rows = self.repo.list_with_books(user_id, limit=limit)
        return self._to_with_books(rows)

    def user_stats(self, user_id: int) -> UserRatingStats:
        scores = self.repo.scores_for_user(user_id)
        if not scores:
            return UserRatingStats()
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((s - mean) ** 2 for s in scores) / n
        std = variance**0.5
        lo, hi = min(scores), max(scores)
        return UserRatingStats(
            n_ratings=n,
            mean_rating=round(mean, 2),
            std_rating=round(std, 3) if n > 1 else 0.0,
            min_score=lo,
            max_score=hi,
            rating_range=round(hi - lo, 2),
        )

    def browse_with_books(
        self,
        user_id: int,
        *,
        q: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        sort: str = "updated_desc",
        page: int = 1,
        page_size: int = 15,
    ) -> RatingBrowseResult:
        page = max(page, 1)
        page_size = max(1, min(page_size, 100))
        offset = (page - 1) * page_size
        total = self.repo.count_browse(
            user_id, q=q, min_score=min_score, max_score=max_score
        )
        rows = self.repo.browse_ratings(
            user_id,
            q=q,
            min_score=min_score,
            max_score=max_score,
            sort=sort,
            offset=offset,
            limit=page_size,
        )
        pages = max(1, (total + page_size - 1) // page_size)
        return RatingBrowseResult(
            items=self._to_with_books(rows),
            total=total,
            page=page,
            page_size=page_size,
            pages=pages,
        )
