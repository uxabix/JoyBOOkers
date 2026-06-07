"""Rating repository — DS1 user-item interactions."""

from __future__ import annotations

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from app.db.models.book import Book
from app.db.models.rating import Rating
from app.repositories.base import BaseRepository


class RatingRepository(BaseRepository[Rating]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, Rating)

    def list_for_user(self, user_id: int, *, limit: int = 500) -> list[Rating]:
        stmt: Select[tuple[Rating]] = (
            select(Rating)
            .where(Rating.user_id == user_id)
            .order_by(Rating.rated_at.desc().nullslast())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())

    def count_for_user(self, user_id: int) -> int:
        stmt = select(func.count()).select_from(Rating).where(Rating.user_id == user_id)
        return int(self.session.scalar(stmt) or 0)

    def get_user_book_rating(self, user_id: int, book_id: int) -> Rating | None:
        stmt = select(Rating).where(Rating.user_id == user_id, Rating.book_id == book_id)
        return self.session.scalars(stmt).first()

    def rated_source_book_ids(self, user_id: int) -> list[str]:
        stmt = (
            select(Book.source_book_id)
            .join(Rating, Rating.book_id == Book.id)
            .where(Rating.user_id == user_id)
        )
        return [str(row) for row in self.session.scalars(stmt).all()]

    def book_ids_for_user(self, user_id: int) -> list[int]:
        stmt = select(Rating.book_id).where(Rating.user_id == user_id)
        return list(self.session.scalars(stmt).all())

    def scores_for_user(self, user_id: int) -> list[float]:
        stmt = select(Rating.score).where(Rating.user_id == user_id)
        return [float(s) for s in self.session.scalars(stmt).all()]

    def list_with_books(self, user_id: int, *, limit: int = 100) -> list[Rating]:
        return self.browse_ratings(user_id, limit=limit)

    def _browse_filters(
        self,
        stmt: Select,
        *,
        q: str | None,
        min_score: float | None,
        max_score: float | None,
    ) -> Select:
        if q and q.strip():
            pattern = f"%{q.strip()}%"
            stmt = stmt.where(
                or_(
                    Book.title.ilike(pattern),
                    Book.author.ilike(pattern),
                )
            )
        if min_score is not None:
            stmt = stmt.where(Rating.score >= min_score)
        if max_score is not None:
            stmt = stmt.where(Rating.score <= max_score)
        return stmt

    def _browse_order(self, sort: str):
        mapping = {
            "title_asc": Book.title.asc(),
            "title_desc": Book.title.desc(),
            "score_asc": Rating.score.asc(),
            "score_desc": Rating.score.desc(),
            "updated_asc": Rating.updated_at.asc(),
            "updated_desc": Rating.updated_at.desc(),
        }
        return mapping.get(sort, Rating.updated_at.desc())

    def count_browse(
        self,
        user_id: int,
        *,
        q: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
    ) -> int:
        stmt = (
            select(func.count(Rating.id))
            .select_from(Rating)
            .join(Book, Rating.book_id == Book.id)
            .where(Rating.user_id == user_id)
        )
        stmt = self._browse_filters(stmt, q=q, min_score=min_score, max_score=max_score)
        return int(self.session.scalar(stmt) or 0)

    def browse_ratings(
        self,
        user_id: int,
        *,
        q: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
        sort: str = "updated_desc",
        offset: int = 0,
        limit: int = 15,
    ) -> list[Rating]:
        stmt = (
            select(Rating)
            .join(Book, Rating.book_id == Book.id)
            .where(Rating.user_id == user_id)
        )
        stmt = self._browse_filters(stmt, q=q, min_score=min_score, max_score=max_score)
        stmt = stmt.order_by(self._browse_order(sort)).offset(offset).limit(limit)
        return list(self.session.scalars(stmt).unique().all())
