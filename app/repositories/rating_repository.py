"""Rating repository — DS1 user-item interactions."""

from __future__ import annotations

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

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
