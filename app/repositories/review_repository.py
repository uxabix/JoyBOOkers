"""Amazon review repository — DS4 independent corpus."""

from __future__ import annotations

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.db.models.review import Review
from app.repositories.base import BaseRepository


class ReviewRepository(BaseRepository[Review]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, Review)

    def list_by_asin(self, asin: str, *, limit: int = 50) -> list[Review]:
        stmt: Select[tuple[Review]] = (
            select(Review)
            .where(Review.asin == asin)
            .order_by(Review.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())
