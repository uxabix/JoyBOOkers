"""Recommendation history repository."""

from __future__ import annotations

from sqlalchemy import Select, select
from sqlalchemy.orm import Session, joinedload

from app.db.models.recommendation import Recommendation
from app.repositories.base import BaseRepository


class RecommendationRepository(BaseRepository[Recommendation]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, Recommendation)

    def list_for_user(self, user_id: int, *, limit: int = 50) -> list[Recommendation]:
        stmt: Select[tuple[Recommendation]] = (
            select(Recommendation)
            .options(joinedload(Recommendation.book))
            .where(Recommendation.user_id == user_id)
            .order_by(Recommendation.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).unique().all())
