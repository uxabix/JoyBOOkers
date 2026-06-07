"""Rating service — DS1 interactions."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models.rating import Rating
from app.repositories.rating_repository import RatingRepository
from app.schemas.rating import RatingCreate, RatingRead


class RatingService:
    def __init__(self, session: Session) -> None:
        self.repo = RatingRepository(session)
        self.session = session

    def create(self, payload: RatingCreate) -> RatingRead:
        existing = self.repo.get_user_book_rating(payload.user_id, payload.book_id)
        if existing:
            existing.score = payload.score
            existing.rated_at = payload.rated_at
            self.session.commit()
            self.session.refresh(existing)
            return RatingRead.model_validate(existing)

        rating = Rating(**payload.model_dump())
        self.repo.add(rating)
        self.session.commit()
        self.session.refresh(rating)
        return RatingRead.model_validate(rating)

    def list_for_user(self, user_id: int) -> list[RatingRead]:
        rows = self.repo.list_for_user(user_id)
        return [RatingRead.model_validate(r) for r in rows]
