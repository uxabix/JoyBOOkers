"""User service."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCandidateRead, UserCreate, UserRead


class UserService:
    def __init__(self, session: Session) -> None:
        self.repo = UserRepository(session)
        self.session = session

    def get(self, user_id: int) -> UserRead | None:
        user = self.repo.get(user_id)
        return UserRead.model_validate(user) if user else None

    def get_or_create(self, payload: UserCreate) -> UserRead:
        existing = self.repo.get_by_external_id(payload.external_id)
        if existing:
            return UserRead.model_validate(existing)

        user = User(external_id=payload.external_id, display_name=payload.display_name)
        self.repo.add(user)
        self.session.commit()
        self.session.refresh(user)
        return UserRead.model_validate(user)

    def list_recent(self, *, limit: int = 20) -> list[UserRead]:
        return [UserRead.model_validate(u) for u in self.repo.list_recent(limit=limit)]

    def list_recommendation_candidates(
        self,
        *,
        limit: int = 20,
        min_ratings: int = 3,
    ) -> list[UserCandidateRead]:
        rows = self.repo.list_top_by_ratings(limit=limit, min_ratings=min_ratings)
        return [
            UserCandidateRead(
                id=u.id,
                external_id=u.external_id,
                display_name=u.display_name,
                rating_count=count,
            )
            for u, count in rows
        ]
