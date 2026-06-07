"""User repository — DS1 collaborative-filtering identities."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, User)

    def get_by_external_id(self, external_id: str) -> User | None:
        stmt = select(User).where(User.external_id == external_id)
        return self.session.scalars(stmt).first()

    def list_recent(self, *, limit: int = 20) -> list[User]:
        stmt = select(User).order_by(User.id.desc()).limit(limit)
        return list(self.session.scalars(stmt).all())
