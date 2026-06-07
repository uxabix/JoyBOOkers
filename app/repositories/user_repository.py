"""User repository — DS1 collaborative-filtering identities."""

from __future__ import annotations

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.db.models.rating import Rating
from app.db.models.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, User)

    def get_by_external_id(self, external_id: str) -> User | None:
        stmt = select(User).where(User.external_id == external_id)
        return self.session.scalars(stmt).first()

    def get_by_nickname(self, nickname: str) -> User | None:
        stmt = select(User).where(func.lower(User.nickname) == nickname.lower())
        return self.session.scalars(stmt).first()

    def nickname_taken(self, nickname: str) -> bool:
        return self.get_by_nickname(nickname) is not None

    def list_recent(self, *, limit: int = 20) -> list[User]:
        stmt = select(User).order_by(User.id.desc()).limit(limit)
        return list(self.session.scalars(stmt).all())

    def list_top_by_ratings(self, *, limit: int = 20, min_ratings: int = 1) -> list[tuple[User, int]]:
        stmt = (
            select(User, func.count(Rating.id).label("rating_count"))
            .join(Rating, Rating.user_id == User.id)
            .group_by(User.id)
            .having(func.count(Rating.id) >= min_ratings)
            .order_by(func.count(Rating.id).desc())
            .limit(limit)
        )
        return [(row[0], int(row[1])) for row in self.session.execute(stmt).all()]

    def list_browse(
        self,
        *,
        q: str | None = None,
        dataset_only: bool = True,
        min_ratings: int = 1,
        offset: int = 0,
        limit: int = 24,
    ) -> list[tuple[User, int]]:
        rating_count = func.count(Rating.id)
        stmt = (
            select(User, rating_count.label("rating_count"))
            .outerjoin(Rating, Rating.user_id == User.id)
            .group_by(User.id)
        )
        if dataset_only:
            stmt = stmt.where(User.is_registered.is_(False))
        if q:
            pattern = f"%{q.strip()}%"
            stmt = stmt.where(
                or_(
                    User.external_id.ilike(pattern),
                    User.display_name.ilike(pattern),
                )
            )
        stmt = (
            stmt.having(rating_count >= min_ratings)
            .order_by(rating_count.desc(), User.id.asc())
            .offset(offset)
            .limit(limit)
        )
        return [(row[0], int(row[1])) for row in self.session.execute(stmt).all()]

    def count_browse(
        self,
        *,
        q: str | None = None,
        dataset_only: bool = True,
        min_ratings: int = 1,
    ) -> int:
        subq = (
            select(User.id)
            .outerjoin(Rating, Rating.user_id == User.id)
            .group_by(User.id)
        )
        if dataset_only:
            subq = subq.where(User.is_registered.is_(False))
        if q:
            pattern = f"%{q.strip()}%"
            subq = subq.where(
                or_(
                    User.external_id.ilike(pattern),
                    User.display_name.ilike(pattern),
                )
            )
        subq = subq.having(func.count(Rating.id) >= min_ratings)
        return int(self.session.scalar(select(func.count()).select_from(subq.subquery())) or 0)
