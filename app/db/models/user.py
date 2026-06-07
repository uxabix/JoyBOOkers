"""User model — DS1 imports and registered app users."""

from __future__ import annotations

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    nickname: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True, index=True)
    is_registered: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    ratings: Mapped[list[Rating]] = relationship(back_populates="user", cascade="all, delete-orphan")
    recommendations: Mapped[list[Recommendation]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


from app.db.models.rating import Rating  # noqa: E402
from app.db.models.recommendation import Recommendation  # noqa: E402
