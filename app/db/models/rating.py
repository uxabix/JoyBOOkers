"""Explicit user ratings from DS1 (collaborative filtering only)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Rating(Base, TimestampMixin):
    __tablename__ = "ratings"
    __table_args__ = (UniqueConstraint("user_id", "book_id", name="uq_rating_user_book"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="ds1", nullable=False)
    rated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="ratings")
    book: Mapped[Book] = relationship(back_populates="ratings")


from app.db.models.book import Book  # noqa: E402
from app.db.models.user import User  # noqa: E402
