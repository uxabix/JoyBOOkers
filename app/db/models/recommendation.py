"""Stored recommendation results for auditing and UI history."""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Recommendation(Base, TimestampMixin):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), index=True)
    algorithm: Mapped[str] = mapped_column(String(64), index=True)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int | None] = mapped_column(nullable=True)

    user: Mapped[User | None] = relationship(back_populates="recommendations")
    book: Mapped[Book] = relationship(back_populates="recommendations")


from app.db.models.book import Book  # noqa: E402
from app.db.models.user import User  # noqa: E402
