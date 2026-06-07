"""Book models — DS2 content catalog + DS3 enrichment."""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


class Book(Base, TimestampMixin):
    """Content-catalog book (DS2 Goodreads 100k — not the 2M DS1 catalog)."""

    __tablename__ = "books"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_book_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    match_key: Mapped[str] = mapped_column(String(512), index=True)
    title: Mapped[str] = mapped_column(String(512), index=True)
    author: Mapped[str | None] = mapped_column(String(512), nullable=True)
    genre: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    isbn: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    isbn13: Mapped[str | None] = mapped_column(String(32), nullable=True)
    pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    rating_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    db_avg_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    goodreads_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    enrichment: Mapped[BookEnrichment | None] = relationship(
        back_populates="book",
        uselist=False,
        cascade="all, delete-orphan",
    )
    ratings: Mapped[list[Rating]] = relationship(back_populates="book", cascade="all, delete-orphan")
    recommendations: Mapped[list[Recommendation]] = relationship(
        back_populates="book",
        cascade="all, delete-orphan",
    )


class BookEnrichment(Base, TimestampMixin):
    """DS3 tags and characters merged onto DS2 books via match_key."""

    __tablename__ = "book_enrichments"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    book_id: Mapped[int] = mapped_column(ForeignKey("books.id", ondelete="CASCADE"), unique=True)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    characters: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra_genres: Mapped[str | None] = mapped_column(Text, nullable=True)
    series: Mapped[str | None] = mapped_column(String(512), nullable=True)

    book: Mapped[Book] = relationship(back_populates="enrichment")


from app.db.models.rating import Rating  # noqa: E402
from app.db.models.recommendation import Recommendation  # noqa: E402
