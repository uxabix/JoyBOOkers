"""Book catalog repository — DS2 + DS3 enrichment."""

from __future__ import annotations

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.db.models.book import Book, BookEnrichment
from app.repositories.base import BaseRepository


class BookRepository(BaseRepository[Book]):
    def __init__(self, session: Session) -> None:
        super().__init__(session, Book)

    def get_with_enrichment(self, book_id: int) -> Book | None:
        stmt = (
            select(Book)
            .options(joinedload(Book.enrichment))
            .where(Book.id == book_id)
        )
        return self.session.scalars(stmt).unique().first()

    def get_by_source_id(self, source_book_id: str) -> Book | None:
        stmt = select(Book).where(Book.source_book_id == source_book_id)
        return self.session.scalars(stmt).first()

    def existing_source_ids(self, source_ids: list[str]) -> set[str]:
        if not source_ids:
            return set()
        stmt = select(Book.source_book_id).where(Book.source_book_id.in_(source_ids))
        return {str(row) for row in self.session.scalars(stmt).all()}

    def search(
        self,
        *,
        q: str | None = None,
        genre: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[Book], int]:
        stmt: Select[tuple[Book]] = select(Book)
        count_stmt = select(func.count()).select_from(Book)

        if q:
            pattern = f"%{q}%"
            filt = or_(Book.title.ilike(pattern), Book.author.ilike(pattern))
            stmt = stmt.where(filt)
            count_stmt = count_stmt.where(filt)

        if genre:
            pattern = f"%{genre}%"
            stmt = stmt.where(Book.genre.ilike(pattern))
            count_stmt = count_stmt.where(Book.genre.ilike(pattern))

        total = int(self.session.scalar(count_stmt) or 0)
        rows = list(
            self.session.scalars(stmt.order_by(Book.title).offset(offset).limit(limit)).all()
        )
        return rows, total

    def upsert_enrichment(self, book_id: int, **fields: str | None) -> BookEnrichment:
        book = self.get(book_id)
        if book is None:
            raise ValueError(f"Book {book_id} not found")

        enrichment = book.enrichment
        if enrichment is None:
            enrichment = BookEnrichment(book_id=book_id, **fields)
            self.session.add(enrichment)
        else:
            for key, value in fields.items():
                setattr(enrichment, key, value)

        self.session.flush()
        return enrichment
