"""Author aggregates from the book catalog."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models.book import Book


class AuthorRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _author_order(self, sort: str, author_col, book_count, rating_count, catalog_avg, db_avg):
        mapping = {
            "name_asc": author_col.asc(),
            "name_desc": author_col.desc(),
            "book_count_desc": book_count.desc(),
            "book_count_asc": book_count.asc(),
            "rating_count_desc": rating_count.desc(),
            "rating_count_asc": rating_count.asc(),
            "catalog_rating_desc": catalog_avg.desc().nullslast(),
            "db_rating_desc": db_avg.desc().nullslast(),
        }
        return mapping.get(sort, book_count.desc())

    def list_authors(
        self,
        *,
        q: str | None = None,
        min_books: int = 1,
        sort: str = "book_count_desc",
        offset: int = 0,
        limit: int = 24,
    ) -> list[tuple[str, int, int, float | None, float | None]]:
        book_count = func.count(Book.id)
        rating_count = func.coalesce(func.sum(Book.rating_count), 0)
        catalog_avg = func.avg(Book.avg_rating)
        db_avg = func.avg(Book.db_avg_rating)

        stmt = (
            select(Book.author, book_count, rating_count, catalog_avg, db_avg)
            .where(Book.author.isnot(None), func.trim(Book.author) != "")
            .group_by(Book.author)
        )
        if q and q.strip():
            stmt = stmt.where(Book.author.ilike(f"%{q.strip()}%"))
        stmt = stmt.having(book_count >= min_books)
        stmt = stmt.order_by(
            self._author_order(sort, Book.author, book_count, rating_count, catalog_avg, db_avg)
        )
        stmt = stmt.offset(offset).limit(limit)
        return [(row[0], int(row[1]), int(row[2]), row[3], row[4]) for row in self.session.execute(stmt).all()]

    def count_authors(self, *, q: str | None = None, min_books: int = 1) -> int:
        stmt = (
            select(Book.author)
            .where(Book.author.isnot(None), func.trim(Book.author) != "")
            .group_by(Book.author)
            .having(func.count(Book.id) >= min_books)
        )
        if q and q.strip():
            stmt = stmt.where(Book.author.ilike(f"%{q.strip()}%"))
        return int(self.session.scalar(select(func.count()).select_from(stmt.subquery())) or 0)
