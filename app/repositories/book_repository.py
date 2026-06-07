"""Book catalog repository — DS2 + DS3 enrichment."""

from __future__ import annotations

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.db.models.book import Book, BookEnrichment
from app.db.models.rating import Rating
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

    def list_starter_books(self, user_id: int, *, limit: int = 15) -> list[Book]:
        rated_subq = select(Rating.book_id).where(Rating.user_id == user_id)
        stmt = (
            select(Book)
            .where(Book.id.notin_(rated_subq))
            .where(Book.rating_count > 0)
            .order_by(Book.rating_count.desc(), Book.db_avg_rating.desc().nullslast())
            .limit(limit)
        )
        rows = list(self.session.scalars(stmt).all())
        if len(rows) >= limit:
            return rows
        extra_stmt = select(Book).where(Book.id.notin_(rated_subq)).order_by(Book.title)
        if rows:
            extra_stmt = extra_stmt.where(Book.id.notin_([b.id for b in rows]))
        extra = extra_stmt.limit(limit - len(rows))
        rows.extend(self.session.scalars(extra).all())
        return rows[:limit]

    def existing_source_ids(self, source_ids: list[str]) -> set[str]:
        if not source_ids:
            return set()
        stmt = select(Book.source_book_id).where(Book.source_book_id.in_(source_ids))
        return {str(row) for row in self.session.scalars(stmt).all()}

    def _apply_filters(
        self,
        stmt: Select,
        *,
        q: str | None,
        genre: str | None,
        author: str | None,
        min_ratings: int,
        max_ratings: int | None,
        min_db_rating: float | None,
        max_db_rating: float | None,
        min_catalog_rating: float | None,
    ) -> Select:
        if q:
            pattern = f"%{q}%"
            stmt = stmt.where(or_(Book.title.ilike(pattern), Book.author.ilike(pattern)))
        if genre:
            stmt = stmt.where(Book.genre.ilike(f"%{genre}%"))
        if author:
            stmt = stmt.where(Book.author == author)
        if min_ratings > 0:
            stmt = stmt.where(Book.rating_count >= min_ratings)
        if max_ratings is not None:
            stmt = stmt.where(Book.rating_count <= max_ratings)
        if min_db_rating is not None:
            stmt = stmt.where(Book.db_avg_rating >= min_db_rating)
        if max_db_rating is not None:
            stmt = stmt.where(Book.db_avg_rating <= max_db_rating)
        if min_catalog_rating is not None:
            stmt = stmt.where(Book.avg_rating >= min_catalog_rating)
        return stmt

    def _order_by(self, sort: str):
        mapping = {
            "title_asc": Book.title.asc(),
            "title_desc": Book.title.desc(),
            "rating_count_desc": Book.rating_count.desc(),
            "rating_count_asc": Book.rating_count.asc(),
            "db_rating_desc": Book.db_avg_rating.desc().nullslast(),
            "db_rating_asc": Book.db_avg_rating.asc().nullslast(),
            "catalog_rating_desc": Book.avg_rating.desc().nullslast(),
            "catalog_rating_asc": Book.avg_rating.asc().nullslast(),
        }
        return mapping.get(sort, Book.title.asc())

    def search(
        self,
        *,
        q: str | None = None,
        genre: str | None = None,
        author: str | None = None,
        min_ratings: int = 0,
        max_ratings: int | None = None,
        min_db_rating: float | None = None,
        max_db_rating: float | None = None,
        min_catalog_rating: float | None = None,
        sort: str = "title_asc",
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[Book], int]:
        stmt = select(Book)
        stmt = self._apply_filters(
            stmt,
            q=q,
            genre=genre,
            author=author,
            min_ratings=min_ratings,
            max_ratings=max_ratings,
            min_db_rating=min_db_rating,
            max_db_rating=max_db_rating,
            min_catalog_rating=min_catalog_rating,
        )
        total = int(self.session.scalar(select(func.count()).select_from(stmt.subquery())) or 0)
        rows = list(
            self.session.scalars(
                stmt.order_by(self._order_by(sort)).offset(offset).limit(limit)
            ).all()
        )
        return rows, total

    def rating_stats_for_book(self, book_id: int) -> tuple[int, float | None]:
        book = self.get(book_id)
        if book is None:
            return 0, None
        return int(book.rating_count or 0), book.db_avg_rating

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
