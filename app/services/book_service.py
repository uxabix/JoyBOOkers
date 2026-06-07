"""Book catalog service — DS2 metadata + DS3 enrichment."""

from __future__ import annotations

import math

from sqlalchemy.orm import Session

from app.db.models.book import Book
from app.repositories.book_repository import BookRepository
from app.schemas.book import BookCreate, BookRead, BookSearchParams
from app.schemas.common import PaginatedResponse


class BookService:
    def __init__(self, session: Session) -> None:
        self.repo = BookRepository(session)
        self.session = session

    def get(self, book_id: int) -> BookRead | None:
        book = self.repo.get_with_enrichment(book_id)
        return BookRead.model_validate(book) if book else None

    def create(self, payload: BookCreate) -> BookRead:
        book = Book(**payload.model_dump())
        self.repo.add(book)
        self.session.commit()
        self.session.refresh(book)
        return BookRead.model_validate(book)

    def search(self, params: BookSearchParams) -> PaginatedResponse[BookRead]:
        offset = (params.page - 1) * params.page_size
        rows, total = self.repo.search(
            q=params.q,
            genre=params.genre,
            offset=offset,
            limit=params.page_size,
        )
        pages = max(1, math.ceil(total / params.page_size))
        return PaginatedResponse[BookRead](
            items=[BookRead.model_validate(b) for b in rows],
            total=total,
            page=params.page,
            page_size=params.page_size,
            pages=pages,
        )
