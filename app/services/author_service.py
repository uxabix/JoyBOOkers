"""Author browse service."""

from __future__ import annotations

import math

from sqlalchemy.orm import Session

from app.repositories.author_repository import AuthorRepository
from app.repositories.book_repository import BookRepository
from app.schemas.author import AuthorBrowseParams, AuthorBrowseResult, AuthorSummary
from app.schemas.book import BookRead, BookSearchParams
from app.schemas.common import PaginatedResponse
from app.services.book_service import BookService


class AuthorService:
    def __init__(self, session: Session) -> None:
        self.repo = AuthorRepository(session)
        self.books = BookRepository(session)
        self.session = session

    def browse(self, params: AuthorBrowseParams) -> AuthorBrowseResult:
        offset = (params.page - 1) * params.page_size
        rows = self.repo.list_authors(
            q=params.q,
            min_books=params.min_books,
            sort=params.sort,
            offset=offset,
            limit=params.page_size,
        )
        total = self.repo.count_authors(q=params.q, min_books=params.min_books)
        items = [
            AuthorSummary(
                name=name,
                book_count=book_count,
                rating_count=rating_count,
                catalog_avg_rating=round(catalog_avg, 2) if catalog_avg is not None else None,
                db_avg_rating=round(db_avg, 2) if db_avg is not None else None,
            )
            for name, book_count, rating_count, catalog_avg, db_avg in rows
        ]
        pages = max(1, math.ceil(total / params.page_size))
        return AuthorBrowseResult(
            items=items,
            total=total,
            page=params.page,
            page_size=params.page_size,
            pages=pages,
        )

    def get_books(self, author_name: str, params: BookSearchParams) -> PaginatedResponse[BookRead]:
        params = params.model_copy(update={"author": author_name})
        return BookService(self.session).search(params)
