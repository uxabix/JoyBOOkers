"""Author API."""

from __future__ import annotations

from urllib.parse import unquote

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_author_service
from app.schemas.author import AuthorBrowseParams, AuthorBrowseResult
from app.schemas.book import BookRead, BookSearchParams
from app.schemas.common import PaginatedResponse
from app.services.author_service import AuthorService

router = APIRouter()


@router.get("", response_model=AuthorBrowseResult)
def browse_authors(
    q: str | None = Query(default=None),
    sort: str = Query(default="book_count_desc"),
    min_books: int = Query(default=1, ge=1),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    service: AuthorService = Depends(get_author_service),
) -> AuthorBrowseResult:
    return service.browse(AuthorBrowseParams(q=q, sort=sort, min_books=min_books, page=page, page_size=page_size))


@router.get("/by/{author_name:path}/books", response_model=PaginatedResponse[BookRead])
def author_books(
    author_name: str,
    q: str | None = Query(default=None),
    genre: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    sort: str = Query(default="title_asc"),
    min_ratings: int = Query(default=0, ge=0),
    service: AuthorService = Depends(get_author_service),
) -> PaginatedResponse[BookRead]:
    name = unquote(author_name)
    params = BookSearchParams(
        q=q,
        genre=genre,
        page=page,
        page_size=page_size,
        sort=sort,
        min_ratings=min_ratings,
    )
    return service.get_books(name, params)
