"""Author browse schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AuthorSummary(BaseModel):
    name: str
    book_count: int = 0
    rating_count: int = 0
    catalog_avg_rating: float | None = None
    db_avg_rating: float | None = None


class AuthorBrowseParams(BaseModel):
    q: str | None = None
    sort: str = Field(default="book_count_desc")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=24, ge=1, le=100)
    min_books: int = Field(default=1, ge=1)


class AuthorBrowseResult(BaseModel):
    items: list[AuthorSummary]
    total: int
    page: int
    page_size: int
    pages: int
