"""Book API schemas — DS2 catalog + DS3 enrichment."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class BookEnrichmentRead(ORMModel):
    tags: str | None = None
    characters: str | None = None
    extra_genres: str | None = None
    series: str | None = None


class BookRead(ORMModel):
    id: int
    source_book_id: str
    match_key: str
    title: str
    author: str | None = None
    genre: str | None = None
    description: str | None = None
    isbn: str | None = None
    isbn13: str | None = None
    pages: int | None = None
    avg_rating: float | None = None
    image_url: str | None = None
    goodreads_url: str | None = None
    enrichment: BookEnrichmentRead | None = None


class BookCreate(BaseModel):
    source_book_id: str
    match_key: str
    title: str
    author: str | None = None
    genre: str | None = None
    description: str | None = None
    isbn: str | None = None
    isbn13: str | None = None
    pages: int | None = None
    avg_rating: float | None = None
    image_url: str | None = None
    goodreads_url: str | None = None


class BookSearchParams(BaseModel):
    q: str | None = Field(default=None, description="Search title or author")
    genre: str | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
