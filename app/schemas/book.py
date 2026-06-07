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
    rating_count: int = 0
    db_avg_rating: float | None = None


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
    author: str | None = Field(default=None, description="Exact author name filter")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    sort: str = Field(default="title_asc")
    min_ratings: int = Field(default=0, ge=0)
    max_ratings: int | None = Field(default=None, ge=0)
    min_db_rating: float | None = Field(default=None, ge=1.0, le=5.0)
    max_db_rating: float | None = Field(default=None, ge=1.0, le=5.0)
    min_catalog_rating: float | None = Field(default=None, ge=0.0, le=5.0)
