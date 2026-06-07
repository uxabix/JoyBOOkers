"""URL query builders for web filters."""

from __future__ import annotations

from urllib.parse import quote, urlencode

from app.schemas.author import AuthorBrowseParams
from app.schemas.book import BookSearchParams


def book_query(params: BookSearchParams, **overrides) -> str:
    data = params.model_dump()
    data.update(overrides)
    cleaned: dict[str, str | int | float | bool] = {}
    data.pop("page", None)
    for key, value in data.items():
        if value is None or value == "":
            continue
        if key == "min_ratings" and value == 0:
            continue
        cleaned[key] = value
    return urlencode(cleaned)


def author_query(params: AuthorBrowseParams, **overrides) -> str:
    data = params.model_dump()
    data.update(overrides)
    data.pop("page", None)
    cleaned = {k: v for k, v in data.items() if v not in (None, "") and not (k == "min_books" and v == 1)}
    return urlencode(cleaned)


def author_path(name: str) -> str:
    return f"/authors/by/{quote(name, safe='')}"
