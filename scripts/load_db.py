#!/usr/bin/env python3
"""Load processed pipeline artifacts into SQLite for the FastAPI application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import select

from app.config import Settings, get_settings
from app.db.models.book import Book, BookEnrichment
from app.db.models.rating import Rating
from app.db.models.user import User
from app.db.session import configure_engine, get_session_factory, init_db
from bookrec.io_utils import read_table


def _as_str(value) -> str:
    if value is None:
        return ""
    return str(value)


def _genre_text(value) -> str | None:
    text = _as_str(value).strip()
    return text or None


def _match_key(row) -> str:
    title = _as_str(row.get("title", "")).strip().lower()
    author = _as_str(row.get("authors", "")).strip().lower()
    if title or author:
        return f"{title}|{author}"
    return _as_str(row.get("book_key", row.get("source_book_id", "")))


def load_books(session, catalog_path: Path, *, batch_size: int, limit: int | None) -> dict[str, int]:
    df = read_table(catalog_path)
    if limit:
        df = df.head(limit)

    source_to_id: dict[str, int] = {}
    batch: list[Book] = []

    for row in df.to_dict(orient="records"):
        source_id = _as_str(row["source_book_id"])
        book = Book(
            source_book_id=source_id,
            match_key=_match_key(row),
            title=_as_str(row.get("title", source_id))[:512],
            author=_as_str(row.get("authors"))[:512] or None,
            genre=_genre_text(row.get("genres")),
            description=_as_str(row.get("content_text"))[:5000] or None,
        )
        batch.append(book)
        if len(batch) >= batch_size:
            session.add_all(batch)
            session.flush()
            for b in batch:
                source_to_id[b.source_book_id] = b.id
            session.commit()
            batch.clear()

    if batch:
        session.add_all(batch)
        session.flush()
        for b in batch:
            source_to_id[b.source_book_id] = b.id
        session.commit()

    enrich_batch: list[BookEnrichment] = []
    for row in df.to_dict(orient="records"):
        source_id = _as_str(row["source_book_id"])
        book_id = source_to_id.get(source_id)
        if book_id is None:
            continue
        tags = _genre_text(row.get("tags"))
        chars = _genre_text(row.get("characters"))
        if not tags and not chars:
            continue
        enrich_batch.append(
            BookEnrichment(book_id=book_id, tags=tags, characters=chars)
        )
        if len(enrich_batch) >= batch_size:
            session.add_all(enrich_batch)
            session.commit()
            enrich_batch.clear()
    if enrich_batch:
        session.add_all(enrich_batch)
        session.commit()

    return source_to_id


def ensure_ds1_books(session, interactions, source_to_id: dict[str, int], batch_size: int) -> dict[str, int]:
    needed = sorted(set(interactions["book_id"].astype(str).unique()))
    missing = [bid for bid in needed if bid not in source_to_id]
    batch: list[Book] = []
    for source_id in missing:
        batch.append(
            Book(
                source_book_id=source_id,
                match_key=f"ds1:{source_id}",
                title=f"DS1 book {source_id}",
            )
        )
        if len(batch) >= batch_size:
            session.add_all(batch)
            session.flush()
            for b in batch:
                source_to_id[b.source_book_id] = b.id
            session.commit()
            batch.clear()
    if batch:
        session.add_all(batch)
        session.flush()
        for b in batch:
            source_to_id[b.source_book_id] = b.id
        session.commit()
    return source_to_id


def load_users(session, interactions, *, batch_size: int) -> dict[str, int]:
    external_to_id: dict[str, int] = {}
    user_ids = sorted(set(interactions["user_id"].astype(str).unique()))
    batch: list[User] = []
    for ext in user_ids:
        batch.append(User(external_id=ext, display_name=f"User {ext}"))
        if len(batch) >= batch_size:
            session.add_all(batch)
            session.flush()
            for u in batch:
                external_to_id[u.external_id] = u.id
            session.commit()
            batch.clear()
    if batch:
        session.add_all(batch)
        session.flush()
        for u in batch:
            external_to_id[u.external_id] = u.id
        session.commit()
    return external_to_id


def load_ratings(
    session,
    interactions,
    user_map: dict[str, int],
    book_map: dict[str, int],
    *,
    batch_size: int,
    limit: int | None,
) -> int:
    if limit:
        interactions = interactions.head(limit)
    n = 0
    batch: list[Rating] = []
    for row in interactions.itertuples(index=False):
        ext_user = str(row.user_id)
        source_book = str(row.book_id)
        user_id = user_map.get(ext_user)
        book_id = book_map.get(source_book)
        if user_id is None or book_id is None:
            continue
        batch.append(Rating(user_id=user_id, book_id=book_id, score=float(row.rating), source="ds1"))
        n += 1
        if len(batch) >= batch_size:
            session.add_all(batch)
            session.commit()
            batch.clear()
    if batch:
        session.add_all(batch)
        session.commit()
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load JoyBookers SQLite database from processed artifacts")
    parser.add_argument("--books-limit", type=int, default=None, help="Cap catalog rows (default: all)")
    parser.add_argument("--ratings-limit", type=int, default=None, help="Cap rating rows (default: all)")
    parser.add_argument("--skip-ratings", action="store_true")
    parser.add_argument("--skip-users", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    catalog_path = settings.content_catalog_path
    interactions_path = settings.interactions_path

    if not catalog_path.is_file():
        print(f"Missing catalog: {catalog_path}", file=sys.stderr)
        return 1

    configure_engine(settings.database_url)
    init_db()
    Session = get_session_factory()

    with Session() as session:
        existing = session.scalar(select(Book.id).limit(1))
        if existing and not args.skip_ratings:
            print("Database already has books. Use a fresh DB or delete data/joybookers.db to reload.")

        print(f"Loading books from {catalog_path} ...")
        book_map = load_books(session, catalog_path, batch_size=settings.db_batch_size, limit=args.books_limit)
        print(f"  books: {len(book_map)}")

        if not args.skip_users and interactions_path.is_file():
            interactions = read_table(interactions_path)
            book_map = ensure_ds1_books(session, interactions, book_map, settings.db_batch_size)
            print(f"  books after DS1 stubs: {len(book_map)}")

            print(f"Loading users from {interactions_path} ...")
            user_map = load_users(session, interactions, batch_size=settings.db_batch_size)
            print(f"  users: {len(user_map)}")

            if not args.skip_ratings:
                print("Loading ratings ...")
                n_ratings = load_ratings(
                    session,
                    interactions,
                    user_map,
                    book_map,
                    batch_size=settings.db_batch_size,
                    limit=args.ratings_limit,
                )
                print(f"  ratings: {n_ratings}")
        else:
            print("Skipped users/ratings (missing interactions file or --skip-ratings).")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
