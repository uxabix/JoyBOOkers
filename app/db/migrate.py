"""Lightweight SQLite schema migrations (add columns if missing)."""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.db.book_stats import backfill_all_book_rating_stats, stats_need_backfill
from app.logging_config import get_logger

logger = get_logger(__name__)

_USER_COLUMNS = {
    "nickname": "VARCHAR(64)",
    "is_registered": "BOOLEAN DEFAULT 0 NOT NULL",
    "cluster_id": "INTEGER",
}

_BOOK_COLUMNS = {
    "rating_count": "INTEGER DEFAULT 0 NOT NULL",
    "db_avg_rating": "FLOAT",
}


def migrate_schema(engine: Engine) -> None:
    if not str(engine.url).startswith("sqlite"):
        return
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "users" in tables:
        _migrate_users(engine, inspector)
    if "books" in tables:
        backfill = _migrate_books(engine, inspector)
        if backfill or stats_need_backfill(engine):
            backfill_all_book_rating_stats(engine)


def _migrate_users(engine: Engine, inspector) -> None:
    existing = {c["name"] for c in inspector.get_columns("users")}
    with engine.begin() as conn:
        for name, ddl in _USER_COLUMNS.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {name} {ddl}"))
                logger.info("Added column users.%s", name)
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_users_nickname_lower "
                "ON users (nickname COLLATE NOCASE)"
            )
        )


def _migrate_books(engine: Engine, inspector) -> bool:
    existing = {c["name"] for c in inspector.get_columns("books")}
    added = False
    with engine.begin() as conn:
        for name, ddl in _BOOK_COLUMNS.items():
            if name not in existing:
                conn.execute(text(f"ALTER TABLE books ADD COLUMN {name} {ddl}"))
                logger.info("Added column books.%s", name)
                added = True
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_books_rating_count ON books (rating_count)")
        )
    return added
