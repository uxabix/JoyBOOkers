"""Lightweight SQLite schema migrations (add columns if missing)."""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.logging_config import get_logger

logger = get_logger(__name__)

_USER_COLUMNS = {
    "nickname": "VARCHAR(64)",
    "is_registered": "BOOLEAN DEFAULT 0 NOT NULL",
    "cluster_id": "INTEGER",
}


def migrate_schema(engine: Engine) -> None:
    if not str(engine.url).startswith("sqlite"):
        return
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return
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
