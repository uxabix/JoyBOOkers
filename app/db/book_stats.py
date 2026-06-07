"""Denormalized per-book rating stats for fast catalog queries."""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models.book import Book
from app.db.models.rating import Rating
from app.logging_config import get_logger

logger = get_logger(__name__)


def refresh_book_rating_stats(session: Session, book_id: int) -> None:
    row = session.execute(
        select(func.count(Rating.id), func.avg(Rating.score)).where(Rating.book_id == book_id)
    ).one()
    count = int(row[0] or 0)
    avg = float(row[1]) if row[1] is not None else None
    session.execute(
        update(Book)
        .where(Book.id == book_id)
        .values(rating_count=count, db_avg_rating=avg)
    )


def backfill_all_book_rating_stats(engine: Engine) -> int:
    """Rebuild books.rating_count / db_avg_rating from the ratings table."""
    Session = sessionmaker(bind=engine)
    with Session() as session:
        session.execute(update(Book).values(rating_count=0, db_avg_rating=None))
        rows = session.execute(
            select(
                Rating.book_id,
                func.count(Rating.id),
                func.avg(Rating.score),
            ).group_by(Rating.book_id)
        ).all()
        updated = 0
        for book_id, count, avg in rows:
            session.execute(
                update(Book)
                .where(Book.id == book_id)
                .values(rating_count=int(count), db_avg_rating=float(avg) if avg is not None else None)
            )
            updated += 1
        session.commit()
        logger.info("Backfilled rating stats for %s books", updated)
        return updated


def stats_need_backfill(engine: Engine) -> bool:
    """True when ratings exist but denormalized columns look empty."""
    with engine.connect() as conn:
        from sqlalchemy import text

        rating_rows = conn.execute(text("SELECT COUNT(*) FROM ratings")).scalar() or 0
        if rating_rows == 0:
            return False
        books_with_stats = conn.execute(text("SELECT COUNT(*) FROM books WHERE rating_count > 0")).scalar() or 0
        return books_with_stats == 0
