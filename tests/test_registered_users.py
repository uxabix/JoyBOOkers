"""Tests for registered user flows."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models.book import Book
from app.db.session import get_db


def _seed_books(db: Session) -> tuple[int, int]:
    books = [
        Book(
            source_book_id="test-book-1",
            match_key="test book 1",
            title="Test Book One",
            author="Test Author",
            genre="Fantasy",
            avg_rating=4.2,
            rating_count=100,
            db_avg_rating=4.2,
        ),
        Book(
            source_book_id="test-book-2",
            match_key="test book 2",
            title="Test Book Two",
            author="Test Author",
            genre="Fantasy, Adventure",
            avg_rating=4.5,
            rating_count=80,
            db_avg_rating=4.5,
        ),
    ]
    db.add_all(books)
    db.commit()
    for b in books:
        db.refresh(b)
    return books[0].id, books[1].id


def test_register_login_and_profile(client: TestClient) -> None:
    reg = client.post("/api/v1/auth/register", json={"nickname": "test_reader"})
    assert reg.status_code == 201
    data = reg.json()
    assert data["nickname"] == "test_reader"
    assert data["is_registered"] is True

    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["nickname"] == "test_reader"

    page = client.get("/me", follow_redirects=True)
    assert page.status_code == 200
    assert "test_reader" in page.text


def test_register_duplicate_nickname(client: TestClient) -> None:
    client.post("/api/v1/auth/register", json={"nickname": "dupe_user"})
    dup = client.post("/api/v1/auth/register", json={"nickname": "dupe_user"})
    assert dup.status_code == 409


def test_rate_and_recommend(client: TestClient) -> None:
    db_gen = client.app.dependency_overrides.get(get_db)
    assert db_gen is not None
    db = next(db_gen())
    book_id, _other_id = _seed_books(db)
    db.close()

    client.post("/api/v1/auth/register", json={"nickname": "rater_one"})
    user_id = client.get("/api/v1/auth/me").json()["id"]

    rating = client.post(
        "/api/v1/ratings/me",
        json={"book_id": book_id, "score": 5.0},
    )
    assert rating.status_code == 201
    assert rating.json()["score"] == 5.0

    recs = client.get(f"/api/v1/users/{user_id}/recommendations")
    assert recs.status_code == 200
    payload = recs.json()
    assert payload["items"]
    assert payload["algorithm"] == "hybrid"
    assert payload.get("profile") is not None
    assert payload["profile"]["rating_count"] >= 1
