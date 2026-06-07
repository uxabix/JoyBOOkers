"""Smoke tests for the FastAPI backend."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["database"] == "ok"


def test_home_page(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Book Recommendation System" in response.text


def test_sentiment_predict(client: TestClient) -> None:
    response = client.post(
        "/api/v1/sentiment/predict",
        json={"text": "I love this book, it is great and wonderful."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in {"positive", "negative", "neutral"}


def test_analytics_page(client: TestClient) -> None:
    response = client.get("/analytics")
    assert response.status_code == 200


def test_clustering_page(client: TestClient) -> None:
    response = client.get("/clustering")
    assert response.status_code == 200
