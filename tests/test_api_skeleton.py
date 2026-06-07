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
    assert "Przypisanie nowego profilu" in response.text


def test_clustering_predict(client: TestClient) -> None:
    response = client.post(
        "/clustering/predict",
        data={"rating_1": 5, "rating_2": 5, "rating_3": 4, "rating_4": 5, "rating_5": 5},
    )
    assert response.status_code == 200
    assert "Cluster" in response.text or "klaster" in response.text.lower()


def test_clustering_predict_invalid(client: TestClient) -> None:
    response = client.post("/clustering/predict", data={"rating_1": 9})
    assert response.status_code == 200
    assert "1–5" in response.text or "1-5" in response.text


def test_analytics_rubric_sections(client: TestClient) -> None:
    response = client.get("/analytics")
    assert response.status_code == 200
    assert "Selekcja cech" in response.text
    assert "Wartości odstające" in response.text
