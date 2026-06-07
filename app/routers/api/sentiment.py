"""Sentiment API — independent DS4 Amazon reviews."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.dependencies import get_sentiment_service
from app.schemas.review import ReviewCreate, ReviewRead
from app.schemas.sentiment import SentimentPredictRequest, SentimentPredictResponse
from app.services.sentiment_service import SentimentService

router = APIRouter()


@router.post("/predict", response_model=SentimentPredictResponse)
def predict_sentiment(
    payload: SentimentPredictRequest,
    service: SentimentService = Depends(get_sentiment_service),
) -> SentimentPredictResponse:
    return service.predict(payload)


@router.post("/reviews", response_model=ReviewRead, status_code=status.HTTP_201_CREATED)
def analyze_review(
    payload: ReviewCreate,
    service: SentimentService = Depends(get_sentiment_service),
) -> ReviewRead:
    return service.analyze_and_store(payload)
