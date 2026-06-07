"""Sentiment analysis schemas — DS4 independent NLP."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SentimentPredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)


class SentimentPredictResponse(BaseModel):
    label: str
    score: float
    model: str
