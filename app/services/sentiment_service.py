"""Sentiment analysis service — independent DS4 pipeline."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models.review import Review
from app.ml.sentiment import SentimentEngine
from app.repositories.review_repository import ReviewRepository
from app.schemas.review import ReviewCreate, ReviewRead
from app.schemas.sentiment import SentimentPredictRequest, SentimentPredictResponse


class SentimentService:
    def __init__(self, session: Session, engine: SentimentEngine) -> None:
        self.session = session
        self.engine = engine
        self.reviews = ReviewRepository(session)

    def predict(self, payload: SentimentPredictRequest) -> SentimentPredictResponse:
        if not self.engine.is_loaded:
            self.engine.load()
        label, score = self.engine.predict(payload.text)
        return SentimentPredictResponse(label=label, score=score, model="ds4_sentiment")

    def analyze_and_store(self, payload: ReviewCreate) -> ReviewRead:
        prediction = self.predict(SentimentPredictRequest(text=payload.review_text))
        review = Review(
            asin=payload.asin,
            reviewer_id=payload.reviewer_id,
            review_text=payload.review_text,
            star_rating=payload.star_rating,
            sentiment_label=prediction.label,
            sentiment_score=prediction.score,
        )
        self.reviews.add(review)
        self.session.commit()
        self.session.refresh(review)
        return ReviewRead.model_validate(review)
