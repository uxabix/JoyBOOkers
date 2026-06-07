"""Sentiment analysis — TF-IDF + Logistic Regression on DS4 Amazon reviews."""

from bookrec.ml.sentiment.evaluate import evaluate_sentiment_model
from bookrec.ml.sentiment.train import train_sentiment_model

__all__ = ["evaluate_sentiment_model", "train_sentiment_model"]
