"""Sentiment classifier — independent DS4 Amazon reviews pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from app.logging_config import get_logger

logger = get_logger(__name__)


class SentimentEngine:
    """Loads sklearn pipeline (TF-IDF + LogisticRegression) from bookrec ML training."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path
        self._pipeline = None
        self._label_names: dict[int, str] = {0: "negative", 1: "positive"}

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> bool:
        if self.model_path is None or not self.model_path.exists():
            logger.warning("Sentiment model not found at %s", self.model_path)
            return False
        bundle = joblib.load(self.model_path)
        if isinstance(bundle, dict) and "pipeline" in bundle:
            self._pipeline = bundle["pipeline"]
            self._label_names = bundle.get("label_names", self._label_names)
        else:
            self._pipeline = bundle
        logger.info("Loaded sentiment model from %s", self.model_path)
        return True

    def predict(self, text: str) -> tuple[str, float]:
        if self._pipeline is None:
            positive_words = {"great", "love", "excellent", "amazing", "wonderful"}
            negative_words = {"bad", "terrible", "awful", "hate", "boring"}
            lower = text.lower()
            pos = sum(1 for w in positive_words if w in lower)
            neg = sum(1 for w in negative_words if w in lower)
            if pos > neg:
                return "positive", min(0.55 + 0.1 * pos, 0.99)
            if neg > pos:
                return "negative", min(0.55 + 0.1 * neg, 0.99)
            return "neutral", 0.5

        raw_label = self._pipeline.predict([text])[0]
        if isinstance(raw_label, (int, np.integer)):
            label = self._label_names.get(int(raw_label), str(raw_label))
        else:
            label = str(raw_label)

        score = 0.75
        if hasattr(self._pipeline, "predict_proba"):
            proba = self._pipeline.predict_proba([text])[0]
            classes = list(self._pipeline.classes_)
            idx = classes.index(raw_label)
            score = float(proba[idx])
        return label, score
