"""Collaborative filtering via Surprise (DS1 ratings only)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

from app.logging_config import get_logger

logger = get_logger(__name__)


class CollaborativeFilteringEngine:
    """Wraps Surprise SVD trained on DS1 user-item ratings."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path
        self._model: SVD | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        if self.model_path is None or not self.model_path.exists():
            logger.warning("CF model not found at %s", self.model_path)
            return False
        with self.model_path.open("rb") as fh:
            self._model = pickle.load(fh)
        logger.info("Loaded Surprise model from %s", self.model_path)
        return True

    def train_from_ratings_df(self, ratings: pd.DataFrame, *, save: bool = True) -> dict[str, float]:
        """Train SVD on columns: user_id, book_id, rating (1-5)."""
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings[["user_id", "book_id", "rating"]].astype({"user_id": str, "book_id": str}),
            reader,
        )
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        model = SVD(n_factors=50, n_epochs=20, random_state=42)
        model.fit(trainset)
        predictions = model.test(testset)
        rmse = float(accuracy.rmse(predictions, verbose=False))

        self._model = model
        if save and self.model_path is not None:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with self.model_path.open("wb") as fh:
                pickle.dump(model, fh)

        return {"rmse": rmse, "train_size": trainset.n_ratings}

    def predict(self, user_id: str, book_id: str) -> float | None:
        if self._model is None:
            return None
        return float(self._model.predict(str(user_id), str(book_id)).est)

    def recommend(
        self,
        user_id: str,
        candidate_book_ids: list[str],
        *,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        if self._model is None:
            return []
        scored = [
            (bid, float(self._model.predict(str(user_id), str(bid)).est))
            for bid in candidate_book_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]
