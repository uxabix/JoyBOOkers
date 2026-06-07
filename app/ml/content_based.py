"""Content-based recommendations — sklearn cosine on sparse TF-IDF/BoW (DS2+DS3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.logging_config import get_logger
from app.ml.sparse_loader import SparseMatrixBundle, load_sparse_bundle

logger = get_logger(__name__)


class ContentRecommendationEngine:
    """Similar-book retrieval using precomputed sparse content vectors."""

    def __init__(
        self,
        matrix_path: Path,
        *,
        book_ids_path: Path | None = None,
    ) -> None:
        self.matrix_path = matrix_path
        self.book_ids_path = book_ids_path
        self._bundle: SparseMatrixBundle | None = None

    @property
    def is_loaded(self) -> bool:
        return self._bundle is not None

    def load(self) -> bool:
        self._bundle = load_sparse_bundle(self.matrix_path, book_ids_path=self.book_ids_path)
        return self._bundle is not None

    def similar_books(
        self,
        source_book_id: str,
        *,
        limit: int = 10,
        exclude_self: bool = True,
    ) -> list[tuple[str, float]]:
        if self._bundle is None:
            return []

        idx = self._bundle.index_for_book(source_book_id)
        if idx is None:
            logger.debug("Book %s not in content matrix", source_book_id)
            return []

        query = self._bundle.matrix[idx]
        sims = cosine_similarity(query, self._bundle.matrix).ravel()

        if exclude_self:
            sims[idx] = -1.0

        top_idx = np.argpartition(-sims, min(limit, len(sims) - 1))[:limit]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: list[tuple[str, float]] = []
        for i in top_idx:
            if sims[i] <= 0:
                continue
            results.append((str(self._bundle.book_ids[i]), float(sims[i])))
        return results[:limit]
