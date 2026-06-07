"""Content-based recommendations — sparse dot product on L2-normalized TF-IDF rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from app.logging_config import get_logger
from app.ml.sparse_loader import SparseMatrixBundle, load_sparse_bundle

logger = get_logger(__name__)


class ContentRecommendationEngine:
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
        scores = self._bundle.matrix @ query.T

        if sparse.issparse(scores):
            sims = np.asarray(scores.toarray()).ravel()
        else:
            sims = np.asarray(scores).ravel()

        if exclude_self:
            sims[idx] = -1.0

        k = min(limit, max(0, len(sims) - 1))
        if k == 0:
            return []

        top_idx = np.argpartition(-sims, k)[: k + 1]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results: list[tuple[str, float]] = []
        for i in top_idx:
            if sims[i] <= 0:
                continue
            results.append((str(self._bundle.book_ids[i]), float(sims[i])))
            if len(results) >= limit:
                break
        return results

    def build_user_vector(
        self,
        rated: list[tuple[str, float]],
        *,
        max_books: int = 10,
    ) -> sparse.csr_matrix | None:
        """Weighted average TF-IDF vector from rated books (rating as weight)."""
        if self._bundle is None or not rated:
            return None

        rows: list[sparse.csr_matrix] = []
        weights: list[float] = []
        for source_id, score in rated[:max_books]:
            idx = self._bundle.index_for_book(str(source_id))
            if idx is None:
                continue
            rows.append(self._bundle.matrix[idx])
            weights.append(max(float(score), 1.0))

        if not rows:
            return None

        w = np.asarray(weights, dtype=np.float64)
        w /= w.sum()
        combined = sum(w[i] * rows[i] for i in range(len(rows)))
        norm = sparse.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined.tocsr()

    def score_candidates(
        self,
        user_vector: sparse.csr_matrix,
        source_book_ids: list[str],
    ) -> dict[str, float]:
        """Cosine similarity between user profile vector and candidate books."""
        if self._bundle is None or not source_book_ids:
            return {}

        indices: list[int] = []
        ids: list[str] = []
        for sid in source_book_ids:
            idx = self._bundle.index_for_book(str(sid))
            if idx is not None:
                indices.append(idx)
                ids.append(str(sid))

        if not indices:
            return {}

        sub = self._bundle.matrix[indices]
        scores = sub @ user_vector.T
        if sparse.issparse(scores):
            sims = np.asarray(scores.toarray(), dtype=np.float64).ravel()
        else:
            sims = np.asarray(scores, dtype=np.float64).ravel()
        sims = np.maximum(sims, 0.0)
        return {ids[i]: float(sims[i]) for i in range(len(ids))}
