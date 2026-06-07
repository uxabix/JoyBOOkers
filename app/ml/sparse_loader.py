"""Load scipy sparse feature matrices produced by the bookrec pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SparseMatrixBundle:
    """In-memory sparse content vectors for DS2 (+ DS3 enrichment text)."""

    matrix: sparse.csr_matrix
    book_ids: np.ndarray
    feature_names: list[str] | None = None

    def index_for_book(self, source_book_id: str) -> int | None:
        matches = np.where(self.book_ids.astype(str) == str(source_book_id))[0]
        return int(matches[0]) if len(matches) else None


def load_sparse_bundle(
    matrix_path: Path,
    *,
    book_ids_path: Path | None = None,
) -> SparseMatrixBundle | None:
    """Load .npz artifact with optional sidecar book-id ordering."""
    if not matrix_path.exists():
        logger.warning("Sparse matrix not found: %s", matrix_path)
        return None

    matrix = sparse.load_npz(matrix_path)
    data = np.load(matrix_path, allow_pickle=True)

    ids_path = book_ids_path or matrix_path.with_name("book_ids.npy")
    if ids_path.exists():
        book_ids = np.load(ids_path, allow_pickle=True)
    elif "book_ids" in data:
        book_ids = data["book_ids"]
    else:
        book_ids = np.arange(matrix.shape[0])

    feature_names = None
    if "feature_names" in data:
        feature_names = list(data["feature_names"])

    logger.info("Loaded sparse matrix %s shape=%s", matrix_path.name, matrix.shape)
    return SparseMatrixBundle(matrix=matrix, book_ids=book_ids, feature_names=feature_names)
