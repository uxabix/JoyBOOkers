"""Load scipy sparse feature matrices produced by the bookrec pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SparseMatrixBundle:
    matrix: sparse.csr_matrix
    book_ids: np.ndarray
    feature_names: list[str] | None = None
    _id_index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._id_index = {
            str(bid): idx for idx, bid in enumerate(self.book_ids.astype(str))
        }

    def index_for_book(self, source_book_id: str) -> int | None:
        return self._id_index.get(str(source_book_id))


def load_sparse_bundle(
    matrix_path: Path,
    *,
    book_ids_path: Path | None = None,
) -> SparseMatrixBundle | None:
    if not matrix_path.exists():
        logger.warning("Sparse matrix not found: %s", matrix_path)
        return None

    matrix = sparse.load_npz(matrix_path)
    ids_path = book_ids_path or matrix_path.with_name("book_ids.npy")

    if ids_path.exists():
        book_ids = np.load(ids_path, allow_pickle=True)
    else:
        with np.load(matrix_path, allow_pickle=True) as data:
            if "book_ids" in data:
                book_ids = data["book_ids"]
            else:
                book_ids = np.arange(matrix.shape[0])

    logger.info("Loaded sparse matrix %s shape=%s", matrix_path.name, matrix.shape)
    return SparseMatrixBundle(matrix=matrix, book_ids=book_ids)
