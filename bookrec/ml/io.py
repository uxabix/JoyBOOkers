"""Model and sparse-matrix persistence."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, path: Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def save_joblib(obj: Any, path: Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    return path


def load_joblib(path: Path) -> Any:
    return joblib.load(path)


def save_sparse_matrix(
    matrix: csr_matrix,
    path: Path,
    *,
    book_ids: np.ndarray | list | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save CSR matrix; optional book_ids.npy and metadata.json sidecars."""
    path = Path(path)
    ensure_dir(path.parent)
    save_npz(path, matrix)
    if book_ids is not None:
        np.save(path.with_name("book_ids.npy"), np.asarray(book_ids))
    if metadata is not None:
        meta_path = path.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2, default=str)
    return path


def load_sparse_matrix(path: Path) -> tuple[csr_matrix, np.ndarray | None, dict[str, Any] | None]:
    path = Path(path)
    matrix = load_npz(path)
    ids_path = path.with_name("book_ids.npy")
    book_ids = np.load(ids_path, allow_pickle=True) if ids_path.exists() else None
    meta_path = path.with_suffix(".meta.json")
    metadata = None
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as fh:
            metadata = json.load(fh)
    return matrix, book_ids, metadata
