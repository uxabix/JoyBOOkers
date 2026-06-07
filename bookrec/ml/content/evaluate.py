"""Evaluate content vectors via genre-coherence of top-K neighbors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from bookrec.io_utils import write_json
from bookrec.ml.content.preprocess import _list_to_text, load_content_catalog
from bookrec.ml.io import load_sparse_matrix
from bookrec.paths import MODEL_CONTENT_DIR, MODEL_EVAL_DIR


def _genre_sets(genres_cell: Any) -> set[str]:
    text = _list_to_text(genres_cell)
    if not text:
        return set()
    return {g.lower().strip("[]'\"") for g in text.split() if g.strip("[]'\"")}


def evaluate_content_vectors(
    *,
    model_dir: Path | None = None,
    features_dir: Path | None = None,
    out_dir: Path | None = None,
    sample_size: int = 500,
    top_k: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Mean genre overlap@K and mean cosine similarity on a random book sample."""
    model_dir = Path(model_dir or MODEL_CONTENT_DIR)
    out = Path(out_dir or MODEL_EVAL_DIR / "content")
    out.mkdir(parents=True, exist_ok=True)

    matrix_path = model_dir / "tfidf_combined.npz"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Combined TF-IDF matrix not found at {matrix_path}")

    matrix, book_ids, _ = load_sparse_matrix(matrix_path)
    catalog = load_content_catalog(features_dir)
    genre_by_id = {
        str(row.source_book_id): _genre_sets(row.genres)
        for row in catalog.itertuples(index=False)
    }

    n = matrix.shape[0]
    rng = np.random.default_rng(random_state)
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)

    overlaps: list[float] = []
    mean_sims: list[float] = []

    for idx in sample_idx:
        query = matrix[idx]
        sims = cosine_similarity(query, matrix).ravel()
        sims[idx] = -1.0
        top = np.argpartition(-sims, min(top_k, n - 1))[:top_k]
        top = top[np.argsort(-sims[top])]
        source_id = str(book_ids[idx]) if book_ids is not None else str(idx)
        q_genres = genre_by_id.get(source_id, set())
        if not q_genres:
            continue
        hit_overlaps: list[float] = []
        for j in top:
            if sims[j] <= 0:
                continue
            nid = str(book_ids[j]) if book_ids is not None else str(j)
            n_genres = genre_by_id.get(nid, set())
            if n_genres:
                hit_overlaps.append(len(q_genres & n_genres) / max(len(q_genres), 1))
            mean_sims.append(float(sims[j]))
        if hit_overlaps:
            overlaps.append(float(np.mean(hit_overlaps)))

    report: dict[str, Any] = {
        "algorithm": "TF-IDF cosine similarity",
        "sample_size": int(len(sample_idx)),
        "top_k": top_k,
        "mean_genre_overlap_at_k": float(np.mean(overlaps)) if overlaps else 0.0,
        "mean_neighbor_cosine": float(np.mean(mean_sims)) if mean_sims else 0.0,
        "matrix_path": str(matrix_path),
        "n_books": int(n),
    }
    write_json(report, out / "evaluate_report.json")
    return report
