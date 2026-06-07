"""Build sparse TF-IDF feature blocks from DS2+DS3 content catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from bookrec.constraints import DEFAULT_MAX_TEXT_FEATURES, MAX_CONTENT_BOOKS
from bookrec.io_utils import read_table, write_json
from bookrec.paths import MODEL_CONTENT_DIR, PROC_FEATURES


def _resolve_catalog(path_stem: Path) -> Path:
    for suffix in (".parquet", ".csv"):
        candidate = path_stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return path_stem.with_suffix(".parquet")


def load_content_catalog(features_dir: Path | None = None) -> pd.DataFrame:
    """Load content catalog produced by bookrec.features.content."""
    base = features_dir or (PROC_FEATURES / "content")
    catalog_path = _resolve_catalog(base / "content_catalog")
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Content catalog not found at {catalog_path}. "
            "Run data pipeline features stage (DS2+DS3) first."
        )
    return read_table(catalog_path)


def _list_to_text(values: Any) -> str:
    """Flatten list / numpy / parquet-serialized genre-tag cells to space-separated text."""
    if values is None or (isinstance(values, float) and np.isnan(values)):
        return ""
    if isinstance(values, np.ndarray):
        if values.size == 0:
            return ""
        parts = [_list_to_text(v) for v in values.tolist()]
        return " ".join(p for p in parts if p)
    if isinstance(values, list):
        return " ".join(_list_to_text(v) for v in values if _list_to_text(v))
    text = str(values).strip()
    if text in ("", "[]", "['[]']", '[""]', "array([], dtype=object)"):
        return ""
    # Parquet sometimes stores nested lists as a single quoted string.
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].replace("'", " ").replace('"', " ").replace("\\n", " ")
        inner = " ".join(inner.split())
        if inner in ("", "[]"):
            return ""
        return inner
    return text


def _fit_tfidf(texts: list[str], max_features: int) -> tuple[csr_matrix, TfidfVectorizer | None]:
    """Fit TF-IDF; return empty CSR when all documents are blank or tokenization fails."""
    if not any(t.strip() for t in texts):
        empty = csr_matrix((len(texts), 0), dtype=np.float32)
        return empty, None
    vectorizer = TfidfVectorizer(max_features=max_features, dtype=np.float32)
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        empty = csr_matrix((len(texts), 0), dtype=np.float32)
        return empty, None
    return csr_matrix(matrix), vectorizer


def build_tfidf_blocks(
    catalog: pd.DataFrame,
    *,
    max_author_features: int = 3000,
    max_genre_features: int = 500,
    max_tag_features: int = 5000,
    max_content_features: int = DEFAULT_MAX_TEXT_FEATURES,
) -> dict[str, Any]:
    """Separate TF-IDF matrices for authors, genres, tags; combined CSR matrix."""
    if len(catalog) > MAX_CONTENT_BOOKS:
        raise ValueError(
            f"Catalog has {len(catalog)} rows; university scope limits content to {MAX_CONTENT_BOOKS}"
        )

    authors = catalog["authors"].fillna("").astype(str).tolist()
    genres = catalog["genres"].map(_list_to_text).tolist()
    tags = catalog["tags"].map(_list_to_text).tolist()
    content = catalog["content_text"].fillna("").astype(str).tolist()

    author_mat, author_vec = _fit_tfidf(authors, max_author_features)
    genre_mat, genre_vec = _fit_tfidf(genres, max_genre_features)
    tag_mat, tag_vec = _fit_tfidf(tags, max_tag_features)
    content_mat, content_vec = _fit_tfidf(content, max_content_features)

    combined = hstack([author_mat, genre_mat, tag_mat, content_mat], format="csr")
    combined = normalize(combined, norm="l2", axis=1, copy=False)

    book_ids = catalog["source_book_id"].astype(str).to_numpy()
    return {
        "book_ids": book_ids,
        "author_matrix": csr_matrix(author_mat),
        "genre_matrix": csr_matrix(genre_mat),
        "tag_matrix": csr_matrix(tag_mat),
        "content_matrix": csr_matrix(content_mat),
        "combined_matrix": csr_matrix(combined),
        "vectorizers": {
            k: v
            for k, v in {
                "authors": author_vec,
                "genres": genre_vec,
                "tags": tag_vec,
                "content": content_vec,
            }.items()
            if v is not None
        },
        "stats": {
            "n_books": int(len(catalog)),
            "author_features": int(author_mat.shape[1]),
            "genre_features": int(genre_mat.shape[1]),
            "tag_features": int(tag_mat.shape[1]),
            "content_features": int(content_mat.shape[1]),
            "combined_features": int(combined.shape[1]),
            "combined_nnz": int(combined.nnz),
        },
    }


def prepare_content_training_data(
    features_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate catalog availability before TF-IDF training."""
    out = Path(out_dir or MODEL_CONTENT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    catalog = load_content_catalog(features_dir)
    report = {
        "dataset": "ds2_goodreads_100k + ds3_goodreads_best",
        "catalog_rows": int(len(catalog)),
        "ds1_excluded": True,
        "matrix_format": "scipy.sparse.csr",
    }
    write_json(report, out / "preprocess_report.json")
    return report
