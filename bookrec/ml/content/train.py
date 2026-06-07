"""Train (build) sparse TF-IDF content vectors for cosine-similarity retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bookrec.io_utils import write_json
from bookrec.ml.content.preprocess import build_tfidf_blocks, load_content_catalog
from bookrec.ml.io import save_joblib, save_sparse_matrix
from bookrec.paths import MODEL_CONTENT_DIR, PROC_FEATURES


def train_content_vectors(
    *,
    features_dir: Path | None = None,
    out_dir: Path | None = None,
    max_author_features: int = 3000,
    max_genre_features: int = 500,
    max_tag_features: int = 5000,
    max_content_features: int = 5000,
) -> dict[str, Any]:
    """Persist TF-IDF sparse matrices and fitted vectorizers."""
    out = Path(out_dir or MODEL_CONTENT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    catalog = load_content_catalog(features_dir or (PROC_FEATURES / "content"))
    blocks = build_tfidf_blocks(
        catalog,
        max_author_features=max_author_features,
        max_genre_features=max_genre_features,
        max_tag_features=max_tag_features,
        max_content_features=max_content_features,
    )

    paths: dict[str, str] = {}
    matrix_specs = {
        "tfidf_authors.npz": blocks["author_matrix"],
        "tfidf_genres.npz": blocks["genre_matrix"],
        "tfidf_tags.npz": blocks["tag_matrix"],
        "tfidf_content.npz": blocks["content_matrix"],
        "tfidf_combined.npz": blocks["combined_matrix"],
    }
    for name, mat in matrix_specs.items():
        p = save_sparse_matrix(
            mat,
            out / name,
            book_ids=blocks["book_ids"],
            metadata={"feature_block": name.replace("tfidf_", "").replace(".npz", "")},
        )
        paths[name] = str(p)

    vec_path = save_joblib(blocks["vectorizers"], out / "tfidf_vectorizers.joblib")
    paths["vectorizers"] = str(vec_path)

    report: dict[str, Any] = {
        "algorithm": "TF-IDF + cosine similarity",
        "dataset": "ds2_goodreads_100k + ds3_goodreads_best",
        "similarity": "cosine (L2-normalized sparse rows)",
        "stats": blocks["stats"],
        "paths": paths,
    }
    write_json(report, out / "train_report.json")
    return report
