"""University-project performance limits (8–16 GB RAM)."""

from __future__ import annotations

# Dataset roles — see data/raw/README.txt
DS1_ROLE = "collaborative_filtering_only"
DS2_ROLE = "content_based_primary"
DS3_ROLE = "content_enrichment"
DS4_ROLE = "nlp_independent"

# Do not build content vectors for the full DS1 catalog (1.5M+ books).
MAX_CONTENT_BOOKS = 160_000  # DS2 ~100k + DS3-only rows after merge
MAX_DENSE_MATRIX_ROWS = 100_000
DEFAULT_MAX_TEXT_FEATURES = 5_000


def schema_roles_summary() -> dict[str, str]:
    """Human-readable dataset → task mapping for pipeline reports."""
    return {
        "ds1_goodreads_2m": "Collaborative filtering (SVD), rating analysis, user K-Means",
        "ds2_goodreads_100k": "Content-based recommendation (TF-IDF / sparse BoW)",
        "ds3_goodreads_best": "Enrichment: tags, characters (merged into DS2 where match_key aligns)",
        "ds4_amazon_reviews": "Independent NLP / sentiment (Logistic Regression)",
        "excluded": "No TF-IDF/BoW on full DS1 catalog (1.5M+ books)",
    }
