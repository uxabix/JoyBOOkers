"""Expected column mappings and usable fields per dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetSchema:
    """Documents raw → clean column flow for one source."""

    source_id: str
    display_name: str
    raw_file_patterns: tuple[str, ...]
    id_column_candidates: tuple[str, ...]
    title_column_candidates: tuple[str, ...]
    author_column_candidates: tuple[str, ...]
    usable_columns: dict[str, str]  # raw_name → role description
    preprocessing_notes: tuple[str, ...] = ()
    feature_outputs: tuple[str, ...] = ()


def _cols(**mapping: str) -> dict[str, str]:
    return mapping


DS1_GOODREADS_2M = DatasetSchema(
    source_id="ds1_goodreads_2m",
    display_name="Goodreads Book Datasets With User Rating 2M",
    raw_file_patterns=("book*.csv", "user_rating_*.csv", "user-rating.csv"),
    id_column_candidates=("id", "goodreads_book_id", "bookid"),
    title_column_candidates=("name", "title"),
    author_column_candidates=("authors", "author"),
    usable_columns=_cols(
        id="canonical book id (catalog)",
        name="book title",
        authors="pipe-separated authors",
        publisher="publisher name",
        publishyear="publication year",
        publishmonth="publication month",
        publishday="publication day",
        language="ISO-like language code",
        isbn="ISBN-10/13 as stored",
        pagesnumber="page count",
        catalog_rating="catalog average rating 0-5 (column: rating)",
        countsofreview="number of ratings on Goodreads",
        ratingdist1="count of 1-star ratings (parsed)",
        ratingdist5="count of 5-star ratings (parsed)",
        user_id="rater id (interactions)",
        book_id="book id (interactions)",
        user_rating="user star rating 1-5 (column: rating in ratings file)",
        timestamp="optional unix/ms timestamp",
    ),
    preprocessing_notes=(
        "CF ONLY: user-item matrix, SVD, user K-Means — not used for content TF-IDF.",
        "Books: drop invalid ids, dedupe on id, bound numeric fields.",
        "Ratings: support explicit ids OR title+text-rating Kaggle export.",
        "Title matching links text ratings to catalog ids (rapidfuzz).",
    ),
    feature_outputs=(
        "interactions_clean",
        "user_item_matrix",
        "user_clustering_features",
        "svd_train_interactions",
    ),
)

DS2_GOODREADS_100K = DatasetSchema(
    source_id="ds2_goodreads_100k",
    display_name="Goodreads 100k Books",
    raw_file_patterns=(
        "GoodReads_100k_books.csv",
        "goodreads_100k*.csv",
        "books*.csv",
        "*.csv",
    ),
    id_column_candidates=("bookid", "book_id", "id"),
    title_column_candidates=("title", "name"),
    author_column_candidates=("author", "authors"),
    usable_columns=_cols(
        title="book title",
        author="primary author(s)",
        genre="genre list / tags",
        description="book blurb",
        isbn="ISBN",
        isbn13="ISBN-13",
        pages="page count",
        rating="average rating",
        book_format="hardcover/paperback/etc",
        link="Goodreads URL",
        image="cover image URL",
    ),
    preprocessing_notes=(
        "Primary dataset for content-based recommendation (~100k books).",
        "Sparse BoW / TF-IDF; cosine similarity for similar books.",
        "DS3 tags/characters merged via match_key during feature stage.",
    ),
    feature_outputs=("content_catalog", "bow_matrix.npz", "genre_matrix.npz"),
)

DS3_GOODREADS_BEST = DatasetSchema(
    source_id="ds3_goodreads_best",
    display_name="Goodreads Best Books",
    raw_file_patterns=("*.csv", "*.json", "*.jsonl"),
    id_column_candidates=("bookid", "book_id", "id"),
    title_column_candidates=("title", "name"),
    author_column_candidates=("authors", "author"),
    usable_columns=_cols(
        title="book title",
        authors="author list",
        genres="genre tags",
        tags="user tags",
        characters="character names",
        description="summary text",
        avg_rating="average rating",
        num_ratings="rating count",
        num_reviews="review count",
        isbn="ISBN",
        isbn13="ISBN-13",
        language="language",
        series="series name",
        first_publish_date="first edition date",
    ),
    preprocessing_notes=(
        "Enrichment only: tags, characters, extra genres.",
        "Merged into DS2 catalog where match_key aligns — no DS1 linking.",
    ),
    feature_outputs=("ds3_enrichment_rows",),
)

DS4_AMAZON_REVIEWS = DatasetSchema(
    source_id="ds4_amazon_reviews",
    display_name="Amazon Books Reviews",
    raw_file_patterns=("*.json", "*.jsonl", "*.csv", "*.csv.gz"),
    id_column_candidates=("asin", "parent_asin"),
    title_column_candidates=("title", "product_title"),
    author_column_candidates=(),
    usable_columns=_cols(
        reviewtext="review body (NLP)",
        text="review body alt name",
        review_body="review body alt name",
        overall="star rating 1–5",
        rating="star rating alt name",
        summary="short review headline",
        asin="Amazon product id",
        parent_asin="parent product id",
        reviewerid="reviewer id",
        user_id="reviewer id alt",
        unixreviewtime="unix timestamp",
        timestamp="unix timestamp alt",
        verified="verified purchase flag",
        helpful="helpful votes [yes, total]",
    ),
    preprocessing_notes=(
        "Fully independent NLP corpus — do NOT merge with Goodreads.",
        "Labels from star rating: >=4 positive, <=2 negative, drop neutral.",
    ),
    feature_outputs=(
        "nlp_train_reviews",
        "nlp_val_reviews",
        "nlp_test_reviews",
        "sentiment_labels",
    ),
)

ALL_SCHEMAS: dict[str, DatasetSchema] = {
    s.source_id: s
    for s in (DS1_GOODREADS_2M, DS2_GOODREADS_100K, DS3_GOODREADS_BEST, DS4_AMAZON_REVIEWS)
}


def schema_summary() -> dict[str, Any]:
    """JSON-serializable overview for pipeline reports."""
    return {
        sid: {
            "display_name": s.display_name,
            "raw_file_patterns": list(s.raw_file_patterns),
            "usable_columns": s.usable_columns,
            "preprocessing_notes": list(s.preprocessing_notes),
            "feature_outputs": list(s.feature_outputs),
        }
        for sid, s in ALL_SCHEMAS.items()
    }
