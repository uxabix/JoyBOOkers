"""DS1: Goodreads 2M ratings + book metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bookrec.analysis import (
    BOOK_NUMERIC_RULES,
    analyze_dataset,
    detect_interaction_outliers,
)
from bookrec.cleaning import (
    apply_book_column_aliases,
    apply_rating_column_aliases,
    clean_books,
    clean_interactions,
)
from bookrec.datasets import discover_book_shards, discover_rating_shards, load_books_dataset, load_ratings_dataset
from bookrec.ingest.base import save_stage_output
from bookrec.paths import ds1_raw_dir
from bookrec.schemas import DS1_GOODREADS_2M
from bookrec.text_normalization import add_match_keys
from bookrec.title_matching import TitleMatcher


def _resolve_raw_dir(raw_dir: Path | None) -> Path:
    return raw_dir if raw_dir is not None else ds1_raw_dir()


def load_and_analyze_ds1(raw_dir: Path | None = None) -> dict[str, Any]:
    raw = _resolve_raw_dir(raw_dir)
    book_paths = discover_book_shards(raw)
    rating_paths = discover_rating_shards(raw)
    summary: dict[str, Any] = {
        "source": DS1_GOODREADS_2M.source_id,
        "raw_dir": str(raw),
        "book_files": [p.name for p in book_paths],
        "rating_files": [p.name for p in rating_paths],
    }
    if not book_paths:
        summary["error"] = "No book shards found"
        return summary
    books_raw, _ = load_books_dataset(book_paths)
    books_raw = apply_book_column_aliases(books_raw)
    summary["books"] = analyze_dataset(
        books_raw, "ds1_books_raw", schema_usable=DS1_GOODREADS_2M.usable_columns, numeric_rules=BOOK_NUMERIC_RULES
    )
    if rating_paths:
        ratings_raw, _ = load_ratings_dataset(rating_paths)
        ratings_raw = apply_rating_column_aliases(ratings_raw)
        summary["ratings"] = analyze_dataset(
            ratings_raw, "ds1_ratings_raw", schema_usable=DS1_GOODREADS_2M.usable_columns
        )
    else:
        summary["ratings"] = {"error": "No rating shards found"}
    return summary


def preprocess_ds1(
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    *,
    fuzzy_threshold: int = 88,
    enable_fuzzy: bool = True,
    min_user_ratings: int = 0,
    min_book_ratings: int = 0,
) -> dict[str, Any]:
    """Clean books + interactions; add match keys for cross-dataset linking."""
    from bookrec.paths import PROC_DS1

    raw = _resolve_raw_dir(raw_dir)
    out = out_dir or PROC_DS1
    book_paths = discover_book_shards(raw)
    rating_paths = discover_rating_shards(raw)
    if not book_paths or not rating_paths:
        raise FileNotFoundError(f"DS1 requires book and rating CSVs under {raw}")

    books_raw, books_load = load_books_dataset(book_paths)
    books_raw = apply_book_column_aliases(books_raw)
    ratings_raw, ratings_load = load_ratings_dataset(rating_paths)
    ratings_raw = apply_rating_column_aliases(ratings_raw)

    books_clean, book_report = clean_books(books_raw)
    books_clean = add_match_keys(books_clean, title_col="name", author_col="authors")
    books_clean["source_name"] = DS1_GOODREADS_2M.source_id
    books_clean["source_book_id"] = books_clean["id"].astype(str)

    valid_ids = set(books_clean["id"].astype("int64").unique())
    matcher = TitleMatcher.from_catalog(books_clean, fuzzy_threshold=fuzzy_threshold) if enable_fuzzy else None

    interactions, inter_report = clean_interactions(
        ratings_raw,
        books_catalog=books_clean,
        valid_book_ids=valid_ids,
        title_matcher=matcher,
        fuzzy_threshold=fuzzy_threshold,
        enable_fuzzy_title_match=enable_fuzzy,
    )

    # Optional sparsity filters for CF training subsets
    if min_user_ratings > 0 or min_book_ratings > 0:
        uc = interactions.groupby("user_id").size()
        bc = interactions.groupby("book_id").size()
        keep_users = set(uc[uc >= min_user_ratings].index) if min_user_ratings > 0 else set(uc.index)
        keep_books = set(bc[bc >= min_book_ratings].index) if min_book_ratings > 0 else set(bc.index)
        before = len(interactions)
        interactions = interactions[
            interactions["user_id"].isin(keep_users) & interactions["book_id"].isin(keep_books)
        ]
        inter_report["filtered_min_user_book_ratings"] = {
            "min_user_ratings": min_user_ratings,
            "min_book_ratings": min_book_ratings,
            "rows_dropped": int(before - len(interactions)),
        }

    inter_report["outliers"] = detect_interaction_outliers(interactions)

    report: dict[str, Any] = {
        "source": DS1_GOODREADS_2M.source_id,
        "load": {"books": books_load, "ratings": ratings_load},
        "books_cleaning": book_report,
        "interactions_cleaning": inter_report,
    }

    paths = {}
    paths["books"] = save_stage_output(books_clean, out, "books_clean", report)["table"]
    paths["interactions"] = save_stage_output(interactions, out, "interactions_clean", inter_report)["table"]

    catalog_for_linking = books_clean[
        ["id", "name", "authors", "title_norm", "title_core", "author_norm", "match_key", "isbn"]
    ].copy()
    paths["catalog"] = save_stage_output(catalog_for_linking, out, "catalog_link_keys", {})["table"]

    return {"report": report, "paths": paths, "books": books_clean, "interactions": interactions}
