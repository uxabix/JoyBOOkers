#!/usr/bin/env python3
"""Stage 1: analyse Kaggle CSVs, clean, build user–item matrix, save EDA figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.cleaning import (
    apply_book_column_aliases,
    apply_rating_column_aliases,
    clean_books,
    clean_interactions,
    dataset_profile,
)
from bookrec.datasets import (
    discover_book_shards,
    discover_rating_shards,
    load_books_dataset,
    load_ratings_dataset,
)
from bookrec.eda import (
    save_avg_book_rating_hist,
    save_books_per_year,
    save_language_distribution,
    save_rating_distribution,
    save_top_books,
    save_top_publishers,
    save_top_users,
)
from bookrec.paths import EDA_DIR, PROCESSED_DIR, RAW_DIR


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def _write_table(df, path_parquet: Path, path_csv_fallback: Path) -> None:
    try:
        df.to_parquet(path_parquet, index=False)
    except Exception:
        df.to_csv(path_csv_fallback, index=False)


def _resolve_book_paths(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        if explicit.is_file():
            return [explicit]
        if explicit.is_dir():
            return discover_book_shards(explicit)
        raise FileNotFoundError(f"Books path not found: {explicit}")
    return discover_book_shards(RAW_DIR)


def _resolve_rating_paths(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        if explicit.is_file():
            return [explicit]
        if explicit.is_dir():
            return discover_rating_shards(explicit)
        raise FileNotFoundError(f"Ratings path not found: {explicit}")
    return discover_rating_shards(RAW_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 — data prep for book recommender")
    parser.add_argument(
        "--books",
        type=Path,
        default=None,
        help="Single book CSV or directory; default: all book*.csv shards in data/raw/",
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        default=None,
        help="Single ratings CSV or directory; default: all user_rating_*.csv shards in data/raw/",
    )
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR, help="Processed output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving figures")
    args = parser.parse_args()

    _ensure_dirs()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    eda_dir = out / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    try:
        book_paths = _resolve_book_paths(args.books)
        rating_paths = _resolve_rating_paths(args.ratings)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not book_paths:
        print("No book CSV shards found.", file=sys.stderr)
        print(
            "Download from Kaggle (goodreads-book-datasets-10m) and place book*.csv under data/raw/.",
            file=sys.stderr,
        )
        return 1
    if not rating_paths:
        print("No ratings CSV shards found.", file=sys.stderr)
        print("Place user_rating_*.csv (or user-rating.csv) under data/raw/.", file=sys.stderr)
        return 1

    print(f"Books:   {len(book_paths)} file(s) — {book_paths[0].name}" + (
        f" … {book_paths[-1].name}" if len(book_paths) > 1 else ""
    ))
    print(f"Ratings: {len(rating_paths)} file(s) — {rating_paths[0].name}" + (
        f" … {rating_paths[-1].name}" if len(rating_paths) > 1 else ""
    ))

    print("Loading book shards...")
    books_raw, books_load_report = load_books_dataset(book_paths)
    books_raw = apply_book_column_aliases(books_raw)

    print("Loading rating shards...")
    ratings_raw, ratings_load_report = load_ratings_dataset(rating_paths)
    ratings_raw = apply_rating_column_aliases(ratings_raw)

    summary: dict = {
        "load": {
            "books": books_load_report,
            "ratings": ratings_load_report,
        },
        "profiles": {
            "books_raw": dataset_profile(books_raw, "books_raw"),
            "ratings_raw": dataset_profile(ratings_raw, "ratings_raw"),
        },
    }

    print("Cleaning books...")
    books_clean, book_report = clean_books(books_raw)
    summary["books_cleaning"] = book_report
    del books_raw

    print("Building valid book id set...")
    valid_ids = set(books_clean["id"].astype("int64").unique())

    print("Cleaning interactions...")
    interactions, inter_report = clean_interactions(
        ratings_raw,
        books_catalog=books_clean,
        valid_book_ids=valid_ids,
    )
    summary["interactions_cleaning"] = inter_report
    del ratings_raw, valid_ids

    summary["profiles"]["books_clean"] = dataset_profile(books_clean, "books_clean")
    summary["profiles"]["interactions_clean"] = dataset_profile(interactions, "interactions_clean")

    books_path_pq = out / "books_clean.parquet"
    books_path_csv = out / "books_clean.csv"
    inter_path_pq = out / "interactions_clean.parquet"
    inter_path_csv = out / "interactions_clean.csv"
    matrix_path_pq = out / "user_item_matrix.parquet"
    matrix_path_csv = out / "user_item_matrix.csv"

    print("Writing cleaned tables...")
    _write_table(books_clean, books_path_pq, books_path_csv)
    _write_table(interactions, inter_path_pq, inter_path_csv)
    _write_table(interactions, matrix_path_pq, matrix_path_csv)

    if not args.no_plots:
        print("Saving EDA figures...")
        save_rating_distribution(
            interactions.assign(rating=interactions["rating"].astype(int)),
            eda_dir / "01_rating_distribution.png",
        )
        save_top_books(interactions, books_clean, eda_dir / "02_top_books.png")
        save_top_users(interactions, eda_dir / "03_top_users.png")
        save_language_distribution(books_clean, eda_dir / "04_languages.png")
        save_books_per_year(books_clean, eda_dir / "05_books_per_year.png")
        save_top_publishers(books_clean, eda_dir / "06_top_publishers.png")
        save_avg_book_rating_hist(books_clean, eda_dir / "07_avg_rating_books_hist.png")

    report_path = out / "stage1_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print("Done.")
    print(f"  Cleaned books:      {books_path_pq if books_path_pq.exists() else books_path_csv}")
    print(f"  User-item matrix:   {matrix_path_pq if matrix_path_pq.exists() else matrix_path_csv}")
    print(f"  Summary JSON:       {report_path}")
    print(f"  Figures:            {eda_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
