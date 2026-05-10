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
    load_books_csv,
    load_ratings_csv,
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
from bookrec.paths import DEFAULT_BOOKS_CSV, DEFAULT_RATINGS_CSV, EDA_DIR, PROCESSED_DIR, RAW_DIR


def _ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def _write_table(df, path_parquet: Path, path_csv_fallback: Path) -> None:
    try:
        df.to_parquet(path_parquet, index=False)
    except Exception:
        df.to_csv(path_csv_fallback, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 — data prep for book recommender")
    parser.add_argument("--books", type=Path, default=DEFAULT_BOOKS_CSV, help="Path to book.csv")
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS_CSV, help="Path to user-rating.csv")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR, help="Processed output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip saving figures")
    args = parser.parse_args()

    _ensure_dirs()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    eda_dir = out / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    if not args.books.is_file():
        print(f"Missing books file: {args.books}", file=sys.stderr)
        print(
            "Download from Kaggle "
            "(goodreads-book-datasets-10m) and place book.csv under data/raw/.",
            file=sys.stderr,
        )
        return 1
    if not args.ratings.is_file():
        print(f"Missing ratings file: {args.ratings}", file=sys.stderr)
        print("Place user-rating.csv under data/raw/.", file=sys.stderr)
        return 1

    print("Loading CSVs...")
    books_raw = apply_book_column_aliases(load_books_csv(args.books))
    ratings_raw = apply_rating_column_aliases(load_ratings_csv(args.ratings))

    summary: dict = {
        "profiles": {
            "books_raw": dataset_profile(books_raw, "books_raw"),
            "ratings_raw": dataset_profile(ratings_raw, "ratings_raw"),
        }
    }

    print("Cleaning books...")
    books_clean, book_report = clean_books(books_raw)
    summary["books_cleaning"] = book_report

    valid_ids = set(books_clean["id"].astype(int).tolist())
    print("Cleaning interactions...")
    interactions, inter_report = clean_interactions(
        ratings_raw,
        books_catalog=books_clean,
        valid_book_ids=valid_ids,
    )
    summary["interactions_cleaning"] = inter_report

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
    # Explicit matrix copy (same as interactions — long format user_id | book_id | rating)
    _write_table(interactions, matrix_path_pq, matrix_path_csv)

    if not args.no_plots:
        print("Saving EDA figures...")
        save_rating_distribution(interactions.assign(rating=interactions["rating"].astype(int)), eda_dir / "01_rating_distribution.png")
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
    print(f"  Cleaned books:     {books_path_pq if books_path_pq.exists() else books_path_csv}")
    print(f"  User-item matrix:   {matrix_path_pq if matrix_path_pq.exists() else matrix_path_csv}")
    print(f"  Summary JSON:      {report_path}")
    print(f"  Figures:           {eda_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
