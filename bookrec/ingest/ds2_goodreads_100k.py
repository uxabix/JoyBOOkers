"""DS2: Goodreads 100k Books — genres and extended metadata."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bookrec.analysis import BOOK_NUMERIC_RULES, analyze_dataset
from bookrec.ingest.base import discover_files, fill_missing_strings, load_csv_shards, save_stage_output, standardize_columns
from bookrec.paths import RAW_DS2
from bookrec.schemas import DS2_GOODREADS_100K
from bookrec.text_normalization import add_match_keys

_GENRE_SPLIT_RE = re.compile(r"[|;,/]+")


def _parse_list_cell(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [p.strip() for p in _GENRE_SPLIT_RE.split(s) if p.strip()]


def _discover(raw_dir: Path) -> list[Path]:
    paths = discover_files(raw_dir, DS2_GOODREADS_100K.raw_file_patterns)
    return paths


def load_and_analyze_ds2(raw_dir: Path | None = None) -> dict[str, Any]:
    raw = raw_dir or RAW_DS2
    paths = _discover(raw)
    summary: dict[str, Any] = {"source": DS2_GOODREADS_100K.source_id, "raw_dir": str(raw), "files": [p.name for p in paths]}
    if not paths:
        summary["error"] = "No CSV files found"
        return summary
    df, load_meta = load_csv_shards(paths)
    summary["load"] = load_meta
    summary["analysis"] = analyze_dataset(
        df, "ds2_raw", schema_usable=DS2_GOODREADS_100K.usable_columns, numeric_rules=BOOK_NUMERIC_RULES
    )
    return summary


def preprocess_ds2(raw_dir: Path | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    from bookrec.paths import PROC_DS2

    raw = raw_dir or RAW_DS2
    out = out_dir or PROC_DS2
    paths = _discover(raw)
    if not paths:
        raise FileNotFoundError(f"DS2: place GoodReads_100k_books.csv under {raw}")

    df, load_meta = load_csv_shards(paths)
    df = standardize_columns(
        df,
        {
            "bookid": "source_book_id",
            "book_id": "source_book_id",
            "name": "title",
            "author": "authors",
            "pages": "pagesnumber",
            "num_pages": "pagesnumber",
            "average_rating": "rating",
            "genre": "genres_raw",
        },
    )
    if "source_book_id" not in df.columns:
        df["source_book_id"] = np.arange(len(df)).astype(str)
    else:
        df["source_book_id"] = df["source_book_id"].astype(str)

    df = fill_missing_strings(df, ["title", "authors", "description", "genres_raw"])
    df = df[df["title"].str.len() > 0]

    if "genres_raw" in df.columns:
        df["genres_list"] = df["genres_raw"].map(_parse_list_cell)
    else:
        df["genres_list"] = [[] for _ in range(len(df))]

    for col, lo, hi in [("pagesnumber", 1, 10000), ("rating", 0.0, 5.0)]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if lo is not None:
                df.loc[df[col] < lo, col] = np.nan
            if hi is not None:
                df.loc[df[col] > hi, col] = np.nan

    for col in ("isbn", "isbn13"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            df.loc[df[col].isin(("", "nan", "None")), col] = np.nan

    df = add_match_keys(df, title_col="title", author_col="authors")
    df["source_name"] = DS2_GOODREADS_100K.source_id
    df = df.drop_duplicates(subset=["source_book_id"], keep="first")

    report: dict[str, Any] = {
        "source": DS2_GOODREADS_100K.source_id,
        "load": load_meta,
        "rows_clean": int(len(df)),
        "rows_missing_title": int((df["title"].str.len() == 0).sum()),
        "rows_with_genres": int(df["genres_list"].map(bool).sum()),
        "unique_authors_norm": int(df["author_norm"].nunique()),
    }

    paths_out = save_stage_output(df, out, "books_clean", report)
    return {"report": report, "paths": paths_out, "books": df}
