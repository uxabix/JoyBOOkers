"""DS3: Goodreads Best Books — tags, characters, rich metadata."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bookrec.analysis import BOOK_NUMERIC_RULES, analyze_dataset
from bookrec.ingest.base import (
    discover_files,
    fill_missing_strings,
    load_csv_shards,
    load_json_records,
    save_stage_output,
    standardize_columns,
)
from bookrec.ingest.ds2_goodreads_100k import _parse_list_cell
from bookrec.paths import RAW_DS3
from bookrec.schemas import DS3_GOODREADS_BEST
from bookrec.text_normalization import add_match_keys

_LIST_COLS = ("genres", "tags", "characters", "places", "awards")


def _discover(raw_dir: Path) -> tuple[list[Path], str]:
    csvs = discover_files(raw_dir, ("*.csv",))
    if csvs:
        return csvs, "csv"
    jsons = discover_files(raw_dir, ("*.jsonl", "*.json"))
    if jsons:
        return jsons, "json"
    return [], "none"


def load_and_analyze_ds3(raw_dir: Path | None = None) -> dict[str, Any]:
    raw = raw_dir or RAW_DS3
    paths, fmt = _discover(raw)
    summary: dict[str, Any] = {"source": DS3_GOODREADS_BEST.source_id, "raw_dir": str(raw), "format": fmt}
    if not paths:
        summary["error"] = "No files found"
        return summary
    if fmt == "csv":
        df, load_meta = load_csv_shards(paths)
    else:
        df, load_meta = load_json_records(paths, lines=any(p.suffix == ".jsonl" for p in paths))
    summary["load"] = load_meta
    summary["analysis"] = analyze_dataset(
        df, "ds3_raw", schema_usable=DS3_GOODREADS_BEST.usable_columns, numeric_rules=BOOK_NUMERIC_RULES
    )
    return summary


def _parse_all_lists(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([[] for _ in range(len(df))], index=df.index)
    return df[col].map(_parse_list_cell)


def preprocess_ds3(raw_dir: Path | None = None, out_dir: Path | None = None) -> dict[str, Any]:
    from bookrec.paths import PROC_DS3

    raw = raw_dir or RAW_DS3
    out = out_dir or PROC_DS3
    paths, fmt = _discover(raw)
    if not paths:
        raise FileNotFoundError(f"DS3: place best-books CSV/JSON under {raw}")

    if fmt == "csv":
        df, load_meta = load_csv_shards(paths)
    else:
        df, load_meta = load_json_records(paths, lines=any(p.suffix == ".jsonl" for p in paths))

    df = standardize_columns(
        df,
        {
            "bookid": "source_book_id",
            "book_id": "source_book_id",
            "name": "title",
            "author": "authors",
            "avg_rating": "rating",
            "num_ratings": "ratings_count",
            "num_reviews": "text_reviews_count",
        },
    )
    if "source_book_id" not in df.columns:
        df["source_book_id"] = np.arange(len(df)).astype(str)
    else:
        df["source_book_id"] = df["source_book_id"].astype(str)

    df = fill_missing_strings(df, ["title", "authors", "description", "series"])
    df = df[df["title"].str.len() > 0]

    for col in _LIST_COLS:
        df[f"{col}_list"] = _parse_all_lists(df, col)

    for col in ("rating", "ratings_count", "text_reviews_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("isbn", "isbn13"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
            df.loc[df[col].isin(("", "nan", "None")), col] = np.nan

    df = add_match_keys(df, title_col="title", author_col="authors")
    df["source_name"] = DS3_GOODREADS_BEST.source_id
    df = df.drop_duplicates(subset=["source_book_id"], keep="first")

    report: dict[str, Any] = {
        "source": DS3_GOODREADS_BEST.source_id,
        "load": load_meta,
        "rows_clean": int(len(df)),
        "rows_with_tags": int(df.get("tags_list", pd.Series(dtype=object)).map(bool).sum()),
        "rows_with_characters": int(df.get("characters_list", pd.Series(dtype=object)).map(bool).sum()),
    }

    paths_out = save_stage_output(df, out, "books_clean", report)
    return {"report": report, "paths": paths_out, "books": df}
