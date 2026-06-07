"""DS4: Amazon Books Reviews — independent NLP corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bookrec.analysis import analyze_dataset, detect_review_text_outliers, detect_rating_outliers
from bookrec.ingest.base import discover_files, load_csv_shards, load_json_records, save_stage_output, standardize_columns
from bookrec.paths import RAW_DS4
from bookrec.schemas import DS4_AMAZON_REVIEWS
from bookrec.text_normalization import normalize_review_text


def _csv_has_review_text(path: Path) -> bool:
    """Peek header — Amazon Kaggle exports use review/text, not reviewText."""
    import pandas as pd

    cols = {str(c).strip().lower() for c in pd.read_csv(path, nrows=0).columns}
    review_cols = {
        "reviewtext",
        "review_text",
        "review/text",
        "text",
        "review_body",
        "review body",
    }
    return bool(cols & review_cols)


def _discover(raw_dir: Path) -> tuple[list[Path], str]:
    jsons = discover_files(raw_dir, ("*.jsonl", "*.json"))
    if jsons:
        return jsons, "json"
    csvs = discover_files(raw_dir, ("*.csv", "*.csv.gz"))
    if csvs:
        review_csvs = [p for p in csvs if _csv_has_review_text(p)]
        if review_csvs:
            return review_csvs, "csv"
        return csvs, "csv"
    return [], "none"


def _star_to_sentiment(stars: float, pos_thresh: float = 4.0, neg_thresh: float = 2.0) -> int | None:
    if pd.isna(stars):
        return None
    s = float(stars)
    if s >= pos_thresh:
        return 1
    if s <= neg_thresh:
        return 0
    return None


def load_and_analyze_ds4(raw_dir: Path | None = None) -> dict[str, Any]:
    raw = raw_dir or RAW_DS4
    paths, fmt = _discover(raw)
    summary: dict[str, Any] = {"source": DS4_AMAZON_REVIEWS.source_id, "raw_dir": str(raw), "format": fmt}
    if not paths:
        summary["error"] = "No review files found"
        return summary
    if fmt == "json":
        df, load_meta = load_json_records(paths, lines=any(p.suffix == ".jsonl" for p in paths))
    else:
        df, load_meta = load_csv_shards(paths)
    summary["load"] = load_meta
    summary["analysis"] = analyze_dataset(df, "ds4_raw", schema_usable=DS4_AMAZON_REVIEWS.usable_columns)
    return summary


def preprocess_ds4(
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    *,
    positive_threshold: float = 4.0,
    negative_threshold: float = 2.0,
    min_text_len: int = 20,
    max_text_len: int = 15000,
    sample_n: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Clean Amazon reviews for standalone NLP — no Goodreads join required."""
    from bookrec.paths import PROC_DS4

    raw = raw_dir or RAW_DS4
    out = out_dir or PROC_DS4
    paths, fmt = _discover(raw)
    if not paths:
        raise FileNotFoundError(f"DS4: place Amazon review JSON/CSV under {raw}")

    if fmt == "json":
        df, load_meta = load_json_records(paths, lines=any(p.suffix == ".jsonl" for p in paths))
    else:
        df, load_meta = load_csv_shards(paths)

    df = standardize_columns(
        df,
        {
            "reviewtext": "review_text",
            "review/text": "review_text",
            "text": "review_text",
            "review_body": "review_text",
            "overall": "star_rating",
            "review/score": "star_rating",
            "rating": "star_rating",
            "reviewerid": "reviewer_id",
            "user_id": "reviewer_id",
            "unixreviewtime": "timestamp",
            "review/time": "timestamp",
            "parent_asin": "parent_asin",
            "id": "asin",
        },
    )

    if "review_text" not in df.columns:
        raise ValueError("DS4: no review text column (reviewText/text/review_body)")

    raw_n = len(df)
    df["review_text_clean"] = df["review_text"].map(lambda x: normalize_review_text(x, lowercase=False))
    df = df[df["review_text_clean"].str.len() >= min_text_len]
    df = df[df["review_text_clean"].str.len() <= max_text_len]

    df["star_rating"] = pd.to_numeric(df.get("star_rating"), errors="coerce")
    df = df.dropna(subset=["star_rating"])
    df = df[(df["star_rating"] >= 1) & (df["star_rating"] <= 5)]
    df["star_rating"] = df["star_rating"].round().astype("int8")

    df["sentiment_label"] = df["star_rating"].map(
        lambda s: _star_to_sentiment(s, positive_threshold, negative_threshold)
    )
    before_sent = len(df)
    df = df.dropna(subset=["sentiment_label"])
    df["sentiment_label"] = df["sentiment_label"].astype("int8")

    if "reviewer_id" not in df.columns:
        df["reviewer_id"] = np.arange(len(df)).astype(str)
    if "asin" not in df.columns and "parent_asin" in df.columns:
        df["asin"] = df["parent_asin"]

    df["source_name"] = DS4_AMAZON_REVIEWS.source_id
    if "summary" in df.columns:
        df["summary_clean"] = df["summary"].map(lambda x: normalize_review_text(x, lowercase=False))
    else:
        df["summary_clean"] = ""

    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state)

    df = df.drop_duplicates(subset=["reviewer_id", "review_text_clean"], keep="first")

    report: dict[str, Any] = {
        "source": DS4_AMAZON_REVIEWS.source_id,
        "load": load_meta,
        "nlp_independent": True,
        "rows_raw": int(raw_n),
        "rows_clean": int(len(df)),
        "rows_dropped_neutral_sentiment": int(before_sent - len(df)),
        "sentiment_positive": int((df["sentiment_label"] == 1).sum()),
        "sentiment_negative": int((df["sentiment_label"] == 0).sum()),
        "rating_outliers": detect_rating_outliers(df["star_rating"]),
        "text_outliers": detect_review_text_outliers(df, "review_text_clean"),
        "label_thresholds": {"positive": positive_threshold, "negative": negative_threshold},
    }

    cols = [
        "reviewer_id",
        "asin",
        "parent_asin",
        "star_rating",
        "sentiment_label",
        "review_text_clean",
        "summary_clean",
        "timestamp",
        "source_name",
    ]
    out_df = df[[c for c in cols if c in df.columns]].copy()
    paths_out = save_stage_output(out_df, out, "reviews_clean", report)
    return {"report": report, "paths": paths_out, "reviews": out_df}
