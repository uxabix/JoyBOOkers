"""Dataset analysis: profiling, missing values, outliers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bookrec.cleaning import dataset_profile, normalize_column_names


def analyze_missing(df: pd.DataFrame) -> dict[str, Any]:
    """Per-column missing counts and rates."""
    df = normalize_column_names(df)
    n = len(df)
    missing = df.isna().sum()
    return {
        "n_rows": int(n),
        "missing_count": missing.astype(int).to_dict(),
        "missing_rate": {c: float(missing[c] / n) if n else 0.0 for c in df.columns},
        "rows_all_missing": int(df.isna().all(axis=1).sum()),
    }


def identify_usable_columns(
    df: pd.DataFrame,
    schema_usable: dict[str, str],
) -> dict[str, Any]:
    """Map schema roles to actual columns present after normalization."""
    df = normalize_column_names(df)
    present = set(df.columns)
    matched: dict[str, str] = {}
    unmatched_roles: list[str] = []
    extra_columns = sorted(present)

    alias_map = {
        "name": "title",
        "title": "name",
        "bookid": "id",
        "goodreads_book_id": "id",
        "reviewtext": "text",
        "text": "reviewtext",
        "overall": "rating",
        "userid": "user_id",
        "reviewerid": "user_id",
    }

    for role in schema_usable:
        if role in present:
            matched[role] = role
            if role in extra_columns:
                extra_columns.remove(role)
            continue
        alt = alias_map.get(role)
        if alt and alt in present:
            matched[role] = alt
            if alt in extra_columns:
                extra_columns.remove(alt)
        else:
            unmatched_roles.append(role)

    return {
        "matched_columns": matched,
        "unmatched_schema_roles": unmatched_roles,
        "extra_columns": extra_columns,
    }


def detect_rating_outliers(series: pd.Series) -> dict[str, int]:
    s = pd.to_numeric(series, errors="coerce")
    return {
        "below_1": int((s < 1).sum()),
        "above_5": int((s > 5).sum()),
        "null_after_coerce": int(s.isna().sum()),
    }


def detect_numeric_outliers(
    df: pd.DataFrame,
    rules: dict[str, tuple[float | None, float | None]],
) -> dict[str, Any]:
    """Count values outside [lo, hi] per column (None = unbounded)."""
    out: dict[str, Any] = {}
    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= s < lo
        if hi is not None:
            mask |= s > hi
        mask &= s.notna()
        out[col] = {
            "outlier_count": int(mask.sum()),
            "lo": lo,
            "hi": hi,
            "p01": float(s.quantile(0.01)) if s.notna().any() else None,
            "p99": float(s.quantile(0.99)) if s.notna().any() else None,
        }
    return out


def detect_interaction_outliers(interactions: pd.DataFrame) -> dict[str, Any]:
    """Power-user and spam-like interaction patterns."""
    report: dict[str, Any] = {}
    if interactions.empty:
        return report
    user_counts = interactions.groupby("user_id").size()
    book_counts = interactions.groupby("book_id").size()
    report["users_with_1_rating"] = int((user_counts == 1).sum())
    report["users_with_gt_5000_ratings"] = int((user_counts > 5000).sum())
    report["books_with_1_rating"] = int((book_counts == 1).sum())
    report["books_with_gt_10000_ratings"] = int((book_counts > 10000).sum())
    report["max_ratings_per_user"] = int(user_counts.max())
    report["max_ratings_per_book"] = int(book_counts.max())
    report["rating_outliers"] = detect_rating_outliers(interactions["rating"])
    return report


def detect_review_text_outliers(df: pd.DataFrame, text_col: str) -> dict[str, Any]:
    if text_col not in df.columns:
        return {}
    lengths = df[text_col].fillna("").astype(str).str.len()
    return {
        "empty_reviews": int((lengths == 0).sum()),
        "very_short_lt_10": int((lengths.between(1, 9)).sum()),
        "very_long_gt_20000": int((lengths > 20000).sum()),
        "length_p50": float(lengths.median()),
        "length_p99": float(lengths.quantile(0.99)) if len(lengths) else 0.0,
    }


def analyze_dataset(
    df: pd.DataFrame,
    name: str,
    *,
    schema_usable: dict[str, str] | None = None,
    numeric_rules: dict[str, tuple[float | None, float | None]] | None = None,
) -> dict[str, Any]:
    """Full analysis report for one loaded table."""
    profile = dataset_profile(df, name)
    report: dict[str, Any] = {
        "profile": profile,
        "missing": analyze_missing(df),
    }
    if schema_usable:
        report["columns"] = identify_usable_columns(df, schema_usable)
    if numeric_rules:
        report["numeric_outliers"] = detect_numeric_outliers(df, numeric_rules)
    return report


BOOK_NUMERIC_RULES: dict[str, tuple[float | None, float | None]] = {
    "publishyear": (1000, 2030),
    "pagesnumber": (1, 10000),
    "pages": (1, 10000),
    "num_pages": (1, 10000),
    "rating": (0.0, 5.0),
    "average_rating": (0.0, 5.0),
    "avg_rating": (0.0, 5.0),
    "countsofreview": (0, None),
    "ratings_count": (0, None),
}
