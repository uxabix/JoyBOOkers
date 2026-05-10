from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = []
    for c in out.columns:
        s = str(c).strip().lower().replace(" ", "_").replace("-", "_")
        cols.append(s)
    out.columns = cols
    return out


BOOK_COLUMN_ALIASES = {
    "goodreads_book_id": "id",
    "title": "name",
}

RATING_COLUMN_ALIASES = {
    "goodreads_book_id": "book_id",
    "userid": "user_id",
    "bookid": "book_id",
    "item_id": "book_id",
}


def apply_book_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)
    renames = {k: v for k, v in BOOK_COLUMN_ALIASES.items() if k in df.columns and v not in df.columns}
    return df.rename(columns=renames) if renames else df


def apply_rating_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_column_names(df)
    renames: dict[str, str] = {}
    for k, v in RATING_COLUMN_ALIASES.items():
        if k in df.columns and v not in df.columns:
            renames[k] = v
    return df.rename(columns=renames) if renames else df


# Goodreads shelf labels (Kaggle export with ID, Name, Rating text)
GOODREADS_TEXT_RATING_TO_STARS: dict[str, int] = {
    "did not like it": 1,
    "it was ok": 2,
    "liked it": 3,
    "really liked it": 4,
    "it was amazing": 5,
}


def _goodreads_text_to_stars(series: pd.Series) -> pd.Series:
    """Map Goodreads text ratings to 1–5; numeric strings still work; unknown → NaN."""

    def one(val: Any) -> float:
        if pd.isna(val):
            return np.nan
        raw = str(val).strip()
        if not raw:
            return np.nan
        low = raw.lower()
        if "doesn't have any rating" in low or "does not have any rating" in low:
            return np.nan
        if low in GOODREADS_TEXT_RATING_TO_STARS:
            return float(GOODREADS_TEXT_RATING_TO_STARS[low])
        try:
            f = float(raw)
            if 1 <= f <= 5:
                return float(int(round(f)))
        except ValueError:
            pass
        return np.nan

    return series.map(one)


def _is_title_label_rating_table(df: pd.DataFrame) -> bool:
    """Kaggle-style per-row file: user id, book title, text rating (no book_id column)."""
    cols = set(df.columns)
    if "book_id" in cols or "user_id" in cols:
        return False
    return {"id", "name", "rating"}.issubset(cols)


def _clean_interactions_title_labels(
    df: pd.DataFrame,
    books_catalog: pd.DataFrame,
    valid_book_ids: set[int] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join ratings on book title to catalog ids; map text stars to 1–5."""
    report: dict[str, Any] = {"interactions_source_format": "title_labels"}
    raw_n = len(df)
    work = df.replace(r"^\s*$", np.nan, regex=True).copy()
    work = work.rename(columns={"id": "user_id", "name": "book_title"})
    work["user_id"] = pd.to_numeric(work["user_id"], errors="coerce")
    work = work.dropna(subset=["user_id", "book_title"])
    work["user_id"] = work["user_id"].astype("int64")
    work["book_title"] = work["book_title"].astype(str).str.strip()

    work["rating"] = _goodreads_text_to_stars(work["rating"])
    before_rating = len(work)
    work = work.dropna(subset=["rating"])
    report["rows_dropped_unmapped_rating"] = int(before_rating - len(work))

    cat = books_catalog.copy()
    if "name" not in cat.columns or "id" not in cat.columns:
        raise ValueError("books_catalog must contain columns id and name for title matching.")
    cat["_match_name"] = cat["name"].astype(str).str.strip()
    cat = cat.sort_values("id").drop_duplicates(subset=["_match_name"], keep="first")
    lookup = cat.set_index("_match_name")["id"]

    work["book_id"] = work["book_title"].map(lookup)
    before_join = len(work)
    work = work.dropna(subset=["book_id"])
    work["book_id"] = work["book_id"].astype("int64")
    report["rows_dropped_no_title_match"] = int(before_join - len(work))

    work["rating"] = work["rating"].astype("int64")
    work = work[(work["rating"] >= 1) & (work["rating"] <= 5)]

    if "timestamp" in work.columns:
        work["timestamp_parsed"] = _coerce_timestamp(work["timestamp"])
        work = work.sort_values(["user_id", "book_id", "timestamp_parsed"], na_position="first")
    else:
        work = work.sort_values(["user_id", "book_id"])

    before_dedup = len(work)
    work = work.drop_duplicates(subset=["user_id", "book_id"], keep="last")
    report["interactions_duplicate_user_book_dropped"] = int(before_dedup - len(work))

    if valid_book_ids is not None:
        work = work[work["book_id"].isin(valid_book_ids)]
        report["interactions_filtered_unknown_book"] = True
    else:
        report["interactions_filtered_unknown_book"] = False

    out = work[["user_id", "book_id", "rating"]].copy()
    out["user_id"] = out["user_id"].astype("int32")
    out["book_id"] = out["book_id"].astype("int32")
    out["rating"] = out["rating"].astype("int8")

    report["interactions_rows_raw"] = int(raw_n)
    report["interactions_rows_clean"] = int(len(out))
    report["n_users"] = int(out["user_id"].nunique())
    report["n_books_rated"] = int(out["book_id"].nunique())
    denom = max(report["n_users"] * report["n_books_rated"], 1)
    report["density"] = float(len(out) / denom)
    return out, report


def _clean_interactions_explicit_ids(
    df: pd.DataFrame,
    valid_book_ids: set[int] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Classic user_id / book_id / numeric rating matrix."""
    report: dict[str, Any] = {"interactions_source_format": "explicit_ids"}
    raw_n = len(df)
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["book_id"] = pd.to_numeric(df["book_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df.dropna(subset=["user_id", "book_id", "rating"])
    df["user_id"] = df["user_id"].astype("int64")
    df["book_id"] = df["book_id"].astype("int64")
    df["rating"] = df["rating"].round().astype("int64")
    df = df[(df["rating"] >= 1) & (df["rating"] <= 5)]

    if "timestamp" in df.columns:
        df["timestamp_parsed"] = _coerce_timestamp(df["timestamp"])
        df = df.sort_values(["user_id", "book_id", "timestamp_parsed"], na_position="first")
    else:
        df = df.sort_values(["user_id", "book_id"])

    before_dedup = len(df)
    df = df.drop_duplicates(subset=["user_id", "book_id"], keep="last")
    report["interactions_duplicate_user_book_dropped"] = int(before_dedup - len(df))

    if valid_book_ids is not None:
        df = df[df["book_id"].isin(valid_book_ids)]
        report["interactions_filtered_unknown_book"] = True
    else:
        report["interactions_filtered_unknown_book"] = False

    out = df[["user_id", "book_id", "rating"]].copy()
    out["user_id"] = out["user_id"].astype("int32")
    out["book_id"] = out["book_id"].astype("int32")
    out["rating"] = out["rating"].astype("int8")

    report["interactions_rows_raw"] = int(raw_n)
    report["interactions_rows_clean"] = int(len(out))
    report["n_users"] = int(out["user_id"].nunique())
    report["n_books_rated"] = int(out["book_id"].nunique())
    denom = max(report["n_users"] * report["n_books_rated"], 1)
    report["density"] = float(len(out) / denom)
    return out, report


def _parse_rating_dist_cell(value: Any) -> int | None:
    """Parse cells like '1:42' or 'total:1234' → integer after colon."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if ":" not in s:
        m = re.search(r"(\d+)\s*$", s)
        return int(m.group(1)) if m else None
    _, tail = s.rsplit(":", 1)
    tail = tail.strip()
    try:
        return int(tail)
    except ValueError:
        m = re.search(r"(\d+)", tail)
        return int(m.group(1)) if m else None


def parse_rating_distribution_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric columns n_rating_star_1 … n_rating_star_5, n_rating_total_parsed."""
    out = df.copy()
    dist_cols = [f"ratingdist{i}" for i in range(1, 6)]
    for c in dist_cols:
        if c in out.columns:
            out[f"n_rating_star_{c[-1]}"] = out[c].map(_parse_rating_dist_cell)
    if "ratingdisttotal" in out.columns:
        out["n_rating_total_parsed"] = out["ratingdisttotal"].map(_parse_rating_dist_cell)
    return out


def load_books_csv(path: str | Path, **read_csv_kw: Any) -> pd.DataFrame:
    path = Path(path)
    kw = dict(low_memory=False, encoding="utf-8", on_bad_lines="skip")
    kw.update(read_csv_kw)
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        return pd.read_csv(path, **kw)


def load_ratings_csv(path: str | Path, **read_csv_kw: Any) -> pd.DataFrame:
    path = Path(path)
    kw = dict(low_memory=False, encoding="utf-8", on_bad_lines="skip")
    kw.update(read_csv_kw)
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        kw["encoding"] = "latin-1"
        return pd.read_csv(path, **kw)


def clean_books(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return cleaned books and a small report dict."""
    report: dict[str, Any] = {}
    raw_n = len(df)
    df = normalize_column_names(df)

    # Expected id column for books
    id_col = "id" if "id" in df.columns else None
    if id_col is None:
        raise ValueError("Books table must contain an 'id' column (after normalization).")

    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Core numeric / text constraints
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce")
    df = df[df[id_col].notna() & (df[id_col] > 0)]
    df[id_col] = df[id_col].astype("int64")

    if "name" in df.columns:
        df = df[df["name"].notna() & (df["name"].astype(str).str.len() > 0)]

    def _valid_numeric_bounds(series: pd.Series, lo: float | None, hi: float | None) -> pd.Series:
        ok = pd.Series(True, index=series.index)
        present = series.notna()
        if lo is not None:
            ok &= ~present | (series >= lo)
        if hi is not None:
            ok &= ~present | (series <= hi)
        return ok

    num_cols = [
        ("publishyear", 1000, 3000),
        ("publishmonth", 1, 12),
        ("publishday", 1, 31),
        ("pagesnumber", 1, None),
        ("rating", 0.0, 5.0),
        ("countsofreview", 0, None),
    ]
    for col, lo, hi in num_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[_valid_numeric_bounds(df[col], lo, hi)]

    df = parse_rating_distribution_columns(df)
    before_id_dedup = len(df)
    df = df.drop_duplicates(subset=[id_col], keep="first")

    report["books_rows_input"] = int(raw_n)
    report["books_rows_clean"] = int(len(df))
    report["books_duplicate_ids_dropped"] = int(before_id_dedup - len(df))
    return df, report


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().mean() > 0.9 and (s.dropna() > 1e12).mean() > 0.5:
        return pd.to_datetime(s, unit="ms", errors="coerce")
    if s.notna().mean() > 0.9 and (s.dropna() > 1e9).mean() > 0.5:
        return pd.to_datetime(s, unit="s", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def clean_interactions(
    df: pd.DataFrame,
    books_catalog: pd.DataFrame | None = None,
    valid_book_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build user — book — rating table.

    Supports:
    - Explicit matrix: user_id, book_id, rating (optional timestamp).
    - Kaggle text file: id (user), name (book title), rating (Goodreads text labels).
    """
    df = apply_rating_column_aliases(df)

    if {"user_id", "book_id", "rating"}.issubset(df.columns):
        return _clean_interactions_explicit_ids(df, valid_book_ids)

    if _is_title_label_rating_table(df):
        if books_catalog is None:
            raise ValueError(
                "Ratings file looks like ID + book title + text rating (no book_id). "
                "Provide books_catalog to match titles to ids, or use a CSV with "
                "user_id, book_id, rating columns."
            )
        return _clean_interactions_title_labels(df, books_catalog, valid_book_ids)

    cols = ", ".join(sorted(df.columns))
    raise ValueError(
        "Unrecognized ratings table layout. Columns: "
        f"{cols}. Expected either user_id + book_id + rating, or id + name + rating "
        "(Goodreads text stars)."
    )


def dataset_profile(df: pd.DataFrame, name: str) -> dict[str, Any]:
    """Shape, dtypes, missing counts, duplicate keys."""
    df = normalize_column_names(df)
    profile: dict[str, Any] = {
        "name": name,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_per_column": df.isna().sum().astype(int).to_dict(),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024**2)),
    }
    if "id" in df.columns and name.startswith("book"):
        d = pd.to_numeric(df["id"], errors="coerce")
        profile["duplicate_ids"] = int(d.duplicated().sum())
    if {"user_id", "book_id"}.issubset(df.columns):
        key = df[["user_id", "book_id"]].astype(str).agg("|".join, axis=1)
        profile["duplicate_user_book_pairs"] = int(key.duplicated().sum())
    return profile
