"""Match books across Goodreads datasets via title + author normalization."""

from __future__ import annotations

from typing import Any

import pandas as pd

from bookrec.text_normalization import normalize_author_for_match, normalize_title_core, normalize_title_for_match
from bookrec.title_matching import TitleMatcher


def _isbn_key(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    s = s.str.replace(r"[^0-9Xx]", "", regex=True)
    return s.where(~s.isin(("", "nan", "None")), other=pd.NA)


def link_books_to_catalog(
    external: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    source_name: str,
    source_id_col: str = "source_book_id",
    title_col: str = "title",
    author_col: str = "authors",
    fuzzy_threshold: int = 88,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Link external source rows to DS1 catalog book ids.

    Strategy (in order):
    1. ISBN-13 / ISBN exact match on catalog
    2. match_key = title_core || author_norm exact
    3. Title-only fuzzy match (TitleMatcher) when author missing or no hit
    """
    stats: dict[str, Any] = {
        "source_name": source_name,
        "rows_input": int(len(external)),
        "match_isbn": 0,
        "match_key_exact": 0,
        "match_title_fuzzy": 0,
        "unmatched": 0,
    }
    work = external.copy()
    work["canonical_book_id"] = pd.NA
    work["match_method"] = pd.NA
    work["match_confidence"] = pd.NA

    cat = catalog.copy()
    if "id" not in cat.columns:
        raise ValueError("catalog must have id column (DS1 book id)")

    # --- ISBN blocking ---
    isbn_cols = [c for c in ("isbn13", "isbn") if c in work.columns and c in cat.columns]
    if isbn_cols:
        for isbn_col in isbn_cols:
            work_isbn = _isbn_key(work[isbn_col])
            cat_isbn = _isbn_key(cat[isbn_col])
            isbn_map = cat.dropna(subset=[isbn_col]).copy()
            isbn_map["_isbn_k"] = _isbn_key(isbn_map[isbn_col])
            isbn_map = isbn_map.dropna(subset=["_isbn_k"]).drop_duplicates("_isbn_k", keep="first")
            lookup = isbn_map.set_index("_isbn_k")["id"]
            miss = work["canonical_book_id"].isna()
            if miss.any():
                matched_ids = work_isbn[miss].map(lookup)
                hit = matched_ids.notna()
                work.loc[miss & hit, "canonical_book_id"] = matched_ids[hit].astype("Int64")
                work.loc[miss & hit, "match_method"] = f"isbn_{isbn_col}"
                work.loc[miss & hit, "match_confidence"] = 1.0
        stats["match_isbn"] = int((work["match_method"].astype(str).str.startswith("isbn")).sum())

    # --- match_key exact ---
    if "match_key" in work.columns and "match_key" in cat.columns:
        miss = work["canonical_book_id"].isna()
        if miss.any():
            key_map = cat.drop_duplicates("match_key", keep="first").set_index("match_key")["id"]
            mk = work.loc[miss, "match_key"].map(key_map)
            hit = mk.notna()
            idx = work.index[miss][hit.values]
            work.loc[idx, "canonical_book_id"] = mk[hit].astype("Int64").values
            work.loc[idx, "match_method"] = "match_key_exact"
            work.loc[idx, "match_confidence"] = 0.95
            stats["match_key_exact"] = int(hit.sum())

    # --- title_core + author_norm compound key ---
    miss = work["canonical_book_id"].isna()
    if miss.any() and title_col in work.columns:
        if "title_core" not in work.columns:
            work["title_core"] = work[title_col].map(normalize_title_core)
        if author_col in work.columns and "author_norm" not in work.columns:
            work["author_norm"] = work[author_col].map(normalize_author_for_match)
        elif "author_norm" not in work.columns:
            work["author_norm"] = ""
        if "title_core" not in cat.columns:
            cat["title_core"] = cat.get("name", cat.get("title", pd.Series(dtype=str))).map(normalize_title_core)
        if "author_norm" not in cat.columns:
            cat["author_norm"] = cat.get("authors", pd.Series(dtype=str)).map(normalize_author_for_match)
        cat["_compound"] = cat["title_core"] + "||" + cat["author_norm"]
        work.loc[miss, "_compound"] = work.loc[miss, "title_core"] + "||" + work.loc[miss, "author_norm"]
        cmap = cat.drop_duplicates("_compound", keep="first").set_index("_compound")["id"]
        comp = work.loc[miss, "_compound"].map(cmap)
        hit = comp.notna()
        idx = work.index[miss][hit.values]
        work.loc[idx, "canonical_book_id"] = comp[hit].astype("Int64").values
        work.loc[idx, "match_method"] = "title_author_exact"
        work.loc[idx, "match_confidence"] = 0.92
        stats["match_key_exact"] += int(hit.sum())

    # --- fuzzy title ---
    miss = work["canonical_book_id"].isna()
    if miss.any() and title_col in work.columns:
        matcher = TitleMatcher.from_catalog(
            cat.rename(columns={"name": "name"}) if "name" in cat.columns else cat.assign(name=cat[title_col]),
            fuzzy_threshold=fuzzy_threshold,
        )
        titles = work.loc[miss, title_col]
        ids, fuzzy_stats = matcher.match_titles(titles)
        hit = ids.notna()
        idx = work.index[miss][hit.values]
        work.loc[idx, "canonical_book_id"] = ids[hit].astype("Int64").values
        work.loc[idx, "match_method"] = "title_fuzzy"
        work.loc[idx, "match_confidence"] = fuzzy_threshold / 100.0
        stats["match_title_fuzzy"] = int(hit.sum())
        stats["fuzzy_detail"] = fuzzy_stats

    stats["unmatched"] = int(work["canonical_book_id"].isna().sum())
    stats["match_rate"] = float(1.0 - stats["unmatched"] / max(stats["rows_input"], 1))

    links = work[[source_id_col, "canonical_book_id", "match_method", "match_confidence"]].copy()
    links = links.rename(columns={source_id_col: "source_book_id"})
    links["source_name"] = source_name
    links = links.dropna(subset=["canonical_book_id"])
    links["canonical_book_id"] = links["canonical_book_id"].astype("int64")
    return links, stats


def link_external_books(
    ds2: pd.DataFrame | None,
    ds3: pd.DataFrame | None,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Produce combined book_links table for DS2 and DS3."""
    frames: list[pd.DataFrame] = []
    all_stats: dict[str, Any] = {}
    if ds2 is not None and len(ds2):
        links, st = link_books_to_catalog(ds2, catalog, source_name="ds2_goodreads_100k")
        frames.append(links)
        all_stats["ds2"] = st
    if ds3 is not None and len(ds3):
        links, st = link_books_to_catalog(ds3, catalog, source_name="ds3_goodreads_best")
        frames.append(links)
        all_stats["ds3"] = st
    if not frames:
        return pd.DataFrame(), all_stats
    combined = pd.concat(frames, ignore_index=True)
    return combined, all_stats
