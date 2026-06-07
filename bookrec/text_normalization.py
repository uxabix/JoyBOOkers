"""Normalize titles, authors, and review text for matching and NLP."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Re-export title helpers used across the pipeline.
from bookrec.title_matching import normalize_title_core, normalize_title_for_match

__all__ = [
    "normalize_title_for_match",
    "normalize_title_core",
    "normalize_author_for_match",
    "normalize_author_primary",
    "normalize_review_text",
    "add_match_keys",
]

_AUTHOR_SPLIT_RE = re.compile(r"[|/;]+")
_WS_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^\w\s'-]", flags=re.UNICODE)
_HONORIFIC_RE = re.compile(
    r"^(dr|prof|mr|mrs|ms|miss|sir|dame)\.?\s+",
    flags=re.IGNORECASE,
)
_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv)\.?$", flags=re.IGNORECASE)


def _unicode_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def split_authors(raw: str) -> list[str]:
    """Split Goodreads-style author lists (pipe, slash, comma)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    parts = _AUTHOR_SPLIT_RE.split(str(raw).strip())
    return [p.strip() for p in parts if p and p.strip()]


def normalize_author_primary(raw: str) -> str:
    """First author only, normalized for display keys."""
    authors = split_authors(raw)
    if not authors:
        return ""
    return normalize_author_for_match(authors[0])


def normalize_author_for_match(raw: str) -> str:
    """Aggressive author key for cross-dataset blocking."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip().lower()
    if not s:
        return ""
    s = _unicode_fold(s)
    s = _HONORIFIC_RE.sub("", s)
    s = _SUFFIX_RE.sub("", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    # "lastname, firstname" → "firstname lastname" (heuristic)
    if "," in s:
        left, _, right = s.partition(",")
        if right.strip() and left.strip():
            s = f"{right.strip()} {left.strip()}"
    return s


def normalize_review_text(text: str, *, lowercase: bool = False) -> str:
    """Clean review body for NLP (independent of book linkage)."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).strip()
    if not s:
        return ""
    s = _HTML_TAG_RE.sub(" ", s)
    s = _unicode_fold(s)
    s = _WS_RE.sub(" ", s).strip()
    if lowercase:
        s = s.lower()
    return s


def add_match_keys(
    df: pd.DataFrame,
    *,
    title_col: str = "title",
    author_col: str | None = "authors",
) -> pd.DataFrame:
    """Add title_norm, title_core, author_norm, match_key columns."""
    out = df.copy()
    if title_col not in out.columns:
        raise ValueError(f"Missing title column: {title_col}")
    titles = out[title_col].astype(str)
    out["title_norm"] = titles.map(normalize_title_for_match)
    out["title_core"] = titles.map(normalize_title_core)
    if author_col and author_col in out.columns:
        out["author_norm"] = out[author_col].map(normalize_author_primary)
    else:
        out["author_norm"] = ""
    out["match_key"] = out["title_core"] + "||" + out["author_norm"]
    return out
