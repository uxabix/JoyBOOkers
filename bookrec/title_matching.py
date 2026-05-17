"""Match rating rows to catalog books by title (normalized + fuzzy)."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

# Strip series / edition hints in trailing parentheses.
_TRAILING_PAREN_RE = re.compile(r"\s*\([^)]*#\s*\d+[^)]*\)\s*$")
_TRAILING_ANY_PAREN_RE = re.compile(r"\s*\([^)]{1,120}\)\s*$")
_QUOTE_RE = re.compile(r"[`´']")
_WS_RE = re.compile(r"\s+")


def normalize_title_for_match(title: str) -> str:
    """Lowercase, collapse whitespace, normalize quotes."""
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return ""
    s = str(title).strip().lower()
    if not s:
        return ""
    s = _QUOTE_RE.sub("'", s)
    s = _WS_RE.sub(" ", s)
    return s


def normalize_title_core(title: str) -> str:
    """Aggressive key: drop series parentheses and subtitle after colon."""
    s = normalize_title_for_match(title)
    if not s:
        return ""
    s = _TRAILING_PAREN_RE.sub("", s).strip()
    s = _TRAILING_ANY_PAREN_RE.sub("", s).strip()
    if ":" in s:
        left, _, _right = s.partition(":")
        if len(left) >= 8:
            s = left.strip()
    return s


def _block_key(normalized: str, length: int = 36) -> str:
    if not normalized:
        return ""
    return normalized[:length] if len(normalized) > length else normalized


@dataclass
class TitleMatcher:
    """Catalog title index for exact and fuzzy matching."""

    exact: dict[str, int] = field(default_factory=dict)
    core: dict[str, int] = field(default_factory=dict)
    key_to_id: dict[str, int] = field(default_factory=dict)
    blocks: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    fuzzy_threshold: int = 88
    block_key_len: int = 36
    max_candidates_per_query: int = 800

    @classmethod
    def from_catalog(
        cls,
        catalog: pd.DataFrame,
        *,
        fuzzy_threshold: int = 88,
        block_key_len: int = 36,
    ) -> TitleMatcher:
        if "id" not in catalog.columns or "name" not in catalog.columns:
            raise ValueError("catalog must contain id and name columns")

        matcher = cls(fuzzy_threshold=fuzzy_threshold, block_key_len=block_key_len)
        cat = catalog[["id", "name"]].copy()
        cat["id"] = pd.to_numeric(cat["id"], errors="coerce")
        cat = cat.dropna(subset=["id"])
        cat["id"] = cat["id"].astype("int64")
        cat = cat.sort_values("id")

        for _raw, book_id in zip(cat["name"].astype(str), cat["id"], strict=True):
            raw = _raw.strip()
            if not raw:
                continue
            norm = normalize_title_for_match(raw)
            core = normalize_title_core(raw)
            bid = int(book_id)
            if norm and norm not in matcher.exact:
                matcher.exact[norm] = bid
            if core and core not in matcher.core:
                matcher.core[core] = bid
            for key in (norm, core):
                if not key:
                    continue
                matcher.key_to_id.setdefault(key, bid)
                bk = _block_key(key, block_key_len)
                matcher.blocks.setdefault(bk, []).append((key, bid))

        return matcher

    def _candidate_keys(self, normalized: str) -> list[str]:
        """Collect catalog title keys from block index (and shorter prefixes)."""
        if not normalized:
            return []
        seen: set[str] = set()
        keys: list[str] = []
        lengths = (self.block_key_len, 28, 20, 14)
        for n in lengths:
            bk = _block_key(normalized, n)
            for title_key, _bid in self.blocks.get(bk, []):
                if title_key not in seen:
                    seen.add(title_key)
                    keys.append(title_key)
            if len(keys) >= self.max_candidates_per_query:
                break
        if len(keys) > self.max_candidates_per_query:
            # Prefer similar length to reduce false positives.
            q_len = len(normalized)
            keys.sort(key=lambda k: abs(len(k) - q_len))
            keys = keys[: self.max_candidates_per_query]
        return keys

    def fuzzy_match_one(self, normalized: str) -> tuple[int | None, float | None]:
        candidates = self._candidate_keys(normalized)
        if not candidates:
            return None, None
        result = process.extractOne(
            normalized,
            candidates,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=self.fuzzy_threshold,
        )
        if result is None:
            return None, None
        matched_key, score, _idx = result
        book_id = self.key_to_id.get(matched_key)
        return (int(book_id) if book_id is not None else None), float(score)

    def match_titles(self, titles: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
        """Return book_id series (nullable int64) and match statistics."""
        stats: dict[str, Any] = {
            "title_match_fuzzy_threshold": self.fuzzy_threshold,
            "title_match_exact_normalized": 0,
            "title_match_exact_core": 0,
            "title_match_fuzzy": 0,
            "title_match_unmatched": 0,
        }

        norm = titles.map(normalize_title_for_match)
        core = titles.map(normalize_title_core)

        book_id = norm.map(self.exact)
        exact_norm_mask = book_id.notna()
        stats["title_match_exact_normalized"] = int(exact_norm_mask.sum())

        miss = ~exact_norm_mask
        if miss.any():
            book_id = book_id.astype("object")
            core_hits = core[miss].map(self.core)
            book_id.loc[miss] = book_id.loc[miss].where(book_id.loc[miss].notna(), core_hits)
        exact_core_mask = book_id.notna() & ~exact_norm_mask
        stats["title_match_exact_core"] = int(exact_core_mask.sum())

        still_miss = book_id.isna()
        if still_miss.any() and self.fuzzy_threshold > 0:
            unique_norm = norm[still_miss].drop_duplicates()
            fuzzy_map: dict[str, int] = {}
            fuzzy_scores: list[float] = []
            total_unique = len(unique_norm)
            for i, n in enumerate(unique_norm, start=1):
                if i == 1 or i % 5000 == 0 or i == total_unique:
                    print(
                        f"    Fuzzy title match: {i}/{total_unique} unique titles...",
                        flush=True,
                    )
                if not n:
                    continue
                bid, score = self.fuzzy_match_one(n)
                if bid is not None:
                    fuzzy_map[n] = bid
                    if score is not None:
                        fuzzy_scores.append(score)

            if fuzzy_map:
                fuzzy_series = norm[still_miss].map(fuzzy_map)
                book_id = book_id.astype("object")
                book_id.loc[still_miss] = book_id.loc[still_miss].where(
                    book_id.loc[still_miss].notna(), fuzzy_series
                )
                stats["title_match_fuzzy"] = int(fuzzy_series.notna().sum())
                if fuzzy_scores:
                    stats["title_match_fuzzy_score_avg"] = float(sum(fuzzy_scores) / len(fuzzy_scores))
                    stats["title_match_fuzzy_score_min"] = float(min(fuzzy_scores))

        stats["title_match_unmatched"] = int(book_id.isna().sum())
        return book_id.astype("Int64"), stats
