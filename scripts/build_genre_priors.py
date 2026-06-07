#!/usr/bin/env python3
"""Build global + per-cluster genre priors for cold-start users."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml.user_profile import _split_genres  # noqa: E402
from bookrec.io_utils import read_table  # noqa: E402
from bookrec.paths import PROC_FEATURES  # noqa: E402


def _normalize(counter: Counter[str], *, top_n: int = 40) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    items = counter.most_common(top_n)
    subtotal = sum(c for _, c in items) or 1.0
    return {g: c / subtotal for g, c in items}


def build_genre_priors() -> dict:
    catalog_path = PROC_FEATURES / "content" / "content_catalog.parquet"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Missing catalog: {catalog_path}")
    catalog = read_table(catalog_path)
    for candidate in ("genres", "genre", "extra_genres"):
        if candidate in catalog.columns:
            genre_col = candidate
            break
    else:
        raise ValueError("content_catalog has no genre/genres column")

    book_id_col = "book_id" if "book_id" in catalog.columns else "source_book_id"
    weight_col = "rating_count" if "rating_count" in catalog.columns else None

    global_counter: Counter[str] = Counter()
    book_genres: dict[str, list[str]] = {}
    for row in catalog.itertuples(index=False):
        bid = str(getattr(row, book_id_col))
        genres = _split_genres(getattr(row, genre_col, None))
        book_genres[bid] = genres
        w = float(getattr(row, weight_col, 1) or 1) if weight_col else 1.0
        for g in genres:
            global_counter[g] += w

    clusters: dict[str, dict[str, float]] = {}
    affinity_path = PROC_FEATURES / "clustering" / "cluster_affinity.json"
    if affinity_path.is_file():
        with affinity_path.open(encoding="utf-8") as fh:
            affinity = json.load(fh)
        for cluster_id, books in affinity.items():
            counter: Counter[str] = Counter()
            for book_id, score in books.items():
                for g in book_genres.get(str(book_id), []):
                    counter[g] += float(score)
            clusters[str(cluster_id)] = _normalize(counter)

    return {"global": _normalize(global_counter), "clusters": clusters}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build genre priors JSON for cold-start.")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROC_FEATURES / "clustering" / "genre_priors.json",
    )
    args = parser.parse_args()
    priors = build_genre_priors()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(priors, fh, indent=2)
    print(
        f"Wrote genre priors: global={len(priors['global'])} genres, "
        f"clusters={len(priors['clusters'])} -> {args.out}"
    )


if __name__ == "__main__":
    main()
