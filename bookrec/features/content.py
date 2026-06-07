"""Content-based features — DS2 (primary) + DS3 (enrichment) only.

DS1 (1.5M+ book catalog) is NOT used here. See bookrec/constraints.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix, save_npz

from bookrec.constraints import DEFAULT_MAX_TEXT_FEATURES, MAX_CONTENT_BOOKS
from bookrec.features._text_vectors import build_sparse_bow
from bookrec.io_utils import write_json, write_table


def _list_cell(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [str(value)] if str(value).strip() else []


def _row_to_record(
    *,
    book_key: str,
    source_book_id: str,
    title: str,
    authors: str,
    desc: str,
    genres: list[str],
    tags: list[str],
    chars: list[str],
    source: str,
    ds3_enriched: bool,
) -> dict[str, Any]:
    genre_text = " ".join(genres + tags)
    char_text = " ".join(chars[:15])
    content_text = " ".join(filter(None, [title, authors, desc, genre_text, char_text]))
    return {
        "book_key": book_key,
        "source_book_id": source_book_id,
        "title": title,
        "authors": authors,
        "content_text": content_text,
        "genres": genres,
        "tags": tags,
        "characters": chars,
        "source": source,
        "ds3_enriched": ds3_enriched,
    }


def _merge_ds2_ds3_catalog(
    ds2: pd.DataFrame,
    ds3: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """DS2 base catalog; enrich matching rows with DS3 tags/characters via match_key."""
    stats: dict[str, Any] = {"ds2_rows": int(len(ds2)), "ds3_enriched": 0, "ds3_only_rows": 0}
    base = ds2.copy()
    if "source_book_id" not in base.columns:
        base["source_book_id"] = base.index.astype(str)

    ds3_idx: pd.DataFrame | None = None
    if ds3 is not None and len(ds3) and "match_key" in ds3.columns:
        ds3_idx = ds3.drop_duplicates("match_key", keep="first").set_index("match_key")

    records: list[dict[str, Any]] = []
    enriched_count = 0

    for _, row in base.iterrows():
        title = str(row.get("title", ""))
        authors = str(row.get("authors", row.get("author", "")))
        desc = str(row.get("description", ""))
        genres = _list_cell(row.get("genres_list"))
        tags: list[str] = []
        chars: list[str] = []
        key = str(row.get("match_key", ""))
        if ds3_idx is not None and key and key in ds3_idx.index:
            d3 = ds3_idx.loc[key]
            if isinstance(d3, pd.DataFrame):
                d3 = d3.iloc[0]
            tags = _list_cell(d3.get("tags_list"))
            extra_genres = _list_cell(d3.get("genres_list"))
            genres = list(dict.fromkeys(genres + extra_genres))
            chars = _list_cell(d3.get("characters_list"))
            if tags or chars or extra_genres:
                enriched_count += 1
        records.append(
            _row_to_record(
                book_key=f"ds2:{row['source_book_id']}",
                source_book_id=str(row["source_book_id"]),
                title=title,
                authors=authors,
                desc=desc,
                genres=genres,
                tags=tags,
                chars=chars,
                source="ds2_goodreads_100k",
                ds3_enriched=bool(tags or chars),
            )
        )

    stats["ds3_enriched"] = enriched_count

    if ds3 is not None and len(ds3) and "match_key" in ds3.columns:
        ds2_keys = set(base["match_key"].dropna().astype(str))
        ds3_only = ds3[~ds3["match_key"].astype(str).isin(ds2_keys)]
        stats["ds3_only_rows"] = int(len(ds3_only))
        for _, row in ds3_only.iterrows():
            records.append(
                _row_to_record(
                    book_key=f"ds3:{row['source_book_id']}",
                    source_book_id=str(row["source_book_id"]),
                    title=str(row.get("title", "")),
                    authors=str(row.get("authors", "")),
                    desc=str(row.get("description", "")),
                    genres=_list_cell(row.get("genres_list")),
                    tags=_list_cell(row.get("tags_list")),
                    chars=_list_cell(row.get("characters_list")),
                    source="ds3_goodreads_best",
                    ds3_enriched=False,
                )
            )

    catalog = pd.DataFrame(records)
    stats["catalog_rows"] = int(len(catalog))
    return catalog, stats


def _genre_multihot_sparse(genre_lists: pd.Series, top_n: int = 200) -> tuple[csr_matrix, list[str]]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for gl in genre_lists:
        if isinstance(gl, list):
            counter.update(g.lower() for g in gl if g)
    top_genres = [g for g, _ in counter.most_common(top_n)]
    g2i = {g: i for i, g in enumerate(top_genres)}
    mat = lil_matrix((len(genre_lists), len(top_genres)), dtype=np.float32)
    for i, gl in enumerate(genre_lists):
        if not isinstance(gl, list):
            continue
        for g in gl:
            j = g2i.get(str(g).lower())
            if j is not None:
                mat[i, j] = 1.0
    return csr_matrix(mat), top_genres


def build_content_features(
    ds2_books: pd.DataFrame | None,
    ds3_books: pd.DataFrame | None,
    out_dir: Path,
    *,
    max_text_features: int = DEFAULT_MAX_TEXT_FEATURES,
    max_books: int = MAX_CONTENT_BOOKS,
) -> dict[str, Any]:
    """Sparse BoW for content-based recommendation (DS2 + optional DS3 enrichment)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ds2_books is None or not len(ds2_books):
        report = {
            "error": "DS2 (Goodreads 100k) is required for content-based features",
            "hint": "Place GoodReads_100k_books.csv under data/raw/ds2_goodreads_100k/",
        }
        write_json(report, out_dir / "content_features_report.json")
        return report

    content_df, merge_stats = _merge_ds2_ds3_catalog(ds2_books, ds3_books)

    if len(content_df) > max_books:
        report = {
            "error": f"Content catalog has {len(content_df)} rows; limit is {max_books}",
            "hint": "University scope uses DS2 (~100k) only; do not pass DS1 catalog.",
        }
        write_json(report, out_dir / "content_features_report.json")
        return report

    texts = content_df["content_text"].fillna("").tolist()
    bow, vocabulary = build_sparse_bow(texts, max_features=max_text_features)
    genre_mat, genre_labels = _genre_multihot_sparse(content_df["genres"])

    paths: dict[str, str] = {}
    paths["content_catalog"] = str(write_table(content_df, out_dir / "content_catalog"))
    save_npz(out_dir / "bow_matrix.npz", bow)
    save_npz(out_dir / "genre_matrix.npz", genre_mat)
    paths["bow_matrix"] = str(out_dir / "bow_matrix.npz")
    paths["genre_matrix"] = str(out_dir / "genre_matrix.npz")

    with (out_dir / "vocabulary.json").open("w", encoding="utf-8") as f:
        json.dump(vocabulary, f)
    with (out_dir / "genre_labels.json").open("w", encoding="utf-8") as f:
        json.dump(genre_labels, f)

    report = {
        "dataset_scope": "ds2_primary_ds3_enrichment",
        "ds1_excluded": True,
        "matrix_format": "scipy.sparse.csr",
        "n_books": int(len(content_df)),
        "n_vocab": len(vocabulary),
        "n_genres": len(genre_labels),
        "bow_nnz": int(bow.nnz),
        "merge": merge_stats,
        "sources": content_df["source"].value_counts().to_dict(),
        "paths": paths,
    }
    write_json(report, out_dir / "content_features_report.json")
    return report
