"""Offline feature matrix for training / evaluating hybrid weights."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.ml.collaborative import CollaborativeFilteringEngine
from app.ml.content_based import ContentRecommendationEngine
from app.ml.user_profile import _split_genres, genre_match_score
from bookrec.io_utils import read_table
from bookrec.paths import MODEL_CLUSTERING_DIR, PROC_FEATURES, PROC_SPLITS


def _norm_cf(score: float) -> float:
    return max(0.0, min(1.0, (float(score) - 1.0) / 4.0))


def _load_cluster_map() -> dict[str, int]:
    for path in (
        MODEL_CLUSTERING_DIR / "user_cluster_assignments.parquet",
        MODEL_CLUSTERING_DIR / "user_cluster_assignments.csv",
    ):
        if path.is_file():
            df = read_table(path)
            return dict(zip(df["user_id"].astype(str), df["cluster_id"].astype(int), strict=False))
    return {}


def _load_book_genres() -> dict[str, str | None]:
    catalog = read_table(PROC_FEATURES / "content" / "content_catalog.parquet")
    col = "book_id" if "book_id" in catalog.columns else "source_book_id"
    genre_col = next((c for c in ("genres", "genre", "extra_genres") if c in catalog.columns), None)
    genres = catalog[genre_col] if genre_col else None
    if genres is None:
        return {str(b): None for b in catalog[col].astype(str)}
    return dict(zip(catalog[col].astype(str), genres, strict=False))


def _load_cluster_affinity() -> dict[int, dict[str, float]]:
    path = PROC_FEATURES / "clustering" / "cluster_affinity.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return {int(k): {str(b): float(s) for b, s in v.items()} for k, v in raw.items()}


def _load_genre_priors() -> dict:
    path = PROC_FEATURES / "clustering" / "genre_priors.json"
    if not path.is_file():
        return {"global": {}, "clusters": {}}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _genre_weights_from_history(
    history: list[tuple[str, float, str | None]],
    *,
    cluster_id: int,
    priors: dict,
) -> dict[str, float]:
    if history:
        weights: dict[str, float] = {}
        for _, score, genre in history:
            for g in _split_genres(genre):
                weights[g] = weights.get(g, 0.0) + score
        total = sum(weights.values())
        return {g: w / total for g, w in weights.items()} if total > 0 else {}
    global_w = priors.get("global") or {}
    cluster_w = (priors.get("clusters") or {}).get(str(cluster_id), {})
    blended: dict[str, float] = {}
    for g in set(global_w) | set(cluster_w):
        blended[g] = 0.35 * float(global_w.get(g, 0)) + 0.65 * float(cluster_w.get(g, 0))
    total = sum(blended.values())
    return {g: w / total for g, w in blended.items()} if total > 0 else {}


def build_training_frame(
    *,
    sample_size: int,
    cf_engine: CollaborativeFilteringEngine,
    content_engine: ContentRecommendationEngine,
) -> tuple[np.ndarray, np.ndarray]:
    interactions = read_table(PROC_SPLITS / "cf_train.parquet")
    if len(interactions) > sample_size:
        interactions = interactions.sample(n=sample_size, random_state=42)

    cluster_map = _load_cluster_map()
    book_genres = _load_book_genres()
    cluster_affinity = _load_cluster_affinity()
    genre_priors = _load_genre_priors()
    pop_counts = interactions.groupby("book_id").size().astype(float).to_dict()
    max_pop = max(pop_counts.values()) if pop_counts else 1.0

    user_history: dict[str, list[tuple[str, float, str | None]]] = defaultdict(list)
    for row in interactions.itertuples(index=False):
        uid = str(row.user_id)
        bid = str(row.book_id)
        user_history[uid].append((bid, float(row.rating), book_genres.get(bid)))

    rows_x: list[list[float]] = []
    rows_y: list[float] = []

    for row in interactions.itertuples(index=False):
        uid = str(row.user_id)
        bid = str(row.book_id)
        rating = float(row.rating)
        history = [(b, s, g) for b, s, g in user_history[uid] if b != bid]
        cluster_id = cluster_map.get(uid, 1)
        genre_w = _genre_weights_from_history(history, cluster_id=cluster_id, priors=genre_priors)

        cf_raw = cf_engine.predict(uid, bid)
        cf = _norm_cf(cf_raw) if cf_raw is not None else 0.0

        content = 0.0
        if history and content_engine.is_loaded:
            pairs = [(b, s) for b, s, _ in history[:8]]
            vec = content_engine.build_user_vector(pairs)
            if vec is not None:
                scores = content_engine.score_candidates(vec, [bid])
                content = scores.get(bid, 0.0)

        cluster = cluster_affinity.get(cluster_id, {}).get(bid, 0.0)
        pop_norm = float(pop_counts.get(bid, 0)) / max_pop
        genre = genre_match_score(genre_w, book_genres.get(bid))

        rows_x.append([cf, content, cluster, pop_norm, genre])
        rows_y.append((rating - 1.0) / 4.0)

    return np.asarray(rows_x, dtype=np.float64), np.asarray(rows_y, dtype=np.float64)
