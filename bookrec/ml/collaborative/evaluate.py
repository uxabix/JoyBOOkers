"""Evaluate Surprise SVD on held-out DS1 interactions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from surprise import accuracy

from bookrec.io_utils import write_json
from bookrec.ml.collaborative.preprocess import load_cf_splits
from bookrec.ml.io import load_pickle
from bookrec.ml.metrics import precision_recall_at_k, regression_metrics
from bookrec.paths import MODEL_CF_DIR, MODEL_EVAL_DIR


def evaluate_svd(
    *,
    splits_dir: Path | None = None,
    model_dir: Path | None = None,
    out_dir: Path | None = None,
    top_k: int = 10,
    max_users_for_ranking: int = 500,
) -> dict[str, Any]:
    """RMSE/MAE on test set + optional Precision@K / Recall@K."""
    model_dir = Path(model_dir or MODEL_CF_DIR)
    out = Path(out_dir or MODEL_EVAL_DIR / "collaborative")
    out.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "svd_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"SVD model not found at {model_path}. Run collaborative train first.")

    model = load_pickle(model_path)
    train_df, test_df = load_cf_splits(splits_dir)

    if test_df.empty:
        report = {"error": "empty_test_split", "hint": "Re-run splits stage with more DS1 ratings"}
        write_json(report, out / "evaluate_report.json")
        return report

    testset = [
        (str(row.user_id), str(row.book_id), float(row.rating))
        for row in test_df.itertuples(index=False)
    ]
    predictions = model.test(testset)
    rmse = float(accuracy.rmse(predictions, verbose=False))
    mae = float(accuracy.mae(predictions, verbose=False))
    reg = regression_metrics(
        [p.r_ui for p in predictions],
        [p.est for p in predictions],
    )

    # Ranking metrics: recommend unseen books per user from train catalog
    train_user_books: dict[str, set[str]] = defaultdict(set)
    all_books: set[str] = set()
    for row in train_df.itertuples(index=False):
        uid, bid = str(row.user_id), str(row.book_id)
        train_user_books[uid].add(bid)
        all_books.add(bid)

    test_ground_truth: dict[str, set[str]] = defaultdict(set)
    for row in test_df.itertuples(index=False):
        uid, bid = str(row.user_id), str(row.book_id)
        if bid not in train_user_books[uid]:
            test_ground_truth[uid].add(bid)

    recommendations: dict[str, list[str]] = {}
    candidate_books = sorted(all_books)
    for i, user_id in enumerate(sorted(test_ground_truth.keys())):
        if i >= max_users_for_ranking:
            break
        seen = train_user_books.get(user_id, set())
        candidates = [b for b in candidate_books if b not in seen]
        scored = [(bid, model.predict(user_id, bid).est) for bid in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        recommendations[user_id] = [bid for bid, _ in scored[:top_k]]

    ranking = precision_recall_at_k(test_ground_truth, recommendations, k=top_k)

    report: dict[str, Any] = {
        "algorithm": "Surprise.SVD",
        "test_rows": int(len(test_df)),
        "rmse": rmse,
        "mae": mae,
        "regression": reg,
        "ranking": ranking,
        "model_path": str(model_path),
    }
    write_json(report, out / "evaluate_report.json")
    return report
