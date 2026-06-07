"""Integration tests for ML training pipeline (synthetic / fixture data)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bookrec.features.clustering import build_user_clustering_features
from bookrec.features.content import build_content_features
from bookrec.ingest.ds1_goodreads_2m import preprocess_ds1
from bookrec.io_utils import write_table
from bookrec.ml.clustering.evaluate import evaluate_user_clusters
from bookrec.ml.clustering.train import train_user_clusters
from bookrec.ml.collaborative.evaluate import evaluate_svd
from bookrec.ml.collaborative.train import train_svd
from bookrec.ml.content.evaluate import evaluate_content_vectors
from bookrec.ml.content.train import train_content_vectors
from bookrec.ml.sentiment.evaluate import evaluate_sentiment_model
from bookrec.ml.sentiment.train import train_sentiment_model
from bookrec.splits import save_all_splits
from bookrec.text_normalization import add_match_keys

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "stage1"


@pytest.fixture
def ml_workspace(tmp_path: Path) -> dict[str, Path]:
    """Minimal processed features + splits for all four ML modules."""
    proc = tmp_path / "processed"
    features = proc / "features"
    splits = proc / "splits"
    models = proc / "models"

    ds1 = preprocess_ds1(raw_dir=FIXTURES, out_dir=proc / "ds1", fuzzy_threshold=80)
    interactions = ds1["interactions"]
    build_user_clustering_features(interactions, features / "clustering", min_ratings=1)

    ds2 = add_match_keys(
        pd.DataFrame(
            {
                "source_book_id": ["1", "2", "3"],
                "title": ["Dune", "Foundation", "Neuromancer"],
                "authors": ["Frank Herbert", "Isaac Asimov", "William Gibson"],
                "description": ["Desert planet", "Galactic empire", "Cyberpunk"],
                "genres_list": [
                    ["Science Fiction", "Adventure"],
                    ["Science Fiction"],
                    ["Science Fiction", "Cyberpunk"],
                ],
                "tags": [[], [], ["ai", "hacker"]],
            }
        )
    )
    build_content_features(ds2, None, features / "content", max_text_features=50)

    reviews = pd.DataFrame(
        {
            "review_text_clean": [
                "Absolutely loved this book, brilliant and wonderful storytelling.",
                "Terrible pacing and awful characters, hated every page.",
                "Great world building and excellent prose throughout.",
                "Boring and bad, would not recommend to anyone.",
                "Amazing twist ending, loved the protagonist journey.",
                "Worst book I have read, horrible and disappointing.",
            ]
            * 3,
            "sentiment_label": [1, 0, 1, 0, 1, 0] * 3,
            "asin": ["B001"] * 18,
        }
    )
    save_all_splits(interactions, reviews, splits, random_state=42)

    return {
        "proc": proc,
        "features": features,
        "splits": splits,
        "models": models,
        "cf": models / "collaborative",
        "content": models / "content",
        "sentiment": models / "sentiment",
        "clustering": models / "clustering",
        "eval": models / "evaluation",
    }


def test_train_and_evaluate_collaborative(ml_workspace: dict[str, Path]) -> None:
    train_report = train_svd(
        splits_dir=ml_workspace["splits"],
        out_dir=ml_workspace["cf"],
        n_factors=5,
        n_epochs=5,
    )
    assert train_report["validation_metrics"]["rmse"] >= 0
    eval_report = evaluate_svd(
        splits_dir=ml_workspace["splits"],
        model_dir=ml_workspace["cf"],
        out_dir=ml_workspace["eval"] / "collaborative",
        max_users_for_ranking=10,
    )
    assert "rmse" in eval_report


def test_train_and_evaluate_content(ml_workspace: dict[str, Path]) -> None:
    train_report = train_content_vectors(
        features_dir=ml_workspace["features"] / "content",
        out_dir=ml_workspace["content"],
        max_content_features=50,
        max_tag_features=20,
    )
    assert train_report["stats"]["n_books"] == 3
    eval_report = evaluate_content_vectors(
        model_dir=ml_workspace["content"],
        features_dir=ml_workspace["features"] / "content",
        out_dir=ml_workspace["eval"] / "content",
        sample_size=3,
        top_k=2,
    )
    assert eval_report["n_books"] == 3


def test_train_and_evaluate_sentiment(ml_workspace: dict[str, Path]) -> None:
    train_report = train_sentiment_model(
        splits_dir=ml_workspace["splits"],
        out_dir=ml_workspace["sentiment"],
        max_features=500,
    )
    assert train_report["validation_metrics"]["accuracy"] >= 0
    eval_report = evaluate_sentiment_model(
        splits_dir=ml_workspace["splits"],
        model_dir=ml_workspace["sentiment"],
        out_dir=ml_workspace["eval"] / "sentiment",
    )
    assert eval_report["metrics"]["accuracy"] >= 0


def test_train_and_evaluate_clustering(ml_workspace: dict[str, Path]) -> None:
    train_report = train_user_clusters(
        features_dir=ml_workspace["features"] / "clustering",
        out_dir=ml_workspace["clustering"],
        n_clusters=2,
    )
    assert train_report["n_clusters"] == 2
    eval_report = evaluate_user_clusters(
        features_dir=ml_workspace["features"] / "clustering",
        model_dir=ml_workspace["clustering"],
        out_dir=ml_workspace["eval"] / "clustering",
    )
    assert eval_report["n_users"] >= 1
