"""Tests for data-driven cluster profile descriptions."""

from __future__ import annotations

import pandas as pd

from bookrec.ml.clustering.profiles import build_cluster_profiles_detail, describe_cluster_pl
from bookrec.ml.clustering.viz import build_clustering_visualizations


def test_describe_cluster_high_activity_generous() -> None:
    profile = {
        "n_users": 100,
        "share_of_users_pct": 40.0,
        "n_ratings_mean": 150.0,
        "mean_rating": 4.2,
        "std_rating": 0.7,
        "activity_mix": {"low": 0.05, "medium": 0.15, "high": 0.8},
        "rating_distribution_pct": {"1": 0.02, "2": 0.05, "3": 0.15, "4": 0.38, "5": 0.40},
        "generous_share_pct": 78.0,
        "critical_share_pct": 7.0,
    }
    baseline = {"n_ratings_mean": 50.0, "mean_rating": 3.8, "std_rating": 0.85}
    title, desc = describe_cluster_pl(profile, baseline)
    assert "Aktywni" in title or "aktywn" in title.lower()
    assert "4" in desc or "5" in desc
    assert "★" in desc


def test_build_cluster_profiles_detail_includes_distribution() -> None:
    users = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4"],
            "n_ratings": [10, 12, 3, 4],
            "mean_rating": [4.5, 4.2, 3.0, 3.2],
            "std_rating": [0.5, 0.6, 0.8, 0.7],
            "rating_range": [2.0, 3.0, 2.0, 1.0],
            "min_rating": [3.0, 2.0, 2.0, 2.0],
            "max_rating": [5.0, 5.0, 4.0, 3.0],
            "activity_level": ["high", "high", "low", "low"],
        }
    )
    cluster_ids = pd.Series([0, 0, 1, 1])
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3", "u4"],
            "rating": [5, 4, 5, 3, 2],
        }
    )
    detail = build_cluster_profiles_detail(
        user_features=users,
        cluster_ids=cluster_ids,
        interactions=interactions,
    )
    assert "0" in detail["clusters"]
    assert "1" in detail["clusters"]
    assert detail["clusters"]["0"]["rating_distribution_pct"]["5"] > 0
    assert detail["cluster_descriptions"]["0"]["title"]
    assert detail["cluster_descriptions"]["0"]["description"]


def test_clustering_visualizations_pca_and_histogram() -> None:
    import pandas as pd
    import numpy as np

    users = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(12)],
            "n_ratings": [4, 5, 8, 9, 15, 20, 4, 6, 100, 120, 18, 22],
            "mean_rating": [4.0] * 12,
            "std_rating": [0.5] * 12,
            "rating_range": [1.0] * 12,
            "activity_level": ["low"] * 4 + ["medium"] * 4 + ["high"] * 4,
        }
    )
    scaled = pd.DataFrame(
        {
            "user_id": users["user_id"],
            "f1": np.random.randn(12),
            "f2": np.random.randn(12),
        }
    )
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    viz = build_clustering_visualizations(
        scaled_df=scaled,
        feature_cols=["f1", "f2"],
        cluster_ids=labels,
        user_features=users,
        max_pca_points=20,
    )
    assert viz["pca_scatter"]["points_by_cluster"]
    assert len(viz["pca_scatter"]["explained_variance_pct"]) == 2
    assert viz["n_ratings_histogram"]["bin_labels"]
    assert sum(viz["n_ratings_histogram"]["by_cluster"]["0"]) == 3
