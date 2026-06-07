"""Unit tests for unified hybrid recommendation logic."""

from __future__ import annotations

from app.ml.user_profile import (
    RatedBook,
    UserProfile,
    blend_signal_weights,
    genre_match_score,
)


def test_blend_weights_cold_start_user() -> None:
    profile = UserProfile(user_id=1, external_id="reg:abc", is_registered=True, rated_books=[])
    weights = blend_signal_weights(profile)
    assert weights["cf"] == 0.0
    assert weights["content"] == 0.0
    assert weights["cluster"] > 0
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_blend_weights_redistributes_cf_when_unavailable() -> None:
    profile = UserProfile(
        user_id=2,
        external_id="42",
        is_registered=False,
        rated_books=[RatedBook(1, "b1", 5.0)] * 12,
        cf_available=False,
    )
    weights = blend_signal_weights(profile)
    assert weights["cf"] == 0.0
    assert weights["content"] > 0.35
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_blend_weights_includes_cf_for_dataset_user() -> None:
    profile = UserProfile(
        user_id=3,
        external_id="42",
        is_registered=False,
        rated_books=[RatedBook(1, "b1", 5.0)] * 12,
        cf_available=True,
    )
    weights = blend_signal_weights(profile)
    assert weights["cf"] > 0


def test_genre_match_score() -> None:
    weights = {"fantasy": 0.7, "romance": 0.3}
    assert genre_match_score(weights, "Fantasy, Adventure") > 0.5
    assert genre_match_score(weights, "History") == 0.0


def test_profile_strength_sparse_vs_rich() -> None:
    sparse = UserProfile(
        user_id=4,
        external_id="reg:x",
        is_registered=True,
        rated_books=[RatedBook(1, "b1", 4.0)],
        profile_strength=0.1,
    )
    rich = UserProfile(
        user_id=5,
        external_id="99",
        is_registered=False,
        rated_books=[RatedBook(i, f"b{i}", 4.0) for i in range(10)],
        profile_strength=1.0,
        cf_available=True,
    )
    assert blend_signal_weights(sparse)["content"] > blend_signal_weights(rich)["pop"]
