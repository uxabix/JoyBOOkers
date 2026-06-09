"""Tests for CF retrain merge and scheduler."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from app.config import Settings
from app.ml.cf_retrain import merge_app_ratings_into_cf_train
from app.ml.user_profile import UserProfileBuilder
from app.services.cf_retrain_scheduler import CfRetrainScheduler


def test_merge_app_ratings_overlay_wins_on_duplicate() -> None:
    base = pd.DataFrame(
        {
            "user_id": ["u1", "u1"],
            "book_id": ["b1", "b2"],
            "rating": [3.0, 4.0],
        }
    )
    overlay = pd.DataFrame(
        {
            "user_id": ["reg:abc", "u1"],
            "book_id": ["b9", "b1"],
            "rating": [5.0, 1.0],
        }
    )
    merged = merge_app_ratings_into_cf_train(base, overlay)
    assert len(merged) == 3
    assert float(merged.loc[merged["book_id"] == "b1", "rating"].iloc[0]) == 1.0
    assert set(merged["user_id"]) == {"u1", "reg:abc"}


def test_cf_available_for_registered_user_in_train_set() -> None:
    builder = UserProfileBuilder(
        session=MagicMock(),
        clustering=MagicMock(is_loaded=True, predict_cluster=lambda _: 1, cluster_label=lambda _: "x"),
        settings=Settings(min_cf_ratings_per_user=3),
        cf_known_user_ids={"reg:demo"},
        genre_priors=None,
    )
    builder.users.get = MagicMock(
        return_value=MagicMock(external_id="reg:demo", is_registered=True, cluster_id=1)
    )
    builder.ratings.list_for_user = MagicMock(
        return_value=[
            MagicMock(book_id=1, score=5.0),
            MagicMock(book_id=2, score=4.0),
            MagicMock(book_id=3, score=4.0),
        ]
    )
    builder.books.get = MagicMock(
        side_effect=lambda bid: MagicMock(
            id=bid,
            source_book_id=f"b{bid}",
            genre="Fantasy",
        )
    )

    profile = builder.build(1)
    assert profile is not None
    assert profile.cf_available is True
    assert profile.is_registered is True


def test_scheduler_triggers_retrain_at_threshold(tmp_path: Path) -> None:
    reports: list[dict] = []
    registry = MagicMock()
    registry.reload_cf.return_value = True

    def fake_retrain(settings: Settings, session) -> dict:
        reports.append({"ok": True})
        return {"app_ratings_exported": 10, "validation_rmse": 0.15}

    settings = Settings(
        log_dir=tmp_path,
        cf_retrain_enabled=True,
        cf_retrain_threshold=3,
        cf_retrain_min_interval_seconds=0,
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
    )

    scheduler = CfRetrainScheduler(
        settings,
        registry_getter=lambda: registry,
        retrain_fn=fake_retrain,
    )

    for _ in range(3):
        scheduler.record_app_change()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if reports:
            break
        time.sleep(0.05)

    assert reports, "expected background retrain to run"
    registry.reload_cf.assert_called_once()
    assert scheduler.status()["dirty_count"] == 0


def test_scheduler_disabled_is_noop() -> None:
    scheduler = CfRetrainScheduler(
        Settings(cf_retrain_enabled=False, cf_retrain_threshold=1),
        retrain_fn=lambda *_: (_ for _ in ()).throw(AssertionError("should not run")),
    )
    scheduler.record_app_change()
    assert scheduler.status()["dirty_count"] == 0
