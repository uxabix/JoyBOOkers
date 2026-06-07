"""Smoke test DS1 preprocess on stage1 fixtures."""

from pathlib import Path

from bookrec.ingest.ds1_goodreads_2m import load_and_analyze_ds1, preprocess_ds1
from bookrec.features.interactions import build_interaction_features

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "stage1"


def test_analyze_ds1_fixtures():
    report = load_and_analyze_ds1(FIXTURES)
    assert "books" in report
    assert report["books"]["profile"]["n_rows"] >= 3


def test_preprocess_and_features(tmp_path):
    result = preprocess_ds1(raw_dir=FIXTURES, out_dir=tmp_path, fuzzy_threshold=80)
    assert result["interactions"].shape[0] >= 3
    feat = build_interaction_features(result["interactions"], result["books"], tmp_path / "features")
    assert feat["n_users"] >= 2
