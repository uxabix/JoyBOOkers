"""Load DS4 NLP splits for sentiment training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bookrec.io_utils import read_table, write_json
from bookrec.paths import MODEL_SENTIMENT_DIR, PROC_SPLITS


def _resolve_split(path_stem: Path) -> Path:
    for suffix in (".parquet", ".csv"):
        candidate = path_stem.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Split not found: {path_stem}.{{parquet,csv}}")


def load_nlp_splits(
    splits_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = splits_dir or PROC_SPLITS
    train = read_table(_resolve_split(base / "nlp_train"))
    val = read_table(_resolve_split(base / "nlp_val"))
    test = read_table(_resolve_split(base / "nlp_test"))
    return train, val, test


def prepare_sentiment_training_data(
    splits_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    out = Path(out_dir or MODEL_SENTIMENT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = load_nlp_splits(splits_dir)
    report = {
        "dataset": "ds4_amazon_reviews",
        "independent_of_goodreads": True,
        "label_mapping": {"positive": 1, "negative": 0},
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "text_column": "review_text_clean",
        "label_column": "sentiment_label",
    }
    write_json(report, out / "preprocess_report.json")
    return report
