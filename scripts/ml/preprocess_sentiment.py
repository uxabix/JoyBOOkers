#!/usr/bin/env python3
"""Validate DS4 NLP splits before sentiment training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.sentiment.preprocess import prepare_sentiment_training_data


def main() -> int:
    report = prepare_sentiment_training_data()
    print(f"Sentiment preprocess OK: {report['train_rows']} train reviews")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
