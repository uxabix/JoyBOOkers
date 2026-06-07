#!/usr/bin/env python3
"""Train TF-IDF + Logistic Regression sentiment model (DS4)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.sentiment.train import train_sentiment_model


def main() -> int:
    report = train_sentiment_model()
    print(f"Sentiment model trained. Val accuracy: {report['validation_metrics']['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
