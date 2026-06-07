#!/usr/bin/env python3
"""Evaluate sentiment classifier on DS4 test split."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.sentiment.evaluate import evaluate_sentiment_model


def main() -> int:
    report = evaluate_sentiment_model()
    print(f"Test accuracy: {report['metrics']['accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
