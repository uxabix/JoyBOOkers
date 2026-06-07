#!/usr/bin/env python3
"""Evaluate content TF-IDF retrieval quality."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.content.evaluate import evaluate_content_vectors


def main() -> int:
    report = evaluate_content_vectors()
    print(f"Genre overlap@{report['top_k']}: {report['mean_genre_overlap_at_k']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
