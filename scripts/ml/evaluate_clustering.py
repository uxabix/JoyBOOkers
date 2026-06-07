#!/usr/bin/env python3
"""Evaluate user K-Means clustering."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.clustering.evaluate import evaluate_user_clusters


def main() -> int:
    report = evaluate_user_clusters()
    print(f"Silhouette: {report['silhouette']} | Inertia: {report['inertia']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
