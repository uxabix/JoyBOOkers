#!/usr/bin/env python3
"""Validate DS1 user behaviour features before K-Means."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.clustering.preprocess import prepare_clustering_training_data


def main() -> int:
    report = prepare_clustering_training_data()
    print(f"Clustering preprocess OK: {report['n_users']} users")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
