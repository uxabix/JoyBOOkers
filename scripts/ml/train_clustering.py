#!/usr/bin/env python3
"""Train K-Means user behaviour clusters (DS1)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.clustering.train import train_user_clusters


def main() -> int:
    report = train_user_clusters()
    print(f"K-Means trained with k={report['n_clusters']}, silhouette={report['silhouette']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
