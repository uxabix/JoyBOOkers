#!/usr/bin/env python3
"""Train Surprise SVD collaborative filtering model (DS1)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.collaborative.train import train_svd


def main() -> int:
    report = train_svd()
    print(f"SVD trained. Validation RMSE: {report['validation_metrics']['rmse']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
