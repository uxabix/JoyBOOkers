#!/usr/bin/env python3
"""Evaluate Surprise SVD on DS1 test split."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.collaborative.evaluate import evaluate_svd


def main() -> int:
    report = evaluate_svd()
    if "error" in report:
        print(report)
        return 1
    print(f"Test RMSE: {report['rmse']:.4f} | MAE: {report['mae']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
