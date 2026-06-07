#!/usr/bin/env python3
"""Validate DS1 CF splits before SVD training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.collaborative.preprocess import prepare_cf_training_data


def main() -> int:
    report = prepare_cf_training_data()
    print(f"CF preprocess OK: {report['train_rows']} train rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
