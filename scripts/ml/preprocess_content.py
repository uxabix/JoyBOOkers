#!/usr/bin/env python3
"""Validate DS2+DS3 content catalog before TF-IDF training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.content.preprocess import prepare_content_training_data


def main() -> int:
    report = prepare_content_training_data()
    print(f"Content preprocess OK: {report['catalog_rows']} catalog rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
