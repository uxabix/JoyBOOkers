#!/usr/bin/env python3
"""Build sparse TF-IDF content vectors (DS2 + DS3)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.content.train import train_content_vectors


def main() -> int:
    report = train_content_vectors()
    print(f"Content vectors built for {report['stats']['n_books']} books.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
