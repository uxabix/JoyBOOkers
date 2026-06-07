#!/usr/bin/env python3
"""Run full ML pipeline: preprocess → train → evaluate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
