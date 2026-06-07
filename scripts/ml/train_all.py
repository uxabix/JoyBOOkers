#!/usr/bin/env python3
"""Train all ML models."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.ml.runner import train_all
from bookrec.reports_export import export_reports
from bookrec.io_utils import write_json
from bookrec.paths import MODEL_EVAL_DIR


def main() -> int:
    report = train_all()
    MODEL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    write_json(report, MODEL_EVAL_DIR / "train_all.json")
    export_reports()
    print("Training complete. Reports exported to reports/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
