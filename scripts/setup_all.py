#!/usr/bin/env python3
"""Run full JoyBookers pipeline: data → ML → hybrid artifacts → reports → SQLite."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Full JoyBookers setup for local demo / defense.")
    parser.add_argument("--books-limit", type=int, default=20_000)
    parser.add_argument("--ratings-limit", type=int, default=50_000)
    parser.add_argument("--skip-data", action="store_true", help="Skip run_data_pipeline (processed data exists).")
    parser.add_argument("--skip-ml", action="store_true", help="Skip ML train/eval (models exist).")
    parser.add_argument("--skip-hybrid-eval", action="store_true", help="Skip evaluate_hybrid_baselines.py.")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_data:
        _run([py, "scripts/run_data_pipeline.py", "--stages", "all"], label="Data pipeline")

    if not args.skip_ml:
        _run([py, "scripts/ml/run_ml_pipeline.py", "--stages", "all"], label="ML pipeline")

    _run([py, "scripts/build_cluster_affinity.py"], label="Cluster affinity")
    _run([py, "scripts/build_genre_priors.py"], label="Genre priors")
    _run([py, "scripts/train_hybrid_weights.py"], label="Hybrid Ridge weights")

    if not args.skip_hybrid_eval:
        _run([py, "scripts/evaluate_hybrid_baselines.py"], label="Hybrid baseline evaluation")

    _run([py, "scripts/export_reports.py"], label="Export reports")

    _run(
        [
            py,
            "scripts/load_db.py",
            "--books-limit",
            str(args.books_limit),
            "--ratings-limit",
            str(args.ratings_limit),
        ],
        label="Load SQLite",
    )

    print("\nSetup complete. Start the app with:")
    print("  uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
