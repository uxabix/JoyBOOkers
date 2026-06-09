#!/usr/bin/env python3
"""Merge in-app ratings into CF train split and retrain Surprise SVD."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.db.session import configure_engine, get_session_factory, init_db
from app.ml.cf_retrain import run_cf_retrain


def main() -> int:
    settings = get_settings()
    configure_engine(settings.database_url)
    init_db()
    Session = get_session_factory()
    with Session() as session:
        report = run_cf_retrain(settings, session)
    print(
        f"CF retrain OK: {report['app_ratings_exported']} app ratings, "
        f"{report['train_rows']} train rows, RMSE={report['validation_rmse']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
