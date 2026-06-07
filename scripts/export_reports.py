#!/usr/bin/env python3
"""Copy analysis / training JSON reports into git-tracked reports/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bookrec.reports_export import export_reports


def main() -> int:
    manifest = export_reports()
    print(f"Exported {manifest['n_files']} files to reports/")
    if manifest["missing_sources"]:
        print("Missing sources (not yet run):", ", ".join(manifest["missing_sources"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
