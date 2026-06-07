"""Base helpers for dataset ingestors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bookrec.cleaning import normalize_column_names
from bookrec.io_utils import discover_files, read_table, write_json, write_table


def load_csv_shards(paths: list[Path], **read_kw: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not paths:
        raise FileNotFoundError("No input files.")
    frames: list[pd.DataFrame] = []
    rows: dict[str, int] = {}
    for i, path in enumerate(paths, start=1):
        print(f"  [{i}/{len(paths)}] {path.name} ...", flush=True)
        part = read_table(path, **read_kw)
        rows[path.name] = len(part)
        frames.append(part)
    combined = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return combined, {
        "n_files": len(paths),
        "files": [p.name for p in paths],
        "rows_per_file": rows,
        "rows_total": int(len(combined)),
    }


def load_json_records(paths: list[Path], lines: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load JSON Lines or JSON array files into a DataFrame."""
    if not paths:
        raise FileNotFoundError("No JSON files.")
    records: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"files": [], "rows_per_file": {}}
    for path in paths:
        meta["files"].append(path.name)
        n_before = len(records)
        if path.suffix.lower() == ".gz":
            import gzip

            opener = gzip.open(path, "rt", encoding="utf-8")
        else:
            opener = path.open("r", encoding="utf-8")
        with opener as f:
            if lines:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
                else:
                    records.append(data)
        meta["rows_per_file"][path.name] = len(records) - n_before
    df = pd.DataFrame(records)
    meta["rows_total"] = int(len(df))
    meta["n_files"] = len(paths)
    return df, meta


def standardize_columns(df: pd.DataFrame, renames: dict[str, str]) -> pd.DataFrame:
    df = normalize_column_names(df)
    mapping = {k: v for k, v in renames.items() if k in df.columns and v not in df.columns}
    return df.rename(columns=mapping) if mapping else df


def fill_missing_strings(df: pd.DataFrame, cols: list[str], fill: str = "") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].fillna(fill).astype(str).str.strip()
            out.loc[out[c] == "", c] = fill
    return out


def save_stage_output(
    df: pd.DataFrame,
    out_dir: Path,
    basename: str,
    report: dict[str, Any],
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = write_table(df, out_dir / basename)
    report_path = out_dir / f"{basename}_report.json"
    write_json(report, report_path)
    return {"table": str(table_path), "report": str(report_path)}
