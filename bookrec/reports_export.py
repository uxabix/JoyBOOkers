"""Export pipeline / ML JSON reports (and EDA plots) into git-tracked reports/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from bookrec.paths import (
    EDA_DIR,
    MODEL_CF_DIR,
    MODEL_CLUSTERING_DIR,
    MODEL_CONTENT_DIR,
    MODEL_EVAL_DIR,
    MODEL_SENTIMENT_DIR,
    PROC_ANALYSIS,
    PROC_FEATURES,
    PROC_SPLITS,
    PROJECT_ROOT,
)

REPORTS_DIR = PROJECT_ROOT / "reports"

_EVAL_MODULES = ("collaborative", "content", "sentiment", "clustering")
_TRAIN_MODULE_DIRS = {
    "collaborative": MODEL_CF_DIR,
    "content": MODEL_CONTENT_DIR,
    "sentiment": MODEL_SENTIMENT_DIR,
    "clustering": MODEL_CLUSTERING_DIR,
}

# JSON artifacts to mirror under reports/ (source → dest relative to reports/).
_REPORT_GLOBS: list[tuple[Path, str]] = [
    (PROC_ANALYSIS, "data_pipeline"),
    (PROC_FEATURES, "features"),
    (PROC_SPLITS, "splits"),
    (MODEL_CF_DIR, "ml/collaborative"),
    (MODEL_CONTENT_DIR, "ml/content"),
    (MODEL_SENTIMENT_DIR, "ml/sentiment"),
    (MODEL_CLUSTERING_DIR, "ml/clustering"),
    (MODEL_EVAL_DIR, "ml/evaluation"),
    (MODEL_EVAL_DIR / "hybrid", "ml/evaluation/hybrid"),
    (EDA_DIR, "eda"),
]


def _sanitize_value(value: Any, root: Path) -> Any:
    root_s = str(root)
    root_alt = root_s.replace("\\", "/")

    def _scrub(text: str) -> str:
        out = text
        for prefix in (root_s, root_alt, root_s.lower(), root_alt.lower()):
            if prefix and prefix in out:
                out = out.replace(prefix, "").lstrip("\\/")
                out = out.replace(prefix.replace("\\", "/"), "").lstrip("/")
        return out.replace("\\", "/")

    if isinstance(value, str):
        if value.startswith(root_s) or value.replace("\\", "/").startswith(root_alt):
            return _scrub(value)
        if root_s in value or root_alt in value:
            return _scrub(value)
        return value
    if isinstance(value, dict):
        return {k: _sanitize_value(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(v, root) for v in value]
    return value


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_evaluate_all_summary() -> dict[str, Any]:
    """Merge latest per-module evaluate_report.json into one snapshot."""
    summary: dict[str, Any] = {}
    for name in _EVAL_MODULES:
        path = MODEL_EVAL_DIR / name / "evaluate_report.json"
        payload = _read_json_if_exists(path)
        if payload is not None:
            summary[name] = _sanitize_value(payload, PROJECT_ROOT)
    return summary


def build_train_all_summary() -> dict[str, Any]:
    """Merge latest per-module train_report.json into one snapshot."""
    summary: dict[str, Any] = {}
    for name, model_dir in _TRAIN_MODULE_DIRS.items():
        payload = _read_json_if_exists(model_dir / "train_report.json")
        if payload is not None:
            summary[name] = _sanitize_value(payload, PROJECT_ROOT)
    return summary


def _write_summary_artifacts() -> list[str]:
    """Refresh aggregated JSON snapshots before mirroring to reports/."""
    MODEL_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    evaluate_all = build_evaluate_all_summary()
    if evaluate_all:
        path = MODEL_EVAL_DIR / "evaluate_all.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(evaluate_all, fh, indent=2, ensure_ascii=False)
        written.append(str(path.relative_to(PROJECT_ROOT)))

    train_all = build_train_all_summary()
    if train_all:
        path = MODEL_EVAL_DIR / "train_all.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(train_all, fh, indent=2, ensure_ascii=False)
        written.append(str(path.relative_to(PROJECT_ROOT)))

    return written


def export_reports(*, reports_dir: Path | None = None) -> dict[str, Any]:
    """Copy JSON reports and EDA PNGs into reports/ for version control."""
    dest_root = Path(reports_dir or REPORTS_DIR)
    dest_root.mkdir(parents=True, exist_ok=True)

    aggregated = _write_summary_artifacts()

    copied: list[str] = []
    missing: list[str] = []

    for source_root, rel_prefix in _REPORT_GLOBS:
        if not source_root.is_dir():
            missing.append(str(source_root))
            continue

        patterns = ("*.json", "*.png") if source_root == EDA_DIR else ("*.json",)
        for pattern in patterns:
            for src in sorted(source_root.rglob(pattern)):
                if not src.is_file():
                    continue
                rel = src.relative_to(source_root)
                dest = dest_root / rel_prefix / rel
                dest.parent.mkdir(parents=True, exist_ok=True)

                if src.suffix.lower() == ".json":
                    with src.open(encoding="utf-8") as fh:
                        data = json.load(fh)
                    data = _sanitize_value(data, PROJECT_ROOT)
                    with dest.open("w", encoding="utf-8") as fh:
                        json.dump(data, fh, indent=2, ensure_ascii=False)
                else:
                    shutil.copy2(src, dest)

                copied.append(str(dest.relative_to(dest_root)))

    manifest = {
        "exported_from": "data/processed",
        "n_files": len(copied),
        "files": copied,
        "missing_sources": missing,
        "aggregated_refreshed": aggregated,
    }
    manifest_path = dest_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return manifest
