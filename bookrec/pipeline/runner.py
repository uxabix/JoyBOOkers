"""Run analyze → preprocess → features → splits.

University scope (see bookrec/constraints.py):
  DS1 → collaborative filtering + user clustering only
  DS2+DS3 → content-based recommendation
  DS4 → independent NLP / sentiment
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from bookrec.constraints import schema_roles_summary
from bookrec.features.clustering import build_user_clustering_features
from bookrec.features.content import build_content_features
from bookrec.features.interactions import build_interaction_features
from bookrec.features.nlp_corpus import build_nlp_corpus_features
from bookrec.ingest import (
    load_and_analyze_ds1,
    load_and_analyze_ds2,
    load_and_analyze_ds3,
    load_and_analyze_ds4,
    preprocess_ds1,
    preprocess_ds2,
    preprocess_ds3,
    preprocess_ds4,
)
from bookrec.io_utils import read_table, write_json
from bookrec.paths import (
    PROC_ANALYSIS,
    PROC_DS1,
    PROC_DS2,
    PROC_DS3,
    PROC_DS4,
    PROC_FEATURES,
    PROC_SPLITS,
    RAW_DS2,
    RAW_DS3,
    RAW_DS4,
    ds1_raw_dir,
)
from bookrec.schemas import schema_summary
from bookrec.reports_export import export_reports
from bookrec.splits import save_all_splits


def stage_analyze(
    *,
    raw_ds1: Path | None = None,
    raw_ds2: Path | None = None,
    raw_ds3: Path | None = None,
    raw_ds4: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Analyze all datasets; write JSON reports."""
    out = out_dir or PROC_ANALYSIS
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "project_scope": schema_roles_summary(),
        "schemas": schema_summary(),
        "datasets": {},
    }

    analyzers = [
        ("ds1", lambda: load_and_analyze_ds1(raw_ds1)),
        ("ds2", lambda: load_and_analyze_ds2(raw_ds2 or RAW_DS2)),
        ("ds3", lambda: load_and_analyze_ds3(raw_ds3 or RAW_DS3)),
        ("ds4", lambda: load_and_analyze_ds4(raw_ds4 or RAW_DS4)),
    ]
    for key, fn in analyzers:
        print(f"Analyzing {key}...", flush=True)
        try:
            report["datasets"][key] = fn()
        except Exception as exc:
            report["datasets"][key] = {"error": str(exc)}

    write_json(report, out / "analyze_all.json")
    return report


def stage_preprocess(
    *,
    raw_ds1: Path | None = None,
    skip_ds2: bool = False,
    skip_ds3: bool = False,
    skip_ds4: bool = False,
    ds4_sample: int | None = None,
    fuzzy_threshold: int = 88,
) -> dict[str, Any]:
    """Clean all available datasets."""
    results: dict[str, Any] = {}

    print("Preprocessing DS1 (CF + clustering)...", flush=True)
    results["ds1"] = preprocess_ds1(
        raw_dir=raw_ds1 or ds1_raw_dir(),
        out_dir=PROC_DS1,
        fuzzy_threshold=fuzzy_threshold,
    )

    if not skip_ds2 and RAW_DS2.is_dir() and any(RAW_DS2.glob("*.csv")):
        print("Preprocessing DS2 (content-based primary)...", flush=True)
        try:
            results["ds2"] = preprocess_ds2(out_dir=PROC_DS2)
        except FileNotFoundError as exc:
            results["ds2"] = {"skipped": str(exc)}
    else:
        results["ds2"] = {"skipped": "no raw files"}

    if not skip_ds3 and RAW_DS3.is_dir():
        print("Preprocessing DS3 (content enrichment)...", flush=True)
        try:
            results["ds3"] = preprocess_ds3(out_dir=PROC_DS3)
        except FileNotFoundError as exc:
            results["ds3"] = {"skipped": str(exc)}
    else:
        results["ds3"] = {"skipped": "no raw files"}

    if not skip_ds4 and RAW_DS4.is_dir():
        print("Preprocessing DS4 (NLP, independent)...", flush=True)
        try:
            results["ds4"] = preprocess_ds4(out_dir=PROC_DS4, sample_n=ds4_sample)
        except FileNotFoundError as exc:
            results["ds4"] = {"skipped": str(exc)}
    else:
        results["ds4"] = {"skipped": "no raw files"}

    write_json({k: v.get("report", v) for k, v in results.items()}, PROC_ANALYSIS / "preprocess_summary.json")
    return results


def stage_features(preprocess_results: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build module-specific features — no cross-dataset mega-merge."""
    report: dict[str, Any] = {"project_scope": schema_roles_summary()}

    inter_path = PROC_DS1 / "interactions_clean.parquet"
    if not inter_path.exists():
        inter_path = PROC_DS1 / "interactions_clean.csv"
    books_path = PROC_DS1 / "books_clean.parquet"
    if not books_path.exists():
        books_path = PROC_DS1 / "books_clean.csv"

    if inter_path.exists():
        interactions = read_table(inter_path)
        # CF: user-item matrix stats (DS1)
        if books_path.exists():
            books = read_table(books_path)
            report["interactions"] = build_interaction_features(
                interactions, books, PROC_FEATURES / "interactions"
            )
        # Clustering: users only (DS1)
        report["clustering"] = build_user_clustering_features(
            interactions, PROC_FEATURES / "clustering"
        )

    ds2 = ds3 = None
    p2 = PROC_DS2 / "books_clean.parquet"
    if not p2.exists():
        p2 = PROC_DS2 / "books_clean.csv"
    if p2.exists():
        ds2 = read_table(p2)
    p3 = PROC_DS3 / "books_clean.parquet"
    if not p3.exists():
        p3 = PROC_DS3 / "books_clean.csv"
    if p3.exists():
        ds3 = read_table(p3)

    # Content-based: DS2 + DS3 only (~100k–150k rows, sparse matrices)
    report["content"] = build_content_features(ds2, ds3, PROC_FEATURES / "content")

    rev_path = PROC_DS4 / "reviews_clean.parquet"
    if not rev_path.exists():
        rev_path = PROC_DS4 / "reviews_clean.csv"
    if rev_path.exists():
        reviews = read_table(rev_path)
        report["nlp"] = build_nlp_corpus_features(reviews, PROC_FEATURES / "nlp")

    write_json(report, PROC_FEATURES / "features_summary.json")
    return report


def stage_splits() -> dict[str, Any]:
    interactions = reviews = None
    ip = PROC_DS1 / "interactions_clean.parquet"
    if not ip.exists():
        ip = PROC_DS1 / "interactions_clean.csv"
    if ip.exists():
        interactions = read_table(ip)
    rp = PROC_DS4 / "reviews_clean.parquet"
    if not rp.exists():
        rp = PROC_DS4 / "reviews_clean.csv"
    if rp.exists():
        reviews = read_table(rp)
    return save_all_splits(interactions, reviews, PROC_SPLITS)


def run_pipeline(
    stages: list[str] | None = None,
    *,
    raw_ds1: Path | None = None,
    skip_ds2: bool = False,
    skip_ds3: bool = False,
    skip_ds4: bool = False,
    ds4_sample: int | None = None,
    fuzzy_threshold: int = 88,
) -> dict[str, Any]:
    all_stages = ["analyze", "preprocess", "features", "splits"]
    stages = stages or all_stages
    summary: dict[str, Any] = {"stages_run": stages, "project_scope": schema_roles_summary()}

    if "analyze" in stages:
        summary["analyze"] = stage_analyze(raw_ds1=raw_ds1)

    pre = None
    if "preprocess" in stages:
        pre = stage_preprocess(
            raw_ds1=raw_ds1,
            skip_ds2=skip_ds2,
            skip_ds3=skip_ds3,
            skip_ds4=skip_ds4,
            ds4_sample=ds4_sample,
            fuzzy_threshold=fuzzy_threshold,
        )
        summary["preprocess"] = {k: v.get("report", v) for k, v in pre.items()}

    # Legacy alias: "resolve" is a no-op (DS1↔DS2 linking removed)
    if "resolve" in stages:
        summary["resolve"] = {
            "skipped": True,
            "reason": "University scope: no global Goodreads merge. DS3 enriches DS2 inside content features.",
        }

    if "features" in stages:
        summary["features"] = stage_features(pre)

    if "splits" in stages:
        summary["splits"] = stage_splits()

    write_json(summary, PROC_ANALYSIS / "pipeline_summary.json")
    export_reports()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="JoyBookers data pipeline (university scope: DS1=CF, DS2+DS3=content, DS4=NLP)"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["analyze", "preprocess", "resolve", "features", "splits", "all"],
        default=["all"],
        help="'resolve' is deprecated (no-op); DS3→DS2 merge happens in features/content",
    )
    parser.add_argument("--raw-ds1", type=Path, default=None)
    parser.add_argument("--skip-ds2", action="store_true")
    parser.add_argument("--skip-ds3", action="store_true")
    parser.add_argument("--skip-ds4", action="store_true")
    parser.add_argument("--ds4-sample", type=int, default=None, help="Subsample Amazon reviews")
    parser.add_argument("--title-fuzzy-threshold", type=int, default=88)
    args = parser.parse_args(argv)

    if "all" in args.stages:
        stages = ["analyze", "preprocess", "features", "splits"]
    else:
        stages = args.stages

    run_pipeline(
        stages,
        raw_ds1=args.raw_ds1,
        skip_ds2=args.skip_ds2,
        skip_ds3=args.skip_ds3,
        skip_ds4=args.skip_ds4,
        ds4_sample=args.ds4_sample,
        fuzzy_threshold=args.title_fuzzy_threshold,
    )
    print("Pipeline complete. See data/processed/analysis/pipeline_summary.json")
    return 0
