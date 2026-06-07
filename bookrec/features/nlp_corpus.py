"""NLP feature prep for Amazon reviews (DS4) — fully independent."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bookrec.io_utils import write_json, write_table

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def build_nlp_corpus_features(
    reviews: pd.DataFrame,
    out_dir: Path,
    *,
    max_vocab: int = 15000,
    min_df: int = 2,
) -> dict[str, Any]:
    """Document-term stats and vocabulary for Logistic Regression training."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = reviews.copy()
    df["char_len"] = df["review_text_clean"].str.len()
    df["word_count"] = df["review_text_clean"].str.split().str.len()

    # Vocabulary from training corpus (full clean set before split)
    doc_freq: dict[str, int] = {}
    for text in df["review_text_clean"]:
        tokens = set(_TOKEN_RE.findall(str(text).lower()))
        for t in tokens:
            doc_freq[t] = doc_freq.get(t, 0) + 1
    vocab = sorted(
        [t for t, c in doc_freq.items() if c >= min_df],
        key=lambda t: (-doc_freq[t], t),
    )[:max_vocab]

    paths: dict[str, str] = {}
    paths["reviews_enriched"] = str(write_table(df, out_dir / "reviews_enriched"))
    with (out_dir / "nlp_vocabulary.json").open("w", encoding="utf-8") as f:
        json.dump(vocab, f)
    paths["vocabulary"] = str(out_dir / "nlp_vocabulary.json")

    report = {
        "n_reviews": int(len(df)),
        "n_positive": int((df["sentiment_label"] == 1).sum()),
        "n_negative": int((df["sentiment_label"] == 0).sum()),
        "vocab_size": len(vocab),
        "word_count_median": float(df["word_count"].median()),
        "word_count_p99": float(df["word_count"].quantile(0.99)),
        "independent_of_goodreads": True,
        "paths": paths,
    }
    write_json(report, out_dir / "nlp_corpus_report.json")
    return report
