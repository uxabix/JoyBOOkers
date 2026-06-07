"""Evaluate sentiment classifier on DS4 test split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bookrec.io_utils import write_json
from bookrec.ml.io import load_joblib
from bookrec.ml.metrics import classification_metrics
from bookrec.ml.sentiment.preprocess import load_nlp_splits
from bookrec.paths import MODEL_EVAL_DIR, MODEL_SENTIMENT_DIR


def evaluate_sentiment_model(
    *,
    splits_dir: Path | None = None,
    model_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    model_dir = Path(model_dir or MODEL_SENTIMENT_DIR)
    out = Path(out_dir or MODEL_EVAL_DIR / "sentiment")
    out.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "sentiment_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Sentiment model not found at {model_path}")

    bundle = load_joblib(model_path)
    pipeline = bundle["pipeline"]
    text_col = bundle.get("text_column", "review_text_clean")

    _, _, test_df = load_nlp_splits(splits_dir)
    X_test = test_df[text_col].fillna("").astype(str)
    y_test = test_df["sentiment_label"].astype(int).to_numpy()
    y_pred = pipeline.predict(X_test)
    metrics = classification_metrics(y_test, y_pred)

    report: dict[str, Any] = {
        "algorithm": "TfidfVectorizer + LogisticRegression",
        "test_rows": int(len(test_df)),
        "metrics": metrics,
        "model_path": str(model_path),
    }
    write_json(report, out / "evaluate_report.json")
    return report
