"""Train TF-IDF + Logistic Regression sentiment classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from bookrec.io_utils import write_json
from bookrec.ml.io import save_joblib
from bookrec.ml.metrics import classification_metrics
from bookrec.ml.sentiment.preprocess import load_nlp_splits
from bookrec.paths import MODEL_SENTIMENT_DIR


def train_sentiment_model(
    *,
    splits_dir: Path | None = None,
    out_dir: Path | None = None,
    max_features: int = 20000,
    min_df: int = 2,
    ngram_range: tuple[int, int] = (1, 2),
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit sklearn Pipeline on DS4 train split; evaluate on validation."""
    out = Path(out_dir or MODEL_SENTIMENT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    train_df, val_df, _ = load_nlp_splits(splits_dir)
    text_col = "review_text_clean"
    label_col = "sentiment_label"

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    ngram_range=ngram_range,
                    dtype="float32",
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train = train_df[text_col].fillna("").astype(str)
    y_train = train_df[label_col].astype(int)
    pipeline.fit(X_train, y_train)

    X_val = val_df[text_col].fillna("").astype(str)
    y_val = val_df[label_col].astype(int)
    y_pred = pipeline.predict(X_val)
    val_metrics = classification_metrics(y_val.to_numpy(), y_pred)

    # Human-readable labels for API
    label_names = {0: "negative", 1: "positive"}
    model_path = save_joblib(
        {"pipeline": pipeline, "label_names": label_names, "text_column": text_col},
        out / "sentiment_pipeline.joblib",
    )

    report: dict[str, Any] = {
        "algorithm": "TfidfVectorizer + LogisticRegression",
        "dataset": "ds4_amazon_reviews",
        "hyperparameters": {
            "max_features": max_features,
            "min_df": min_df,
            "ngram_range": list(ngram_range),
            "C": C,
            "max_iter": max_iter,
        },
        "train_rows": int(len(train_df)),
        "validation_metrics": val_metrics,
        "paths": {"model": str(model_path)},
    }
    write_json(report, out / "train_report.json")
    return report
