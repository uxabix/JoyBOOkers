"""Train Surprise SVD collaborative filtering model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from surprise import SVD
from surprise.model_selection import train_test_split

from bookrec.io_utils import write_json
from bookrec.ml.collaborative.preprocess import load_cf_splits, to_surprise_dataset
from bookrec.ml.io import save_pickle
from bookrec.ml.metrics import regression_metrics
from bookrec.paths import MODEL_CF_DIR


def train_svd(
    *,
    splits_dir: Path | None = None,
    out_dir: Path | None = None,
    n_factors: int = 100,
    n_epochs: int = 20,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
    random_state: int = 42,
    internal_val_ratio: float = 0.1,
) -> dict[str, Any]:
    """Train SVD on DS1 CF train split; persist model and metadata."""
    out = Path(out_dir or MODEL_CF_DIR)
    out.mkdir(parents=True, exist_ok=True)

    train_df, _ = load_cf_splits(splits_dir)
    dataset, reader = to_surprise_dataset(train_df)
    trainset, valset = train_test_split(dataset, test_size=internal_val_ratio, random_state=random_state)

    model = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
    )
    model.fit(trainset)

    val_preds = model.test(valset)
    y_true = [p.r_ui for p in val_preds]
    y_pred = [p.est for p in val_preds]
    val_metrics = regression_metrics(y_true, y_pred)

    model_path = save_pickle(model, out / "svd_model.pkl")
    reader_path = save_pickle(reader, out / "surprise_reader.pkl")

    report: dict[str, Any] = {
        "algorithm": "Surprise.SVD",
        "dataset": "ds1_goodreads_2m",
        "hyperparameters": {
            "n_factors": n_factors,
            "n_epochs": n_epochs,
            "lr_all": lr_all,
            "reg_all": reg_all,
            "random_state": random_state,
        },
        "train_interactions": int(trainset.n_ratings),
        "validation_metrics": val_metrics,
        "paths": {
            "model": str(model_path),
            "reader": str(reader_path),
        },
    }
    write_json(report, out / "train_report.json")
    return report
