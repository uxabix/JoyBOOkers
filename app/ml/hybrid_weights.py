"""Learned hybrid signal weights — Ridge regression on interaction history."""

from __future__ import annotations

import json
from pathlib import Path

from app.logging_config import get_logger
from app.ml.signals import FEATURE_ORDER
from bookrec.ml.io import load_joblib, save_joblib

logger = get_logger(__name__)


class HybridWeightModel:
    """Ridge model: signal features → normalized rating target."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._model = None
        self._metrics: dict[str, float] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> bool:
        if self.path is None or not self.path.is_file():
            logger.warning("Hybrid weight model not found: %s", self.path)
            self._loaded = True
            return False
        bundle = load_joblib(self.path)
        if isinstance(bundle, dict) and "model" in bundle:
            self._model = bundle["model"]
            self._metrics = dict(bundle.get("metrics") or {})
        else:
            self._model = bundle
        self._loaded = True
        logger.info("Loaded hybrid weight model from %s", self.path)
        return True

    def score(self, raw_features: dict[str, float]) -> float:
        """Predict normalized rating from raw signals (same scale as training)."""
        if self._model is None:
            raise RuntimeError("Hybrid weight model is not loaded")
        row = [[float(raw_features.get(k, 0.0)) for k in FEATURE_ORDER]]
        pred = float(self._model.predict(row)[0])
        return max(0.0, min(1.0, pred))

    def coefficients(self) -> dict[str, float]:
        if self._model is None:
            return {}
        coefs = list(self._model.coef_.ravel())
        raw = {k: float(c) for k, c in zip(FEATURE_ORDER, coefs, strict=False)}
        total = sum(abs(v) for v in raw.values()) or 1.0
        return {k: abs(v) / total for k, v in raw.items()}

    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    @staticmethod
    def save_bundle(
        path: Path,
        *,
        model,
        metrics: dict[str, float],
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": model,
            "feature_order": FEATURE_ORDER,
            "metrics": metrics,
            "coefficients": {
                k: float(c)
                for k, c in zip(FEATURE_ORDER, model.coef_.ravel(), strict=False)
            },
        }
        save_joblib(bundle, path)
        report_path = path.with_suffix(".json")
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "algorithm": "Ridge",
                    "feature_order": FEATURE_ORDER,
                    "metrics": metrics,
                    "coefficients": bundle["coefficients"],
                    "intercept": float(model.intercept_),
                },
                fh,
                indent=2,
            )
        return path
