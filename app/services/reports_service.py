"""Load pipeline / ML JSON reports for analytics and clustering dashboards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)


class ReportsService:
    """Read version-controlled reports/ artifacts for the web UI."""

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = Path(reports_dir)

    @property
    def is_available(self) -> bool:
        return self.reports_dir.is_dir()

    def _read_json(self, relative: str) -> dict[str, Any] | None:
        path = self.reports_dir / relative
        if not path.is_file():
            logger.debug("Report not found: %s", path)
            return None
        try:
            with path.open(encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return None

    def list_eda_images(self) -> list[str]:
        eda = self.reports_dir / "eda"
        if not eda.is_dir():
            return []
        return sorted(p.name for p in eda.glob("*.png"))

    def get_analytics_context(self) -> dict[str, Any]:
        evaluate_all = self._read_json("ml/evaluation/evaluate_all.json") or {}
        features = self._read_json("features/features_summary.json") or {}
        preprocess = self._read_json("data_pipeline/preprocess_summary.json") or {}
        manifest = self._read_json("manifest.json") or {}

        cf = evaluate_all.get("collaborative", {})
        content = evaluate_all.get("content", {})
        sentiment = evaluate_all.get("sentiment", {})
        clustering = evaluate_all.get("clustering", {})

        interactions = features.get("interactions", {})
        content_feat = features.get("content", {})
        nlp = features.get("nlp", {})

        ds1 = preprocess.get("ds1", {}).get("interactions_cleaning", {})
        ds3 = preprocess.get("ds3", {})

        chart_data = {
            "ml_comparison": {
                "labels": ["CF RMSE", "Content overlap %", "Sentiment acc. %", "Silhouette %"],
                "values": [
                    float(cf.get("rmse", 0) or 0),
                    float(content.get("mean_genre_overlap_at_k", 0) or 0) * 100,
                    float((sentiment.get("metrics") or {}).get("accuracy", 0) or 0) * 100,
                    float(clustering.get("silhouette", 0) or 0) * 100,
                ],
            },
            "dataset_volumes": {
                "labels": ["DS1 ratings", "DS1 users", "Content books", "NLP reviews"],
                "values": [
                    int(interactions.get("n_interactions", ds1.get("interactions_rows_clean", 0)) or 0),
                    int(interactions.get("n_users", ds1.get("n_users", 0)) or 0),
                    int(content_feat.get("n_books", content.get("n_books", 0)) or 0),
                    int(nlp.get("n_reviews", 0) or 0),
                ],
            },
        }

        return {
            "reports_available": self.is_available,
            "evaluate_all": evaluate_all,
            "features": features,
            "preprocess": preprocess,
            "manifest": manifest,
            "eda_images": self.list_eda_images(),
            "metrics": {
                "cf_rmse": cf.get("rmse"),
                "cf_mae": cf.get("mae"),
                "content_books": content.get("n_books"),
                "content_overlap": content.get("mean_genre_overlap_at_k"),
                "content_cosine": content.get("mean_neighbor_cosine"),
                "sentiment_accuracy": (sentiment.get("metrics") or {}).get("accuracy"),
                "sentiment_f1": (sentiment.get("metrics") or {}).get("f1_macro"),
                "cluster_silhouette": clustering.get("silhouette"),
                "cluster_k": clustering.get("n_clusters"),
                "interactions": interactions.get("n_interactions"),
                "catalog_books": content_feat.get("n_books"),
                "nlp_reviews": nlp.get("n_reviews"),
                "ds3_rows": ds3.get("rows_clean"),
            },
            "chart_data": chart_data,
        }

    def get_clustering_context(self) -> dict[str, Any]:
        evaluate_all = self._read_json("ml/evaluation/evaluate_all.json") or {}
        train = self._read_json("ml/clustering/train_report.json") or {}
        features = self._read_json("features/clustering/clustering_features_report.json") or {}
        clustering = evaluate_all.get("clustering", {})

        profiles = clustering.get("cluster_profiles_mean", {})
        sizes = train.get("cluster_sizes", {})
        silhouette_by_k = train.get("silhouette_by_k", {})

        chart_data = {
            "cluster_sizes": {
                "labels": [f"Cluster {k}" for k in sorted(sizes, key=lambda x: int(x))],
                "values": [int(sizes[k]) for k in sorted(sizes, key=lambda x: int(x))],
            },
            "silhouette_by_k": {
                "labels": [str(k) for k in sorted(silhouette_by_k, key=lambda x: int(x))],
                "values": [float(silhouette_by_k[k]) for k in sorted(silhouette_by_k, key=lambda x: int(x))],
            },
            "profile_ratings": {
                "labels": [f"Cluster {k}" for k in sorted((profiles.get("n_ratings") or {}), key=lambda x: int(x))],
                "mean_ratings": [
                    float((profiles.get("mean_rating") or {}).get(k, 0))
                    for k in sorted((profiles.get("n_ratings") or {}), key=lambda x: int(x))
                ],
                "n_ratings": [
                    float((profiles.get("n_ratings") or {}).get(k, 0))
                    for k in sorted((profiles.get("n_ratings") or {}), key=lambda x: int(x))
                ],
            },
        }

        return {
            "reports_available": self.is_available,
            "clustering": clustering,
            "train": train,
            "features": features,
            "chart_data": chart_data,
        }
