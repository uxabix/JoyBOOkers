"""Load pipeline / ML JSON reports for analytics and clustering dashboards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.logging_config import get_logger
from bookrec.constraints import DEFAULT_MAX_TEXT_FEATURES, MAX_CONTENT_BOOKS

logger = get_logger(__name__)


def _build_outlier_summary(
    preprocess: dict[str, Any],
    analyze_all: dict[str, Any],
) -> dict[str, Any]:
    """Interaction + catalog outlier counts for the analytics dashboard."""
    interaction_outliers = preprocess.get("ds1", {}).get("interactions_cleaning", {}).get("outliers", {})
    book_outliers = (
        (analyze_all.get("datasets") or {}).get("ds1", {}).get("books", {}).get("numeric_outliers", {})
    )
    return {
        "interactions": interaction_outliers,
        "catalog_numeric": book_outliers,
    }


def _build_feature_selection_rows(
    features: dict[str, Any],
    train_all: dict[str, Any],
) -> list[dict[str, Any]]:
    """Human-readable feature-selection summary for rubric defense."""
    content_train = train_all.get("content", {}) or {}
    content_stats = content_train.get("stats", {}) or {}
    sentiment_train = train_all.get("sentiment", {}) or {}
    sentiment_hp = sentiment_train.get("hyperparameters", {}) or {}
    clustering_feat = features.get("clustering", {}) or {}
    content_feat = features.get("content", {}) or {}

    rows: list[dict[str, Any]] = [
        {
            "module": "Treść (DS2+DS3)",
            "selected": (
                f"autorzy {content_stats.get('author_features', 3000)}, "
                f"gatunki {content_stats.get('genre_features', 500)}, "
                f"tekst TF-IDF {content_stats.get('content_features', DEFAULT_MAX_TEXT_FEATURES)}"
            ),
            "limit": f"maks. {MAX_CONTENT_BOOKS:,} książek; katalog DS1 wykluczony",
            "note": "Cosinus na rzadkich wierszach TF-IDF znormalizowanych L2",
        },
        {
            "module": "Tagi treści (DS3)",
            "selected": f"{content_stats.get('tag_features', 0)} wymiarów tagów",
            "limit": "Scalenie przez match_key",
            "note": "Puste tagi po scaleniu — blok tagów pominięty w macierzy łączonej",
        },
        {
            "module": "Sentyment (DS4)",
            "selected": f"TF-IDF max_features={sentiment_hp.get('max_features', 20_000)}",
            "limit": f"ngram_range={sentiment_hp.get('ngram_range', [1, 2])}",
            "note": "Niezależny korpus Amazon; etykiety z gwiazdek ≥4 / ≤2",
        },
        {
            "module": "K-Means (użytk. DS1)",
            "selected": ", ".join(clustering_feat.get("feature_columns", [])) or "7 cech behawioralnych",
            "limit": f"min_ratings={clustering_feat.get('min_ratings', 3)}",
            "note": "Użytkownicy poniżej min. ocen wykluczeni z treningu",
        },
        {
            "module": "Współpracujący (SVD)",
            "selected": "user_id × book_id → ocena",
            "limit": "Tylko interakcje DS1",
            "note": f"Zakres katalogu: {content_feat.get('n_books', '—')} książek treściowych oddzielnie od CF",
        },
    ]
    return rows


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
        analyze_all = self._read_json("data_pipeline/analyze_all.json") or {}
        train_all = self._read_json("ml/evaluation/train_all.json") or {}
        hybrid_eval = self._read_json("ml/evaluation/hybrid/baseline_comparison.json") or {}
        manifest = self._read_json("manifest.json") or {}

        cf = evaluate_all.get("collaborative", {})
        cf_ranking = cf.get("ranking", {}) or {}
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

        ridge_hybrid = (hybrid_eval.get("baselines") or {}).get("learned_ridge_hybrid", {})

        return {
            "reports_available": self.is_available,
            "evaluate_all": evaluate_all,
            "features": features,
            "preprocess": preprocess,
            "manifest": manifest,
            "eda_images": self.list_eda_images(),
            "outlier_summary": _build_outlier_summary(preprocess, analyze_all),
            "feature_selection_rows": _build_feature_selection_rows(features, train_all),
            "hybrid_eval": hybrid_eval,
            "metrics": {
                "cf_rmse": cf.get("rmse"),
                "cf_mae": cf.get("mae"),
                "cf_precision_at_k": cf_ranking.get("precision_at_k"),
                "cf_recall_at_k": cf_ranking.get("recall_at_k"),
                "cf_ranking_k": cf_ranking.get("k"),
                "hybrid_rmse": ridge_hybrid.get("rmse"),
                "hybrid_mae": ridge_hybrid.get("mae"),
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
        profiles_detail = clustering.get("cluster_profiles_detail") or {}
        cluster_descriptions = clustering.get("cluster_descriptions") or {}
        cluster_baseline = clustering.get("cluster_baseline") or {}
        sizes = train.get("cluster_sizes", {}) or clustering.get("cluster_sizes", {})
        silhouette_by_k = train.get("silhouette_by_k", {})

        cluster_labels: dict[int, str] = {}
        for cid_str, payload in cluster_descriptions.items():
            cluster_labels[int(cid_str)] = str(payload.get("title", f"Klaster {cid_str}"))
        for cid_str, payload in profiles_detail.items():
            cid = int(cid_str)
            if cid not in cluster_labels and payload.get("title"):
                cluster_labels[cid] = str(payload["title"])

        rating_dist_charts: dict[str, dict[str, list]] = {}
        for cid_str, payload in profiles_detail.items():
            dist = payload.get("rating_distribution_pct") or {}
            rating_dist_charts[cid_str] = {
                "labels": [f"{s}★" for s in ("1", "2", "3", "4", "5")],
                "values": [round(float(dist.get(s, 0)) * 100, 1) for s in ("1", "2", "3", "4", "5")],
            }

        viz = clustering.get("cluster_visualizations") or {}
        pca_viz = viz.get("pca_scatter") or {}
        n_ratings_viz = viz.get("n_ratings_histogram") or {}

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
            "rating_distributions": rating_dist_charts,
            "pca_scatter": pca_viz,
            "n_ratings_histogram": n_ratings_viz,
        }

        return {
            "reports_available": self.is_available,
            "clustering": clustering,
            "train": train,
            "features": features,
            "profiles_detail": profiles_detail,
            "cluster_descriptions": cluster_descriptions,
            "cluster_baseline": cluster_baseline,
            "cluster_labels": cluster_labels,
            "cluster_visualizations": viz,
            "chart_data": chart_data,
        }
