"""Application configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from bookrec.paths import (
    DATA_DIR,
    MODEL_CF_DIR,
    MODEL_CLUSTERING_DIR,
    MODEL_CONTENT_DIR,
    MODEL_SENTIMENT_DIR,
    PROC_FEATURES,
    PROC_SPLITS,
    PROJECT_ROOT,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Book Recommendation System"
    app_version: str = "0.1.0"
    debug: bool = False

    database_url: str = Field(
        default=f"sqlite:///{(DATA_DIR / 'joybookers.db').as_posix()}",
        description="SQLAlchemy database URL",
    )

    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"

    features_dir: Path = PROC_FEATURES
    cf_model_path: Path = MODEL_CF_DIR / "svd_model.pkl"
    cf_train_path: Path = PROC_SPLITS / "cf_train.parquet"
    content_bow_path: Path = PROC_FEATURES / "content" / "bow_matrix.npz"
    content_tfidf_path: Path = MODEL_CONTENT_DIR / "tfidf_combined.npz"
    content_catalog_path: Path = PROC_FEATURES / "content" / "content_catalog.parquet"
    sentiment_model_path: Path = MODEL_SENTIMENT_DIR / "sentiment_pipeline.joblib"
    clustering_model_path: Path = MODEL_CLUSTERING_DIR / "kmeans_model.joblib"
    interactions_path: Path = PROC_FEATURES / "interactions" / "interactions_indexed.parquet"

    default_recommendation_limit: int = 10
    min_cf_ratings_per_user: int = 3
    cf_candidate_limit: int = 2000

    ml_eager_load: bool = True
    db_batch_size: int = 1000

    templates_dir: Path = Path(__file__).resolve().parent / "templates"
    static_dir: Path = Path(__file__).resolve().parent / "static"
    reports_dir: Path = PROJECT_ROOT / "reports"


@lru_cache
def get_settings() -> Settings:
    return Settings()
