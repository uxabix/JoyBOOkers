from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EDA_DIR = PROCESSED_DIR / "eda"

# Per-dataset raw directories (place Kaggle downloads here).
RAW_DS1 = RAW_DIR / "ds1_goodreads_2m"
RAW_DS2 = RAW_DIR / "ds2_goodreads_100k"
RAW_DS3 = RAW_DIR / "ds3_goodreads_best"
RAW_DS4 = RAW_DIR / "ds4_amazon_reviews"

# Processed outputs per stage.
PROC_DS1 = PROCESSED_DIR / "ds1"
PROC_DS2 = PROCESSED_DIR / "ds2"
PROC_DS3 = PROCESSED_DIR / "ds3"
PROC_DS4 = PROCESSED_DIR / "ds4"
PROC_CANONICAL = PROCESSED_DIR / "canonical"
PROC_FEATURES = PROCESSED_DIR / "features"
PROC_SPLITS = PROCESSED_DIR / "splits"
PROC_ANALYSIS = PROCESSED_DIR / "analysis"
PROC_MODELS = PROCESSED_DIR / "models"

# Trained model artifacts (written by bookrec.ml training scripts).
MODEL_CF_DIR = PROC_MODELS / "collaborative"
MODEL_CONTENT_DIR = PROC_MODELS / "content"
MODEL_SENTIMENT_DIR = PROC_MODELS / "sentiment"
MODEL_CLUSTERING_DIR = PROC_MODELS / "clustering"
MODEL_EVAL_DIR = PROC_MODELS / "evaluation"

# Legacy single-file names (still supported when present in data/raw/).
LEGACY_BOOKS_CSV = RAW_DIR / "book.csv"
LEGACY_RATINGS_CSV = RAW_DIR / "user-rating.csv"

# Default: load all Kaggle shards from data/raw/ or ds1 subfolder.
BOOK_SHARD_GLOB = "book*.csv"
RATING_SHARD_GLOB = "user_rating_*.csv"


def ds1_raw_dir() -> Path:
    """Prefer ds1 subfolder; fall back to flat data/raw/."""
    if RAW_DS1.is_dir() and any(RAW_DS1.glob("book*.csv")):
        return RAW_DS1
    if RAW_DIR.is_dir() and any(RAW_DIR.glob("book*.csv")):
        return RAW_DIR
    return RAW_DS1
