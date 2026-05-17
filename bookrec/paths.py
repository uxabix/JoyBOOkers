from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EDA_DIR = PROCESSED_DIR / "eda"

# Legacy single-file names (still supported when present).
LEGACY_BOOKS_CSV = RAW_DIR / "book.csv"
LEGACY_RATINGS_CSV = RAW_DIR / "user-rating.csv"

# Default: load all Kaggle shards from data/raw/.
BOOK_SHARD_GLOB = "book*.csv"
RATING_SHARD_GLOB = "user_rating_*.csv"
