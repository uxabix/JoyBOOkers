from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EDA_DIR = PROCESSED_DIR / "eda"

DEFAULT_BOOKS_CSV = RAW_DIR / "book.csv"
DEFAULT_RATINGS_CSV = RAW_DIR / "user-rating.csv"
