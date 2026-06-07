"""Per-dataset ingest and preprocess modules."""

from bookrec.ingest.ds1_goodreads_2m import preprocess_ds1, load_and_analyze_ds1
from bookrec.ingest.ds2_goodreads_100k import preprocess_ds2, load_and_analyze_ds2
from bookrec.ingest.ds3_goodreads_best import preprocess_ds3, load_and_analyze_ds3
from bookrec.ingest.ds4_amazon_reviews import preprocess_ds4, load_and_analyze_ds4

__all__ = [
    "preprocess_ds1",
    "preprocess_ds2",
    "preprocess_ds3",
    "preprocess_ds4",
    "load_and_analyze_ds1",
    "load_and_analyze_ds2",
    "load_and_analyze_ds3",
    "load_and_analyze_ds4",
]
