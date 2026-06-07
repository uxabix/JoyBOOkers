"""Content-based recommendation — TF-IDF sparse vectors (DS2 + DS3)."""

from bookrec.ml.content.evaluate import evaluate_content_vectors
from bookrec.ml.content.train import train_content_vectors

__all__ = ["evaluate_content_vectors", "train_content_vectors"]
