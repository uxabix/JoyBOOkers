"""Machine-learning adapters — sparse matrices, Surprise, sklearn."""

from app.ml.collaborative import CollaborativeFilteringEngine
from app.ml.content_based import ContentRecommendationEngine
from app.ml.sentiment import SentimentEngine
from app.ml.sparse_loader import SparseMatrixBundle, load_sparse_bundle

__all__ = [
    "CollaborativeFilteringEngine",
    "ContentRecommendationEngine",
    "SentimentEngine",
    "SparseMatrixBundle",
    "load_sparse_bundle",
]
