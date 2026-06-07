"""ORM models — import all so Alembic / init_db register metadata."""

from app.db.models.book import Book, BookEnrichment
from app.db.models.model_artifact import ModelArtifact
from app.db.models.rating import Rating
from app.db.models.recommendation import Recommendation
from app.db.models.review import Review
from app.db.models.user import User

__all__ = [
    "Book",
    "BookEnrichment",
    "ModelArtifact",
    "Rating",
    "Recommendation",
    "Review",
    "User",
]
