"""Unified user representation for hybrid recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scipy import sparse

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from app.config import Settings
from app.ml.user_clustering import UserClusteringEngine
from app.repositories.book_repository import BookRepository
from app.repositories.rating_repository import RatingRepository
from app.repositories.user_repository import UserRepository


@dataclass(frozen=True)
class RatedBook:
    book_id: int
    source_book_id: str
    score: float
    genre: str | None = None


@dataclass
class UserProfile:
    user_id: int
    external_id: str
    is_registered: bool
    rated_books: list[RatedBook] = field(default_factory=list)
    cluster_id: int = 1
    cluster_label: str | None = None
    genre_weights: dict[str, float] = field(default_factory=dict)
    profile_strength: float = 0.0
    cf_available: bool = False
    content_vector: sparse.csr_matrix | None = None


def _split_genres(genre: str | None) -> list[str]:
    if not genre or not genre.strip():
        return []
    parts = [g.strip().lower() for g in genre.replace(";", ",").split(",")]
    return [g for g in parts if g]


def _genre_weights_from_ratings(rated: list[RatedBook]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in rated:
        for genre in _split_genres(item.genre):
            weights[genre] = weights.get(genre, 0.0) + item.score
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {g: w / total for g, w in weights.items()}


def genre_match_score(genre_weights: dict[str, float], book_genre: str | None) -> float:
    if not genre_weights:
        return 0.0
    tags = _split_genres(book_genre)
    if not tags:
        return 0.0
    return sum(genre_weights.get(tag, 0.0) for tag in tags)


def blend_signal_weights(profile: UserProfile) -> dict[str, float]:
    """Adaptive hybrid weights — same function for all user types."""
    n = len(profile.rated_books)
    if n == 0:
        base = {"cf": 0.0, "content": 0.0, "cluster": 0.35, "pop": 0.65, "genre": 0.0}
    elif n <= 2:
        base = {"cf": 0.0, "content": 0.50, "cluster": 0.20, "pop": 0.20, "genre": 0.10}
    elif n < 10:
        base = {"cf": 0.25, "content": 0.45, "cluster": 0.15, "pop": 0.10, "genre": 0.05}
    else:
        base = {"cf": 0.40, "content": 0.35, "cluster": 0.15, "pop": 0.05, "genre": 0.05}

    if not profile.cf_available:
        cf_share = base["cf"]
        base["cf"] = 0.0
        base["content"] += cf_share * 0.6
        base["cluster"] += cf_share * 0.25
        base["pop"] += cf_share * 0.15

    total = sum(base.values())
    return {k: v / total for k, v in base.items()}


class UserProfileBuilder:
    def __init__(
        self,
        session: Session,
        clustering: UserClusteringEngine,
        settings: Settings,
        *,
        cf_known_user_ids: set[str] | None = None,
    ) -> None:
        self.session = session
        self.clustering = clustering
        self.settings = settings
        self.users = UserRepository(session)
        self.ratings = RatingRepository(session)
        self.books = BookRepository(session)
        self._cf_known_user_ids = cf_known_user_ids

    def build(
        self,
        user_id: int,
        *,
        content_vector: sparse.csr_matrix | None = None,
    ) -> UserProfile | None:
        user = self.users.get(user_id)
        if user is None:
            return None

        rating_rows = self.ratings.list_for_user(user_id, limit=500)
        rated_books: list[RatedBook] = []
        for row in rating_rows:
            book = self.books.get(row.book_id)
            if book is None:
                continue
            rated_books.append(
                RatedBook(
                    book_id=book.id,
                    source_book_id=str(book.source_book_id),
                    score=float(row.score),
                    genre=book.genre,
                )
            )

        scores = [b.score for b in rated_books]
        if not self.clustering.is_loaded:
            self.clustering.load()
        cluster_id = self.clustering.predict_cluster(scores) if self.clustering.is_loaded else 1
        cluster_label = self.clustering.cluster_label(cluster_id) if self.clustering.is_loaded else None

        n = len(rated_books)
        profile_strength = min(1.0, n / 10.0)
        cf_available = (
            not user.is_registered
            and n >= self.settings.min_cf_ratings_per_user
            and self._cf_known_user_ids is not None
            and str(user.external_id) in self._cf_known_user_ids
        )

        return UserProfile(
            user_id=user_id,
            external_id=str(user.external_id),
            is_registered=bool(user.is_registered),
            rated_books=rated_books,
            cluster_id=cluster_id,
            cluster_label=cluster_label,
            genre_weights=_genre_weights_from_ratings(rated_books),
            profile_strength=profile_strength,
            cf_available=cf_available,
            content_vector=content_vector,
        )
