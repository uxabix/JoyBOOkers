"""Unified hybrid recommender — single scoring path for all users."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models.recommendation import Recommendation
from app.ml.cluster_affinity import ClusterAffinityStore
from app.ml.collaborative import CollaborativeFilteringEngine
from app.ml.content_based import ContentRecommendationEngine
from app.ml.genre_priors import GenrePriorStore
from app.ml.hybrid_scoring import HybridScoringEngine, ScoredCandidate
from app.ml.hybrid_weights import HybridWeightModel
from app.ml.signals import heuristic_signal_keys, ml_signal_keys, signal_kind_map
from app.ml.user_clustering import UserClusteringEngine
from app.ml.user_profile import UserProfileBuilder, blend_signal_weights
from app.repositories.book_repository import BookRepository
from app.repositories.recommendation_repository import RecommendationRepository
from app.repositories.user_repository import UserRepository
from app.schemas.book import BookRead
from app.schemas.recommendation import (
    RecommendationItem,
    RecommendationResponse,
    ScoreBreakdown,
    UserProfileSummary,
)
from app.services.collaborative_filtering_service import CollaborativeFilteringService


class RecommendationService:
    def __init__(
        self,
        session: Session,
        cf_service: CollaborativeFilteringService,
        cf_engine: CollaborativeFilteringEngine,
        content_engine: ContentRecommendationEngine,
        clustering_engine: UserClusteringEngine,
        cluster_affinity: ClusterAffinityStore,
        genre_priors: GenrePriorStore,
        weight_model: HybridWeightModel,
        settings: Settings,
    ) -> None:
        self.session = session
        self.cf = cf_service
        self.cf_engine = cf_engine
        self.content_engine = content_engine
        self.clustering_engine = clustering_engine
        self.cluster_affinity = cluster_affinity
        self.settings = settings
        self.books = BookRepository(session)
        self.users = UserRepository(session)
        self.history = RecommendationRepository(session)

        if not self.cf_engine.is_loaded:
            self.cf_engine.load()
        known_users = self.cf_engine.known_user_ids() if self.cf_engine.is_loaded else set()
        if not weight_model.is_loaded:
            weight_model.load()
        self.weight_model = weight_model
        self.profile_builder = UserProfileBuilder(
            session,
            clustering_engine,
            settings,
            cf_known_user_ids=known_users,
            genre_priors=genre_priors,
        )
        self.hybrid = HybridScoringEngine(
            session,
            cf_engine,
            cf_service,
            content_engine,
            cluster_affinity,
            settings,
            weight_model=weight_model,
        )

    def recommend_for_user(
        self,
        user_id: int,
        *,
        limit: int | None = None,
        algorithm: str = "auto",
    ) -> RecommendationResponse:
        limit = limit or self.settings.default_recommendation_limit
        profile = self.profile_builder.build(user_id)
        if profile is None:
            return RecommendationResponse(user_id=user_id, algorithm=algorithm, items=[])

        algo = algorithm
        if algo in ("auto", "hybrid"):
            scored = self.hybrid.recommend(profile, limit=limit)
            algo = "hybrid"
        elif algo == "collaborative":
            scored = self.hybrid.recommend_cf_only(profile, limit=limit)
            algo = "collaborative"
        elif algo == "content":
            scored = self.hybrid.recommend_content_only(profile, limit=limit)
            algo = "content"
        else:
            scored = self.hybrid.recommend(profile, limit=limit)
            algo = "hybrid"

        weight_source = (
            self.hybrid.last_weight_source
            if algo == "hybrid"
            else ("learned" if self.weight_model.is_loaded and profile.cf_available else "manual")
        )
        weights = (
            self.weight_model.coefficients()
            if weight_source == "learned" and self.weight_model.is_loaded
            else blend_signal_weights(profile)
        )
        summary = UserProfileSummary(
            cluster_id=profile.cluster_id,
            cluster_label=profile.cluster_label,
            rating_count=len(profile.rated_books),
            profile_strength=profile.profile_strength,
            top_genres=sorted(profile.genre_weights, key=profile.genre_weights.get, reverse=True)[:5],
            genre_prior_active=profile.genre_prior_active,
            cf_available=profile.cf_available,
            weight_source=weight_source,
            weights_used=weights,
            ml_signals=ml_signal_keys(),
            heuristic_signals=heuristic_signal_keys(),
        )

        items = self._to_items(scored, algorithm=algo)
        if items:
            self._persist(user_id=user_id, items=items)
        return RecommendationResponse(
            user_id=user_id,
            algorithm=algo,
            items=items,
            profile=summary,
        )

    def explain_empty(self, user_id: int) -> str:
        user = self.users.get(user_id)
        if user is None:
            return (
                f"Użytkownik #{user_id} nie istnieje. Użyj wewnętrznego ID z bazy z tabeli poniżej "
                "(nie zewnętrznego ID DS1)."
            )
        profile = self.profile_builder.build(user_id)
        if profile is None:
            return f"Użytkownik #{user_id} nie istnieje."

        if not self.books.list_starter_books(user_id, limit=1):
            return (
                "Katalog książek jest pusty. Uruchom `python scripts/load_db.py`, "
                "aby załadować książki, i spróbuj ponownie."
            )
        return (
            "Scoring hybrydowy zakończył się, ale nie znaleziono dopasowań w katalogu. "
            "Spróbuj innego użytkownika lub oceń więcej książek i odśwież wynik."
        )

    def similar_books(self, book_id: int, *, limit: int | None = None) -> RecommendationResponse:
        limit = limit or self.settings.default_recommendation_limit
        if not self.content_engine.is_loaded:
            self.content_engine.load()
        book = self.books.get(book_id)
        if book is None:
            return RecommendationResponse(seed_book_id=book_id, algorithm="content", items=[])

        raw = self.content_engine.similar_books(book.source_book_id, limit=limit)
        pairs: list[tuple[int, float]] = []
        for source_id, score in raw:
            match = self.books.get_by_source_id(source_id)
            if match:
                pairs.append((match.id, score))

        scored = [
            ScoredCandidate(
                book_id=bid,
                source_book_id="",
                final_score=score,
                content=score,
            )
            for bid, score in pairs
        ]
        items = self._to_items(scored, algorithm="content")
        return RecommendationResponse(seed_book_id=book_id, algorithm="content", items=items)

    def _to_items(self, scored: list[ScoredCandidate], *, algorithm: str) -> list[RecommendationItem]:
        items: list[RecommendationItem] = []
        for rank, row in enumerate(scored, start=1):
            book = self.books.get_with_enrichment(row.book_id)
            if book is None:
                continue
            items.append(
                RecommendationItem(
                    book=BookRead.model_validate(book),
                    score=row.final_score,
                    algorithm=algorithm,
                    rank=rank,
                    score_breakdown=ScoreBreakdown(
                        cf=row.cf,
                        content=row.content,
                        cluster=row.cluster,
                        popularity=row.pop,
                        genre=row.genre,
                        signal_kinds=signal_kind_map(),
                    ),
                    explanations=row.explanations or [],
                )
            )
        return items

    def _persist(self, *, user_id: int, items: list[RecommendationItem]) -> None:
        for item in items:
            rec = Recommendation(
                user_id=user_id,
                book_id=item.book.id,
                algorithm=item.algorithm,
                score=item.score,
                rank=item.rank,
            )
            self.history.add(rec)
        self.session.commit()
