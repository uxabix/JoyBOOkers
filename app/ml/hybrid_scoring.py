"""Unified hybrid scoring — single path for all user types."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import Settings
from app.ml.cluster_affinity import ClusterAffinityStore
from app.ml.collaborative import CollaborativeFilteringEngine
from app.ml.content_based import ContentRecommendationEngine
from app.ml.explanations import build_explanations
from app.ml.hybrid_weights import HybridWeightModel
from app.ml.user_profile import (
    UserProfile,
    blend_signal_weights,
    genre_match_score,
)
from app.repositories.book_repository import BookRepository
from app.services.collaborative_filtering_service import CollaborativeFilteringService


@dataclass
class ScoredCandidate:
    book_id: int
    source_book_id: str
    final_score: float
    cf: float = 0.0
    content: float = 0.0
    cluster: float = 0.0
    pop: float = 0.0
    genre: float = 0.0
    explanations: list[str] | None = None


class HybridScoringEngine:
    def __init__(
        self,
        session: Session,
        cf_engine: CollaborativeFilteringEngine,
        cf_service: CollaborativeFilteringService,
        content_engine: ContentRecommendationEngine,
        cluster_affinity: ClusterAffinityStore,
        settings: Settings,
        *,
        weight_model: HybridWeightModel | None = None,
    ) -> None:
        self.session = session
        self.cf_engine = cf_engine
        self.cf_service = cf_service
        self.content_engine = content_engine
        self.cluster_affinity = cluster_affinity
        self.settings = settings
        self.weight_model = weight_model
        self.books = BookRepository(session)

    def recommend(
        self,
        profile: UserProfile,
        *,
        limit: int,
    ) -> list[ScoredCandidate]:
        if not self.content_engine.is_loaded:
            self.content_engine.load()
        if not self.cluster_affinity.is_loaded:
            self.cluster_affinity.load()

        content_vector = profile.content_vector
        if content_vector is None and profile.rated_books:
            rated_pairs = [(b.source_book_id, b.score) for b in profile.rated_books]
            content_vector = self.content_engine.build_user_vector(rated_pairs)

        rated_ids = {b.book_id for b in profile.rated_books}
        candidates = self._gather_candidates(profile, rated_ids, limit=limit)

        if not candidates:
            return []

        manual_weights = blend_signal_weights(profile)
        use_learned = self.weight_model is not None and self.weight_model.is_loaded
        cf_scores = self._cf_scores(profile, candidates, limit=limit)
        content_scores = self._content_scores(content_vector, candidates)
        cluster_raw = {sid: self.cluster_affinity.score(profile.cluster_id, sid) for sid in candidates}
        pop_raw = {sid: self._popularity_raw(sid) for sid in candidates}
        genre_raw = {sid: self._genre_raw(profile, sid) for sid in candidates}

        cf_norm = self._normalize_dict(cf_scores)
        content_norm = self._normalize_dict(content_scores)
        cluster_norm = self._normalize_dict(cluster_raw)
        pop_norm = self._normalize_dict(pop_raw)
        genre_norm = self._normalize_dict(genre_raw)

        max_pop = max(pop_raw.values()) if pop_raw else 1.0
        use_ridge = use_learned and profile.cf_available
        used_learned = False
        scored: list[ScoredCandidate] = []
        for source_id, book_id in candidates.items():
            breakdown = {
                "cf": cf_norm.get(source_id, 0.0),
                "content": content_norm.get(source_id, 0.0),
                "cluster": cluster_norm.get(source_id, 0.0),
                "pop": pop_norm.get(source_id, 0.0),
                "genre": genre_norm.get(source_id, 0.0),
            }
            manual_score = sum(manual_weights[k] * breakdown[k] for k in breakdown)
            if use_ridge:
                ridge_feats = self._ridge_features(
                    source_id,
                    cf_scores=cf_scores,
                    content_scores=content_scores,
                    cluster_raw=cluster_raw,
                    pop_raw=pop_raw,
                    genre_raw=genre_raw,
                    max_pop=max_pop,
                )
                ridge_score = self.weight_model.score(ridge_feats)  # type: ignore[union-attr]
                if ridge_score > 0:
                    final = ridge_score
                    used_learned = True
                else:
                    final = manual_score
            else:
                final = manual_score

            book = self.books.get_by_source_id(source_id)
            explanations = build_explanations(
                profile,
                breakdown,
                book_genre=book.genre if book else None,
            )
            scored.append(
                ScoredCandidate(
                    book_id=book_id,
                    source_book_id=source_id,
                    final_score=final,
                    cf=breakdown["cf"],
                    content=breakdown["content"],
                    cluster=breakdown["cluster"],
                    pop=breakdown["pop"],
                    genre=breakdown["genre"],
                    explanations=explanations,
                )
            )

        scored.sort(key=lambda x: x.final_score, reverse=True)
        self._last_weight_source = "learned" if used_learned else "manual"
        return scored[:limit]

    @property
    def last_weight_source(self) -> str:
        return getattr(self, "_last_weight_source", "manual")

    @staticmethod
    def _ridge_features(
        source_id: str,
        *,
        cf_scores: dict[str, float],
        content_scores: dict[str, float],
        cluster_raw: dict[str, float],
        pop_raw: dict[str, float],
        genre_raw: dict[str, float],
        max_pop: float,
    ) -> dict[str, float]:
        """Raw signal vector — must match training in hybrid_training.py."""
        cf_val = cf_scores.get(source_id)
        return {
            "cf": HybridScoringEngine._norm_cf(cf_val) if cf_val is not None else 0.0,
            "content": float(content_scores.get(source_id, 0.0)),
            "cluster": float(cluster_raw.get(source_id, 0.0)),
            "pop": float(pop_raw.get(source_id, 0.0)) / max_pop if max_pop > 0 else 0.0,
            "genre": float(genre_raw.get(source_id, 0.0)),
        }

    def recommend_cf_only(self, profile: UserProfile, *, limit: int) -> list[ScoredCandidate]:
        if not profile.cf_available:
            return []
        pairs = self.cf_service.recommend_for_user(profile.user_id, limit=limit)
        return [
            ScoredCandidate(
                book_id=book_id,
                source_book_id=self._source_id(book_id) or "",
                final_score=self._norm_cf(score),
                cf=self._norm_cf(score),
            )
            for book_id, score in pairs
        ]

    def recommend_content_only(self, profile: UserProfile, *, limit: int) -> list[ScoredCandidate]:
        if not self.content_engine.is_loaded:
            self.content_engine.load()
        rated_ids = {b.book_id for b in profile.rated_books}
        candidates = self._content_candidates(profile, rated_ids, limit=limit * 5)
        if not candidates:
            return self._popular_fallback(profile, limit=limit)

        content_vector = self.content_engine.build_user_vector(
            [(b.source_book_id, b.score) for b in profile.rated_books],
        )
        if content_vector is None:
            return self._popular_fallback(profile, limit=limit)

        scores = self.content_engine.score_candidates(content_vector, list(candidates.keys()))
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [
            ScoredCandidate(
                book_id=candidates[sid],
                source_book_id=sid,
                final_score=score,
                content=score,
            )
            for sid, score in ranked
        ]

    def _gather_candidates(
        self,
        profile: UserProfile,
        rated_ids: set[int],
        *,
        limit: int,
    ) -> dict[str, int]:
        """source_book_id → internal book_id."""
        out: dict[str, int] = {}
        cap = self.settings.hybrid_candidate_limit

        if profile.cf_available:
            for book_id, _ in self.cf_service.recommend_for_user(
                profile.user_id, limit=min(limit * 5, cap // 2)
            ):
                if book_id in rated_ids:
                    continue
                book = self.books.get(book_id)
                if book:
                    out[str(book.source_book_id)] = book.id

        for item in profile.rated_books[:5]:
            if not self.content_engine.is_loaded:
                continue
            for sid, _ in self.content_engine.similar_books(item.source_book_id, limit=limit * 2):
                if sid in out:
                    continue
                match = self.books.get_by_source_id(sid)
                if match and match.id not in rated_ids:
                    out[sid] = match.id
                if len(out) >= cap:
                    break
            if len(out) >= cap:
                break

        for sid, _ in self.cluster_affinity.top_books(profile.cluster_id, limit=80):
            if sid in out:
                continue
            match = self.books.get_by_source_id(sid)
            if match and match.id not in rated_ids:
                out[sid] = match.id

        for book in self.books.list_starter_books(profile.user_id, limit=40):
            sid = str(book.source_book_id)
            if sid not in out and book.id not in rated_ids:
                out[sid] = book.id

        return dict(list(out.items())[:cap])

    def _content_candidates(
        self,
        profile: UserProfile,
        rated_ids: set[int],
        *,
        limit: int,
    ) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in profile.rated_books[:5]:
            for sid, _ in self.content_engine.similar_books(item.source_book_id, limit=limit):
                if sid in out:
                    continue
                match = self.books.get_by_source_id(sid)
                if match and match.id not in rated_ids:
                    out[sid] = match.id
        return out

    def _cf_scores(
        self,
        profile: UserProfile,
        candidates: dict[str, int],
        *,
        limit: int,
    ) -> dict[str, float]:
        if not profile.cf_available or not self.cf_engine.is_loaded:
            return {}

        pairs = self.cf_service.recommend_for_user(profile.user_id, limit=limit * 5)
        scores = {self._source_id(bid) or "": score for bid, score in pairs}

        missing = [sid for sid in candidates if sid not in scores]
        for sid in missing[:200]:
            pred = self.cf_engine.predict(profile.external_id, sid)
            if pred is not None:
                scores[sid] = pred
        return {k: v for k, v in scores.items() if k}

    def _content_scores(self, user_vector, candidates: dict[str, int]) -> dict[str, float]:
        if user_vector is None:
            return {}
        return self.content_engine.score_candidates(user_vector, list(candidates.keys()))

    def _popularity_raw(self, source_book_id: str) -> float:
        book = self.books.get_by_source_id(source_book_id)
        if book is None:
            return 0.0
        return float(book.rating_count or 0)

    def _genre_raw(self, profile: UserProfile, source_book_id: str) -> float:
        book = self.books.get_by_source_id(source_book_id)
        if book is None:
            return 0.0
        return genre_match_score(profile.genre_weights, book.genre)

    def _popular_fallback(self, profile: UserProfile, *, limit: int) -> list[ScoredCandidate]:
        rows = self.books.list_starter_books(profile.user_id, limit=limit)
        return [
            ScoredCandidate(
                book_id=b.id,
                source_book_id=str(b.source_book_id),
                final_score=float(b.db_avg_rating or b.avg_rating or 4.0) / 5.0,
                pop=1.0,
            )
            for b in rows
        ]

    @staticmethod
    def _normalize_dict(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        lo = min(values.values())
        hi = max(values.values())
        if hi <= lo:
            return {k: 1.0 if v > 0 else 0.0 for k, v in values.items()}
        return {k: (v - lo) / (hi - lo) for k, v in values.items()}

    @staticmethod
    def _norm_cf(score: float) -> float:
        return max(0.0, min(1.0, (float(score) - 1.0) / 4.0))

    def _source_id(self, book_id: int) -> str | None:
        book = self.books.get(book_id)
        return str(book.source_book_id) if book else None
