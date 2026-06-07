"""User service — registration, login, profiles."""

from __future__ import annotations

import uuid

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.db.models.user import User
from app.ml.user_clustering import UserClusteringEngine
from app.repositories.user_repository import UserRepository
from app.repositories.rating_repository import RatingRepository
from app.schemas.user import UserBrowseResult, UserCandidateRead, UserCreate, UserProfile, UserRead, UserRegister


class UserService:
    def __init__(self, session: Session, clustering_engine: UserClusteringEngine | None = None) -> None:
        self.repo = UserRepository(session)
        self.session = session
        self.clustering = clustering_engine

    def get(self, user_id: int) -> UserRead | None:
        user = self.repo.get(user_id)
        return UserRead.model_validate(user) if user else None

    def _resolve_cluster(self, user_id: int, scores: list[float], stored: int | None) -> tuple[int | None, str | None]:
        if self.clustering is None:
            return (stored, None) if stored is not None else (None, None)
        if scores:
            cluster_id = self.clustering.predict_cluster(scores)
            return cluster_id, self.clustering.cluster_label(cluster_id)
        if stored is not None:
            return stored, self.clustering.cluster_label(stored)
        return None, None

    def get_profile(self, user_id: int, *, sync_cluster: bool = False) -> UserProfile | None:
        user = self.repo.get(user_id)
        if user is None:
            return None

        ratings_repo = RatingRepository(self.session)
        scores = ratings_repo.scores_for_user(user_id)
        n = len(scores)
        cluster_id, label = self._resolve_cluster(user_id, scores, user.cluster_id)

        if sync_cluster and cluster_id is not None and user.cluster_id != cluster_id:
            user.cluster_id = cluster_id
            self.session.commit()

        return UserProfile(
            id=user.id,
            external_id=user.external_id,
            display_name=user.display_name,
            nickname=user.nickname,
            is_registered=user.is_registered,
            cluster_id=cluster_id,
            rating_count=n,
            cluster_label=label,
        )

    def browse_users(
        self,
        *,
        q: str | None = None,
        dataset_only: bool = True,
        min_ratings: int = 1,
        page: int = 1,
        page_size: int = 24,
    ) -> UserBrowseResult:
        page = max(page, 1)
        offset = (page - 1) * page_size
        rows = self.repo.list_browse(
            q=q,
            dataset_only=dataset_only,
            min_ratings=min_ratings,
            offset=offset,
            limit=page_size,
        )
        total = self.repo.count_browse(q=q, dataset_only=dataset_only, min_ratings=min_ratings)
        items: list[UserCandidateRead] = []
        for user, count in rows:
            scores = RatingRepository(self.session).scores_for_user(user.id) if count else []
            cluster_id, label = self._resolve_cluster(user.id, scores, user.cluster_id)
            items.append(
                UserCandidateRead(
                    id=user.id,
                    external_id=user.external_id,
                    display_name=user.display_name,
                    nickname=user.nickname,
                    is_registered=user.is_registered,
                    cluster_id=cluster_id,
                    rating_count=count,
                    cluster_label=label,
                )
            )
        pages = max(1, (total + page_size - 1) // page_size)
        return UserBrowseResult(items=items, total=total, page=page, page_size=page_size, pages=pages)

    def register(self, payload: UserRegister) -> UserRead:
        if self.repo.nickname_taken(payload.nickname):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Nickname '{payload.nickname}' is already taken",
            )
        external_id = f"reg:{uuid.uuid4().hex[:12]}"
        user = User(
            external_id=external_id,
            nickname=payload.nickname,
            display_name=payload.nickname,
            is_registered=True,
            cluster_id=1,
        )
        if self.clustering is not None:
            user.cluster_id = self.clustering.predict_cluster([])
        self.repo.add(user)
        self.session.commit()
        self.session.refresh(user)
        return UserRead.model_validate(user)

    def login(self, nickname: str) -> UserRead:
        user = self.repo.get_by_nickname(nickname)
        if user is None or not user.is_registered:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found. Register first.",
            )
        return UserRead.model_validate(user)

    def get_or_create(self, payload: UserCreate) -> UserRead:
        existing = self.repo.get_by_external_id(payload.external_id)
        if existing:
            return UserRead.model_validate(existing)

        user = User(external_id=payload.external_id, display_name=payload.display_name)
        self.repo.add(user)
        self.session.commit()
        self.session.refresh(user)
        return UserRead.model_validate(user)

    def list_recent(self, *, limit: int = 20) -> list[UserRead]:
        return [UserRead.model_validate(u) for u in self.repo.list_recent(limit=limit)]

    def list_recommendation_candidates(
        self,
        *,
        limit: int = 20,
        min_ratings: int = 3,
    ) -> list[UserCandidateRead]:
        rows = self.repo.list_top_by_ratings(limit=limit, min_ratings=min_ratings)
        out: list[UserCandidateRead] = []
        for u, count in rows:
            scores = RatingRepository(self.session).scores_for_user(u.id) if count else []
            cluster_id, label = self._resolve_cluster(u.id, scores, u.cluster_id)
            out.append(
                UserCandidateRead(
                    id=u.id,
                    external_id=u.external_id,
                    display_name=u.display_name,
                    nickname=u.nickname,
                    is_registered=u.is_registered,
                    cluster_id=cluster_id,
                    rating_count=count,
                    cluster_label=label,
                )
            )
        return out
