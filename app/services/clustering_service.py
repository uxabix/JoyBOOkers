"""Assign registered users to K-Means clusters from their ratings."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.config import Settings
from app.ml.user_clustering import UserClusteringEngine
from app.repositories.rating_repository import RatingRepository
from app.repositories.user_repository import UserRepository


class ClusteringService:
    def __init__(
        self,
        session: Session,
        engine: UserClusteringEngine,
        settings: Settings,
    ) -> None:
        self.session = session
        self.engine = engine
        self.settings = settings
        self.users = UserRepository(session)
        self.ratings = RatingRepository(session)

    def update_user_cluster(self, user_id: int) -> int | None:
        user = self.users.get(user_id)
        if user is None:
            return None
        if not self.engine.is_loaded and not self.engine.load():
            return user.cluster_id

        scores = self.ratings.scores_for_user(user_id)
        cluster_id = self.engine.predict_cluster(scores)
        user.cluster_id = cluster_id
        self.session.commit()
        return cluster_id
