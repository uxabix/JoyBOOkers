"""Recommendation API — CF (DS1) + content (DS2+DS3)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_recommendation_service
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from app.services.recommendation_service import RecommendationService

router = APIRouter()


@router.post("/for-user", response_model=RecommendationResponse)
def recommend_for_user(
    payload: RecommendationRequest,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    return service.recommend_for_user(
        payload.user_id,
        limit=payload.limit,
        algorithm=payload.algorithm,
    )
