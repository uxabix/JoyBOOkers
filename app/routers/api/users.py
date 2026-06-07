"""User API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import get_rating_service, get_recommendation_service, get_user_service
from app.schemas.rating import RatingWithBook
from app.schemas.recommendation import RecommendationResponse
from app.schemas.user import UserBrowseResult, UserCreate, UserProfile, UserRead
from app.services.rating_service import RatingService
from app.services.recommendation_service import RecommendationService
from app.services.user_service import UserService

router = APIRouter()


@router.get("/browse", response_model=UserBrowseResult)
def browse_users(
    q: str | None = None,
    min_ratings: int = 1,
    page: int = 1,
    page_size: int = 24,
    dataset_only: bool = True,
    service: UserService = Depends(get_user_service),
) -> UserBrowseResult:
    return service.browse_users(
        q=q,
        min_ratings=min_ratings,
        page=page,
        page_size=page_size,
        dataset_only=dataset_only,
    )


@router.get("/{user_id}/profile", response_model=UserProfile)
def get_user_profile(
    user_id: int,
    service: UserService = Depends(get_user_service),
) -> UserProfile:
    profile = service.get_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return profile


@router.get("/{user_id}", response_model=UserRead)
def get_user(
    user_id: int,
    service: UserService = Depends(get_user_service),
) -> UserRead:
    user = service.get(user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreate,
    service: UserService = Depends(get_user_service),
) -> UserRead:
    return service.get_or_create(payload)


@router.get("/{user_id}/ratings", response_model=list[RatingWithBook])
def user_ratings_with_books(
    user_id: int,
    service: RatingService = Depends(get_rating_service),
) -> list[RatingWithBook]:
    return service.list_with_books(user_id)


@router.get("/{user_id}/recommendations", response_model=RecommendationResponse)
def user_recommendations(
    user_id: int,
    limit: int = 10,
    algorithm: str = "auto",
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    return service.recommend_for_user(user_id, limit=limit, algorithm=algorithm)
