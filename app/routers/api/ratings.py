"""Rating API — DS1 collaborative filtering."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import get_current_user, get_optional_user
from app.dependencies import get_rating_service
from app.schemas.rating import RatingCreate, RatingRead
from app.schemas.user import UserProfile
from app.services.rating_service import RatingService

router = APIRouter()


@router.post("", response_model=RatingRead, status_code=status.HTTP_201_CREATED)
def create_rating(
    payload: RatingCreate,
    current: UserProfile | None = Depends(get_optional_user),
    service: RatingService = Depends(get_rating_service),
) -> RatingRead:
    if payload.user_id is None:
        if current is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="user_id required or log in")
        payload = payload.model_copy(update={"user_id": current.id})
    return service.create(payload)


@router.post("/me", response_model=RatingRead, status_code=status.HTTP_201_CREATED)
def create_my_rating(
    payload: RatingCreate,
    current: UserProfile = Depends(get_current_user),
    service: RatingService = Depends(get_rating_service),
) -> RatingRead:
    data = payload.model_copy(update={"user_id": current.id})
    return service.create(data)


@router.get("/user/{user_id}", response_model=list[RatingRead])
def list_user_ratings(
    user_id: int,
    service: RatingService = Depends(get_rating_service),
) -> list[RatingRead]:
    return service.list_for_user(user_id)
