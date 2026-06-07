"""Rating API — DS1 collaborative filtering."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.dependencies import get_rating_service
from app.schemas.rating import RatingCreate, RatingRead
from app.services.rating_service import RatingService

router = APIRouter()


@router.post("", response_model=RatingRead, status_code=status.HTTP_201_CREATED)
def create_rating(
    payload: RatingCreate,
    service: RatingService = Depends(get_rating_service),
) -> RatingRead:
    return service.create(payload)


@router.get("/user/{user_id}", response_model=list[RatingRead])
def list_user_ratings(
    user_id: int,
    service: RatingService = Depends(get_rating_service),
) -> list[RatingRead]:
    return service.list_for_user(user_id)
