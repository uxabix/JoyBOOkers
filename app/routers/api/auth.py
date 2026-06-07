"""Registration and session auth API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from fastapi import HTTPException

from app.auth.dependencies import get_current_user
from app.auth.session import clear_session_user, set_session_user_id
from app.dependencies import get_user_service
from app.schemas.user import UserLogin, UserProfile, UserRead, UserRegister
from app.services.user_service import UserService

router = APIRouter()


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def register(
    payload: UserRegister,
    request: Request,
    service: UserService = Depends(get_user_service),
) -> UserRead:
    user = service.register(payload)
    set_session_user_id(request, user.id)
    return user


@router.post("/login", response_model=UserRead)
def login(
    payload: UserLogin,
    request: Request,
    service: UserService = Depends(get_user_service),
) -> UserRead:
    user = service.login(payload.nickname)
    set_session_user_id(request, user.id)
    return user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request) -> None:
    clear_session_user(request)


@router.get("/me", response_model=UserProfile)
def me(current: UserProfile = Depends(get_current_user)) -> UserProfile:
    return current
