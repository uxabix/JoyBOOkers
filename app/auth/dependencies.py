"""Auth dependencies for API routes."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status

from app.auth.session import get_session_user_id
from app.dependencies import get_user_service
from app.schemas.user import UserProfile, UserRead
from app.services.user_service import UserService


def get_optional_user(request: Request, service: UserService = Depends(get_user_service)) -> UserProfile | None:
    if getattr(request.state, "current_user", None) is not None:
        return request.state.current_user
    user_id = get_session_user_id(request)
    if user_id is None:
        return None
    return service.get_profile(user_id)


def get_current_user(
    request: Request,
    service: UserService = Depends(get_user_service),
) -> UserProfile:
    profile = get_optional_user(request, service)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nie jesteś zalogowany. Zarejestruj się lub zaloguj.",
        )
    return profile


def get_current_user_read(current: UserProfile = Depends(get_current_user)) -> UserRead:
    return UserRead.model_validate(current)
