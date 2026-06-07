"""Cookie session helpers for registered users."""

from __future__ import annotations

from fastapi import Request

SESSION_USER_KEY = "user_id"


def get_session_user_id(request: Request) -> int | None:
    raw = request.session.get(SESSION_USER_KEY)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def set_session_user_id(request: Request, user_id: int) -> None:
    request.session[SESSION_USER_KEY] = user_id


def clear_session_user(request: Request) -> None:
    request.session.pop(SESSION_USER_KEY, None)
