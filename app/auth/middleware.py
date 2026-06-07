"""Attach logged-in user profile to each request for templates."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.auth.session import get_session_user_id
from app.db.session import get_session_factory
from app.services.user_service import UserService


class CurrentUserMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request.state.current_user = None
        user_id = get_session_user_id(request)
        if user_id is not None:
            session = get_session_factory()()
            try:
                clustering = getattr(request.app.state, "clustering_engine", None)
                profile = UserService(session, clustering).get_profile(user_id)
                request.state.current_user = profile
            finally:
                session.close()
        return await call_next(request)
