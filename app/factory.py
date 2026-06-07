"""FastAPI application factory — wires routers, static files, and error handlers."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from bookrec.paths import PROJECT_ROOT

from app.auth.middleware import CurrentUserMiddleware
from app.config import Settings, get_settings
from app.errors import register_exception_handlers
from app.logging_config import get_logger
from app.routers.api import api_router
from app.routers.web import web_router
from app.startup import on_shutdown, on_startup

logger = get_logger(__name__)


def register_routers(app: FastAPI) -> None:
    """Mount REST API v1 and server-rendered web UI."""
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(web_router)
    logger.debug("Routers registered: /api/v1/*, web pages")


def register_static_mounts(app: FastAPI, settings: Settings) -> None:
    if settings.static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

    reports_eda = PROJECT_ROOT / "reports" / "eda"
    if reports_eda.is_dir():
        app.mount("/reports-assets/eda", StaticFiles(directory=str(reports_eda)), name="reports-eda")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully wired FastAPI application."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        on_startup(app, settings)
        yield
        on_shutdown(app)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(CurrentUserMiddleware)
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        max_age=settings.session_max_age,
        same_site="lax",
        https_only=False,
    )

    register_static_mounts(app, settings)
    register_routers(app)
    register_exception_handlers(app)

    return app
