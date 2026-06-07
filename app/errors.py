"""Global exception handlers for API JSON and HTML responses."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_config import get_logger
from app.templates_env import get_templates_engine

logger = get_logger(__name__)


def _wants_html(request: Request) -> bool:
    if request.url.path.startswith("/api/"):
        return False
    accept = request.headers.get("accept", "")
    return "text/html" in accept or request.headers.get("HX-Request") == "true"


def register_exception_handlers(app: FastAPI) -> None:
    templates = get_templates_engine()

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse | HTMLResponse:
        if _wants_html(request) and exc.status_code in {404, 403, 400}:
            template = "errors/404.html" if exc.status_code == 404 else "errors/error.html"
            return templates.TemplateResponse(
                request,
                template,
                {
                    "status_code": exc.status_code,
                    "detail": exc.detail,
                    "app_name": getattr(request.app.state, "settings", None) and request.app.state.settings.app_name,
                },
                status_code=exc.status_code,
            )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse | HTMLResponse:
        if _wants_html(request):
            return templates.TemplateResponse(
                request,
                "errors/error.html",
                {
                    "status_code": 422,
                    "detail": "Validation error — check your form input.",
                    "errors": exc.errors(),
                    "app_name": getattr(request.app.state, "settings", None) and request.app.state.settings.app_name,
                },
                status_code=422,
            )
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse | HTMLResponse:
        if isinstance(exc, HTTPException):
            raise exc
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        if _wants_html(request):
            return templates.TemplateResponse(
                request,
                "errors/500.html",
                {
                    "status_code": 500,
                    "detail": "An unexpected error occurred.",
                    "app_name": getattr(request.app.state, "settings", None) and request.app.state.settings.app_name,
                },
                status_code=500,
            )
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
