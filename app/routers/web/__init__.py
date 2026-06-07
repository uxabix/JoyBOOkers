"""Server-rendered HTMX pages."""

from fastapi import APIRouter

from app.routers.web import pages

web_router = APIRouter()
web_router.include_router(pages.router)
