"""Server-rendered HTMX pages."""

from fastapi import APIRouter

from app.routers.web import account, authors, pages, users

web_router = APIRouter()
web_router.include_router(account.router)
web_router.include_router(authors.router)
web_router.include_router(users.router)
web_router.include_router(pages.router)
