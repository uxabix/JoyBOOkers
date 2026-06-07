"""REST API v1."""

from fastapi import APIRouter

from app.routers.api import books, health, ratings, recommendations, sentiment, users

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(books.router, prefix="/books", tags=["books"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(ratings.router, prefix="/ratings", tags=["ratings"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
api_router.include_router(sentiment.router, prefix="/sentiment", tags=["sentiment"])
