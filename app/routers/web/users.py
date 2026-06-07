"""Browse dataset user profiles."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse

from app.config import Settings
from app.dependencies import (
    get_rating_service,
    get_recommendation_service,
    get_settings_dep,
    get_templates,
    get_user_service,
)
from app.routers.web.pages import _ctx
from app.services.rating_service import RatingService
from app.services.recommendation_service import RecommendationService
from app.services.user_service import UserService

router = APIRouter()


@router.get("/users", response_class=HTMLResponse)
def users_list(
    request: Request,
    q: str | None = Query(default=None),
    min_ratings: int = Query(default=1, ge=0),
    page: int = Query(default=1, ge=1),
    service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
):
    result = service.browse_users(q=q, min_ratings=min_ratings, page=page, page_size=24)
    return get_templates(request).TemplateResponse(
        request,
        "users/list.html",
        _ctx(request, settings, result=result, q=q, min_ratings=min_ratings),
    )


@router.get("/users/{user_id}", response_class=HTMLResponse)
def user_detail(
    request: Request,
    user_id: int,
    user_service: UserService = Depends(get_user_service),
    rating_service: RatingService = Depends(get_rating_service),
    rec_service: RecommendationService = Depends(get_recommendation_service),
    settings: Settings = Depends(get_settings_dep),
):
    profile = user_service.get_profile(user_id)
    if profile is None:
        return get_templates(request).TemplateResponse(
            request,
            "users/detail.html",
            _ctx(request, settings, profile=None),
        )
    if profile.is_registered:
        return get_templates(request).TemplateResponse(
            request,
            "users/detail.html",
            _ctx(request, settings, profile=None),
            status_code=404,
        )

    ratings = rating_service.list_with_books(user_id, limit=50)
    recommendations = rec_service.recommend_for_user(user_id, limit=settings.default_recommendation_limit)
    return get_templates(request).TemplateResponse(
        request,
        "users/detail.html",
        _ctx(request, settings, profile=profile, ratings=ratings, recommendations=recommendations),
    )


@router.get("/users/{user_id}/recommendations", response_class=HTMLResponse)
def user_recommendations_partial(
    request: Request,
    user_id: int,
    rec_service: RecommendationService = Depends(get_recommendation_service),
    settings: Settings = Depends(get_settings_dep),
):
    result = rec_service.recommend_for_user(user_id, limit=settings.default_recommendation_limit)
    return get_templates(request).TemplateResponse(
        request,
        "recommendations/_results.html",
        _ctx(request, settings, result=result, empty_reason=None),
    )
