"""Registered user flows — register, login, profile."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth.session import clear_session_user, set_session_user_id
from app.config import Settings
from app.dependencies import (
    get_cold_start_service,
    get_rating_service,
    get_recommendation_service,
    get_settings_dep,
    get_templates,
    get_user_service,
)
from app.routers.web.pages import _ctx
from app.schemas.rating import RatingCreate
from app.schemas.user import UserLogin, UserRegister
from app.services.cold_start_service import ColdStartService
from app.services.rating_service import RatingService
from app.services.recommendation_service import RecommendationService
from app.services.user_service import UserService

router = APIRouter()


@router.get("/register", response_class=HTMLResponse)
def register_page(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
):
    if getattr(request.state, "current_user", None):
        return RedirectResponse("/me", status_code=status.HTTP_303_SEE_OTHER)
    return get_templates(request).TemplateResponse(
        request,
        "account/register.html",
        _ctx(request, settings),
    )


@router.post("/register", response_class=HTMLResponse)
def register_submit(
    request: Request,
    nickname: str = Form(...),
    service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
):
    try:
        user = service.register(UserRegister(nickname=nickname))
    except HTTPException as exc:
        return get_templates(request).TemplateResponse(
            request,
            "account/register.html",
            _ctx(request, settings, error=exc.detail),
            status_code=exc.status_code,
        )
    set_session_user_id(request, user.id)
    return RedirectResponse("/me", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/login", response_class=HTMLResponse)
def login_page(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
):
    if getattr(request.state, "current_user", None):
        return RedirectResponse("/me", status_code=status.HTTP_303_SEE_OTHER)
    return get_templates(request).TemplateResponse(
        request,
        "account/login.html",
        _ctx(request, settings),
    )


@router.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    nickname: str = Form(...),
    service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
):
    try:
        user = service.login(nickname)
    except HTTPException as exc:
        return get_templates(request).TemplateResponse(
            request,
            "account/login.html",
            _ctx(request, settings, error=exc.detail),
            status_code=exc.status_code,
        )
    set_session_user_id(request, user.id)
    return RedirectResponse("/me", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/logout")
def logout(request: Request):
    clear_session_user(request)
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


def _ratings_browse_ctx(
    rating_service: RatingService,
    user_id: int,
    *,
    q: str | None,
    min_score: float | None,
    max_score: float | None,
    sort: str,
    page: int,
    page_size: int,
) -> dict:
    return {
        "rating_stats": rating_service.user_stats(user_id),
        "ratings_result": rating_service.browse_with_books(
            user_id,
            q=q,
            min_score=min_score,
            max_score=max_score,
            sort=sort,
            page=page,
            page_size=page_size,
        ),
        "ratings_q": q or "",
        "ratings_min_score": min_score if min_score is not None else "",
        "ratings_max_score": max_score if max_score is not None else "",
        "ratings_sort": sort,
        "ratings_page": page,
        "ratings_page_size": page_size,
    }


@router.get("/me", response_class=HTMLResponse)
def profile_page(
    request: Request,
    q: str | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=1.0, le=5.0),
    max_score: float | None = Query(default=None, ge=1.0, le=5.0),
    sort: str = Query(default="updated_desc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=15, ge=5, le=50),
    settings: Settings = Depends(get_settings_dep),
    user_service: UserService = Depends(get_user_service),
    rating_service: RatingService = Depends(get_rating_service),
    rec_service: RecommendationService = Depends(get_recommendation_service),
    cold_start: ColdStartService = Depends(get_cold_start_service),
):
    profile = getattr(request.state, "current_user", None)
    if profile is None:
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    recommendations = rec_service.recommend_for_user(profile.id, limit=settings.default_recommendation_limit)
    starters = cold_start.starter_books(profile.id)
    profile = user_service.get_profile(profile.id, sync_cluster=True) or profile

    return get_templates(request).TemplateResponse(
        request,
        "account/profile.html",
        _ctx(
            request,
            settings,
            profile=profile,
            recommendations=recommendations,
            starters=starters,
            **_ratings_browse_ctx(
                rating_service,
                profile.id,
                q=q,
                min_score=min_score,
                max_score=max_score,
                sort=sort,
                page=page,
                page_size=page_size,
            ),
        ),
    )


@router.get("/me/ratings", response_class=HTMLResponse)
def profile_ratings_partial(
    request: Request,
    q: str | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=1.0, le=5.0),
    max_score: float | None = Query(default=None, ge=1.0, le=5.0),
    sort: str = Query(default="updated_desc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=15, ge=5, le=50),
    settings: Settings = Depends(get_settings_dep),
    rating_service: RatingService = Depends(get_rating_service),
):
    profile = getattr(request.state, "current_user", None)
    if profile is None:
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    return get_templates(request).TemplateResponse(
        request,
        "account/_ratings_section.html",
        _ctx(
            request,
            settings,
            profile=profile,
            **_ratings_browse_ctx(
                rating_service,
                profile.id,
                q=q,
                min_score=min_score,
                max_score=max_score,
                sort=sort,
                page=page,
                page_size=page_size,
            ),
        ),
    )


@router.post("/me/rate", response_class=HTMLResponse)
def rate_book(
    request: Request,
    book_id: int = Form(...),
    score: float = Form(...),
    rating_service: RatingService = Depends(get_rating_service),
    settings: Settings = Depends(get_settings_dep),
):
    profile = getattr(request.state, "current_user", None)
    if profile is None:
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    rating = rating_service.create(
        RatingCreate(user_id=profile.id, book_id=book_id, score=score),
    )
    if request.headers.get("HX-Request"):
        return get_templates(request).TemplateResponse(
            request,
            "partials/_rating_widget.html",
            _ctx(
                request,
                settings,
                book_id=book_id,
                book={"id": book_id},
                user_rating=rating,
                success=True,
            ),
        )
    return RedirectResponse(f"/books/{book_id}", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/me/recommendations", response_class=HTMLResponse)
def profile_recommendations_partial(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
    rec_service: RecommendationService = Depends(get_recommendation_service),
):
    profile = getattr(request.state, "current_user", None)
    if profile is None:
        return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)

    result = rec_service.recommend_for_user(profile.id, limit=settings.default_recommendation_limit)
    return get_templates(request).TemplateResponse(
        request,
        "recommendations/_results.html",
        _ctx(request, settings, result=result, empty_reason=None),
    )
