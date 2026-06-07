"""Jinja2 + HTMX web UI."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse

from app.config import Settings
from app.dependencies import (
    get_book_service,
    get_recommendation_service,
    get_reports_service,
    get_sentiment_service,
    get_settings_dep,
    get_templates,
    get_user_service,
)
from app.schemas.book import BookSearchParams
from app.schemas.sentiment import SentimentPredictRequest
from app.schemas.user import UserCreate
from app.services.book_service import BookService
from app.services.recommendation_service import RecommendationService
from app.services.reports_service import ReportsService
from app.services.sentiment_service import SentimentService
from app.services.user_service import UserService

router = APIRouter()


def _ctx(request: Request, settings: Settings, **extra):
    return {"request": request, "app_name": settings.app_name, "version": settings.app_version, **extra}


@router.get("/", response_class=HTMLResponse)
def home(request: Request, settings: Settings = Depends(get_settings_dep)):
    reports = get_reports_service()
    analytics = reports.get_analytics_context()
    return get_templates(request).TemplateResponse(
        request,
        "index.html",
        _ctx(request, settings, metrics=analytics.get("metrics", {}), reports_available=analytics["reports_available"]),
    )


@router.get("/books", response_class=HTMLResponse)
def books_page(
    request: Request,
    q: str | None = Query(default=None),
    genre: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    service: BookService = Depends(get_book_service),
    settings: Settings = Depends(get_settings_dep),
):
    result = service.search(BookSearchParams(q=q, genre=genre, page=page, page_size=12))
    template = "books/_search_results.html" if request.headers.get("HX-Request") else "books/list.html"
    return get_templates(request).TemplateResponse(
        request,
        template,
        _ctx(request, settings, result=result, q=q or "", genre=genre or "", page=page),
    )


@router.get("/books/similar", response_class=HTMLResponse)
def similar_books_page(
    request: Request,
    book_id: int | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    book_service: BookService = Depends(get_book_service),
    rec_service: RecommendationService = Depends(get_recommendation_service),
    settings: Settings = Depends(get_settings_dep),
):
    seed_book = book_service.get(book_id) if book_id else None
    similar = rec_service.similar_books(book_id, limit=limit) if seed_book else None
    partial = request.headers.get("HX-Request")
    if partial and book_id:
        return get_templates(request).TemplateResponse(
            request,
            "books/_similar_results.html",
            _ctx(request, settings, seed_book=seed_book, similar=similar, book_id=book_id, limit=limit),
        )
    return get_templates(request).TemplateResponse(
        request,
        "books/similar.html",
        _ctx(request, settings, seed_book=seed_book, similar=similar, book_id=book_id or "", limit=limit),
    )


@router.post("/books/similar", response_class=HTMLResponse)
def similar_books_submit(
    request: Request,
    book_id: int = Form(...),
    limit: int = Form(default=10),
    book_service: BookService = Depends(get_book_service),
    rec_service: RecommendationService = Depends(get_recommendation_service),
    settings: Settings = Depends(get_settings_dep),
):
    seed_book = book_service.get(book_id)
    similar = rec_service.similar_books(book_id, limit=limit) if seed_book else None
    return get_templates(request).TemplateResponse(
        request,
        "books/_similar_results.html",
        _ctx(request, settings, seed_book=seed_book, similar=similar, book_id=book_id, limit=limit),
    )


@router.get("/books/{book_id}", response_class=HTMLResponse)
def book_detail(
    request: Request,
    book_id: int,
    book_service: BookService = Depends(get_book_service),
    rec_service: RecommendationService = Depends(get_recommendation_service),
    settings: Settings = Depends(get_settings_dep),
):
    book = book_service.get(book_id)
    similar = rec_service.similar_books(book_id, limit=6) if book else None
    return get_templates(request).TemplateResponse(
        request,
        "books/detail.html",
        _ctx(request, settings, book=book, similar=similar),
    )


@router.get("/recommendations", response_class=HTMLResponse)
def recommendations_page(
    request: Request,
    user_service: UserService = Depends(get_user_service),
    settings: Settings = Depends(get_settings_dep),
):
    users = user_service.list_recent(limit=15)
    return get_templates(request).TemplateResponse(
        request,
        "recommendations/user.html",
        _ctx(request, settings, users=users),
    )


@router.post("/recommendations", response_class=HTMLResponse)
def recommendations_submit(
    request: Request,
    user_id: int = Form(...),
    limit: int = Form(default=10),
    algorithm: str = Form(default="auto"),
    service: RecommendationService = Depends(get_recommendation_service),
    settings: Settings = Depends(get_settings_dep),
):
    result = service.recommend_for_user(user_id, limit=limit, algorithm=algorithm)
    return get_templates(request).TemplateResponse(
        request,
        "recommendations/_results.html",
        _ctx(request, settings, result=result),
    )


@router.get("/sentiment", response_class=HTMLResponse)
def sentiment_page(request: Request, settings: Settings = Depends(get_settings_dep)):
    return get_templates(request).TemplateResponse(request, "sentiment/index.html", _ctx(request, settings))


@router.post("/sentiment", response_class=HTMLResponse)
def sentiment_submit(
    request: Request,
    text: str = Form(...),
    service: SentimentService = Depends(get_sentiment_service),
    settings: Settings = Depends(get_settings_dep),
):
    prediction = service.predict(SentimentPredictRequest(text=text))
    return get_templates(request).TemplateResponse(
        request,
        "sentiment/_result.html",
        _ctx(request, settings, prediction=prediction, text=text),
    )


@router.get("/clustering", response_class=HTMLResponse)
def clustering_dashboard(
    request: Request,
    reports: ReportsService = Depends(get_reports_service),
    settings: Settings = Depends(get_settings_dep),
):
    ctx = reports.get_clustering_context()
    return get_templates(request).TemplateResponse(
        request,
        "clustering/dashboard.html",
        _ctx(
            request,
            settings,
            **ctx,
            chart_data_json=json.dumps(ctx.get("chart_data", {})),
        ),
    )


@router.get("/analytics", response_class=HTMLResponse)
def analytics_dashboard(
    request: Request,
    reports: ReportsService = Depends(get_reports_service),
    settings: Settings = Depends(get_settings_dep),
):
    ctx = reports.get_analytics_context()
    return get_templates(request).TemplateResponse(
        request,
        "analytics/dashboard.html",
        _ctx(
            request,
            settings,
            **ctx,
            chart_data_json=json.dumps(ctx.get("chart_data", {})),
        ),
    )


@router.post("/users/quick", response_class=HTMLResponse)
def quick_create_user(
    request: Request,
    external_id: str = Form(...),
    display_name: str | None = Form(default=None),
    service: UserService = Depends(get_user_service),
):
    user = service.get_or_create(UserCreate(external_id=external_id, display_name=display_name))
    return get_templates(request).TemplateResponse(
        request,
        "partials/user_created.html",
        {"request": request, "user": user},
    )
