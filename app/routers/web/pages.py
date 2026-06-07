"""Jinja2 + HTMX web UI."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Query, Request
from fastapi.responses import HTMLResponse

from app.config import Settings
from app.dependencies import (
    get_book_service,
    get_recommendation_service,
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
from app.services.sentiment_service import SentimentService
from app.services.user_service import UserService

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
):
    templates = get_templates()
    return templates.TemplateResponse(
        request,
        "index.html",
        {"app_name": settings.app_name, "version": settings.app_version},
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
    templates = get_templates()
    result = service.search(BookSearchParams(q=q, genre=genre, page=page, page_size=12))
    template = "books/_search_results.html" if request.headers.get("HX-Request") else "books/list.html"
    return templates.TemplateResponse(
        request,
        template,
        {
            "app_name": settings.app_name,
            "result": result,
            "q": q or "",
            "genre": genre or "",
        },
    )


@router.get("/books/{book_id}", response_class=HTMLResponse)
def book_detail(
    request: Request,
    book_id: int,
    book_service: BookService = Depends(get_book_service),
    rec_service: RecommendationService = Depends(get_recommendation_service),
    settings: Settings = Depends(get_settings_dep),
):
    templates = get_templates()
    book = book_service.get(book_id)
    similar = rec_service.similar_books(book_id, limit=6) if book else None
    return templates.TemplateResponse(
        request,
        "books/detail.html",
        {"app_name": settings.app_name, "book": book, "similar": similar},
    )


@router.get("/recommendations", response_class=HTMLResponse)
def recommendations_page(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
):
    templates = get_templates()
    return templates.TemplateResponse(
        request,
        "recommendations/user.html",
        {"app_name": settings.app_name},
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
    templates = get_templates()
    result = service.recommend_for_user(user_id, limit=limit, algorithm=algorithm)
    return templates.TemplateResponse(
        request,
        "recommendations/_results.html",
        {"app_name": settings.app_name, "result": result},
    )


@router.get("/sentiment", response_class=HTMLResponse)
def sentiment_page(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
):
    templates = get_templates()
    return templates.TemplateResponse(
        request,
        "sentiment/index.html",
        {"app_name": settings.app_name},
    )


@router.post("/sentiment", response_class=HTMLResponse)
def sentiment_submit(
    request: Request,
    text: str = Form(...),
    service: SentimentService = Depends(get_sentiment_service),
    settings: Settings = Depends(get_settings_dep),
):
    templates = get_templates()
    prediction = service.predict(SentimentPredictRequest(text=text))
    return templates.TemplateResponse(
        request,
        "sentiment/_result.html",
        {"app_name": settings.app_name, "prediction": prediction, "text": text},
    )


@router.post("/users/quick", response_class=HTMLResponse)
def quick_create_user(
    request: Request,
    external_id: str = Form(...),
    display_name: str | None = Form(default=None),
    service: UserService = Depends(get_user_service),
):
    user = service.get_or_create(UserCreate(external_id=external_id, display_name=display_name))
    templates = get_templates()
    return templates.TemplateResponse(
        request,
        "partials/user_created.html",
        {"user": user},
    )
