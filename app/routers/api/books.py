"""Book catalog API — DS2 + DS3."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import get_book_service, get_recommendation_service
from app.schemas.book import BookCreate, BookRead, BookSearchParams
from app.schemas.common import PaginatedResponse
from app.schemas.recommendation import RecommendationResponse, SimilarBooksRequest
from app.services.book_service import BookService
from app.services.recommendation_service import RecommendationService

router = APIRouter()


@router.get("", response_model=PaginatedResponse[BookRead])
def search_books(
    q: str | None = Query(default=None),
    genre: str | None = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    service: BookService = Depends(get_book_service),
) -> PaginatedResponse[BookRead]:
    params = BookSearchParams(q=q, genre=genre, page=page, page_size=page_size)
    return service.search(params)


@router.get("/{book_id}", response_model=BookRead)
def get_book(
    book_id: int,
    service: BookService = Depends(get_book_service),
) -> BookRead:
    book = service.get(book_id)
    if book is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Book not found")
    return book


@router.post("", response_model=BookRead, status_code=status.HTTP_201_CREATED)
def create_book(
    payload: BookCreate,
    service: BookService = Depends(get_book_service),
) -> BookRead:
    return service.create(payload)


@router.get("/{book_id}/similar", response_model=RecommendationResponse)
def similar_books(
    book_id: int,
    limit: int = Query(default=10, ge=1, le=50),
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    req = SimilarBooksRequest(book_id=book_id, limit=limit)
    return service.similar_books(req.book_id, limit=req.limit)
