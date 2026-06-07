"""Author browse pages."""

from __future__ import annotations

from urllib.parse import unquote

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse

from app.config import Settings
from app.dependencies import get_author_service, get_settings_dep, get_templates
from app.routers.web.pages import _book_search_params, _ctx
from app.routers.web.query_strings import author_path, author_query, book_query
from app.schemas.author import AuthorBrowseParams
from app.schemas.book import BookSearchParams
from app.services.author_service import AuthorService

router = APIRouter()


@router.get("/authors", response_class=HTMLResponse)
def authors_list(
    request: Request,
    q: str | None = Query(default=None),
    sort: str = Query(default="book_count_desc"),
    min_books: int = Query(default=1, ge=1),
    page: int = Query(default=1, ge=1),
    service: AuthorService = Depends(get_author_service),
    settings: Settings = Depends(get_settings_dep),
):
    params = AuthorBrowseParams(q=q, sort=sort, min_books=min_books, page=page, page_size=24)
    result = service.browse(params)
    template = "authors/_list_results.html" if request.headers.get("HX-Request") else "authors/list.html"
    return get_templates(request).TemplateResponse(
        request,
        template,
        _ctx(request, settings, result=result, author_params=params, author_query=author_query(params)),
    )


@router.get("/authors/by/{author_name:path}", response_class=HTMLResponse)
def author_detail(
    request: Request,
    author_name: str,
    service: AuthorService = Depends(get_author_service),
    settings: Settings = Depends(get_settings_dep),
    book_params: BookSearchParams = Depends(_book_search_params),
):
    name = unquote(author_name)
    books = service.get_books(name, book_params)
    partial = request.headers.get("HX-Request")
    base_url = author_path(name)
    ctx = _ctx(
        request,
        settings,
        author_name=name,
        result=books,
        book_params=book_params,
        book_query=book_query(book_params),
        base_url=base_url,
    )
    if partial:
        return get_templates(request).TemplateResponse(request, "books/_search_results.html", ctx)
    return get_templates(request).TemplateResponse(request, "authors/detail.html", ctx)
