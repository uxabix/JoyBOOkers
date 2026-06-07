"""Jinja2 template engine registration and custom filters."""

from __future__ import annotations

from functools import lru_cache

from fastapi.templating import Jinja2Templates

from app.config import Settings, get_settings


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def _truncate_words(text: str | None, count: int = 30) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= count:
        return text
    return " ".join(words[:count]) + "…"


def build_templates_engine(settings: Settings) -> Jinja2Templates:
    """Create a Jinja2 environment with project filters (not cached — Settings is mutable)."""
    templates = Jinja2Templates(directory=str(settings.templates_dir))
    env = templates.env
    env.filters["pct"] = _pct
    env.filters["truncate_words"] = _truncate_words
    env.globals["app_name"] = settings.app_name
    env.globals["app_version"] = settings.app_version
    return templates


@lru_cache
def get_templates_engine() -> Jinja2Templates:
    """Default cached templates using application settings."""
    return build_templates_engine(get_settings())


def clear_templates_cache() -> None:
    get_templates_engine.cache_clear()
