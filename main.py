"""Legacy ASGI entrypoint — prefer: uvicorn app.main:app --reload"""

from app.main import app

__all__ = ["app"]
