"""User API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class UserRead(ORMModel):
    id: int
    external_id: str
    display_name: str | None = None


class UserCandidateRead(UserRead):
    rating_count: int = 0


class UserCreate(BaseModel):
    external_id: str = Field(min_length=1, max_length=64)
    display_name: str | None = Field(default=None, max_length=255)
