"""User API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ORMModel


class UserRead(ORMModel):
    id: int
    external_id: str
    display_name: str | None = None
    nickname: str | None = None
    is_registered: bool = False
    cluster_id: int | None = None


class UserCreate(BaseModel):
    external_id: str = Field(min_length=1, max_length=64)
    display_name: str | None = Field(default=None, max_length=255)


class UserRegister(BaseModel):
    nickname: str = Field(min_length=2, max_length=32)

    @field_validator("nickname")
    @classmethod
    def normalize_nickname(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("nickname cannot be empty")
        if not cleaned.replace("_", "").replace("-", "").isalnum():
            raise ValueError("nickname: letters, digits, _ and - only")
        return cleaned


class UserLogin(BaseModel):
    nickname: str = Field(min_length=2, max_length=32)


class UserCandidateRead(UserRead):
    rating_count: int = 0
    cluster_label: str | None = None


class UserProfile(UserRead):
    rating_count: int = 0
    cluster_label: str | None = None


class UserBrowseResult(BaseModel):
    items: list[UserCandidateRead]
    total: int
    page: int
    page_size: int
    pages: int
