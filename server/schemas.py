"""Pydantic request/response models mirroring rag_core domain objects."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    display_name: str
    language: str


class TokenOut(BaseModel):
    token: str
    user: UserOut


class RegisterIn(BaseModel):
    username: str
    password: str
