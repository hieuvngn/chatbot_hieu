"""Shared application state wired from rag_core, injectable for tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import Request

from rag_core import RagCore, build_rag_core
from rag_core.attachments import AttachmentStore
from rag_core.config import Config, load_config
from rag_core.db import APP_DB_FILENAME, Database
from rag_core.embeddings import Embedder
from rag_core.models import AnswerResult, Session
from server.auth import TokenStore


class CoreLike(Protocol):
    """Structural subset of ``RagCore`` the API relies on."""

    @property
    def has_web_search(self) -> bool: ...

    @property
    def embedder(self) -> Embedder: ...

    def answer(
        self, user_message: str, session: Session, use_web: bool = False
    ) -> AnswerResult: ...


@dataclass
class AppState:
    db: Database
    tokens: TokenStore
    core: CoreLike
    attachments: AttachmentStore
    db_path: Path


def build_state(config: Config | None = None) -> AppState:
    resolved = config if config is not None else load_config()
    db_path = resolved.data_dir / APP_DB_FILENAME
    core: CoreLike = build_rag_core()
    return AppState(
        db=Database(db_path),
        tokens=TokenStore(db_path),
        core=core,
        attachments=AttachmentStore(db_path, core.embedder),
        db_path=db_path,
    )


def get_state(request: Request) -> AppState:
    """FastAPI dependency: returns per-app state, building it lazily."""
    state = getattr(request.app.state, "coursemate", None)
    if state is None:
        state = build_state()
        request.app.state.coursemate = state
    return state
