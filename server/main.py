"""FastAPI application factory and entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator, MutableMapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from server.routes import router

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


class SPAStaticFiles(StaticFiles):
    """StaticFiles that falls back to index.html for client-side routes."""

    async def get_response(
        self, path: str, scope: MutableMapping[str, Any]
    ) -> Any:
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            return await super().get_response("index.html", scope)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Pre-warm AppState at startup so the first request (login) is fast.

    ``build_state()`` runs ``build_rag_core()`` which embeds the entire KB in a
    single batched call (~minutes on cold start). Without this hook, the first
    authenticated request pays the full cost. ``get_state`` keeps its lazy
    fallback for tests that inject state directly.
    """
    from server.state import build_state

    app.state.coursemate = build_state()
    try:
        yield
    finally:
        app.state.coursemate = None


def create_app() -> FastAPI:
    app = FastAPI(title="CourseMate API", lifespan=_lifespan)
    app.include_router(router, prefix="/api")
    if WEB_DIST.is_dir():
        app.mount("/", SPAStaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
