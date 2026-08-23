"""FastAPI application factory and entrypoint."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server.routes import router

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


class SPAStaticFiles(StaticFiles):
    """StaticFiles that falls back to index.html for client-side routes."""

    async def get_response(
        self, path: str, scope: MutableMapping[str, Any]
    ) -> Any:
        response = await super().get_response(path, scope)
        if response.status_code == 404:
            return await super().get_response("index.html", scope)
        return response


def create_app() -> FastAPI:
    app = FastAPI(title="CourseMate API")
    app.include_router(router, prefix="/api")
    if WEB_DIST.is_dir():
        app.mount("/", SPAStaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
