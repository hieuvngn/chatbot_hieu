"""FastAPI application factory and entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from server.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="CourseMate API")
    app.include_router(router, prefix="/api")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
