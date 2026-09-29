"""Stub-backed API server for the browser-free client integration tests.

``web/src/lib/api.integration.node.test.ts`` drives the real ``api.ts`` over
HTTP against this process, so routing, Pydantic validation, multipart
parsing and SQLite all run for real. Only the LLM pipeline is replaced by
a fixed answer, which keeps the suite offline and deterministic.

Run it directly when debugging a failing client test:

    uv run python -m tests.fake_api_server --port 8766
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import uvicorn
from fastapi import FastAPI

from rag_core import StreamDelta, StreamDone, StreamRefused, StreamStart
from rag_core.attachments import AttachmentStore
from rag_core.db import APP_DB_FILENAME, Database
from rag_core.models import (
    AnswerResult,
    Citation,
    Department,
    EntityBundle,
    Instructor,
    Program,
    Session,
    Source,
    Term,
)
from server.auth import TokenStore
from server.main import create_app
from server.state import AppState

DEFAULT_PORT = 8766

ANSWER = "Bảng băm là cấu trúc dữ liệu tra cứu khóa–giá trị."
REWRITE = "Hãy hỏi lại với từ khóa khác."


class StubEmbedder:
    """Deterministic unit vectors — ranking quality is irrelevant here."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class StubCore:
    """CoreLike stand-in: one canned grounded answer, no network calls."""

    has_web_search = False

    def __init__(self) -> None:
        self.embedder = StubEmbedder()
        self.source = Source(
            document_id="DOC-001",
            document_title="Slides Cấu trúc dữ liệu",
            chapter="Chương 3: Bảng băm",
            course_code="CS101",
            kind="slides",
            language="vi",
        )
        self.entities = EntityBundle(
            departments=(Department(id="DEP-CNTT", name="Khoa CNTT", name_en="CS"),),
            instructors=(
                Instructor(
                    id="INS-1",
                    name="TS. Nguyễn Văn A",
                    title="Tiến sĩ",
                    email="a@sv.edu.vn",
                    department_id="DEP-CNTT",
                    bio="",
                    courses=(("CS101", "giảng viên"),),
                ),
            ),
            programs=(
                Program(
                    id="PR-CNTT",
                    name="Kỹ sư CNTT",
                    name_en="CS Engineer",
                    department_id="DEP-CNTT",
                    total_credits=140,
                ),
            ),
            terms=(
                Term(
                    id="T-2025-S1",
                    name="Học kỳ 1 năm 2025",
                    year=2025,
                    season="1",
                    start_date="2025-09-01",
                    end_date="2025-12-31",
                ),
            ),
        )

    def answer(
        self, user_message: str, session: Session, use_web: bool = False
    ) -> AnswerResult:
        return self._result()

    def stream_answer(
        self, user_message: str, session: Session, use_web: bool = False
    ) -> Iterator[Any]:
        if "không rõ" in user_message:
            yield StreamRefused(rephrase_suggestion=REWRITE)
            return
        yield StreamStart(skills_applied=["eli5"], sources=[self.source])
        yield StreamDelta(delta=ANSWER[:10])
        yield StreamDelta(delta=ANSWER[10:])
        yield StreamDone(result=self._result())

    def _result(self) -> AnswerResult:
        return AnswerResult(
            answer=ANSWER,
            citations=[Citation(marker="[1]", source=self.source)],
            sources=[self.source],
            skills_applied=["eli5"],
        )


def build_stub_app(data_dir: Path) -> FastAPI:
    """The real app, with its embedding-heavy lifespan swapped for a stub."""
    app = create_app()
    core = StubCore()
    state = AppState(
        db=Database(data_dir / APP_DB_FILENAME),
        tokens=TokenStore(data_dir / APP_DB_FILENAME),
        core=core,
        attachments=AttachmentStore(data_dir / APP_DB_FILENAME, core.embedder),
        db_path=data_dir / APP_DB_FILENAME,
    )

    @contextlib.asynccontextmanager
    async def stub_lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _app.state.coursemate = state
        try:
            yield
        finally:
            _app.state.coursemate = None

    app.router.lifespan_context = stub_lifespan
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    with TemporaryDirectory() as tmp:
        uvicorn.run(build_stub_app(Path(tmp)), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
