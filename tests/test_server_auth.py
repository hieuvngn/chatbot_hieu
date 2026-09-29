from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient

from rag_core import StreamDone, StreamEvent, StreamStart
from rag_core.attachments import AttachmentStore
from rag_core.db import Database
from rag_core.models import AnswerResult, EntityBundle, Session
from server.auth import TokenStore
from server.main import create_app
from server.state import AppState


class FakeEmbedder:
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


class FakeCore:
    def __init__(self) -> None:
        self.has_web_search = False
        self.embedder = FakeEmbedder()

    def answer(
        self, user_message: str, session: Session, use_web: bool = False
    ) -> AnswerResult:
        raise AssertionError("not used in auth tests")

    @property
    def entities(self) -> EntityBundle:
        return EntityBundle(departments=(), instructors=(), programs=(), terms=())

    def stream_answer(
        self, user_message: str, session: Session, use_web: bool = False
    ) -> Iterator[StreamEvent]:
        yield StreamStart(skills_applied=[], sources=[])
        yield StreamDone(
            result=self.answer(user_message, session, use_web)
        )


def make_client(tmp_path: Path) -> tuple[TestClient, Database]:
    db_path = tmp_path / "app.db"
    db = Database(db_path)
    core = FakeCore()
    state = AppState(
        db=db,
        tokens=TokenStore(db_path),
        core=core,
        attachments=AttachmentStore(db_path, core.embedder),
        db_path=db_path,
    )
    app = create_app()
    app.state.coursemate = state
    return TestClient(app), db


def test_register_login_me(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    registered = client.post("/api/auth/register", json={"username": "hieu", "password": "pw"})
    assert registered.status_code == 201
    body = registered.json()
    assert body["token"]
    assert body["user"]["username"] == "hieu"

    logged_in = client.post("/api/auth/login", json={"username": "hieu", "password": "pw"})
    assert logged_in.status_code == 200
    token = logged_in.json()["token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "hieu"


def test_register_validation_and_duplicates(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    assert client.post("/api/auth/register", json={"username": "  ", "password": "pw"}).status_code == 400
    assert client.post("/api/auth/register", json={"username": "a", "password": " "}).status_code == 400
    client.post("/api/auth/register", json={"username": "a", "password": "pw"})
    dup = client.post("/api/auth/register", json={"username": "a", "password": "other"})
    assert dup.status_code == 400


def test_login_wrong_password(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    client.post("/api/auth/register", json={"username": "a", "password": "pw"})
    bad = client.post("/api/auth/login", json={"username": "a", "password": "nope"})
    assert bad.status_code == 401


def test_me_requires_token(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer deadbeef"}).status_code == 401
