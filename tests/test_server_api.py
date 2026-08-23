from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rag_core.attachments import AttachmentStore
from rag_core.db import Database
from rag_core.models import AnswerResult, Citation, Session, Source
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
        self.calls: list[tuple[str, str, bool]] = []

    def answer(
        self, user_message: str, session: Session, use_web: bool = False
    ) -> AnswerResult:
        self.calls.append((user_message, session.id, use_web))
        source = Source(
            document_id="DOC-001",
            document_title="Slides Cấu trúc dữ liệu",
            chapter="Chương 3: Bảng băm",
            course_code="CS101",
            kind="slides",
            language="vi",
        )
        citation = Citation(marker="[1]", source=source)
        return AnswerResult(
            answer=f"echo: {user_message}", citations=[citation], sources=[source]
        )


def make_client(
    tmp_path: Path,
) -> tuple[TestClient, Database, FakeCore, dict[str, dict[str, str]]]:
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
    client = TestClient(app)

    alice = client.post("/api/auth/register", json={"username": "alice", "password": "pw"}).json()
    bob = client.post("/api/auth/register", json={"username": "bob", "password": "pw"}).json()
    headers = {
        "alice": {"Authorization": f"Bearer {alice['token']}"},
        "bob": {"Authorization": f"Bearer {bob['token']}"},
    }
    return client, db, core, headers


def _create_conversation(client: TestClient, headers: dict[str, str]) -> dict[str, object]:
    res = client.post("/api/conversations", json={}, headers=headers)
    assert res.status_code == 201
    body: dict[str, object] = res.json()
    return body


def test_users_me_profile_update(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    patched = client.patch(
        "/api/users/me", json={"display_name": "Hieu", "language": "en"}, headers=h["alice"]
    )
    assert patched.status_code == 200
    assert patched.json()["display_name"] == "Hieu"
    assert patched.json()["language"] == "en"
    assert client.patch("/api/users/me", json={"language": "fr"}, headers=h["alice"]).status_code == 400


def test_conversation_crud(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = _create_conversation(client, h["alice"])
    assert conv["title"] == "New chat"

    listed = client.get("/api/conversations", headers=h["alice"]).json()
    assert [c["id"] for c in listed] == [conv["id"]]

    renamed = client.patch(
        f"/api/conversations/{conv['id']}", json={"title": "Bảng băm"}, headers=h["alice"]
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Bảng băm"

    assert client.patch(f"/api/conversations/{conv['id']}", json={"title": ""}, headers=h["alice"]).status_code == 400

    assert client.delete(f"/api/conversations/{conv['id']}", headers=h["alice"]).status_code == 204
    assert client.delete(f"/api/conversations/{conv['id']}", headers=h["alice"]).status_code == 404


def test_clear_all_conversations(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    _create_conversation(client, h["alice"])
    _create_conversation(client, h["alice"])
    assert client.delete("/api/conversations", headers=h["alice"]).status_code == 204
    assert client.get("/api/conversations", headers=h["alice"]).json() == []


def test_ownership_is_404(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = _create_conversation(client, h["alice"])
    assert (
        client.patch(f"/api/conversations/{conv['id']}", json={"title": "stolen"}, headers=h["bob"]).status_code
        == 404
    )
    assert client.delete(f"/api/conversations/{conv['id']}", headers=h["bob"]).status_code == 404
    assert client.get(f"/api/conversations/{conv['id']}/messages", headers=h["bob"]).status_code == 404


def test_chat_round_trip_persists(tmp_path: Path) -> None:
    client, _, core, h = make_client(tmp_path)
    conv = _create_conversation(client, h["alice"])

    reply = client.post(
        f"/api/conversations/{conv['id']}/chat",
        json={"message": "giải thích bảng băm"},
        headers=h["alice"],
    )
    assert reply.status_code == 200
    turn = reply.json()
    assert turn["role"] == "assistant"
    assert turn["text"] == "echo: giải thích bảng băm"
    assert turn["citations"][0]["marker"] == "[1]"
    assert turn["citations"][0]["source"]["document_title"] == "Slides Cấu trúc dữ liệu"
    assert turn["refused"] is False
    assert core.calls == [("giải thích bảng băm", conv["id"], False)]

    messages = client.get(f"/api/conversations/{conv['id']}/messages", headers=h["alice"]).json()
    assert [t["role"] for t in messages] == ["user", "assistant"]
    assert messages[0]["text"] == "giải thích bảng băm"


def test_chat_passes_use_web_flag(tmp_path: Path) -> None:
    client, _, core, h = make_client(tmp_path)
    core.has_web_search = True
    conv = _create_conversation(client, h["alice"])
    client.post(
        f"/api/conversations/{conv['id']}/chat",
        json={"message": "xin chào", "use_web": True},
        headers=h["alice"],
    )
    assert core.calls[-1][2] is True


def test_chat_rejects_blank_message(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = _create_conversation(client, h["alice"])
    blank = client.post(
        f"/api/conversations/{conv['id']}/chat", json={"message": "   "}, headers=h["alice"]
    )
    assert blank.status_code == 400


def test_features_endpoint(tmp_path: Path) -> None:
    client, _, core, _ = make_client(tmp_path)
    assert client.get("/api/features").json() == {"has_web_search": False}
    core.has_web_search = True
    assert client.get("/api/features").json() == {"has_web_search": True}
