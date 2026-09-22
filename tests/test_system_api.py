"""System Test Case — API-level end-to-end tests.

These tests exercise the full stack from HTTP → FastAPI → Database → RagCore
pipeline with **real components** (no FakeCore / fake generator). They are
marked ``system`` and skipped by default (``pytest -m "not system"``) because
they require an OpenRouter API key and take several minutes per test.

Run with real LLM:
    OPENROUTER_API_KEY=... uv run pytest tests/test_system_api.py -m system

Skip (no key, no network):
    uv run pytest tests/test_system_api.py  # all skipped
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_core import build_rag_core
from rag_core.attachments import AttachmentStore
from rag_core.config import load_config
from rag_core.db import Database
from rag_core.embeddings import Embedder
from rag_core.models import Session
from server.auth import TokenStore
from server.main import create_app
from server.state import AppState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_real_client(
    tmp_path: Path,
) -> tuple[TestClient, Database, Any]:
    """Build a FastAPI TestClient wired to the real RagCore pipeline."""
    config = load_config()
    core = build_rag_core(config)
    db_path = tmp_path / "app.db"
    db = Database(db_path)
    attachments = AttachmentStore(db_path, core.embedder)
    state = AppState(
        db=db,
        tokens=TokenStore(db_path),
        core=core,
        attachments=attachments,
        db_path=db_path,
    )
    app = create_app()
    app.state.coursemate = state
    return TestClient(app), db, core


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _create_conversation(client: TestClient, headers: dict[str, str]) -> str:
    res = client.post("/api/conversations", json={}, headers=headers)
    assert res.status_code == 201
    return res.json()["id"]


def _register_login(client: TestClient, username: str, password: str) -> str:
    reg = client.post(
        "/api/auth/register", json={"username": username, "password": password}
    )
    if reg.status_code == 201:
        return reg.json()["token"]
    # User may already exist from a previous test — try login.
    login = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert login.status_code == 200, f"register/login failed: {login.json()}"
    return login.json()["token"]


# ---------------------------------------------------------------------------
# Module-level skip / system marker
# ---------------------------------------------------------------------------

REASON_NO_KEY = "OPENROUTER_API_KEY not set — system tests require real LLM"

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason=REASON_NO_KEY,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAuthE2E:
    """Register → login → me flow through real HTTP layer."""

    def test_register_and_login_round_trip(self, tmp_path: Path) -> None:
        client, _, _ = _make_real_client(tmp_path)
        token = _register_login(client, "sys_e2e_user", "sys-pass-123")
        assert token

        headers = _auth_header(token)
        me = client.get("/api/auth/me", headers=headers)
        assert me.status_code == 200
        body = me.json()
        assert body["username"] == "sys_e2e_user"
        assert body["id"] > 0

    def test_register_duplicate_returns_400(self, tmp_path: Path) -> None:
        client, _, _ = _make_real_client(tmp_path)
        _register_login(client, "sys_dup", "pass")
        dup = client.post(
            "/api/auth/register", json={"username": "sys_dup", "password": "pass"}
        )
        assert dup.status_code == 400

    def test_login_wrong_password_returns_401(self, tmp_path: Path) -> None:
        client, _, _ = _make_real_client(tmp_path)
        _register_login(client, "sys_pw", "correct")
        bad = client.post(
            "/api/auth/login", json={"username": "sys_pw", "password": "wrong"}
        )
        assert bad.status_code == 401


class TestChatE2E:
    """Chat flow: create conversation → send message → receive answer with citations."""

    def test_answer_contains_citations(self, tmp_path: Path) -> None:
        """Full pipeline: question → retrieval → judge → generate → citations."""
        client, db, _ = _make_real_client(tmp_path)
        token = _register_login(client, "sys_chat_a", "pass")
        headers = _auth_header(token)
        conv_id = _create_conversation(client, headers)

        response = client.post(
            f"/api/conversations/{conv_id}/chat",
            json={"message": "giải thích bảng băm là gì?"},
            headers=headers,
        )
        assert response.status_code == 200
        turn = response.json()
        assert turn["role"] == "assistant"
        assert turn["text"], "answer text should not be empty"
        # The answer should contain inline citation markers like [1].
        assert any(m in turn["text"] for m in ["[1]", "[2]", "[3]", "[4]", "[5]"]), (
            f"answer should cite sources: {turn['text'][:200]}"
        )

    def test_citations_point_at_valid_sources(self, tmp_path: Path) -> None:
        """Citations in the answer reference real document + chapter."""
        client, db, _ = _make_real_client(tmp_path)
        token = _register_login(client, "sys_chat_b", "pass")
        headers = _auth_header(token)
        conv_id = _create_conversation(client, headers)

        response = client.post(
            f"/api/conversations/{conv_id}/chat",
            json={"message": "cấu trúc dữ liệu là gì?"},
            headers=headers,
        )
        turn = response.json()
        for citation in turn.get("citations", []):
            src = citation["source"]
            assert src["document_id"], f"citation missing document_id: {src}"
            assert src["chapter"], f"citation missing chapter: {src}"
            assert src["course_code"], f"citation missing course_code: {src}"

    def test_chat_persists_across_database_reopen(self, tmp_path: Path) -> None:
        """Messages survive a database reopen (real persistence)."""
        client, db, _ = _make_real_client(tmp_path)
        token = _register_login(client, "sys_chat_c", "pass")
        headers = _auth_header(token)
        conv_id = _create_conversation(client, headers)

        client.post(
            f"/api/conversations/{conv_id}/chat",
            json={"message": "câu hỏi đầu tiên"},
            headers=headers,
        )
        client.post(
            f"/api/conversations/{conv_id}/chat",
            json={"message": "câu hỏi thứ hai"},
            headers=headers,
        )

        # Reopen database from same file.
        db2 = Database(tmp_path / "app.db")
        session = db2.get_session(conv_id)
        # Each chat exchange persists a user turn + an assistant turn.
        assert [t.role for t in session.turns] == ["user", "assistant", "user", "assistant"]
        assert session.turns[0].text == "câu hỏi đầu tiên"
        assert session.turns[2].text == "câu hỏi thứ hai"
        assert session.turns[1].text and session.turns[3].text

    def test_refusal_has_rephrase_suggestion(self, tmp_path: Path) -> None:
        """When the pipeline refuses, rephrase_suggestion is non-empty."""
        client, db, _ = _make_real_client(tmp_path)
        token = _register_login(client, "sys_chat_d", "pass")
        headers = _auth_header(token)
        conv_id = _create_conversation(client, headers)

        # A deliberately unanswerable question to trigger refusal.
        response = client.post(
            f"/api/conversations/{conv_id}/chat",
            json={"message": "zzzznotamenuzzzzzzzzzz"},
            headers=headers,
        )
        turn = response.json()
        if turn.get("refused"):
            assert turn.get("rephrase_suggestion"), "refusal must include rephrase suggestion"


class TestChatE2EStreaming:
    """NDJSON streaming endpoint end-to-end."""

    def test_stream_produces_events(self, tmp_path: Path) -> None:
        client, db, _ = _make_real_client(tmp_path)
        token = _register_login(client, "sys_stream_a", "pass")
        headers = _auth_header(token)
        conv_id = _create_conversation(client, headers)

        response = client.post(
            f"/api/conversations/{conv_id}/chat/stream",
            json={"message": "giải thích bảng băm là gì?"},
            headers=headers,
        )
        assert response.status_code == 200
        lines = response.text.strip().split("\n")
        assert len(lines) >= 2, "stream should have at least start + done events"

        events = [json.loads(line) for line in lines]
        assert "event" in events[0], f"first event: {events[0]}"
        assert events[-1]["event"] in {"done", "refused"}, (
            f"last event should be done/refused: {events[-1]}"
        )


class TestChatE2EWebSearch:
    """Web search toggle merges web results into retrieval."""

    def test_use_web_false_no_web_sources(self, tmp_path: Path) -> None:
        client, db, core = _make_real_client(tmp_path)
        token = _register_login(client, "sys_web_a", "pass")
        headers = _auth_header(token)
        conv_id = _create_conversation(client, headers)

        response = client.post(
            f"/api/conversations/{conv_id}/chat",
            json={"message": "giải thích bảng băm là gì?", "use_web": False},
            headers=headers,
        )
        turn = response.json()
        for citation in turn.get("citations", []):
            assert citation["source"]["kind"] != "web", (
                "use_web=false should not return web citations"
            )
        assert isinstance(core.has_web_search, bool)


class TestConversationCRUDE2E:
    """Conversation management through real HTTP endpoints."""

    def test_create_list_rename_delete(self, tmp_path: Path) -> None:
        client, _, _ = _make_real_client(tmp_path)
        token = _register_login(client, "sys_crud", "pass")
        headers = _auth_header(token)

        # Create
        c1 = client.post("/api/conversations", json={}, headers=headers)
        assert c1.status_code == 201
        c2 = client.post("/api/conversations", json={}, headers=headers)
        assert c2.status_code == 201
        assert c1.json()["id"] != c2.json()["id"]

        # List
        conversations = client.get("/api/conversations", headers=headers)
        assert conversations.status_code == 200
        assert len(conversations.json()) >= 2

        # Rename
        cid = c1.json()["id"]
        renamed = client.patch(
            f"/api/conversations/{cid}",
            json={"title": "Renamed Chat"},
            headers=headers,
        )
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Renamed Chat"

        # Delete
        del_res = client.delete(f"/api/conversations/{cid}", headers=headers)
        assert del_res.status_code == 204
        after = client.get("/api/conversations", headers=headers)
        assert cid not in [c["id"] for c in after.json()]

    def test_conversation_ownership_enforced(self, tmp_path: Path) -> None:
        client, _, _ = _make_real_client(tmp_path)
        alice_token = _register_login(client, "sys_owner_alice", "pass")
        bob_token = _register_login(client, "sys_owner_bob", "pass")

        alice_headers = _auth_header(alice_token)
        bob_headers = _auth_header(bob_token)

        conv = client.post("/api/conversations", json={}, headers=alice_headers)
        conv_id = conv.json()["id"]

        # Bob cannot access Alice's conversation.
        res = client.get(
            f"/api/conversations/{conv_id}/messages", headers=bob_headers
        )
        assert res.status_code == 404


class TestProfileE2E:
    """User profile update through real HTTP endpoints."""

    def test_update_display_name_and_language(self, tmp_path: Path) -> None:
        client, _, _ = _make_real_client(tmp_path)
        token = _register_login(client, "sys_profile", "pass")
        headers = _auth_header(token)

        updated = client.patch(
            "/api/users/me",
            json={"display_name": "Hieu System", "language": "en"},
            headers=headers,
        )
        assert updated.status_code == 200
        assert updated.json()["display_name"] == "Hieu System"
        assert updated.json()["language"] == "en"

    def test_profile_validation(self, tmp_path: Path) -> None:
        client, _, _ = _make_real_client(tmp_path)
        token = _register_login(client, "sys_profile_v", "pass")
        headers = _auth_header(token)

        bad = client.patch(
            "/api/users/me", json={"language": "xx"}, headers=headers
        )
        assert bad.status_code == 400


class TestFeaturesE2E:
    """Feature flags endpoint."""

    def test_features_endpoint(self, tmp_path: Path) -> None:
        client, _, _ = _make_real_client(tmp_path)
        res = client.get("/api/features")
        assert res.status_code == 200
        body = res.json()
        assert "has_web_search" in body
        assert isinstance(body["has_web_search"], bool)
