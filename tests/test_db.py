from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag_core.db import Database
from rag_core.models import Citation, Source, Turn


def make_db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


def test_register_creates_user(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    user = db.register("hieu", "pw")
    assert user.username == "hieu"
    assert user.id > 0


def test_duplicate_username_rejected(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.register("hieu", "pw")
    with pytest.raises(ValueError):
        db.register("hieu", "other")


def test_login_with_correct_password_returns_user(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.register("hieu", "pw")
    user = db.login("hieu", "pw")
    assert user is not None
    assert user.username == "hieu"


def test_login_with_wrong_password_returns_none(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    db.register("hieu", "pw")
    assert db.login("hieu", "wrong") is None


def test_login_unknown_user_returns_none(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    assert db.login("nobody", "pw") is None


def test_session_is_stable_per_user(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    user = db.register("hieu", "pw")
    first = db.get_or_create_session(user.id)
    second = db.get_or_create_session(user.id)
    assert first.id == second.id
    assert first.user_id == str(user.id)


def test_session_differs_between_users(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    alice = db.register("alice", "pw")
    bob = db.register("bob", "pw")
    assert db.get_or_create_session(alice.id).id != db.get_or_create_session(bob.id).id


def test_session_returns_last_six_turns_in_order(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    user = db.register("hieu", "pw")
    session = db.get_or_create_session(user.id)
    for i in range(4):
        db.append_exchange(session.id, f"turn {2 * i}", f"turn {2 * i + 1}")
    resumed = db.get_session(session.id)
    assert len(resumed.turns) == 6
    assert [t.text for t in resumed.turns] == [f"turn {i}" for i in range(2, 8)]
    assert [t.role for t in resumed.turns] == [
        "user", "assistant", "user", "assistant", "user", "assistant",
    ]


def test_session_persists_across_database_instances(tmp_path: Path) -> None:
    path = tmp_path / "test.db"
    db = Database(path)
    user = db.register("hieu", "pw")
    session = db.get_or_create_session(user.id)
    db.append_exchange(session.id, "hello", "world")
    db.close()

    reopened = Database(path)
    logged_in = reopened.login("hieu", "pw")
    assert logged_in is not None
    resumed = reopened.get_or_create_session(logged_in.id)
    assert resumed.id == session.id
    assert [t.text for t in resumed.turns] == ["hello", "world"]


def test_get_session_unknown_id_raises(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    with pytest.raises(KeyError):
        db.get_session("nope")


def make_citation(marker: str) -> Citation:
    return Citation(
        marker=marker,
        source=Source(
            document_id="DOC-001",
            document_title="Bảng băm",
            chapter="Chương 1",
            course_code="CS101",
            kind="textbook",
            language="vi",
        ),
    )


def test_citations_persist_with_exchange(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    user = db.register("hieu", "pw")
    session = db.get_or_create_session(user.id)
    db.append_exchange(
        session.id,
        "giải thích bảng băm",
        "Bảng băm là một cấu trúc dữ liệu [1].",
        citations=[make_citation("1"), make_citation("2")],
    )
    resumed = db.get_session(session.id)
    assistant = resumed.turns[1]
    assert [c.marker for c in assistant.citations] == ["1", "2"]
    assert assistant.citations[0].source.document_id == "DOC-001"
    assert assistant.citations[0].source.chapter == "Chương 1"


def test_refusal_persists_with_exchange(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    user = db.register("hieu", "pw")
    session = db.get_or_create_session(user.id)
    db.append_exchange(
        session.id,
        "hỏi gì đó",
        "",
        refused=True,
        rephrase_suggestion="Hãy thử hỏi lại với từ khóa cụ thể hơn.",
    )
    resumed = db.get_session(session.id)
    assistant = resumed.turns[1]
    assert assistant.refused is True
    assert assistant.rephrase_suggestion == "Hãy thử hỏi lại với từ khóa cụ thể hơn."
    assert assistant.citations == []


def test_exchange_without_metadata_restores_plain_turn(tmp_path: Path) -> None:
    db = make_db(tmp_path)
    user = db.register("hieu", "pw")
    session = db.get_or_create_session(user.id)
    db.append_exchange(session.id, "hello", "world")
    resumed = db.get_session(session.id)
    assistant = resumed.turns[1]
    assert assistant.refused is False
    assert assistant.rephrase_suggestion == ""
    assert assistant.citations == []


def test_old_schema_database_migrates_in_place(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE conversations (
            id TEXT PRIMARY KEY,
            user_id INTEGER UNIQUE NOT NULL REFERENCES users(id),
            created_at TEXT NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id),
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO users (id, username, password, created_at) VALUES (1, 'hieu', 'pw', 'now')"
    )
    conn.execute(
        "INSERT INTO conversations (id, user_id, created_at) VALUES ('c1', 1, 'now')"
    )
    conn.execute(
        "INSERT INTO messages (conversation_id, role, text, created_at) "
        "VALUES ('c1', 'user', 'hello', 'now')"
    )
    conn.commit()
    conn.close()

    db = Database(path)
    session = db.get_session("c1")
    assert [t.text for t in session.turns] == ["hello"]
    db.append_exchange(
        session.id,
        "follow-up",
        "reply [1]",
        citations=[make_citation("1")],
        refused=True,
        rephrase_suggestion="Try again.",
    )
    resumed = db.get_session(session.id)
    assert len(resumed.turns) == 3
    assert resumed.turns[2].citations[0].marker == "1"
    assert resumed.turns[2].refused is True
    assert resumed.turns[2].rephrase_suggestion == "Try again."