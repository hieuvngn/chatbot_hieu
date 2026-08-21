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


def test_register_sets_display_name_and_language(tmp_path: Path) -> None:
    from rag_core.db import Database

    db = Database(tmp_path / "test.db")
    user = db.register("hieu", "pw")
    assert user.display_name == "hieu"
    assert user.language == "vi"
    # login also returns new fields
    logged = db.login("hieu", "pw")
    assert logged is not None
    assert logged.display_name == "hieu"
    assert logged.language == "vi"


def test_get_user_returns_profile(tmp_path: Path) -> None:
    from rag_core.db import Database

    db = Database(tmp_path / "test.db")
    u = db.register("alice", "pw")
    fetched = db.get_user(u.id)
    assert fetched is not None
    assert fetched.username == "alice"
    assert fetched.display_name == "alice"


def test_legacy_db_migrates_users_and_conversations(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.executescript("""
    CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, created_at TEXT NOT NULL);
    CREATE TABLE conversations (id TEXT PRIMARY KEY, user_id INTEGER UNIQUE NOT NULL REFERENCES users(id), created_at TEXT NOT NULL);
    CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL REFERENCES conversations(id), role TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL);
    """)
    conn.execute("INSERT INTO users (id, username, password, created_at) VALUES (1, 'hieu', 'pw', 'now')")
    conn.execute("INSERT INTO conversations (id, user_id, created_at) VALUES ('c1', 1, 'now')")
    conn.commit()
    conn.close()
    from rag_core.db import Database

    db = Database(path)
    u = db.login("hieu", "pw")
    assert u is not None and u.display_name == "hieu" and u.language == "vi"
    # after migrate, should allow second conversation
    s2 = db.create_conversation(u.id, title="Second")
    assert s2.id != "c1"
    assert len(db.list_conversations(u.id)) == 2


def test_create_multiple_conversations_per_user(tmp_path: Path) -> None:
    from rag_core.db import Database

    db = Database(tmp_path / "test.db")
    u = db.register("alice", "pw")
    c1 = db.create_conversation(u.id, title="Chat 1")
    c2 = db.create_conversation(u.id, title="Chat 2")
    assert c1.id != c2.id
    metas = db.list_conversations(u.id)
    assert len(metas) == 2
    assert {m.title for m in metas} == {"Chat 1", "Chat 2"}


def test_list_conversations_ordered_by_updated_at(tmp_path: Path) -> None:
    from rag_core.db import Database
    import time

    db = Database(tmp_path / "test.db")
    u = db.register("bob", "pw")
    c1 = db.create_conversation(u.id, title="Old")
    time.sleep(0.01)
    c2 = db.create_conversation(u.id, title="New")
    # New should be first (DESC)
    metas = db.list_conversations(u.id)
    assert metas[0].id == c2.id
    # after appending to Old, it becomes first
    db.append_exchange(c1.id, "hi", "hello")
    metas2 = db.list_conversations(u.id)
    assert metas2[0].id == c1.id


def test_append_exchange_auto_titles_first_message(tmp_path: Path) -> None:
    from rag_core.db import Database

    db = Database(tmp_path / "test.db")
    u = db.register("hieu", "pw")
    s = db.create_conversation(u.id)  # default "New chat"
    assert db.list_conversations(u.id)[0].title == "New chat"
    db.append_exchange(s.id, "giải thích bảng băm là gì?", "answer")
    assert db.list_conversations(u.id)[0].title == "giải thích bảng băm là gì?"
    # second exchange should NOT overwrite title
    db.append_exchange(s.id, "câu 2", "ans2")
    assert db.list_conversations(u.id)[0].title == "giải thích bảng băm là gì?"


def test_append_truncates_title_40(tmp_path: Path) -> None:
    from rag_core.db import Database

    db = Database(tmp_path / "test.db")
    u = db.register("hieu", "pw")
    s = db.create_conversation(u.id)
    long_text = "a" * 100
    db.append_exchange(s.id, long_text, "ans")
    assert len(db.list_conversations(u.id)[0].title) == 40


def test_update_user_profile(tmp_path: Path) -> None:
    from rag_core.db import Database

    db = Database(tmp_path / "test.db")
    u = db.register("hieu", "pw")
    updated = db.update_user_profile(u.id, display_name="Hiếu Nguyễn", language="en")
    assert updated.display_name == "Hiếu Nguyễn"
    assert updated.language == "en"
    # persist
    assert db.login("hieu", "pw").language == "en"


def test_update_user_profile_validation(tmp_path: Path) -> None:
    from rag_core.db import Database
    import pytest

    db = Database(tmp_path / "test.db")
    u = db.register("hieu", "pw")
    with pytest.raises(ValueError):
        db.update_user_profile(u.id, display_name="")
    with pytest.raises(ValueError):
        db.update_user_profile(u.id, display_name="a" * 51)
    with pytest.raises(ValueError):
        db.update_user_profile(u.id, language="fr")


def test_delete_and_clear(tmp_path: Path) -> None:
    from rag_core.db import Database

    db = Database(tmp_path / "test.db")
    u = db.register("hieu", "pw")
    c1 = db.create_conversation(u.id, title="A")
    c2 = db.create_conversation(u.id, title="B")
    db.append_exchange(c1.id, "hi", "hello")
    db.delete_conversation(c1.id)
    assert len(db.list_conversations(u.id)) == 1
    assert db.list_conversations(u.id)[0].id == c2.id
    db.clear_all_conversations(u.id)
    assert db.list_conversations(u.id) == []


def test_get_or_create_session_backward_compat(tmp_path: Path) -> None:
    from rag_core.db import Database

    db = Database(tmp_path / "test.db")
    u = db.register("hieu", "pw")
    s1 = db.get_or_create_session(u.id)
    s2 = db.get_or_create_session(u.id)
    assert s1.id == s2.id
    # after creating extra, wrapper returns most recent (first in DESC)
    s3 = db.create_conversation(u.id, title="Extra")
    s4 = db.get_or_create_session(u.id)
    assert s4.id == s3.id  # most recent


def _attachment_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return int(conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0])
    finally:
        conn.close()


def _insert_attachment_row(db_path: Path, conv_id: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO attachments (id, conversation_id, filename, file_kind, "
            "size_bytes, chunk_count, language, created_at) "
            "VALUES ('a1', ?, 'note.txt', 'txt', 10, 1, 'vi', '2026-01-01')",
            (conv_id,),
        )
        conn.commit()
    finally:
        conn.close()


def test_delete_conversation_cascades_attachments(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    user = db.register("hieu", "pw")
    conv = db.create_conversation(user.id)
    _insert_attachment_row(db_path, conv.id)
    db.delete_conversation(conv.id)
    assert _attachment_count(db_path) == 0


def test_clear_all_cascades_attachments(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = Database(db_path)
    user = db.register("hieu", "pw")
    conv = db.create_conversation(user.id)
    _insert_attachment_row(db_path, conv.id)
    db.clear_all_conversations(user.id)
    assert _attachment_count(db_path) == 0