from __future__ import annotations

from pathlib import Path

import pytest

from rag_core.db import Database
from rag_core.models import Turn


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