from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rag_core.models import MAX_TURNS, Session, Turn, User


class Database:
    """SQLite persistence for users, one conversation per user, and messages.

    Passwords are stored in plaintext by explicit demo choice — this is a
    demo database, not a security boundary.
    """

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id INTEGER UNIQUE NOT NULL REFERENCES users(id),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL REFERENCES conversations(id),
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def register(self, username: str, password: str) -> User:
        try:
            cursor = self._conn.execute(
                "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                (username, password, self._now()),
            )
        except sqlite3.IntegrityError:
            raise ValueError(f"username already taken: {username}") from None
        self._conn.commit()
        lastrowid = cursor.lastrowid
        assert lastrowid is not None, "AUTOINCREMENT always sets lastrowid"
        return User(id=lastrowid, username=username)

    def login(self, username: str, password: str) -> User | None:
        row = self._conn.execute(
            "SELECT id, username FROM users WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()
        if row is None:
            return None
        return User(id=row["id"], username=row["username"])

    def get_or_create_session(self, user_id: int) -> Session:
        row = self._conn.execute(
            "SELECT id FROM conversations WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is not None:
            return self.get_session(row["id"])
        session_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO conversations (id, user_id, created_at) VALUES (?, ?, ?)",
            (session_id, user_id, self._now()),
        )
        self._conn.commit()
        return Session(id=session_id, user_id=str(user_id), turns=[])

    def get_session(self, session_id: str) -> Session:
        conversation = self._conn.execute(
            "SELECT user_id FROM conversations WHERE id = ?", (session_id,)
        ).fetchone()
        if conversation is None:
            raise KeyError(f"no such conversation: {session_id}")
        rows = self._conn.execute(
            "SELECT role, text FROM messages WHERE conversation_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (session_id, MAX_TURNS),
        ).fetchall()
        turns = [Turn(role=row["role"], text=row["text"]) for row in reversed(rows)]
        return Session(id=session_id, user_id=str(conversation["user_id"]), turns=turns)

    def append_exchange(self, session_id: str, user_text: str, assistant_text: str) -> None:
        """Persist one user message and its assistant reply as two turns."""
        self._conn.executemany(
            "INSERT INTO messages (conversation_id, role, text, created_at) "
            "VALUES (?, ?, ?, ?)",
            [
                (session_id, "user", user_text, self._now()),
                (session_id, "assistant", assistant_text, self._now()),
            ],
        )
        self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()