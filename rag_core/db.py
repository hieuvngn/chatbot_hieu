from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rag_core.models import MAX_TURNS, Citation, Session, Source, Turn, User


class Database:
    """SQLite persistence for users, one conversation per user, and messages.

    Passwords are stored in plaintext by explicit demo choice — this is a
    demo database, not a security boundary.
    """

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._create_schema()
        self._migrate()

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
                created_at TEXT NOT NULL,
                citations TEXT,
                refused INTEGER NOT NULL DEFAULT 0,
                rephrase_suggestion TEXT NOT NULL DEFAULT ''
            );
            """
        )
        self._conn.commit()

    def _migrate(self) -> None:
        """Add the assistant-message metadata columns to pre-existing tables."""
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        for column, ddl in [
            ("citations", "ALTER TABLE messages ADD COLUMN citations TEXT"),
            (
                "refused",
                "ALTER TABLE messages ADD COLUMN refused INTEGER NOT NULL DEFAULT 0",
            ),
            (
                "rephrase_suggestion",
                "ALTER TABLE messages ADD COLUMN rephrase_suggestion TEXT NOT NULL DEFAULT ''",
            ),
        ]:
            if column not in columns:
                self._conn.execute(ddl)
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
            "SELECT role, text, citations, refused, rephrase_suggestion "
            "FROM messages WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, MAX_TURNS),
        ).fetchall()
        turns = [
            Turn(
                role=row["role"],
                text=row["text"],
                citations=self._parse_citations(row["citations"]),
                refused=bool(row["refused"]),
                rephrase_suggestion=row["rephrase_suggestion"] or "",
            )
            for row in reversed(rows)
        ]
        return Session(id=session_id, user_id=str(conversation["user_id"]), turns=turns)

    def append_exchange(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        citations: list[Citation] | None = None,
        refused: bool = False,
        rephrase_suggestion: str = "",
    ) -> None:
        """Persist one user message and its assistant reply as two turns.

        The assistant turn carries the rendering metadata: its Citations and,
        for refusals, the rephrase suggestion.
        """
        citations_json = (
            json.dumps(
                [_citation_to_dict(citation) for citation in citations],
                ensure_ascii=False,
            )
            if citations
            else None
        )
        self._conn.executemany(
            "INSERT INTO messages (conversation_id, role, text, created_at, "
            "citations, refused, rephrase_suggestion) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (session_id, "user", user_text, self._now(), None, 0, ""),
                (
                    session_id,
                    "assistant",
                    assistant_text,
                    self._now(),
                    citations_json,
                    int(refused),
                    rephrase_suggestion,
                ),
            ],
        )
        self._conn.commit()

    @staticmethod
    def _parse_citations(raw: str | None) -> list[Citation]:
        if not raw:
            return []
        entries = json.loads(raw)
        if not isinstance(entries, list):
            raise ValueError("citations column must be a JSON list")
        return [_citation_from_dict(entry) for entry in entries]

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


_CITATION_FIELDS = (
    "marker",
    "document_id",
    "document_title",
    "chapter",
    "course_code",
    "kind",
    "language",
)


def _citation_to_dict(citation: Citation) -> dict[str, str]:
    source = citation.source
    return {
        "marker": citation.marker,
        "document_id": source.document_id,
        "document_title": source.document_title,
        "chapter": source.chapter,
        "course_code": source.course_code,
        "kind": source.kind,
        "language": source.language,
    }


def _citation_from_dict(raw: object) -> Citation:
    if not isinstance(raw, dict) or any(field not in raw for field in _CITATION_FIELDS):
        raise ValueError(
            f"citation entry must be an object with fields {_CITATION_FIELDS}"
        )
    source = Source(
        document_id=raw["document_id"],
        document_title=raw["document_title"],
        chapter=raw["chapter"],
        course_code=raw["course_code"],
        kind=raw["kind"],
        language=raw["language"],
    )
    return Citation(marker=raw["marker"], source=source)