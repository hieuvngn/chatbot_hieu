from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rag_core.models import MAX_TURNS, Citation, ConversationMeta, Session, Source, Turn, User


class Database:
    """SQLite persistence for users, conversations (N per user), and messages.

    Passwords are stored in plaintext by explicit demo choice — this is a
    demo database, not a security boundary.

    Thread-safety:
    Streamlit's ``st.cache_resource`` caches the ``Database`` instance and
    reuses it across script reruns that may execute on different threads.
    The original implementation held a single ``sqlite3.Connection`` (which
    is bound to its creating thread when ``check_same_thread=True``) and
    therefore raised ``ProgrammingError`` on the next rerun.  This class now
    opens a **new connection per operation** with ``check_same_thread=False``
    so it can be safely reused from any thread.  A deprecated long-lived
    handle is still kept as ``self._conn`` for backwards compatibility with
    code that accesses it directly.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(self._connect()) as conn:
            self._create_schema(conn)
            self._migrate(conn)
        # Deprecated handle kept for backwards compat (tests call close(),
        # external code may access db._conn).  Created with
        # check_same_thread=False so cross-thread access does not raise
        # ProgrammingError, but internal methods do NOT use it.
        self._conn = self._connect()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._path), check_same_thread=False, timeout=30.0
        )
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_conversation_index(conn: sqlite3.Connection) -> None:
        """Create index on conversations(user_id, updated_at DESC) if possible.

        Silently ignores OperationalError caused by missing column on legacy
        DBs before migration (e.g. no such column: updated_at); re-raises
        any other OperationalError so real problems are not hidden.
        """
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC)"
            )
        except sqlite3.OperationalError as exc:
            msg = str(exc).lower()
            # Legacy DB before updated_at migration; caller (_migrate) will retry after migration.
            if "no such column" in msg or "no such table" in msg:
                return
            raise

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                display_name TEXT NOT NULL DEFAULT '',
                language TEXT NOT NULL DEFAULT 'vi',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
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
        self._ensure_conversation_index(conn)
        conn.commit()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add the assistant-message metadata columns to pre-existing tables."""
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(messages)").fetchall()
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
                conn.execute(ddl)
        # users columns
        user_cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "display_name" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN display_name TEXT NOT NULL DEFAULT ''")
            conn.execute("UPDATE users SET display_name = username WHERE display_name = ''")
        if "language" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN language TEXT NOT NULL DEFAULT 'vi'")
        # conversations columns
        conv_cols = {r["name"] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()}
        if "title" not in conv_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT 'New chat'")
        if "updated_at" not in conv_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")
            conn.execute("UPDATE conversations SET updated_at = created_at WHERE updated_at = ''")
        # handle UNIQUE on user_id: detect via PRAGMA index_list
        for idx in conn.execute("PRAGMA index_list('conversations')").fetchall():
            if idx["unique"] == 1:
                # check if this index covers user_id
                info = conn.execute(f"PRAGMA index_info('{idx['name']}')").fetchall()
                if any(r["name"] == "user_id" for r in info):
                    # recreate table without UNIQUE
                    conn.executescript("""
                    CREATE TABLE conversations_new (
                        id TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id),
                        title TEXT NOT NULL DEFAULT 'New chat',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    INSERT INTO conversations_new (id, user_id, title, created_at, updated_at)
                        SELECT id, user_id, COALESCE(title, 'New chat'), created_at, COALESCE(updated_at, created_at) FROM conversations;
                    DROP TABLE conversations;
                    ALTER TABLE conversations_new RENAME TO conversations;
                    CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);
                    """)
                    break
        self._ensure_conversation_index(conn)
        conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def register(self, username: str, password: str) -> User:
        with contextlib.closing(self._connect()) as conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO users (username, password, display_name, language, created_at) VALUES (?, ?, ?, ?, ?)",
                    (username, password, username, "vi", self._now()),
                )
            except sqlite3.IntegrityError:
                raise ValueError(f"username already taken: {username}") from None
            conn.commit()
            lastrowid = cursor.lastrowid
            assert lastrowid is not None, "AUTOINCREMENT always sets lastrowid"
            return User(id=lastrowid, username=username, display_name=username, language="vi")

    def login(self, username: str, password: str) -> User | None:
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, username, display_name, language FROM users WHERE username = ? AND password = ?",
                (username, password),
            ).fetchone()
            if row is None:
                return None
            return User(
                id=int(row["id"]),
                username=str(row["username"]),
                display_name=str(row["display_name"] or row["username"]),
                language=str(row["language"] or "vi"),
            )

    def get_user(self, user_id: int) -> User | None:
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, username, display_name, language FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if row is None:
                return None
            return User(
                id=int(row["id"]),
                username=str(row["username"]),
                display_name=str(row["display_name"] or row["username"]),
                language=str(row["language"] or "vi"),
            )

    def create_conversation(self, user_id: int, title: str = "New chat") -> Session:
        session_id = uuid.uuid4().hex
        now = self._now()
        title = title.strip()[:50] or "New chat"
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, title, now, now),
            )
            conn.commit()
        return Session(id=session_id, user_id=str(user_id), turns=[])

    def list_conversations(self, user_id: int) -> list[ConversationMeta]:
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at,
                       (SELECT text FROM messages WHERE conversation_id=c.id AND role='user' ORDER BY id ASC LIMIT 1) AS preview
                FROM conversations c WHERE c.user_id=? ORDER BY c.updated_at DESC
                """,
                (user_id,),
            ).fetchall()
            return [
                ConversationMeta(
                    id=str(r["id"]),
                    user_id=int(r["user_id"]),
                    title=str(r["title"]),
                    created_at=str(r["created_at"]),
                    updated_at=str(r["updated_at"]),
                    preview=str((r["preview"] or "")[:40]),
                )
                for r in rows
            ]

    def update_user_profile(
        self, user_id: int, display_name: str | None = None, language: str | None = None
    ) -> User:
        if display_name is not None:
            if not (1 <= len(display_name.strip()) <= 50):
                raise ValueError("display_name must be 1..50 chars")
            display_name = display_name.strip()
        if language is not None and language not in ("vi", "en"):
            raise ValueError("language must be 'vi' or 'en'")
        with contextlib.closing(self._connect()) as conn:
            if display_name is not None:
                conn.execute("UPDATE users SET display_name=? WHERE id=?", (display_name, user_id))
            if language is not None:
                conn.execute("UPDATE users SET language=? WHERE id=?", (language, user_id))
            conn.commit()
        user = self.get_user(user_id)
        assert user is not None
        return user

    def update_conversation_title(self, session_id: str, title: str) -> None:
        title = title.strip()
        if not (1 <= len(title) <= 50):
            raise ValueError("title must be 1..50 chars")
        with contextlib.closing(self._connect()) as conn:
            cur = conn.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                (title[:50], self._now(), session_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"no such conversation: {session_id}")
            conn.commit()

    def delete_conversation(self, session_id: str) -> None:
        with contextlib.closing(self._connect()) as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id=?", (session_id,))
            cur = conn.execute("DELETE FROM conversations WHERE id=?", (session_id,))
            if cur.rowcount == 0:
                raise KeyError(f"no such conversation: {session_id}")
            conn.commit()

    def clear_all_conversations(self, user_id: int) -> None:
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                "DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id=?)",
                (user_id,),
            )
            conn.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
            conn.commit()

    def get_or_create_session(self, user_id: int) -> Session:
        metas = self.list_conversations(user_id)
        if metas:
            return self.get_session(metas[0].id)
        return self.create_conversation(user_id, title="New chat")

    def get_session(self, session_id: str) -> Session:
        with contextlib.closing(self._connect()) as conn:
            conversation = conn.execute(
                "SELECT user_id FROM conversations WHERE id = ?", (session_id,)
            ).fetchone()
            if conversation is None:
                raise KeyError(f"no such conversation: {session_id}")
            rows = conn.execute(
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
        with contextlib.closing(self._connect()) as conn:
            conn.executemany(
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
            # update updated_at and auto title
            now = self._now()
            cnt = conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE conversation_id=?", (session_id,)
            ).fetchone()["c"]
            if cnt == 2:
                cur_title = conn.execute(
                    "SELECT title FROM conversations WHERE id=?", (session_id,)
                ).fetchone()
                if cur_title and cur_title["title"] == "New chat":
                    new_title = user_text.strip()[:40] or "New chat"
                    conn.execute(
                        "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                        (new_title, now, session_id),
                    )
                else:
                    conn.execute(
                        "UPDATE conversations SET updated_at=? WHERE id=?", (now, session_id)
                    )
            else:
                conn.execute(
                    "UPDATE conversations SET updated_at=? WHERE id=?", (now, session_id)
                )
            conn.commit()

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
