"""Bearer-token sessions backed by a ``tokens`` table in the app database."""

from __future__ import annotations

import contextlib
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class TokenStore:
    """Issues opaque tokens and maps them back to user ids.

    Follows the same connection-per-operation pattern as ``rag_core.db.Database``
    so instances are safe to reuse across threads.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tokens (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def issue(self, user_id: int) -> str:
        token = secrets.token_hex(32)
        now = datetime.now(timezone.utc).isoformat()
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO tokens (token, user_id, created_at) VALUES (?, ?, ?)",
                (token, user_id, now),
            )
            conn.commit()
        return token

    def user_id_for(self, token: str) -> int | None:
        if not token:
            return None
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute("SELECT user_id FROM tokens WHERE token = ?", (token,)).fetchone()
        return int(row["user_id"]) if row is not None else None
