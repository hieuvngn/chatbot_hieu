"""Attachment store: user-uploaded documents as per-conversation knowledge.

Parses PDF/TXT/MD uploads into sections, chunks them with the shared chunking
helper and persists text plus embeddings in SQLite keyed by ``conversation_id``.
"""

from __future__ import annotations

import contextlib
import io
import re
import sqlite3
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

from rag_core.chunking import _chunk_text
from rag_core.embeddings import Embedder
from rag_core.models import Chunk, Source

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_FILES_PER_CONVERSATION = 3
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}

_HEADING_RE = re.compile(r"^#{1,4}\s+(.+?)\s*$", re.MULTILINE)


class Section(NamedTuple):
    text: str
    chapter: str


def _decode(data: bytes, filename: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File '{filename}' không phải UTF-8 hợp lệ.") from exc


def parse_markdown(raw_text: str) -> list[Section]:
    """Split markdown on #-#### headings; headingless prose is 'Nội dung'."""
    sections: list[Section] = []
    chapter = "Nội dung"
    buffer: list[str] = []
    for line in raw_text.splitlines():
        match = _HEADING_RE.match(line)
        if match is not None:
            joined = "\n".join(buffer).strip()
            if joined:
                sections.append(Section(joined, chapter))
            buffer = []
            chapter = match.group(1).strip()[:80]
        else:
            buffer.append(line)
    tail = "\n".join(buffer).strip()
    if tail:
        sections.append(Section(tail, chapter))
    return sections


def parse_pdf(data: bytes) -> list[Section]:
    """Extract each non-empty PDF page as Section(chapter='Trang N')."""
    from pypdf import PdfReader

    sections: list[Section] = []
    try:
        reader = PdfReader(io.BytesIO(data))
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(Section(text, f"Trang {number}"))
    except Exception as exc:
        raise ValueError(
            "Không trích xuất được văn bản từ file (PDF scan thiếu text layer?)."
        ) from exc
    return sections


def parse_file(filename: str, data: bytes) -> list[Section]:
    """Dispatch by extension; raises ValueError on any unusable input."""
    name = filename.strip().lower()
    ext = Path(name).suffix.lstrip(".")
    if ext == "pdf":
        sections = parse_pdf(data)
    elif ext == "txt":
        sections = [Section(_decode(data, name).strip(), "Nội dung")]
    elif ext == "md":
        sections = parse_markdown(_decode(data, name))
    else:
        raise ValueError(
            f"Định dạng .{ext or '?'} không hỗ trợ (chỉ pdf, txt, md)."
        )
    if not sections or not "\n".join(s.text for s in sections).strip():
        raise ValueError(
            "Không trích xuất được văn bản từ file (PDF scan thiếu text layer?)."
        )
    return sections


ATTACHMENTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    filename TEXT NOT NULL,
    file_kind TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    language TEXT NOT NULL DEFAULT 'vi',
    created_at TEXT NOT NULL,
    raw_text TEXT,
    raw_pdf BLOB
);
CREATE TABLE IF NOT EXISTS attachment_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attachment_id TEXT NOT NULL REFERENCES attachments(id),
    conversation_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    chapter TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_att_chunks_conv ON attachment_chunks(conversation_id);
"""


_ATTACHMENT_RAW_COLUMNS = ("raw_text", "raw_pdf")


def _migrate_attachment_raw(conn: sqlite3.Connection) -> None:
    """Idempotently add raw_text / raw_pdf to pre-existing attachments tables."""
    columns = {row["name"] for row in conn.execute("PRAGMA table_info('attachments')").fetchall()}
    for name in _ATTACHMENT_RAW_COLUMNS:
        if name not in columns:
            col_type = "BLOB" if name == "raw_pdf" else "TEXT"
            conn.execute(f"ALTER TABLE attachments ADD COLUMN {name} {col_type}")


@dataclass(frozen=True)
class AttachmentMeta:
    id: str
    conversation_id: str
    filename: str
    file_kind: str
    size_bytes: int
    chunk_count: int
    created_at: str


def _to_blob(vector: list[float]) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def _blob_to_array(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


def _source_for(attachment_id: str, filename: str, chapter: str, language: str) -> Source:
    return Source(
        document_id=f"upload_{attachment_id[:8]}",
        document_title=filename,
        chapter=chapter,
        course_code="",
        kind="upload",
        language=language,
    )


class AttachmentStore:
    """SQLite-backed per-conversation document store (chunks + embeddings)."""

    def __init__(self, path: str | Path, embedder: Embedder) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        with contextlib.closing(self._connect()) as conn:
            conn.executescript(ATTACHMENTS_SCHEMA_SQL)
            _migrate_attachment_raw(conn)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def add(
        self, conversation_id: str, filename: str, data: bytes, language: str = "vi"
    ) -> AttachmentMeta:
        name = filename.strip()
        if not name:
            raise ValueError("Tên file trống.")
        ext = Path(name.lower()).suffix.lstrip(".")
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Định dạng .{ext or '?'} không hỗ trợ (chỉ pdf, txt, md)."
            )
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("File vượt quá giới hạn 5 MB.")
        sections = parse_file(name, data)
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT filename FROM attachments WHERE conversation_id=?",
                (conversation_id,),
            ).fetchall()
        if len(rows) >= MAX_FILES_PER_CONVERSATION:
            raise ValueError(
                f"Mỗi chat tối đa {MAX_FILES_PER_CONVERSATION} tài liệu."
            )
        if any(str(r["filename"]).lower() == name.lower() for r in rows):
            raise ValueError(f"Đã tồn tại tài liệu tên '{name}' trong chat này.")
        attachment_id = uuid.uuid4().hex
        base_source = _source_for(attachment_id, name, "", language)
        chunks: list[Chunk] = []
        chapters: list[str] = []
        for section in sections:
            section_chunks = _chunk_text(
                section.text, replace(base_source, chapter=section.chapter)
            )
            chunks.extend(section_chunks)
            chapters.extend([section.chapter] * len(section_chunks))

        vectors = self._embedder.embed_batch([c.text for c in chunks])

        now = datetime.now(timezone.utc).isoformat()
        raw_text: str | None = _decode(data, name) if ext != "pdf" else None
        raw_pdf: bytes | None = data if ext == "pdf" else None
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO attachments (id, conversation_id, filename, file_kind, "
                "size_bytes, chunk_count, language, created_at, raw_text, raw_pdf) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    attachment_id,
                    conversation_id,
                    name,
                    ext,
                    len(data),
                    len(chunks),
                    language,
                    now,
                    raw_text,
                    raw_pdf,
                ),
            )
            conn.executemany(
                "INSERT INTO attachment_chunks (attachment_id, conversation_id, seq, "
                "chapter, text, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        attachment_id,
                        conversation_id,
                        i,
                        chapters[i],
                        chunks[i].text,
                        _to_blob(vectors[i]),
                    )
                    for i in range(len(chunks))
                ],
            )
            conn.commit()
        return AttachmentMeta(
            id=attachment_id,
            conversation_id=conversation_id,
            filename=name,
            file_kind=ext,
            size_bytes=len(data),
            chunk_count=len(chunks),
            created_at=now,
        )

    def list_for(self, conversation_id: str) -> list[AttachmentMeta]:
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, conversation_id, filename, file_kind, size_bytes, "
                "chunk_count, created_at FROM attachments WHERE conversation_id=? "
                "ORDER BY created_at, id",
                (conversation_id,),
            ).fetchall()
        return [
            AttachmentMeta(
                id=str(r["id"]),
                conversation_id=str(r["conversation_id"]),
                filename=str(r["filename"]),
                file_kind=str(r["file_kind"]),
                size_bytes=int(r["size_bytes"]),
                chunk_count=int(r["chunk_count"]),
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]

    def delete(self, attachment_id: str) -> None:
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                "DELETE FROM attachment_chunks WHERE attachment_id=?", (attachment_id,)
            )
            cur = conn.execute(
                "DELETE FROM attachments WHERE id=?", (attachment_id,)
            )
            conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"no such attachment: {attachment_id}")

    def get_content(self, attachment_id: str) -> list[Section]:
        """Return the original parsed sections for viewer rendering.

        Re-parses from the stored ``raw_text`` / ``raw_pdf`` to guarantee the
        viewer shows the same heading/page boundaries the embedder saw. Raises
        ``KeyError`` if the attachment does not exist.
        """
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT filename, file_kind, raw_text, raw_pdf "
                "FROM attachments WHERE id = ?",
                (attachment_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"no such attachment: {attachment_id}")
        filename = str(row["filename"])
        kind = str(row["file_kind"])
        if kind == "pdf":
            data = bytes(row["raw_pdf"]) if row["raw_pdf"] is not None else b""
        else:
            data = str(row["raw_text"] or "").encode("utf-8")
        return parse_file(filename, data)
    def identity(self, conversation_id: str) -> tuple[str, ...]:
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id FROM attachments WHERE conversation_id=? ORDER BY created_at, id",
                (conversation_id,),
            ).fetchall()
        return tuple(str(r["id"]) for r in rows)

    def load_chunks(self, conversation_id: str) -> list[Chunk]:
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT a.id AS att_id, a.filename, a.language, c.chapter, c.text "
                "FROM attachment_chunks c JOIN attachments a ON a.id=c.attachment_id "
                "WHERE c.conversation_id=? ORDER BY a.created_at, a.id, c.seq",
                (conversation_id,),
            ).fetchall()
        return [
            Chunk(
                source=_source_for(
                    str(r["att_id"]),
                    str(r["filename"]),
                    str(r["chapter"]),
                    str(r["language"]),
                ),
                text=str(r["text"]),
            )
            for r in rows
        ]

    def load_vectors(self, conversation_id: str) -> np.ndarray:
        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT c.embedding FROM attachment_chunks c "
                "JOIN attachments a ON a.id=c.attachment_id "
                "WHERE c.conversation_id=? ORDER BY a.created_at, a.id, c.seq",
                (conversation_id,),
            ).fetchall()
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack([_blob_to_array(bytes(r["embedding"])) for r in rows])
