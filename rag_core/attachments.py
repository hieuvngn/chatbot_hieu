"""Attachment store: user-uploaded documents as per-conversation knowledge.

Parses PDF/TXT/MD uploads into sections, chunks them with the shared chunking
helper and persists text plus embeddings in SQLite keyed by ``conversation_id``.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import NamedTuple

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
    created_at TEXT NOT NULL
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
