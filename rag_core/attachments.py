"""Attachment store: user-uploaded documents as per-conversation knowledge.

Parses PDF/TXT/MD uploads into sections, chunks them with the shared chunking
helper and persists text plus embeddings in SQLite keyed by ``conversation_id``.
"""

from __future__ import annotations

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
