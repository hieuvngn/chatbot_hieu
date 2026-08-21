# Document Upload (Attachment RAG) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cho phép user upload PDF/TXT/MD làm nguồn tri thức theo conversation, truy xuất song song với CourseMate KB qua RRF merge, trả lời kèm citation tới tài liệu + vị trí nguồn.

**Architecture:** `AttachmentStore` mới (parse → chunk → embed → SQLite BLOB theo `conversation_id`) + index thứ cấp tái dùng class `Index` hiện có; `RagCore.answer()` fuse ranking KB và attachment bằng RRF trước `dedupe_by_source`. Judge/Generator/Answer-check/citation không đổi.

**Tech Stack:** Python 3.11+, Streamlit, SQLite (vector BLOB float32 LE), pypdf (mới), FAISS + rank_bm25 (sẵn có), pytest + mypy strict.

**Spec:** `docs/superpowers/specs/2026-08-21-document-upload-design.md`

## Global Constraints

- Giới hạn chốt (verbatim từ spec): **3 file/conversation, ≤ 5 MB/file**, chỉ `pdf/txt/md`, **không thêm rerank**.
- Vector lưu BLOB dtype `<f4` (float32 little-endian) trong bảng `attachment_chunks`.
- `Source.kind = "upload"`; `chapter` = `"Trang {n}"` (PDF) / tiêu đề heading (MD) / `"Nội dung"` (TXT).
- Mọi lỗi người dùng raise `ValueError` với message tiếng Việt; UI hiển thị nguyên văn.
- Không dùng cú pháp PEP 695 (`def f[T]()`) — project yêu cầu Python ≥ 3.11; dùng `typing.TypeVar`.
- Mọi file mới/sửa phải pass `uv run mypy` (strict) và `uv run pytest`.
- Commit style theo repo: `feat(db): …`, `feat(core): …`, `feat(ui): …`.

---

### Task 1: DB schema + cascade delete

**Files:**
- Create: `rag_core/attachments.py` (chỉ phần DDL constant — các task sau điền tiếp)
- Modify: `rag_core/db.py` (`_create_schema`, `delete_conversation`, `clear_all_conversations`)
- Modify: `app.py` (import lại `APP_DB_FILENAME`)
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: không (task đầu tiên).
- Produces:
  - `ATTACHMENTS_SCHEMA_SQL: str` trong `rag_core/attachments.py` — DDL idempotent 2 bảng, do `Database._create_schema()` thực thi.
  - Bảng `attachments(id TEXT PK, conversation_id, filename, file_kind, size_bytes, chunk_count, language, created_at)` và `attachment_chunks(id INTEGER PK, attachment_id, conversation_id, seq, chapter, text, embedding BLOB)` + index `idx_att_chunks_conv`.
  - Hằng số `APP_DB_FILENAME = "app.db"` chuyển sang `rag_core/db.py`.

- [ ] **Step 1: Tạo `rag_core/attachments.py` với DDL constant**

```python
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
```

- [ ] **Step 2: Sửa `Database._create_schema` + chuyển constant**

Trong `rag_core/db.py`, thêm import và constant (đặt cạnh các import models):

```python
from rag_core.attachments import ATTACHMENTS_SCHEMA_SQL

APP_DB_FILENAME = "app.db"
```

Trong `_create_schema`, sau `conn.executescript(...)` hiện có, trước `self._ensure_conversation_index(conn)`:

```python
        conn.executescript(ATTACHMENTS_SCHEMA_SQL)
```

Trong `app.py`: xóa dòng `APP_DB_FILENAME = "app.db"`, đổi import thành:

```python
from rag_core.db import APP_DB_FILENAME, Database
```

- [ ] **Step 3: Cascade delete**

`Database.delete_conversation` — thêm 2 dòng DELETE **trước** dòng `DELETE FROM conversations`:

```python
            conn.execute("DELETE FROM attachment_chunks WHERE conversation_id=?", (session_id,))
            conn.execute("DELETE FROM attachments WHERE conversation_id=?", (session_id,))
```

`Database.clear_all_conversations` — thêm trước `DELETE FROM conversations` tương tự:

```python
            conn.execute(
                "DELETE FROM attachment_chunks WHERE conversation_id IN "
                "(SELECT id FROM conversations WHERE user_id=?)",
                (user_id,),
            )
            conn.execute(
                "DELETE FROM attachments WHERE conversation_id IN "
                "(SELECT id FROM conversations WHERE user_id=?)",
                (user_id,),
            )
```

- [ ] **Step 4: Viết test fail** — thêm vào cuối `tests/test_db.py`:

```python
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
```

- [ ] **Step 5: Chạy test verify fail**

Run: `uv run pytest tests/test_db.py -v -k cascade`
Expected: FAIL — `sqlite3.OperationalError: no such table: attachments`

- [ ] **Step 6: Chạy toàn bộ suite + mypy**

Run: `uv run pytest && uv run mypy`
Expected: PASS tất cả.

- [ ] **Step 7: Commit**

```bash
git add rag_core/db.py rag_core/attachments.py app.py tests/test_db.py
git commit -m "feat(db): add attachments tables + cascade delete on conversation removal"
```

---

### Task 2: Parsers (PDF / TXT / MD)

**Files:**
- Modify: `rag_core/attachments.py`
- Modify: `pyproject.toml` (thêm `pypdf`)
- Test: `tests/test_attachments.py` (file mới)

**Interfaces:**
- Consumes: module `rag_core.attachments`.
- Produces (Task 3 dùng đúng signature này):
  - `class Section(NamedTuple)` với `text: str`, `chapter: str`
  - `parse_file(filename: str, data: bytes) -> list[Section]` — raise `ValueError` cho định dạng không hỗ trợ / decode lỗi / không trích xuất được văn bản.

- [ ] **Step 1: Thêm dependency pypdf**

```bash
uv add "pypdf>=4.0"
```

Expected: `pyproject.toml` có `"pypdf>=4.0"`, `uv.lock` cập nhật.

- [ ] **Step 2: Viết test fail** — tạo `tests/test_attachments.py`:

```python
from __future__ import annotations

import pytest

from rag_core.attachments import parse_file


def _serialize_pdf(objects: dict[int, str]) -> bytes:
    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n{objects[obj_id]}\nendobj\n".encode("latin-1")
    xref_pos = len(out)
    max_id = max(objects)
    out += f"xref\n0 {max_id + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for obj_id in range(1, max_id + 1):
        out += f"{offsets.get(obj_id, 0):010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


def _make_pdf(page_streams: list[str]) -> bytes:
    """Minimal single-font PDF; each page gets one content stream string."""
    n_pages = len(page_streams)
    font_id = 3 + 2 * n_pages
    page_obj_ids = [3 + 2 * i for i in range(n_pages)]
    kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects: dict[int, str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>",
    }
    for i, stream in enumerate(page_streams):
        pid = page_obj_ids[i]
        cid = pid + 1
        objects[pid] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
            f"/Contents {cid} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        if stream:
            body = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
        else:
            body = "<< /Length 0 >>\nstream\n\nendstream"
        objects[cid] = body
    objects[font_id] = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    return _serialize_pdf(objects)


def _text_stream(text: str) -> str:
    return f"BT /F1 12 Tf 40 100 Td ({text}) Tj ET"


def test_parse_txt_single_section() -> None:
    sections = parse_file("notes.txt", "bảng băm là cấu trúc dữ liệu".encode())
    assert len(sections) == 1
    assert sections[0].chapter == "Nội dung"
    assert "bảng băm" in sections[0].text


def test_parse_md_splits_on_headings() -> None:
    md = "# Giới thiệu\nnội dung mở đầu\n## Bảng băm\ncấu trúc dữ liệu\n## Cây AVL\ncân bằng"
    sections = parse_file("doc.md", md.encode())
    assert [(s.chapter, s.text) for s in sections] == [
        ("Giới thiệu", "nội dung mở đầu"),
        ("Bảng băm", "cấu trúc dữ liệu"),
        ("Cây AVL", "cân bằng"),
    ]


def test_parse_md_without_heading_uses_default_chapter() -> None:
    sections = parse_file("doc.md", "văn bản thường\nkhông có heading".encode())
    assert len(sections) == 1
    assert sections[0].chapter == "Nội dung"


def test_parse_pdf_extracts_pages_as_chapters() -> None:
    data = _make_pdf([_text_stream("bang bam page one"), _text_stream("cay avl page two")])
    sections = parse_file("slides.pdf", data)
    assert [s.chapter for s in sections] == ["Trang 1", "Trang 2"]
    assert "bang bam" in sections[0].text


def test_parse_pdf_skips_empty_pages_keeps_numbering() -> None:
    data = _make_pdf([_text_stream("co noi dung"), "", _text_stream("trang ba")])
    sections = parse_file("mixed.pdf", data)
    assert [s.chapter for s in sections] == ["Trang 1", "Trang 3"]


def test_unsupported_extension_raises_value_error() -> None:
    with pytest.raises(ValueError, match="không hỗ trợ"):
        parse_file("photo.png", b"\x89PNG")


def test_invalid_utf8_txt_raises_value_error() -> None:
    with pytest.raises(ValueError):
        parse_file("broken.txt", b"\xff\xfe\xfa")


def test_scanned_pdf_without_text_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Không trích xuất được"):
        parse_file("scan.pdf", _make_pdf([]))
```

- [ ] **Step 3: Chạy test verify fail**

Run: `uv run pytest tests/test_attachments.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_file'`

- [ ] **Step 4: Implement parsers** — thêm vào `rag_core/attachments.py`:

```python
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

    reader = PdfReader(io.BytesIO(data))
    sections: list[Section] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(Section(text, f"Trang {number}"))
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
```

- [ ] **Step 5: Chạy test verify pass**

Run: `uv run pytest tests/test_attachments.py -v`
Expected: PASS 8/8

- [ ] **Step 6: mypy + commit**

Run: `uv run mypy` — Expected: PASS

```bash
git add rag_core/attachments.py pyproject.toml uv.lock tests/test_attachments.py
git commit -m "feat(attachments): add pdf/txt/md section parsers with page/heading metadata"
```

---

### Task 3: AttachmentStore CRUD + embedding persistence

**Files:**
- Modify: `rag_core/attachments.py`
- Test: `tests/test_attachments.py`

**Interfaces:**
- Consumes: `Section`, `parse_file`, `MAX_FILE_BYTES`, `MAX_FILES_PER_CONVERSATION` (Task 2); `ATTACHMENTS_SCHEMA_SQL` (Task 1); `Chunk`, `Source` từ `rag_core.models`; `_chunk_text` từ `rag_core.chunking`; `Embedder` protocol từ `rag_core.embeddings`.
- Produces (Task 5 & 6 dùng đúng signature này):
  - `@dataclass(frozen=True) class AttachmentMeta(id: str, conversation_id: str, filename: str, file_kind: str, size_bytes: int, chunk_count: int, created_at: str)`
  - `class AttachmentStore`:
    - `__init__(self, path: str | Path, embedder: Embedder) -> None`
    - `add(self, conversation_id: str, filename: str, data: bytes, language: str = "vi") -> AttachmentMeta`
    - `list_for(self, conversation_id: str) -> list[AttachmentMeta]`
    - `delete(self, attachment_id: str) -> None` — raise `KeyError` nếu không tồn tại
    - `identity(self, conversation_id: str) -> tuple[str, ...]`
    - `load_chunks(self, conversation_id: str) -> list[Chunk]`
    - `load_vectors(self, conversation_id: str) -> np.ndarray` — shape `(n, dim)`, thứ tự khớp `load_chunks`

Quy ước quan trọng: `document_id` **suy diễn deterministic** từ attachment id để nhất quán giữa các turn:
`document_id = f"upload_{attachment_id[:8]}"` — dùng cả lúc ghi lẫn lúc đọc, KHÔNG random riêng lúc ghi.

- [ ] **Step 1: Viết test fail** — thêm vào `tests/test_attachments.py`:

```python
import hashlib
import math

from rag_core.attachments import AttachmentStore
from rag_core.db import Database


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens embedder; counts batch calls."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.batch_calls = 0

    def _vec(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.md5(token.encode()).digest()
            for i in range(self.dim):
                vector[i] += (digest[i % 16] / 255.0) - 0.5
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_store(tmp_path):
    db_path = tmp_path / "test.db"
    Database(db_path)
    return AttachmentStore(db_path, HashEmbedder())


def test_add_persists_meta_and_chunks(tmp_path) -> None:
    store = make_store(tmp_path)
    meta = store.add("conv1", "note.txt", "bảng băm zzq wub florp".encode())
    assert meta.filename == "note.txt"
    assert meta.file_kind == "txt"
    assert meta.chunk_count >= 1
    assert [m.id for m in store.list_for("conv1")] == [meta.id]


def test_add_rejects_wrong_extension(tmp_path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="không hỗ trợ"):
        store.add("conv1", "img.jpg", b"data")


def test_add_rejects_oversize(tmp_path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="5 MB"):
        store.add("conv1", "big.txt", b"x" * (MAX_FILE_BYTES + 1))


def test_add_rejects_duplicate_filename_case_insensitive(tmp_path) -> None:
    store = make_store(tmp_path)
    store.add("conv1", "Note.txt", "abc".encode())
    with pytest.raises(ValueError, match="tồn tại"):
        store.add("conv1", "NOTE.txt", "abc".encode())


def test_add_rejects_when_limit_reached(tmp_path) -> None:
    store = make_store(tmp_path)
    for i in range(3):
        store.add("conv1", f"f{i}.txt", f"nội dung {i}".encode())
    with pytest.raises(ValueError, match="tối đa"):
        store.add("conv1", "extra.txt", "nội dung".encode())


def test_embed_failure_leaves_no_rows(tmp_path) -> None:
    class BoomEmbedder(HashEmbedder):
        def embed_batch(self, texts):
            raise RuntimeError("api down")

    db_path = tmp_path / "test.db"
    Database(db_path)
    store = AttachmentStore(db_path, BoomEmbedder())
    with pytest.raises(RuntimeError):
        store.add("conv1", "note.txt", "abc".encode())
    assert store.list_for("conv1") == []


def test_load_chunks_and_vectors_roundtrip(tmp_path) -> None:
    store = make_store(tmp_path)
    store.add("conv1", "a.txt", "zzq wub florp nội dung dài hơn một chút".encode())
    chunks = store.load_chunks("conv1")
    vectors = store.load_vectors("conv1")
    assert len(chunks) == vectors.shape[0]
    assert vectors.shape[1] == 32
    first = chunks[0].source
    assert first.kind == "upload"
    assert first.document_title == "a.txt"
    assert first.document_id.startswith("upload_")
    assert first.course_code == ""


def test_document_id_stable_across_reloads(tmp_path) -> None:
    store = make_store(tmp_path)
    store.add("conv1", "a.txt", "nội dung ổn định".encode())
    ids_first = [c.source.document_id for c in store.load_chunks("conv1")]
    ids_second = [c.source.document_id for c in store.load_chunks("conv1")]
    assert ids_first == ids_second


def test_identity_changes_on_add_and_survives_reload(tmp_path) -> None:
    store = make_store(tmp_path)
    before = store.identity("conv1")
    store.add("conv1", "a.txt", "nội dung".encode())
    after = store.identity("conv1")
    fresh = AttachmentStore(tmp_path / "test.db", HashEmbedder())
    assert before == ()
    assert len(after) == 1
    assert fresh.identity("conv1") == after


def test_delete_removes_chunks_and_metadata(tmp_path) -> None:
    store = make_store(tmp_path)
    meta = store.add("conv1", "a.txt", "nội dung".encode())
    store.delete(meta.id)
    assert store.list_for("conv1") == []
    assert store.load_chunks("conv1") == []
    assert store.identity("conv1") == ()
    with pytest.raises(KeyError):
        store.delete(meta.id)


def test_chunks_isolated_between_conversations(tmp_path) -> None:
    store = make_store(tmp_path)
    store.add("conv1", "a.txt", "nội dung một".encode())
    store.add("conv2", "b.txt", "nội dung hai".encode())
    titles1 = [c.source.document_title for c in store.load_chunks("conv1")]
    assert titles1 == ["a.txt"]
```

Thêm import `from rag_core.attachments import MAX_FILE_BYTES` vào khối import hiện có của file test.

- [ ] **Step 2: Chạy test verify fail**

Run: `uv run pytest tests/test_attachments.py -v`
Expected: FAIL — `ImportError: cannot import name 'AttachmentStore'`

- [ ] **Step 3: Implement AttachmentStore** — thêm vào `rag_core/attachments.py`:

Imports bổ sung (gộp vào khối import hiện có):

```python
import contextlib
import sqlite3
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import numpy as np

from rag_core.chunking import _chunk_text
from rag_core.embeddings import Embedder
from rag_core.models import Chunk, Source
```

Body:

```python
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
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO attachments (id, conversation_id, filename, file_kind, "
                "size_bytes, chunk_count, language, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (attachment_id, conversation_id, name, ext, len(data), len(chunks), language, now),
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
```

Lưu ý: mọi lỗi nảy sinh trước khối INSERT cuối cùng (parse, validate, embed) đều xảy ra khi chưa ghi gì → DB luôn nguyên vẹn, thỏa yêu cầu rollback của spec.

- [ ] **Step 4: Chạy test verify pass**

Run: `uv run pytest tests/test_attachments.py -v`
Expected: PASS tất cả

- [ ] **Step 5: Suite + mypy + commit**

```bash
uv run pytest && uv run mypy
git add rag_core/attachments.py tests/test_attachments.py
git commit -m "feat(attachments): add AttachmentStore CRUD with embedded vector persistence"
```

---

### Task 4: Index nhận sẵn vectors + module-level `rrf_merge`

**Files:**
- Modify: `rag_core/index.py`
- Test: `tests/test_attachments.py` (thêm test)

**Interfaces:**
- Consumes: `HashEmbedder` (Task 3 test helper).
- Produces (Task 5 dùng):
  - `Index.__init__(chunks, embedder, vectors=None)` — khi `vectors` cung cấp shape `(len(chunks), dim)`, build dense trực tiếp KHÔNG gọi `embed_batch`; BM25 vẫn build từ text; raise `ValueError` nếu số row lệch.
  - `rrf_merge(rankings: list[list[T]]) -> list[T]` — module-level, TypeVar `T`, fuse nhiều ranking object theo RRF K=60, so sánh bằng identity (`is`) để hỗ trợ dataclass mutable.

- [ ] **Step 1: Viết test fail** — thêm vào `tests/test_attachments.py`:

```python
import numpy as np

from rag_core.index import Index, rrf_merge
from rag_core.models import Chunk, Source


class ExplodingEmbedder:
    def embed_batch(self, texts):
        raise AssertionError("must not re-embed when vectors provided")

    def embed_query(self, text):
        raise AssertionError("must not re-embed when vectors provided")


def _chunk_with_text(text: str) -> Chunk:
    source = Source(
        document_id="d", document_title="t", chapter="c", course_code="", kind="k", language="vi"
    )
    return Chunk(source=source, text=text)


def test_index_accepts_precomputed_vectors_without_embedding() -> None:
    chunks = [_chunk_with_text("zzq wub florp"), _chunk_with_text("hoàn toàn khác biệt")]
    embedder = HashEmbedder()
    matrix = np.asarray(embedder.embed_batch([c.text for c in chunks]), dtype=np.float32)
    idx = Index(chunks, ExplodingEmbedder(), vectors=matrix)
    query_vector = np.asarray(embedder.embed_query("zzq wub florp"), dtype=np.float32)
    ranked = idx.fused_ranking("zzq wub florp", query_vector)
    assert ranked[0] == 0


def test_index_vector_length_mismatch_raises() -> None:
    chunks = [_chunk_with_text("mot"), _chunk_with_text("hai")]
    bad = np.zeros((1, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="vectors"):
        Index(chunks, ExplodingEmbedder(), vectors=bad)


def test_rrf_merge_interleaves_two_rankings() -> None:
    kb = [_chunk_with_text("k1"), _chunk_with_text("k2"), _chunk_with_text("k3")]
    att = [_chunk_with_text("a1"), _chunk_with_text("a2")]
    merged = rrf_merge([kb, att])
    assert merged[0] is kb[0]
    assert sorted(map(id, merged)) == sorted(map(id, kb + att))


def test_rrf_merge_empty_rankings() -> None:
    assert rrf_merge([[], []]) == []
```

- [ ] **Step 2: Chạy test verify fail**

Run: `uv run pytest tests/test_attachments.py -v -k "index or rrf"`
Expected: FAIL — `TypeError: __init__() takes 2 positional arguments but 3 were given` và `ImportError: cannot import name 'rrf_merge'`

- [ ] **Step 3: Implement** — sửa `rag_core/index.py`:

Đầu file, sau imports hiện có:

```python
from typing import Protocol, TypeVar

T = TypeVar("T")
```

Module-level function (đặt cạnh `dedupe_by_source`):

```python
def rrf_merge(rankings: list[list[T]]) -> list[T]:
    """Fuse rankings of arbitrary items via Reciprocal Rank Fusion (K=60).

    Membership compares items with ``is`` so unhashable mutable dataclasses
    work. Every input item appears exactly once in the output.
    """
    unique: list[T] = []

    def slot(item: T) -> int:
        for i, candidate in enumerate(unique):
            if candidate is item:
                return i
        unique.append(item)
        return len(unique) - 1

    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            idx = slot(item)
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [unique[i] for i in ordered]
```

Sửa `Index.__init__` và tách `_dense_from`:

```python
    def __init__(
        self, chunks: list[Chunk], embedder: Embedder, vectors: np.ndarray | None = None
    ) -> None:
        self.chunks = chunks
        if vectors is None:
            raw = np.asarray(
                embedder.embed_batch([chunk.text for chunk in chunks]), dtype=np.float32
            )
        else:
            if vectors.shape[0] != len(chunks):
                raise ValueError(
                    f"vectors rows ({vectors.shape[0]}) must match chunks ({len(chunks)})"
                )
            raw = np.asarray(vectors, dtype=np.float32).copy()
        self._dense = self._dense_from(raw)
        self._lexical = self._build_lexical(chunks)

    @staticmethod
    def _dense_from(matrix: np.ndarray) -> DenseIndex:
        import faiss

        vectors = matrix.copy()
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    @staticmethod
    def _build_dense(chunks: list[Chunk], embedder: Embedder) -> DenseIndex:
        raw = np.asarray(
            embedder.embed_batch([chunk.text for chunk in chunks]), dtype=np.float32
        )
        return Index._dense_from(raw)
```

(`_build_dense` giữ nguyên chữ ký; thân delegate sang `_dense_from`. Method `_rrf_fusion` cũ giữ nguyên vì hoạt động trên list int.)

- [ ] **Step 4: Chạy test verify pass**

Run: `uv run pytest tests/test_attachments.py tests/test_rag_core.py -v`
Expected: PASS (hành vi index cũ không đổi)

- [ ] **Step 5: mypy + commit**

```bash
uv run mypy
git add rag_core/index.py tests/test_attachments.py
git commit -m "feat(index): accept precomputed dense vectors + module-level rrf_merge"
```

---

### Task 5: RagCore tích hợp attachment retrieval

**Files:**
- Modify: `rag_core/__init__.py`
- Modify: `rag_core/intent.py` (prompt classifier — cùng seam answer())
- Test: `tests/test_rag_core.py`

**Interfaces:**
- Consumes: `AttachmentStore` API (Task 3), `Index(chunks, embedder, vectors=…)` + `rrf_merge` (Task 4), `APP_DB_FILENAME` (Task 1).
- Produces:
  - `RagCore.__init__(…, attachment_store: AttachmentStore | None = None)`
  - `RagCore.embedder` property → `Embedder` (Task 6 dùng share embedder cho UI store)
  - `_retrieve_sources(query_text, query_vector, session_id=None)` — merge KB + attachment rồi dedupe top-5; degrade về KB-only khi lỗi.
  - `build_rag_core(config=None)` tự wire `AttachmentStore(config.data_dir / APP_DB_FILENAME, embedder)`.

- [ ] **Step 1: Viết test fail** — thêm vào `tests/test_rag_core.py`:

```python
from rag_core.attachments import AttachmentStore
from rag_core.db import Database


def make_core_with_attachment(
    tmp_path: Path,
) -> tuple[RagCore, gd.Dataset, AttachmentStore]:
    ds = gd.generate(seed=SEED)
    gd.write(ds, tmp_path)
    embedder = RecordingEmbedder()
    db_path = tmp_path / "test.db"
    Database(db_path)
    store = AttachmentStore(db_path, embedder)
    core = RagCore(
        data_dir=tmp_path,
        embedder=embedder,
        generator=FakeGenerator(),
        judge=FakeJudge(levels=["high"]),
        session_rewriter=None,
        attachment_store=store,
    )
    return core, ds, store


UPLOAD_TOKENS = "zzqwub florpquux xylophane"


def test_answer_cites_uploaded_attachment(tmp_path: Path) -> None:
    core, _, store = make_core_with_attachment(tmp_path)
    store.add("sess-upload", "ghichep.txt", UPLOAD_TOKENS.encode())

    result = core.answer(
        f"giải thích {UPLOAD_TOKENS}", Session(id="sess-upload", user_id="u1")
    )

    assert result.sources, "expected at least one source"
    assert any(s.kind == "upload" for s in result.sources)


def test_attachment_source_has_position_chapter(tmp_path: Path) -> None:
    core, _, store = make_core_with_attachment(tmp_path)
    store.add("sess-upload", "ghichep.txt", UPLOAD_TOKENS.encode())

    result = core.answer(UPLOAD_TOKENS, Session(id="sess-upload", user_id="u1"))

    upload_sources = [s for s in result.sources if s.kind == "upload"]
    assert all(s.chapter == "Nội dung" for s in upload_sources)
    assert all(s.document_id.startswith("upload_") for s in upload_sources)


def test_kb_only_when_no_attachments(tmp_path: Path) -> None:
    core, ds, _ = make_core_with_attachment(tmp_path)
    doc, chapter, topic = ground_truth(ds, "CS112")
    query = f"giải thích {topic}" if doc.language == "vi" else f"explain {topic}"

    result = core.answer(query, session())

    assert result.sources
    assert any(s.kind != "upload" for s in result.sources)


def test_attachment_error_degrades_to_kb_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core, ds, store = make_core_with_attachment(tmp_path)
    doc, chapter, topic = ground_truth(ds, "CS112")

    def boom(conversation_id: str) -> tuple[str, ...]:
        raise RuntimeError("db gone")

    monkeypatch.setattr(store, "identity", boom)
    query = f"giải thích {topic}" if doc.language == "vi" else f"explain {topic}"

    result = core.answer(query, session())

    assert result.sources
    assert all(s.kind != "upload" for s in result.sources)


def test_refine_pass_also_searches_attachments(tmp_path: Path) -> None:
    # Judge chấm low lần 1 → Refine viết lại → high lần 2; refined retrieval vẫn thấy attachment.
    ds = gd.generate(seed=SEED)
    gd.write(ds, tmp_path)
    embedder = RecordingEmbedder()
    Database(tmp_path / "test.db")
    store = AttachmentStore(tmp_path / "test.db", embedder)
    store.add("sess-upload", "ghichep.txt", UPLOAD_TOKENS.encode())
    core = RagCore(
        data_dir=tmp_path,
        embedder=embedder,
        generator=FakeGenerator(),
        judge=FakeJudge(levels=["low", "high"]),
        rewriter=FakeRewriter(rewritten=UPLOAD_TOKENS),
        session_rewriter=None,
        attachment_store=store,
    )

    result = core.answer("câu hỏi mơ hồ", Session(id="sess-upload", user_id="u1"))

    assert any(s.kind == "upload" for s in result.sources)
```

Kiểm tra `pytest` đã import sẵn ở đầu file test (có). `FakeRewriter.__init__(rewritten=...)` nhận kwarg `rewritten` (xác nhận ở `tests/test_rag_core.py:131`).

- [ ] **Step 2: Chạy test verify fail**

Run: `uv run pytest tests/test_rag_core.py -v -k "upload or attachment or kb_only or refine_pass"`
Expected: FAIL — `TypeError: RagCore.__init__() got an unexpected keyword argument 'attachment_store'`

- [ ] **Step 3: Implement RagCore** — sửa `rag_core/__init__.py`:

Imports thêm ở đầu:

```python
import logging

from rag_core.attachments import AttachmentStore
from rag_core.db import APP_DB_FILENAME
from rag_core.index import FINAL_TOP_K, Index, dedupe_by_source
from rag_core.index import rrf_merge as rrf_merge_items
```

(Điều chỉnh: file đã có `from rag_core.index import Index` — gộp thành một import duy nhất như trên, xóa dòng cũ.)

Constructor — thêm param `attachment_store: AttachmentStore | None = None` (cuối danh sách param) và field:

```python
        self._attachments = attachment_store
        self._attachment_cache: dict[str, tuple[tuple[str, ...], Index]] = {}
```

Property (đặt cạnh các method public):

```python
    @property
    def embedder(self) -> Embedder:
        """The shared embedder; the UI's AttachmentStore reuses this instance."""
        return self._embedder
```

Thay `_retrieve_sources` cũ bằng:

```python
    def _retrieve_sources(
        self, query_text: str, query_vector: np.ndarray, session_id: str | None = None
    ) -> list[Chunk]:
        """Hybrid retrieval over course KB fused with this session's attachments."""
        kb_ranked = [
            self._index.chunks[i]
            for i in self._index.fused_ranking(query_text, query_vector)
        ]
        if session_id is not None:
            att_ranked = self._attachment_ranking(query_text, query_vector, session_id)
            if att_ranked:
                kb_ranked = rrf_merge_items([kb_ranked, att_ranked])
        return dedupe_by_source(kb_ranked, FINAL_TOP_K)

    def _attachment_ranking(
        self, query_text: str, query_vector: np.ndarray, session_id: str
    ) -> list[Chunk]:
        if self._attachments is None:
            return []
        try:
            identity = self._attachments.identity(session_id)
            if not identity:
                return []
            cached = self._attachment_cache.get(session_id)
            if cached is not None and cached[0] == identity:
                att_index = cached[1]
            else:
                chunks = self._attachments.load_chunks(session_id)
                vectors = self._attachments.load_vectors(session_id)
                att_index = Index(chunks, self._embedder, vectors=vectors)
                self._attachment_cache[session_id] = (identity, att_index)
            return [
                att_index.chunks[i]
                for i in att_index.fused_ranking(query_text, query_vector)
            ]
        except Exception:
            logging.getLogger(__name__).warning(
                "attachment retrieval failed; falling back to course KB only",
                exc_info=True,
            )
            return []
```

`answer()` — đổi dòng retrieve thành:

```python
        retrieved = self._retrieve_sources(query, query_vector, session.id)
```

và dòng gọi gated:

```python
            return self._gated_answer(query, retrieved, self._judge, self._rewriter, session.id)
```

`_gated_answer` — thêm param cuối `session_id: str | None = None`, đổi dòng refine retrieval thành:

```python
        refined_chunks = self._retrieve_sources(refined, refined_vector, session_id)
```

`build_rag_core` — trước `return RagCore(...)`:

```python
    attachment_store = AttachmentStore(config.data_dir / APP_DB_FILENAME, embedder)
```

thêm kwargs `attachment_store=attachment_store,` vào constructor call.

`rag_core/intent.py` — sửa `_CLASSIFIER_SYSTEM` dòng Examples:

```python
    "Examples: 'AI có prerequisite gì?'->COURSE_ADVISOR, 'Tôi còn thiếu gì để học AI?'->COURSE_ADVISOR, "
    "'Tôi nên học môn nào tiếp theo?'->COURSE_ADVISOR, 'Giải thích bảng băm'->KNOWLEDGE_QA, "
    "'Tài liệu này nói về gì?'->KNOWLEDGE_QA, 'Tóm tắt file mình đính kèm'->KNOWLEDGE_QA, "
    "'Thời tiết?'->OTHER"
```

Export `AttachmentStore` trong `__all__` của `rag_core/__init__.py` (thêm `"AttachmentStore"`).

- [ ] **Step 4: Chạy test verify pass**

Run: `uv run pytest tests/test_rag_core.py -v && uv run pytest tests/test_intent.py -v`
Expected: PASS toàn bộ (test cũ không đổi hành vi)

- [ ] **Step 5: Suite + mypy + commit**

```bash
uv run pytest && uv run mypy
git add rag_core/__init__.py rag_core/intent.py tests/test_rag_core.py
git commit -m "feat(core): fuse attachment retrieval into RagCore.answer via RRF with KB-only degrade"
```

---

### Task 6: UI sidebar upload + citation nguồn đính kèm

**Files:**
- Modify: `app.py`
- Modify: `docs/SETUP.md`, `README.md`

**Interfaces:**
- Consumes: `AttachmentStore.add/list_for/delete`, `MAX_FILES_PER_CONVERSATION` (Task 3), `APP_DB_FILENAME` (Task 1), `RagCore.embedder` property + `build_rag_core` wire sẵn (Task 5).

- [ ] **Step 1: Imports + factory trong `app.py`**

Imports:

```python
from rag_core.attachments import MAX_FILES_PER_CONVERSATION, AttachmentStore
from rag_core.db import APP_DB_FILENAME, Database
```

Factory mới (cạnh `get_database`):

```python
@st.cache_resource
def get_attachment_store() -> AttachmentStore:
    config = load_config()
    return AttachmentStore(config.data_dir / APP_DB_FILENAME, get_core().embedder)
```

- [ ] **Step 2: `render_attachments` + gọi trong sidebar**

```python
_KIND_ICONS = {"pdf": "📄", "md": "📝", "txt": "🗒️"}


def render_attachments() -> None:
    session_id = st.session_state.get("active_conversation_id")
    if not isinstance(session_id, str) or not session_id:
        return
    store = get_attachment_store()
    metas = store.list_for(session_id)
    label = f"📎 Tài liệu đính kèm ({len(metas)}/{MAX_FILES_PER_CONVERSATION})"
    with st.sidebar.expander(label, expanded=bool(metas)):
        for meta in metas:
            icon = _KIND_ICONS.get(meta.file_kind, "📄")
            cols = st.columns([4, 1])
            cols[0].caption(
                f"{icon} {meta.filename}\n\n{meta.chunk_count} đoạn · "
                f"{max(1, meta.size_bytes // 1024)} KB"
            )
            with cols[1].popover("🗑️", use_container_width=True):
                st.write(f"Xóa {meta.filename}?")
                if st.button("Xóa", key=f"del_att_{meta.id}", type="primary"):
                    try:
                        store.delete(meta.id)
                    except KeyError:
                        pass
                    st.rerun()
        if len(metas) >= MAX_FILES_PER_CONVERSATION:
            st.caption(
                f"Đã đạt giới hạn {MAX_FILES_PER_CONVERSATION} tài liệu cho chat này."
            )
            return
        uploaded = st.file_uploader(
            "Thêm tài liệu (PDF/TXT/MD)", type=["pdf", "txt", "md"], key="att_uploader"
        )
        if uploaded is not None:
            try:
                user = st.session_state.get("user")
                language = user.language if user is not None else "vi"
                with st.spinner("Đang parse và embedding…"):
                    meta = store.add(session_id, uploaded.name, uploaded.getvalue(), language)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.toast(f"Đã xử lý {meta.filename}: {meta.chunk_count} đoạn.")
                st.rerun()
```

Trong `render_sidebar`, sau khối lịch sử chat (sau vòng lặp `for meta in metas` và `st.divider()` hiện có ngay trước caption user), chèn:

```python
        render_attachments()
        st.divider()
```

(xóa divider trùng lặp nếu sinh ra hai cái liền nhau).

- [ ] **Step 3: Citation hiển thị nguồn đính kèm** — trong `render_citations`, sau dòng `st.write(f"Chapter: …")`:

```python
            origin = "tài liệu đính kèm" if source.kind == "upload" else "kho tài liệu môn học"
            st.write(f"Nguồn: {origin}")
```

- [ ] **Step 4: Cập nhật docs**

`docs/SETUP.md` — thêm mục mới (giữ style hiện có):

```markdown
## Upload tài liệu (PDF/TXT/MD)

- Trong sidebar mở **📎 Tài liệu đính kèm** của chat đang chọn: tối đa 3 file, mỗi file ≤ 5 MB (pdf/txt/md).
- Nội dung được parse, chia đoạn, embedding và lưu theo chat; câu trả lời trích dẫn kèm vị trí (`Trang N` / heading).
- Dependency mới: `pypdf` (đã có trong pyproject; chạy lại `uv sync` là đủ).
```

`README.md` — thêm 1 bullet vào mục tính năng:

```markdown
- Upload PDF/TXT/MD làm nguồn tri thức tạm thời theo cuộc trò chuyện (citation kèm trang/heading).
```

- [ ] **Step 5: Verify thủ công**

```bash
uv run pytest && uv run mypy
uv run streamlit run app.py
```

Checklist (browser):
1. Login → tạo chat mới → sidebar thấy "📎 Tài liệu đính kèm (0/3)".
2. Upload `test.txt` → toast success + meta đúng số đoạn.
3. Upload lại cùng tên → error "Đã tồn tại tài liệu tên…" (không crash).
4. Hỏi câu liên quan nội dung file → citation `[1] test.txt — Nội dung`, expander ghi "Nguồn: tài liệu đính kèm".
5. Xóa file qua popover → biến mất khỏi danh sách.
6. Restart app → hỏi lại câu đó → vẫn cite được (không re-embed).
7. Upload PDF scan rỗng → error "Không trích xuất được…".

- [ ] **Step 6: Commit**

```bash
git add app.py docs/SETUP.md README.md
git commit -m "feat(ui): sidebar attachment upload/delete + attachment-aware citations"
```

---

### Task 7: Smoke test end-to-end

**Files:** không sửa file bắt buộc; chỉ kiểm chứng.

- [ ] **Step 1: Toàn bộ test + typecheck**

```bash
uv run pytest -v && uv run mypy
```
Expected: PASS toàn bộ.

- [ ] **Step 2: Kiểm tra giới hạn bằng script ad-hoc (không commit)**

```bash
uv run python - <<'EOF'
import tempfile
from pathlib import Path

from rag_core.attachments import AttachmentStore
from rag_core.db import Database
from tests.test_attachments import HashEmbedder

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "t.db"
    Database(p)
    s = AttachmentStore(p, HashEmbedder())
    for i in range(3):
        s.add("c", f"f{i}.txt", f"noi dung {i}".encode())
    try:
        s.add("c", "g.txt", b"x")
    except ValueError as e:
        print("OK limit:", e)
EOF
```
Expected: in `OK limit: Mỗi chat tối đa 3 tài liệu.`

- [ ] **Step 3: Review diff tổng đối chiếu spec mục 2–6**

```bash
git log --oneline -7
```
Tìm commit ngay trước `feat(db): add attachments tables…` (commit đầu của plan), rồi:

```bash
git diff <hash-commit-đó>..HEAD --stat
```
Đối chiếu: parsers ✓, store ✓, fusion ✓, UI ✓, cascade ✓, docs ✓.

- [ ] **Step 4: Commit dọn dẹp (nếu còn file chưa track)**

```bash
git status --short
git add -A -- ':!docs/superpowers'
git commit -m "chore: finalize document upload feature" --allow-empty
```
