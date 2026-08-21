from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from rag_core.attachments import MAX_FILE_BYTES, AttachmentStore, parse_file
from rag_core.db import Database


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


def test_corrupt_pdf_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Không trích xuất được"):
        parse_file("broken.pdf", b"%PDF-1.4 this is not really a pdf")


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


def make_store(tmp_path: Path) -> AttachmentStore:
    db_path = tmp_path / "test.db"
    Database(db_path)
    return AttachmentStore(db_path, HashEmbedder())


def test_add_persists_meta_and_chunks(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    meta = store.add("conv1", "note.txt", "bảng băm zzq wub florp".encode())
    assert meta.filename == "note.txt"
    assert meta.file_kind == "txt"
    assert meta.chunk_count >= 1
    assert [m.id for m in store.list_for("conv1")] == [meta.id]


def test_add_rejects_wrong_extension(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="không hỗ trợ"):
        store.add("conv1", "img.jpg", b"data")


def test_add_rejects_oversize(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="5 MB"):
        store.add("conv1", "big.txt", b"x" * (MAX_FILE_BYTES + 1))


def test_add_rejects_duplicate_filename_case_insensitive(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add("conv1", "Note.txt", "abc".encode())
    with pytest.raises(ValueError, match="tồn tại"):
        store.add("conv1", "NOTE.txt", "abc".encode())


def test_add_rejects_when_limit_reached(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    for i in range(3):
        store.add("conv1", f"f{i}.txt", f"nội dung {i}".encode())
    with pytest.raises(ValueError, match="tối đa"):
        store.add("conv1", "extra.txt", "nội dung".encode())


def test_embed_failure_leaves_no_rows(tmp_path: Path) -> None:
    class BoomEmbedder(HashEmbedder):
        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("api down")

    db_path = tmp_path / "test.db"
    Database(db_path)
    store = AttachmentStore(db_path, BoomEmbedder())
    with pytest.raises(RuntimeError):
        store.add("conv1", "note.txt", "abc".encode())
    assert store.list_for("conv1") == []


def test_load_chunks_and_vectors_roundtrip(tmp_path: Path) -> None:
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


def test_document_id_stable_across_reloads(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add("conv1", "a.txt", "nội dung ổn định".encode())
    ids_first = [c.source.document_id for c in store.load_chunks("conv1")]
    ids_second = [c.source.document_id for c in store.load_chunks("conv1")]
    assert ids_first == ids_second


def test_identity_changes_on_add_and_survives_reload(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    before = store.identity("conv1")
    store.add("conv1", "a.txt", "nội dung".encode())
    after = store.identity("conv1")
    fresh = AttachmentStore(tmp_path / "test.db", HashEmbedder())
    assert before == ()
    assert len(after) == 1
    assert fresh.identity("conv1") == after


def test_delete_removes_chunks_and_metadata(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    meta = store.add("conv1", "a.txt", "nội dung".encode())
    store.delete(meta.id)
    assert store.list_for("conv1") == []
    assert store.load_chunks("conv1") == []
    assert store.identity("conv1") == ()
    with pytest.raises(KeyError):
        store.delete(meta.id)


def test_chunks_isolated_between_conversations(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.add("conv1", "a.txt", "nội dung một".encode())
    store.add("conv2", "b.txt", "nội dung hai".encode())
    titles1 = [c.source.document_title for c in store.load_chunks("conv1")]
    assert titles1 == ["a.txt"]
