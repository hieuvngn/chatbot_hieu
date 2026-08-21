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
