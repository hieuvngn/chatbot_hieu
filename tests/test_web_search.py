from __future__ import annotations

import pytest

from rag_core.web_search import (
    WEB_MAX_CHARS,
    WEB_TOP_K,
    FirecrawlWebSearcher,
    parse_search_payload,
)

FLAT_PAYLOAD = {
    "success": True,
    "data": [
        {
            "url": "https://example.com/hash-tables",
            "title": "Hash tables explained",
            "description": "A primer on hash tables.",
            "markdown": "# Hash tables\n\nBucket arrays.",
        }
    ],
}

GROUPED_PAYLOAD = {
    "success": True,
    "data": {
        "web": [
            {"url": "https://example.com/btree", "title": "B-trees", "markdown": "B-tree pages."}
        ]
    },
}


def test_parse_flat_shape_maps_web_source_fields() -> None:
    chunks = parse_search_payload(FLAT_PAYLOAD)

    assert len(chunks) == 1
    source = chunks[0].source
    assert source.document_id == "https://example.com/hash-tables"
    assert source.document_title == "Hash tables explained"
    assert source.chapter == "https://example.com/hash-tables"
    assert source.course_code == ""
    assert source.kind == "web"
    assert source.language == ""
    assert chunks[0].text.startswith("# Hash tables")


def test_parse_grouped_web_shape() -> None:
    chunks = parse_search_payload(GROUPED_PAYLOAD)

    assert len(chunks) == 1
    assert chunks[0].source.document_id == "https://example.com/btree"


def test_parse_failure_payload_returns_empty() -> None:
    assert parse_search_payload({"success": False, "error": "boom"}) == []
    assert parse_search_payload({"data": "junk"}) == []


def test_parse_truncates_markdown_to_limit() -> None:
    payload = {
        "success": True,
        "data": [{"url": "https://e.com/x", "title": "x", "markdown": "a" * (WEB_MAX_CHARS + 500)}],
    }

    chunks = parse_search_payload(payload)

    assert len(chunks[0].text) == WEB_MAX_CHARS


def test_parse_falls_back_to_description_then_drops_empty() -> None:
    payload = {
        "success": True,
        "data": [
            {"url": "https://e.com/desc-only", "title": "t", "markdown": "", "description": "desc here"},
            {"url": "", "title": "no url"},
            {"url": "https://e.com/no-content", "title": "empty"},
        ],
    }

    chunks = parse_search_payload(payload)

    assert [c.text for c in chunks] == ["desc here"]
    assert chunks[0].source.document_title == "t"


def test_constants_match_spec() -> None:
    assert WEB_TOP_K == 5
    assert WEB_MAX_CHARS == 4000


def test_search_parses_via_post_json(monkeypatch: pytest.MonkeyPatch) -> None:
    searcher = FirecrawlWebSearcher(api_key="fc-test")
    seen: dict[str, object] = {}

    def fake_post(path: str, body: dict[str, object]) -> dict[str, object]:
        seen["path"], seen["body"] = path, body
        return FLAT_PAYLOAD

    monkeypatch.setattr(searcher, "_post_json", fake_post)

    chunks = searcher.search("bảng băm là gì", k=3)

    assert len(chunks) == 1
    assert seen["path"] == "/search"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["query"] == "bảng băm là gì"
    assert body["limit"] == 3
    scrape = body["scrapeOptions"]
    assert isinstance(scrape, dict)
    assert scrape == {"formats": ["markdown"], "onlyMainContent": True}


def test_search_swallows_transport_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    searcher = FirecrawlWebSearcher(api_key="fc-test")

    def boom(path: str, body: dict[str, object]) -> dict[str, object]:
        raise OSError("network down")

    monkeypatch.setattr(searcher, "_post_json", boom)

    assert searcher.search("anything") == []
