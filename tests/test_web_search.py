from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

import generate_data as gd
from rag_core import RagCore, Session
from rag_core.intent import Extraction, Intent
from rag_core.judge import Judgment, Level
from rag_core.models import Chunk, Source
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


DIM = 32
SEED = 42


class HashEmbedder:
    """Deterministic md5 vectors — ranking quality irrelevant for these tests."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        vec = [(digest[i % 16] / 255.0) - 0.5 for i in range(DIM)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class FakeGenerator:
    def generate(
        self,
        question: str,
        chunks: list[Chunk],
        feedback: str | None = None,
        *,
        skill_instructions: str = "",
    ) -> str:
        return f"Trả lời về {chunks[0].source.document_title}: {chunks[0].text[:40]} [1]"


class ScriptedJudge:
    """Returns verdicts from a script; records every call."""

    def __init__(self, levels: list[Level], suggestion: str = "Hãy hỏi lại với từ khóa khác.") -> None:
        self.levels = levels
        self.suggestion = suggestion
        self.calls: list[tuple[str, list[Chunk]]] = []

    def assess(self, query: str, chunks: list[Chunk]) -> Judgment:
        self.calls.append((query, list(chunks)))
        level = self.levels[min(len(self.calls) - 1, len(self.levels) - 1)]
        return Judgment(level=level, rephrase_suggestion=self.suggestion)


class FixedRewriter:
    def __init__(self, rewritten: str = "hash table") -> None:
        self.rewritten = rewritten
        self.calls: list[str] = []

    def rewrite(self, query: str) -> str:
        self.calls.append(query)
        return self.rewritten


WEB_TOKENS = "zzqwub florpquux xylophane"


def web_chunks() -> list[Chunk]:
    return [
        Chunk(
            source=Source(
                document_id=f"https://example.com/{i}",
                document_title=f"Web result {i}",
                chapter=f"https://example.com/{i}",
                course_code="",
                kind="web",
                language="",
            ),
            text=f"{WEB_TOKENS} page content number {i}.",
        )
        for i in range(1, 4)
    ]


class FakeWebSearcher:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []

    def search(self, query: str, k: int = WEB_TOP_K) -> list[Chunk]:
        if self.fail:
            raise RuntimeError("searcher down")
        self.queries.append(query)
        return web_chunks()


class AlwaysOtherClassifier:
    def classify(self, query: str) -> Intent:
        return "OTHER"


class NoopExtractor:
    def extract(self, query: str) -> Extraction:
        return Extraction(completed_courses=[], target_course=None, current_semester=None)


def make_web_core(
    tmp_path: Path,
    searcher: FakeWebSearcher | None = None,
    judge: ScriptedJudge | None = None,
    rewriter: FixedRewriter | None = None,
    classifier: AlwaysOtherClassifier | None = None,
) -> RagCore:
    ds = gd.generate(seed=SEED)
    gd.write(ds, tmp_path)
    if classifier is not None:
        return RagCore(
            tmp_path,
            HashEmbedder(),
            FakeGenerator(),
            judge=judge,
            rewriter=rewriter,
            web_searcher=searcher,
            classifier=classifier,
            extractor=NoopExtractor(),
        )
    return RagCore(
        tmp_path,
        HashEmbedder(),
        FakeGenerator(),
        judge=judge,
        rewriter=rewriter,
        web_searcher=searcher,
    )


def fresh_session() -> Session:
    return Session(id="s-web", user_id="u-web", turns=[])


def test_use_web_merges_web_results_into_retrieval(tmp_path: Path) -> None:
    searcher = FakeWebSearcher()
    core = make_web_core(tmp_path, searcher=searcher)

    result = core.answer(f"giải thích bảng băm {WEB_TOKENS}", fresh_session(), use_web=True)

    assert any(s.kind == "web" for s in result.sources), (
        f"web sources missing: {[s.kind for s in result.sources]}"
    )
    assert searcher.queries, "toggle ON must call the searcher once at retrieval"


def test_use_web_false_never_calls_searcher(tmp_path: Path) -> None:
    searcher = FakeWebSearcher()
    core = make_web_core(tmp_path, searcher=searcher)

    core.answer("giải thích bảng băm là gì?", fresh_session())

    assert searcher.queries == []


def test_searcher_failure_degrades_to_kb_only(tmp_path: Path) -> None:
    core = make_web_core(tmp_path, searcher=FakeWebSearcher(fail=True))

    result = core.answer("giải thích bảng băm là gì?", fresh_session(), use_web=True)

    assert not result.refused
    assert result.sources, "degradation must still retrieve from the KB"
    assert all(s.kind != "web" for s in result.sources)


def test_has_web_search_reflects_wiring(tmp_path: Path) -> None:
    assert make_web_core(tmp_path).has_web_search is False
    assert make_web_core(tmp_path, searcher=FakeWebSearcher()).has_web_search is True


def test_use_web_bypasses_other_intent_refusal(tmp_path: Path) -> None:
    searcher = FakeWebSearcher()
    core = make_web_core(
        tmp_path, searcher=searcher, classifier=AlwaysOtherClassifier()
    )

    result = core.answer("thời tiết hôm nay thế nào", fresh_session(), use_web=True)

    assert any(s.kind == "web" for s in result.sources)
    assert "Tôi chỉ hỗ trợ" not in result.answer


def test_other_without_web_still_refuses(tmp_path: Path) -> None:
    core = make_web_core(
        tmp_path, searcher=FakeWebSearcher(), classifier=AlwaysOtherClassifier()
    )

    result = core.answer("thời tiết hôm nay thế nào", fresh_session(), use_web=False)

    assert "Tôi chỉ hỗ trợ" in result.answer


class CiteAllGenerator:
    """Cites every provided source so per-kind citation assertions are deterministic."""

    def generate(
        self,
        question: str,
        chunks: list[Chunk],
        feedback: str | None = None,
        *,
        skill_instructions: str = "",
    ) -> str:
        return " ".join(f"[{i}]" for i in range(1, len(chunks) + 1))


def test_corrective_web_pass_answers_after_failed_refine(tmp_path: Path) -> None:
    searcher = FakeWebSearcher()
    judge = ScriptedJudge(levels=["low", "low", "high"])
    rewriter = FixedRewriter(rewritten="hash tables")
    ds = gd.generate(seed=SEED)
    gd.write(ds, tmp_path)
    core = RagCore(
        tmp_path,
        HashEmbedder(),
        CiteAllGenerator(),
        judge=judge,
        rewriter=rewriter,
        web_searcher=searcher,
    )

    result = core.answer("câu hỏi mơ hồ", fresh_session())

    assert not result.refused
    assert result.answer
    assert len(judge.calls) == 3, "initial + refine + corrective assessment"
    assert searcher.queries == ["hash tables"], "corrective searches the rewritten query"
    merged_sources = judge.calls[2][1]
    assert any(c.source.kind == "web" for c in merged_sources)
    assert any(c.source.kind != "web" for c in merged_sources), (
        "corrective merges local and web, it does not replace local"
    )
    assert any(c.source.kind == "web" for c in result.citations)


def test_corrective_still_failing_falls_back_to_refusal(tmp_path: Path) -> None:
    searcher = FakeWebSearcher()
    judge = ScriptedJudge(levels=["low", "low", "low"])
    rewriter = FixedRewriter()
    ds = gd.generate(seed=SEED)
    gd.write(ds, tmp_path)
    core = RagCore(
        tmp_path,
        HashEmbedder(),
        FakeGenerator(),
        judge=judge,
        rewriter=rewriter,
        web_searcher=searcher,
    )

    result = core.answer("câu hỏi mơ hồ", fresh_session())

    assert result.refused
    assert result.citations == []
    assert len(judge.calls) == 3


def test_corrective_skipped_when_no_searcher(tmp_path: Path) -> None:
    judge = ScriptedJudge(levels=["low", "low"])
    rewriter = FixedRewriter()
    ds = gd.generate(seed=SEED)
    gd.write(ds, tmp_path)
    core = RagCore(
        tmp_path,
        HashEmbedder(),
        FakeGenerator(),
        judge=judge,
        rewriter=rewriter,
    )

    result = core.answer("giải thích bảng băm là gì?", fresh_session())

    assert result.refused
    assert len(judge.calls) == 2, "behavior identical to pre-web-search pipeline"


def test_corrective_with_empty_search_result_keeps_refusal(tmp_path: Path) -> None:
    class EmptySearcher:
        def search(self, query: str, k: int = WEB_TOP_K) -> list[Chunk]:
            return []

    judge = ScriptedJudge(levels=["low", "low"])
    rewriter = FixedRewriter()
    ds = gd.generate(seed=SEED)
    gd.write(ds, tmp_path)
    core = RagCore(
        tmp_path,
        HashEmbedder(),
        FakeGenerator(),
        judge=judge,
        rewriter=rewriter,
        web_searcher=EmptySearcher(),
    )

    result = core.answer("câu hỏi mơ hồ", fresh_session())

    assert result.refused
    assert len(judge.calls) == 2, "empty web result must not waste a third judgment"


def test_load_config_reads_firecrawl_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from rag_core.config import load_config

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    config = load_config(env_file="/nonexistent/.env")
    assert config.firecrawl_api_key == ""

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-demo")
    config = load_config(env_file="/nonexistent/.env")
    assert config.firecrawl_api_key == "fc-demo"

    monkeypatch.setenv("FIRECRAWL_API_KEY", "  fc-spaces  ")
    config = load_config(env_file="/nonexistent/.env")
    assert config.firecrawl_api_key == "fc-spaces"


def test_worst_case_toggle_plus_corrective_makes_two_calls(tmp_path: Path) -> None:
    searcher = FakeWebSearcher()
    judge = ScriptedJudge(levels=["low", "low", "high"])
    rewriter = FixedRewriter(rewritten="hash tables")
    ds = gd.generate(seed=SEED)
    gd.write(ds, tmp_path)
    core = RagCore(
        tmp_path,
        HashEmbedder(),
        CiteAllGenerator(),
        judge=judge,
        rewriter=rewriter,
        web_searcher=searcher,
    )

    result = core.answer("câu hỏi mơ hồ", fresh_session(), use_web=True)

    assert not result.refused
    assert len(searcher.queries) == 2, (
        "worst case must respect the 2-call budget: retrieval + corrective"
    )
