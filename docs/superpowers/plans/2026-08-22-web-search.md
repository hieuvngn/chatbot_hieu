# Firecrawl Web Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add web search via the Firecrawl Search API to CourseMate — automatic corrective search when the CRAG judge still rejects retrieval after Refine, plus a manual sidebar toggle that merges web results into every retrieval.

**Architecture:** New `WebSearcher` protocol + `FirecrawlWebSearcher` client (`rag_core/web_search.py`, stdlib urllib, no new dependency). Web results are converted to regular `Chunk`s with `Source(kind="web", chapter=url)` so judge/generator/answer-check/citations/persistence run unchanged. Two hook points in `RagCore`: merge in `_retrieve_sources` when `use_web=True`, and one corrective web pass in `_gated_answer` before fallback/refusal.

**Tech Stack:** Python 3.11+, stdlib `urllib.request` (HTTP), pytest, mypy strict, Streamlit (UI only).

**Spec:** `docs/superpowers/specs/2026-08-22-web-search-design.md`

## Global Constraints

- No new pip dependency — Firecrawl called with stdlib `urllib.request`.
- Constants verbatim from spec: `WEB_TOP_K = 5`, `WEB_MAX_CHARS = 4000`, `FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v2"`, timeout 30s.
- Every searcher failure mode returns `[]` and logs a warning — web is strictly additive, never breaks an answer.
- Web chunks map to `Source(document_id=url, document_title=title or url, chapter=url, course_code="", kind="web", language="")`; text = `(markdown or description)[:4000]`.
- Missing `FIRECRAWL_API_KEY` → feature silently off; all existing behavior/tests unchanged.
- Max 2 Firecrawl calls per answer.
- All code passes `uv run mypy` strict (files already include `rag_core`, `app.py`, `tests`).
- Run tests with `uv run pytest`. Commit messages follow repo convention (`feat(scope): ...`).
- `docs/superpowers/` is gitignored by policy — spec/plan files are committed with `git add -f`.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `rag_core/web_search.py` | Create | `WebSearcher` protocol, response parsing, `FirecrawlWebSearcher` HTTP client |
| `tests/test_web_search.py` | Create | Parser tests + RagCore integration tests with fake searcher |
| `rag_core/__init__.py` | Modify | `web_searcher` wiring, `has_web_search`, `answer(use_web=)`, retrieval merge, corrective path |
| `rag_core/config.py` | Modify | `firecrawl_api_key` field + env read |
| `.env.example` | Modify | Documented optional key |
| `app.py` | Modify | Sidebar toggle + clickable web citations + pass `use_web` to `answer()` |
| `README.md` | Modify | User-facing docs section |

---

### Task 1: `FirecrawlWebSearcher` module

**Files:**
- Create: `rag_core/web_search.py`
- Test: `tests/test_web_search.py`

**Interfaces:**
- Consumes: `Chunk`, `Source` from `rag_core.models`.
- Produces:
  - `WEB_TOP_K: int = 5`, `WEB_MAX_CHARS: int = 4000`
  - `class WebSearcher(Protocol)`: `search(self, query: str, k: int = WEB_TOP_K) -> list[Chunk]`
  - `parse_search_payload(payload: dict[str, Any]) -> list[Chunk]` (module function)
  - `FirecrawlWebSearcher(api_key: str, base_url: str = FIRECRAWL_BASE_URL, timeout_s: float = 30.0)`: `.search(query, k) -> list[Chunk]`, `._post_json(path: str, body: dict[str, Any]) -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web_search.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag_core.web_search'`

- [ ] **Step 3: Implement `rag_core/web_search.py`**

```python
"""Web search via the Firecrawl Search API.

``WebSearcher`` is the pipeline seam: given a query it returns web results as
regular Chunks (Source kind="web"), so citations flow through the pipeline
unchanged. Any failure degrades to an empty list — web search is strictly
additive and never breaks an answer.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Protocol

from rag_core.models import Chunk, Source

WEB_TOP_K = 5
WEB_MAX_CHARS = 4000
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v2"
DEFAULT_TIMEOUT_S = 30.0


class WebSearcher(Protocol):
    """Searches the web and returns results as pipeline-ready Chunks."""

    def search(self, query: str, k: int = WEB_TOP_K) -> list[Chunk]: ...


def _web_chunk(
    url: str, title: str | None, description: str | None, markdown: str | None
) -> Chunk | None:
    text = (markdown or "").strip() or (description or "").strip()
    if not url or not text:
        return None
    return Chunk(
        source=Source(
            document_id=url,
            document_title=(title or "").strip() or url,
            chapter=url,
            course_code="",
            kind="web",
            language="",
        ),
        text=text[:WEB_MAX_CHARS],
    )


def parse_search_payload(payload: dict[str, Any]) -> list[Chunk]:
    """Flatten both known /search response shapes into Chunks.

    v1-style responses carry a flat ``data`` list; v2 groups entries under
    ``data.web``. Anything unexpected yields an empty list.
    """
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return []
    data = payload.get("data")
    if isinstance(data, dict):
        entries: list[Any] = list(data.get("web") or [])
    elif isinstance(data, list):
        entries = list(data)
    else:
        entries = []
    chunks: list[Chunk] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        chunk = _web_chunk(
            url=str(entry.get("url") or ""),
            title=entry.get("title"),
            description=entry.get("description"),
            markdown=entry.get("markdown"),
        )
        if chunk is not None:
            chunks.append(chunk)
    return chunks


class FirecrawlWebSearcher:
    """Concrete WebSearcher backed by POST {base_url}/search."""

    def __init__(
        self,
        api_key: str,
        base_url: str = FIRECRAWL_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def search(self, query: str, k: int = WEB_TOP_K) -> list[Chunk]:
        try:
            payload = self._post_json(
                "/search",
                {
                    "query": query,
                    "limit": k,
                    "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
                },
            )
            return parse_search_payload(payload)
        except Exception:
            logging.getLogger(__name__).warning(
                "firecrawl web search failed", exc_info=True
            )
            return []

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_web_search.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Type-check**

Run: `uv run mypy rag_core/web_search.py tests/test_web_search.py`
Expected: no issues

- [ ] **Step 6: Commit**

```bash
git add rag_core/web_search.py tests/test_web_search.py
git commit -m "feat(core): FirecrawlWebSearcher converts /v2/search results into pipeline chunks"
```

---

### Task 2: RagCore toggle path — `answer(use_web=)`, retrieval merge, OTHER bypass

**Files:**
- Modify: `rag_core/__init__.py`
- Test: `tests/test_web_search.py` (append)

**Interfaces:**
- Consumes: `WebSearcher` protocol from Task 1.
- Produces:
  - `RagCore.__init__(..., web_searcher: WebSearcher | None = None)`
  - `RagCore.has_web_search -> bool` (property)
  - `RagCore.answer(user_message: str, session: Session, use_web: bool = False) -> AnswerResult`
  - Internal: `_safe_web_search(query_text: str) -> list[Chunk]` used again by Task 3.

- [ ] **Step 1: Append failing tests to `tests/test_web_search.py`**

Replace the entire import block at the top of the file with:

```python
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
```

Add shared fakes and helpers below the existing parser tests:

```python
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
```

Also add these two import lines to the existing import block at the top of the file:

```python
from rag_core.answer_check import CheckVerdict  # noqa: F401  (not needed — skip this line)
from rag_core.intent import Extraction, Intent
from rag_core.judge import Judgment
from rag_core.models import Chunk, Source
```

(Only keep `Extraction, Intent`, `Judgment`, and extend the models import to include `Source` — do NOT add the CheckVerdict line.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web_search.py -v`
Expected: new tests FAIL — `TypeError: RagCore.__init__() got an unexpected keyword argument 'web_searcher'` (parser tests from Task 1 still PASS)

- [ ] **Step 3: Implement in `rag_core/__init__.py`**

3a. Add import next to the other rag_core imports:

```python
from rag_core.web_search import WebSearcher
```

3b. Constructor — add parameter after `skill_selector` and store it (in `RagCore.__init__`, around line 53–92):

```python
        skills: list[Skill] | None = None,
        skill_selector: SkillSelector | None = None,
        web_searcher: WebSearcher | None = None,
    ) -> None:
```

and at the end of `__init__` body:

```python
        self._skill_selector = skill_selector
        self._web_searcher = web_searcher
```

3c. Property (place right after the `embedder` property, around line 94–97):

```python
    @property
    def has_web_search(self) -> bool:
        """Whether a Firecrawl searcher is wired in; the UI gates its toggle on this."""
        return self._web_searcher is not None
```

3d. Replace `_retrieve_sources` (currently lines 106–118) with the generalized version plus the safe helper:

```python
    def _retrieve_sources(
        self,
        query_text: str,
        query_vector: np.ndarray,
        session_id: str | None = None,
        use_web: bool = False,
    ) -> list[Chunk]:
        """Hybrid retrieval over course KB fused with attachments and optional web results."""
        kb_ranked = [
            self._index.chunks[i]
            for i in self._index.fused_ranking(query_text, query_vector)
        ]
        ranked_lists = [kb_ranked]
        if session_id is not None:
            att_ranked = self._attachment_ranking(query_text, query_vector, session_id)
            if att_ranked:
                ranked_lists.append(att_ranked)
        if use_web:
            web_chunks = self._safe_web_search(query_text)
            if web_chunks:
                ranked_lists.append(web_chunks)
        fused = kb_ranked if len(ranked_lists) == 1 else rrf_merge_items(ranked_lists)
        return dedupe_by_source(fused, FINAL_TOP_K)

    def _safe_web_search(self, query_text: str) -> list[Chunk]:
        if self._web_searcher is None:
            return []
        try:
            return self._web_searcher.search(query_text)
        except Exception:
            logging.getLogger(__name__).warning(
                "web search failed; continuing without web sources", exc_info=True
            )
            return []
```

3e. Update `answer()` signature and OTHER condition (currently lines 148–161):

```python
    def answer(
        self, user_message: str, session: Session, use_web: bool = False
    ) -> AnswerResult:
        query = self._rewrite_for_session(user_message, session)
        use_web_effective = bool(use_web) and self.has_web_search
        if self._classifier is not None and self._extractor is not None and self._advisor is not None:
            intent = self._classifier.classify(query)
            if intent == "COURSE_ADVISOR":
                return self._advisor_branch(query)
            if (
                intent == "OTHER"
                and not self._session_has_attachments(session.id)
                and not use_web_effective
            ):
                return self._other_result(query)
        query_vector = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        retrieved = self._retrieve_sources(
            query, query_vector, session.id, use_web=use_web_effective
        )

        if self._judge is not None and self._rewriter is not None:
            return self._gated_answer(query, retrieved, self._judge, self._rewriter, session.id)
        return self._generate_result(query, retrieved)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_web_search.py -v`
Expected: PASS (all, including Task 1's)

- [ ] **Step 5: Regression — whole suite must stay green**

Run: `uv run pytest`
Expected: PASS (existing behavior unchanged because default `use_web=False`, `web_searcher=None`)

- [ ] **Step 6: Type-check**

Run: `uv run mypy rag_core tests`
Expected: no issues

- [ ] **Step 7: Commit**

```bash
git add rag_core/__init__.py tests/test_web_search.py
git commit -m "feat(core): use_web toggle merges Firecrawl results into hybrid retrieval"
```

---

### Task 3: Corrective web pass in `_gated_answer`

**Files:**
- Modify: `rag_core/__init__.py` (`_gated_answer`, around lines 319–343)
- Test: `tests/test_web_search.py` (append)

**Interfaces:**
- Consumes: `_safe_web_search` from Task 2; existing `Judge.assess`, `_is_acceptable`.
- Produces: internal `_corrective_merge(refined_query: str, refined_chunks: list[Chunk]) -> list[Chunk]` (RRF-fuses web chunks into refined retrieval, deduped top-5).

- [ ] **Step 1: Append failing tests to `tests/test_web_search.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web_search.py -v`
Expected: 4 new tests FAIL — judge receives only 2 calls today; `result.refused` where answer expected

- [ ] **Step 3: Implement the corrective block**

In `_gated_answer`, replace the tail (currently lines 330–343):

```python
        judgment = judge.assess(question, chunks)
        if judgment.is_high:
            return self._generate_result(question, chunks)
        # medium/low → one Refine (even permissive keeps the single retry)
        refined = rewriter.rewrite(question)
        refined_vector = np.asarray(
            self._embedder.embed_query(refined), dtype=np.float32
        )
        refined_chunks = self._retrieve_sources(refined, refined_vector, session_id)
        judgment = judge.assess(refined, refined_chunks)
        if self._is_acceptable(judgment):
            return self._generate_result(refined, refined_chunks)
        # CRAG correction: one web-search pass before giving up
        if self._web_searcher is not None:
            merged = self._corrective_merge(refined, refined_chunks)
            if merged:
                judgment = judge.assess(refined, merged)
                if self._is_acceptable(judgment):
                    return self._generate_result(refined, merged)
        if self._enable_fallback:
            return self._fallback_result(refined)
        return self._refused_result(
            judgment.rephrase_suggestion or DEFAULT_REPHRASE_SUGGESTION
        )
```

and add the helper method right after `_gated_answer`:

```python
    def _corrective_merge(
        self, refined_query: str, refined_chunks: list[Chunk]
    ) -> list[Chunk]:
        """One CRAG correction step: fuse a fresh web search into the refined retrieval."""
        web_chunks = self._safe_web_search(refined_query)
        if not web_chunks:
            return []
        merged = rrf_merge_items([refined_chunks, web_chunks])
        return dedupe_by_source(merged, FINAL_TOP_K)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_web_search.py -v`
Expected: PASS (all)

- [ ] **Step 5: Full regression + types**

Run: `uv run pytest && uv run mypy`
Expected: PASS — existing gated-pipeline tests unchanged (no searcher wired → corrective skipped)

- [ ] **Step 6: Commit**

```bash
git add rag_core/__init__.py tests/test_web_search.py
git commit -m "feat(core): corrective Firecrawl web pass before fallback/refusal"
```

---

### Task 4: Config key + build wiring

**Files:**
- Modify: `rag_core/config.py`
- Modify: `rag_core/__init__.py` (`build_rag_core`)
- Modify: `.env.example`
- Test: append to `tests/test_web_search.py`

**Interfaces:**
- Consumes: `FirecrawlWebSearcher`, `WebSearcher` from Task 1; `Config.firecrawl_api_key` defined here.
- Produces: `Config.firecrawl_api_key: str = ""`; env var `FIRECRAWL_API_KEY` (optional); `build_rag_core()` wires a real searcher iff the key is non-empty.

- [ ] **Step 1: Write the failing config test**

Append to `tests/test_web_search.py`:

```python
def test_load_config_reads_firecrawl_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from rag_core.config import load_config

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    config = load_config(env_file="/nonexistent/.env")
    assert config.firecrawl_api_key == ""

    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-demo")
    config = load_config(env_file="/nonexistent/.env")
    assert config.firecrawl_api_key == "fc-demo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web_search.py::test_load_config_reads_firecrawl_key -v`
Expected: FAIL — `Config` has no attribute `firecrawl_api_key`

- [ ] **Step 3: Implement**

In `rag_core/config.py` — add the field to `Config` (after `enable_fallback`, around line 23):

```python
@dataclass(frozen=True)
class Config:
    api_key: str
    data_dir: Path = DEFAULT_DATA_DIR
    llm_model: str = DEFAULT_LLM_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dim: int = DEFAULT_EMBED_DIM
    base_url: str = DEFAULT_BASE_URL
    allow_medium: bool = True
    enable_fallback: bool = True
    firecrawl_api_key: str = ""
```

and add to the returned `Config(...)` in `load_config()` (around line 57–66):

```python
        allow_medium=_parse_bool("RAG_ALLOW_MEDIUM", True),
        enable_fallback=_parse_bool("RAG_ENABLE_FALLBACK", True),
        firecrawl_api_key=os.environ.get("FIRECRAWL_API_KEY", ""),
    )
```

In `rag_core/__init__.py`, extend the Task 2 import line to:

```python
from rag_core.web_search import FirecrawlWebSearcher, WebSearcher
```

In the same file's `build_rag_core()` — create the searcher just before the `return RagCore(...)` (after `attachment_store = ...`, around line 475):

```python
    web_searcher: WebSearcher | None = (
        FirecrawlWebSearcher(config.firecrawl_api_key)
        if config.firecrawl_api_key
        else None
    )
```

and add `web_searcher=web_searcher,` to the `return RagCore(...)` argument list (around line 484–500).

In `.env.example`, append at the end:

```sh
# Web search via Firecrawl (optional) — get a key at https://firecrawl.dev
# Without it the chatbot runs exactly as before; get_core().has_web_search stays false.
FIRECRAWL_API_KEY=
```

- [ ] **Step 4: Run tests + types**

Run: `uv run pytest tests/test_web_search.py && uv run mypy`
Expected: PASS

- [ ] **Step 5: Smoke-check the wiring offline (optional but cheap)**

Run: `FIRECRAWL_API_KEY=fake uv run python -c "from rag_core.config import load_config; print(load_config(env_file='/nonexistent/.env').firecrawl_api_key)"`
Expected: prints `fake` (full end-to-end construction needs OPENROUTER_API_KEY; the wiring itself is covered by mypy + Task 2/3 unit tests)

- [ ] **Step 6: Commit**

```bash
git add rag_core/config.py rag_core/__init__.py .env.example tests/test_web_search.py
git commit -m "feat(config): optional FIRECRAWL_API_KEY wires FirecrawlWebSearcher into build_rag_core"
```

---

### Task 5: Streamlit UI — sidebar toggle + clickable web citations

**Files:**
- Modify: `app.py` (`render_citations` ~line 101, `render_sidebar` ~line 236, `render_chat` ~line 298)

**Interfaces:**
- Consumes: `core.has_web_search` property and `answer(..., use_web=)` from Task 2 (Task 5 depends on Tasks 2 & 4 being merged).
- Produces: user-facing toggle `"Tìm kiếm web"` persisted in `st.session_state["use_web"]`.

No unit tests exist for `app.py` in this repo (Streamlit widgets need a live runtime) — verification is mypy strict + manual smoke run below.

- [ ] **Step 1: Toggle in `render_sidebar`**

Replace (lines 235–237):

```python
        st.divider()
        render_attachments()
        st.divider()
```

with:

```python
        st.divider()
        render_attachments()
        st.divider()
        if get_core().has_web_search:
            st.toggle(
                "Tìm kiếm web",
                key="use_web",
                help="Mỗi câu hỏi được bổ sung kết quả từ web qua Firecrawl.",
            )
        else:
            st.caption("🌐 Thêm FIRECRAWL_API_KEY vào .env để bật tìm kiếm web.")
        st.divider()
```

- [ ] **Step 2: Pass the flag in `render_chat`**

Replace (line 298):

```python
            result = get_core().answer(prompt, session)
```

with:

```python
            result = get_core().answer(
                prompt, session, use_web=bool(st.session_state.get("use_web"))
            )
```

- [ ] **Step 3: Clickable link inside web-citation expanders**

In `render_citations` (lines 101–114), replace the expander body:

```python
        with st.expander(label):
            st.write(f"Document: {source.document_title} ({source.document_id})")
            st.write(f"Chapter: {source.chapter}")
            origin = "tài liệu đính kèm" if source.kind == "upload" else "kho tài liệu môn học"
            st.write(f"Nguồn: {origin}")
            st.write(f"Course: {source.course_code}")
            st.write(f"Kind: {source.kind} ({source.language})")
```

with:

```python
        with st.expander(label):
            st.write(f"Document: {source.document_title} ({source.document_id})")
            if source.kind == "web":
                st.markdown(f"Nguồn: [{source.chapter}]({source.chapter})")
            else:
                st.write(f"Chapter: {source.chapter}")
                origin = (
                    "tài liệu đính kèm" if source.kind == "upload" else "kho tài liệu môn học"
                )
                st.write(f"Nguồn: {origin}")
                st.write(f"Course: {source.course_code}")
                st.write(f"Kind: {source.kind} ({source.language})")
```

(The expander label `[n] title — chapter(url)` already renders the URL via the unchanged `chapter` field.)

- [ ] **Step 4: Type-check**

Run: `uv run mypy app.py`
Expected: no issues

- [ ] **Step 5: Manual smoke run (requires OPENROUTER_API_KEY; FIRECRAWL_API_KEY optional)**

```sh
uv run streamlit run app.py
```

Checklist:
1. With no `FIRECRAWL_API_KEY` → sidebar shows the 🌐 caption, chat behaves exactly as before.
2. Set `FIRECRAWL_API_KEY=<real key>` → restart → sidebar shows the toggle.
3. Toggle ON, ask something the corpus lacks (e.g. current events) → answer cites expanders labeled `[n] … — https://…`; clicking opens the page.
4. Toggle OFF → normal answers, no extra latency.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat(ui): web search sidebar toggle and clickable web citations"
```

---

### Task 6: README + final verification gate

**Files:**
- Modify: `README.md` (after the "CRAG judge" section, before "Answer check")

- [ ] **Step 1: Add README section**

Insert after the CRAG demo-script block (before `## Answer check`):

```markdown
## Web search (Firecrawl, optional)

When local retrieval is judged insufficient even after its single Refine, the
pipeline performs one corrective web search through the
[Firecrawl Search API](https://docs.firecrawl.dev/features/search) and answers
from the merged sources (CRAG-style correction). A sidebar toggle ("Tìm kiếm
web") additionally merges web results into every retrieval.

- Configure with `FIRECRAWL_API_KEY=...` in `.env` (see `.env.example`). Without
  a key the feature is silently disabled and the bot behaves exactly as before.
- Web results appear as ordinary citations with `kind="web"`; each citation
  links back to the source URL. At most 2 Firecrawl calls are made per answer
  (5 results each, page content truncated to 4000 chars).
```

- [ ] **Step 2: Final gate — everything must be green**

Run: `uv run pytest && uv run mypy`
Expected: full suite PASS, mypy clean across `app.py`, `generate_data.py`, `eval.py`, `rag_core`, `tests`.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document optional Firecrawl web search"
```
