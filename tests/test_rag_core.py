from __future__ import annotations

import hashlib
import math
from pathlib import Path

import generate_data as gd
from rag_core import RagCore, Session
from rag_core.models import Chunk, Source

SEED = 42
DIM = 64
INTRO_TITLES = {"Giới thiệu", "Introduction", "Tóm tắt", "Summary"}

STOPWORDS = {
    "và", "của", "là", "trong", "một", "các", "cho", "với", "có", "được",
    "những", "này", "đó", "thì", "mà", "để", "khi", "đã", "sẽ", "không",
    "the", "and", "of", "to", "in", "a", "an", "is", "are", "for", "with",
    "this", "that", "on", "at", "it", "as", "be", "or", "by", "from", "which",
    "what", "how", "why", "cần", "nào", "của", "về", "bài", "giảng", "sinh",
    "viên", "học", "phần", "tài", "liệu", "nội", "dung", "môn",
}


class RecordingEmbedder:
    def __init__(self) -> None:
        self.batch_calls = 0
        self.batch_sizes: list[int] = []
        self.query_calls = 0
        self._cache: dict[str, list[float]] = {}
        self._idf: dict[str, float] = {}
        self._seen_docs = 0

    def _token_vec(self, token: str) -> list[float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return [(digest[i % 16] / 255.0) - 0.5 for i in range(DIM)]

    def _learn_corpus(self, texts: list[str]) -> None:
        self._seen_docs = len(texts)
        df: dict[str, int] = {}
        for text in texts:
            for token in set(_content_tokens(text)):
                df[token] = df.get(token, 0) + 1
        n = max(self._seen_docs, 1)
        self._idf = {token: math.log(n / (count + 1)) + 1.0 for token, count in df.items()}

    def _embed(self, text: str) -> list[float]:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vector = [0.0] * DIM
        for token in _content_tokens(text):
            weight = self._idf.get(token, 1.0)
            tv = self._token_vec(token)
            for i in range(DIM):
                vector[i] += weight * tv[i]
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        vector = [x / norm for x in vector]
        self._cache[text] = vector
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        self.batch_sizes.append(len(texts))
        self._learn_corpus(texts)
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._embed(text)


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Source]]] = []

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        sources = [chunk.source for chunk in chunks]
        self.calls.append((question, sources))
        first_sentence = chunks[0].text.split(".")[0]
        return f"Trả lời về {sources[0].document_title}: {first_sentence}. [1]"


def _content_tokens(text: str) -> list[str]:
    return [t for t in text.lower().split() if t not in STOPWORDS]


def make_core(tmp_path: Path, embedder: RecordingEmbedder | None = None,
              generator: FakeGenerator | None = None) -> tuple[RagCore, gd.Dataset]:
    ds = gd.generate(seed=SEED)
    gd.write(ds, tmp_path)
    core = RagCore(
        data_dir=tmp_path,
        embedder=embedder or RecordingEmbedder(),
        generator=generator or FakeGenerator(),
    )
    return core, ds


def session() -> Session:
    return Session(id="s1", user_id="u1", turns=[])


def ground_truth(ds: gd.Dataset, course_code: str) -> tuple[gd.Document, gd.Chapter, str]:
    doc = next(d for d in ds.documents if d.course_code == course_code)
    chapter = next(ch for ch in doc.chapters if ch.title not in INTRO_TITLES)
    topic = chapter.title.split(": ", 1)[1]
    return doc, chapter, topic


def test_answer_returns_answer_with_sources_attached(tmp_path: Path) -> None:
    core, _ = make_core(tmp_path)
    result = core.answer("giải thích bảng băm là gì?", session())
    assert result.answer
    assert result.sources
    assert len(result.sources) <= 5


def test_answer_is_grounded_in_ground_truth_source(tmp_path: Path) -> None:
    core, ds = make_core(tmp_path)
    doc, chapter, topic = ground_truth(ds, "CS112")
    query = f"giải thích {topic}" if doc.language == "vi" else f"explain {topic}"

    result = core.answer(query, session())

    assert topic in result.answer
    assert any(
        s.document_id == doc.id and s.chapter == chapter.title
        for s in result.sources
    ), f"ground-truth source {doc.id}/{chapter.title} missing from {[s.document_id for s in result.sources]}"


def test_answer_cites_ground_truth_source(tmp_path: Path) -> None:
    core, ds = make_core(tmp_path)
    doc, chapter, topic = ground_truth(ds, "CS112")
    query = f"giải thích {topic}" if doc.language == "vi" else f"explain {topic}"

    result = core.answer(query, session())

    assert any(
        c.source.document_id == doc.id and c.source.chapter == chapter.title
        for c in result.citations
    ), f"no citation points at {doc.id}/{chapter.title}"


def test_sources_carry_document_and_chapter_metadata(tmp_path: Path) -> None:
    core, _ = make_core(tmp_path)
    result = core.answer("giải thích bảng băm là gì?", session())
    for s in result.sources:
        assert s.document_id
        assert s.chapter


def test_course_advisory_grounded_in_course_data(tmp_path: Path) -> None:
    core, _ = make_core(tmp_path)
    result = core.answer("CS112 cần học môn nào trước?", session())
    assert any(s.document_id == "CS112" for s in result.sources), (
        f"course CS112 missing from sources: {[s.document_id for s in result.sources]}"
    )


def test_all_chunks_embedded_in_single_batched_request(tmp_path: Path) -> None:
    embedder = RecordingEmbedder()
    make_core(tmp_path, embedder=embedder)
    assert embedder.batch_calls == 1, "index embedding must be a single batched request"
    assert embedder.batch_sizes == [embedder.batch_sizes[0]]
    assert embedder.query_calls == 0, "no query embedding during ingest"


def test_query_embedded_once_per_call(tmp_path: Path) -> None:
    embedder = RecordingEmbedder()
    core, _ = make_core(tmp_path, embedder=embedder)
    core.answer("giải thích bảng băm là gì?", session())
    core.answer("giải thích cây nhị phân là gì?", session())
    assert embedder.query_calls == 2


def test_answer_is_deterministic_for_same_query(tmp_path: Path) -> None:
    core, _ = make_core(tmp_path)
    first = core.answer("giải thích bảng băm là gì?", session())
    second = core.answer("giải thích bảng băm là gì?", session())
    assert [s.document_id for s in first.sources] == [s.document_id for s in second.sources]
    assert first.answer == second.answer


def test_citations_reference_returned_sources(tmp_path: Path) -> None:
    core, _ = make_core(tmp_path)
    result = core.answer("giải thích bảng băm là gì?", session())
    source_ids = {(s.document_id, s.chapter) for s in result.sources}
    for c in result.citations:
        assert (c.source.document_id, c.source.chapter) in source_ids


def test_hybrid_ranking_favors_lexical_matches(tmp_path: Path) -> None:
    core, ds = make_core(tmp_path)
    doc, chapter, topic = ground_truth(ds, "CS112")
    query = f"giải thích {topic}" if doc.language == "vi" else f"explain {topic}"

    result = core.answer(query, session())

    assert result.sources[0].document_id == doc.id
    assert result.sources[0].chapter == chapter.title


def test_generator_receives_top_five_ordered_sources(tmp_path: Path) -> None:
    generator = FakeGenerator()
    core, _ = make_core(tmp_path, generator=generator)
    core.answer("giải thích bảng băm là gì?", session())
    assert len(generator.calls) == 1
    question, sources = generator.calls[0]
    assert question == "giải thích bảng băm là gì?"
    assert len(sources) <= 5
    assert len(sources) == len(set((s.document_id, s.chapter) for s in sources))


def test_single_chunk_per_chapter_maps_metadata(tmp_path: Path) -> None:
    core, ds = make_core(tmp_path)
    doc = next(d for d in ds.documents if d.language == "vi")
    chapter = next(ch for ch in doc.chapters if ch.title not in INTRO_TITLES)
    topic = chapter.title.split(": ", 1)[1]

    result = core.answer(f"giải thích {topic}", session())

    assert result.sources[0].document_id == doc.id
    assert result.sources[0].chapter == chapter.title
    assert result.sources[0].course_code == doc.course_code
    assert result.sources[0].kind == doc.kind
    assert result.sources[0].language == doc.language