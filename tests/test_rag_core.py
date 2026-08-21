from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

import generate_data as gd
from rag_core import RagCore, Session
from rag_core.attachments import AttachmentStore
from rag_core.db import Database
from rag_core.answer_check import CheckVerdict, DEFAULT_UNSUPPORTED_FEEDBACK
from rag_core.judge import Judgment, Level
from rag_core.models import Chunk, Source, Turn

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
        self.feedbacks: list[str | None] = []

    def generate(
        self, question: str, chunks: list[Chunk], feedback: str | None = None
    ) -> str:
        sources = [chunk.source for chunk in chunks]
        self.calls.append((question, sources))
        self.feedbacks.append(feedback)
        first_sentence = chunks[0].text.split(".")[0]
        return f"Trả lời về {sources[0].document_title}: {first_sentence}. [1]"


class FakeJudge:
    """Returns verdicts from a script; records every call."""

    def __init__(self, levels: list[Level], suggestion: str = "Hãy thử hỏi với từ khóa khác.") -> None:
        self.levels = levels
        self.suggestion = suggestion
        self.calls: list[tuple[str, list[Chunk]]] = []

    def assess(self, query: str, chunks: list[Chunk]) -> Judgment:
        self.calls.append((query, list(chunks)))
        level = self.levels[min(len(self.calls) - 1, len(self.levels) - 1)]
        return Judgment(level=level, rephrase_suggestion=self.suggestion)


class FakeRewriter:
    """Returns a fixed rewrite and records the queries it was asked to rewrite."""

    def __init__(self, rewritten: str = "bảng băm") -> None:
        self.rewritten = rewritten
        self.calls: list[str] = []

    def rewrite(self, query: str) -> str:
        self.calls.append(query)
        return self.rewritten


class FakeChecker:
    """Returns verdicts from a script; records every check call."""

    def __init__(self, verdicts: list[CheckVerdict]) -> None:
        self.verdicts = verdicts
        self.calls: list[tuple[str, str, list[Chunk]]] = []

    def check(self, question: str, answer: str, chunks: list[Chunk]) -> CheckVerdict:
        self.calls.append((question, answer, list(chunks)))
        return self.verdicts[min(len(self.calls) - 1, len(self.verdicts) - 1)]


class FakeSessionRewriter:
    """Returns a fixed rewrite and records the messages/history it saw."""

    def __init__(self, rewritten: str = "bảng băm cấu trúc dữ liệu") -> None:
        self.rewritten = rewritten
        self.calls: list[tuple[str, list[Turn]]] = []

    def rewrite(self, message: str, turns: list[Turn]) -> str:
        self.calls.append((message, list(turns)))
        return self.rewritten


def _content_tokens(text: str) -> list[str]:
    return [t for t in text.lower().split() if t not in STOPWORDS]


def make_core(tmp_path: Path, embedder: RecordingEmbedder | None = None,
              generator: FakeGenerator | None = None,
              judge: FakeJudge | None = None,
              rewriter: FakeRewriter | None = None,
              checker: FakeChecker | None = None,
              session_rewriter: FakeSessionRewriter | None = None) -> tuple[RagCore, gd.Dataset]:
    ds = gd.generate(seed=SEED)
    gd.write(ds, tmp_path)
    core = RagCore(
        data_dir=tmp_path,
        embedder=embedder or RecordingEmbedder(),
        generator=generator or FakeGenerator(),
        judge=judge,
        rewriter=rewriter,
        checker=checker,
        session_rewriter=session_rewriter,
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


def test_high_judgment_answers_without_refine(tmp_path: Path) -> None:
    judge = FakeJudge(levels=["high"])
    rewriter = FakeRewriter()
    generator = FakeGenerator()
    core, _ = make_core(tmp_path, generator=generator, judge=judge, rewriter=rewriter)

    result = core.answer("giải thích bảng băm là gì?", session())

    assert not result.refused
    assert result.answer
    assert len(judge.calls) == 1
    assert rewriter.calls == []
    assert len(generator.calls) == 1
    question, _ = generator.calls[0]
    assert question == "giải thích bảng băm là gì?"


def test_judge_scores_the_top_five_sources(tmp_path: Path) -> None:
    judge = FakeJudge(levels=["high"])
    generator = FakeGenerator()
    core, _ = make_core(tmp_path, generator=generator, judge=judge, rewriter=FakeRewriter())

    core.answer("giải thích bảng băm là gì?", session())

    judge_query, judged = judge.calls[0]
    assert judge_query == "giải thích bảng băm là gì?"
    assert len(judged) == 5, "the judge must assess the top-5 Sources"
    assert len({(c.source.document_id, c.source.chapter) for c in judged}) == len(judged)
    _, gen_sources = generator.calls[0]
    assert [c.source for c in judged] == gen_sources


def test_medium_judgment_triggers_one_refine_then_grounded_answer(tmp_path: Path) -> None:
    judge = FakeJudge(levels=["medium", "high"])
    rewriter = FakeRewriter(rewritten="bảng băm cấu trúc dữ liệu")
    generator = FakeGenerator()
    embedder = RecordingEmbedder()
    core, _ = make_core(tmp_path, embedder=embedder, generator=generator,
                        judge=judge, rewriter=rewriter)

    result = core.answer("giải thích bảng băm là gì?", session())

    assert not result.refused
    assert result.answer
    assert len(judge.calls) == 2
    assert [q for q, _ in judge.calls] == [
        "giải thích bảng băm là gì?",
        "bảng băm cấu trúc dữ liệu",
    ]
    assert rewriter.calls == ["giải thích bảng băm là gì?"]
    assert embedder.query_calls == 2, "the refine must re-embed the rewritten query"
    assert len(generator.calls) == 1
    question, _ = generator.calls[0]
    assert question == "bảng băm cấu trúc dữ liệu"


def test_ambiguous_query_refines_to_grounded_answer(tmp_path: Path) -> None:
    judge = FakeJudge(levels=["low", "high"])
    rewriter = FakeRewriter(rewritten="bảng băm")
    generator = FakeGenerator()
    core, _ = make_core(tmp_path, generator=generator, judge=judge, rewriter=rewriter)

    result = core.answer("giải thích bảng băm là gì?", session())

    assert not result.refused
    assert result.answer
    assert len(generator.calls) == 1
    question, _ = generator.calls[0]
    assert question == "bảng băm"


def test_low_judgment_refuses_after_single_refine(tmp_path: Path) -> None:
    judge = FakeJudge(levels=["low", "low"])
    rewriter = FakeRewriter()
    generator = FakeGenerator()
    core, _ = make_core(tmp_path, generator=generator, judge=judge, rewriter=rewriter)

    result = core.answer("giải thích bảng băm là gì?", session())

    assert result.refused
    assert result.answer == ""
    assert result.citations == []
    assert result.sources == []
    assert result.rephrase_suggestion
    assert generator.calls == [], "a refusal must not call the generator"
    assert len(judge.calls) == 2
    assert rewriter.calls == ["giải thích bảng băm là gì?"]


def test_not_high_after_refine_refuses_without_second_refine(tmp_path: Path) -> None:
    judge = FakeJudge(levels=["medium", "medium"])
    rewriter = FakeRewriter()
    core, _ = make_core(tmp_path, judge=judge, rewriter=rewriter)

    result = core.answer("giải thích bảng băm là gì?", session())

    assert result.refused
    assert len(judge.calls) == 2
    assert len(rewriter.calls) == 1, "medium/low must trigger exactly one refine"


def test_draft_answer_checked_claim_by_claim_against_sources(tmp_path: Path) -> None:
    checker = FakeChecker(verdicts=[CheckVerdict(supported=True)])
    generator = FakeGenerator()
    core, _ = make_core(tmp_path, generator=generator, checker=checker)

    result = core.answer("giải thích bảng băm là gì?", session())

    assert not result.refused
    assert len(generator.calls) == 1, "a supported draft must not be regenerated"
    question, sources = generator.calls[0]
    checked_question, checked_answer, checked_chunks = checker.calls[0]
    assert len(checker.calls) == 1
    assert checked_question == question
    assert checked_answer == result.answer, "the returned answer is the checked draft"
    assert [c.source for c in checked_chunks] == sources, (
        "the check must verify the draft against the same Sources that were generated from"
    )


def test_valid_answer_returned_unchanged_with_citations(tmp_path: Path) -> None:
    checker = FakeChecker(verdicts=[CheckVerdict(supported=True)])
    core, _ = make_core(tmp_path, checker=checker)

    result = core.answer("giải thích bảng băm là gì?", session())

    assert not result.refused
    assert result.citations, "a passing draft keeps its Citations"
    source_ids = {(s.document_id, s.chapter) for s in result.sources}
    for citation in result.citations:
        assert (citation.source.document_id, citation.source.chapter) in source_ids


def test_unsupported_draft_triggers_exactly_one_regeneration_with_feedback(tmp_path: Path) -> None:
    feedback = "The claim 'X' is not supported by any cited source."
    checker = FakeChecker(
        verdicts=[
            CheckVerdict(supported=False, feedback=feedback),
            CheckVerdict(supported=True),
        ]
    )
    generator = FakeGenerator()
    core, _ = make_core(tmp_path, generator=generator, checker=checker)

    result = core.answer("giải thích bảng băm là gì?", session())

    assert not result.refused
    assert result.answer
    assert len(generator.calls) == 2, "a failed check must trigger exactly one regeneration"
    assert generator.feedbacks == [None, feedback]
    question1, sources1 = generator.calls[0]
    question2, sources2 = generator.calls[1]
    assert question1 == question2, "regeneration keeps the same question"
    assert sources1 == sources2, "regeneration uses the same Sources"
    assert len(checker.calls) == 2, "the regenerated answer must be checked again"
    assert checker.calls[1][1] == result.answer, "the returned answer is the verified regeneration"


def test_unsupported_draft_without_feedback_still_regenerates(tmp_path: Path) -> None:
    checker = FakeChecker(
        verdicts=[CheckVerdict(supported=False), CheckVerdict(supported=True)]
    )
    generator = FakeGenerator()
    core, _ = make_core(tmp_path, generator=generator, checker=checker)

    result = core.answer("giải thích bảng băm là gì?", session())

    assert not result.refused
    assert generator.feedbacks == [None, DEFAULT_UNSUPPORTED_FEEDBACK]


def test_second_failure_produces_refusal_with_rephrase_suggestion(tmp_path: Path) -> None:
    checker = FakeChecker(
        verdicts=[
            CheckVerdict(supported=False, feedback="Claim 'X' is unsupported."),
            CheckVerdict(supported=False, feedback="Claim 'X' is unsupported."),
        ]
    )
    generator = FakeGenerator()
    core, _ = make_core(tmp_path, generator=generator, checker=checker)

    result = core.answer("giải thích bảng băm là gì?", session())

    assert result.refused
    assert result.answer == ""
    assert result.citations == []
    assert result.sources == []
    assert result.rephrase_suggestion
    assert len(generator.calls) == 2, "exactly one regeneration before refusing"
    assert len(checker.calls) == 2


def test_answer_check_runs_after_judge_gate(tmp_path: Path) -> None:
    judge = FakeJudge(levels=["high"])
    checker = FakeChecker(verdicts=[CheckVerdict(supported=True)])
    generator = FakeGenerator()
    core, _ = make_core(
        tmp_path,
        generator=generator,
        judge=judge,
        rewriter=FakeRewriter(),
        checker=checker,
    )

    result = core.answer("giải thích bảng băm là gì?", session())

    assert not result.refused
    assert len(judge.calls) == 1
    assert len(generator.calls) == 1
    _, _, checked_chunks = checker.calls[0]
    assert [c.source for c in checked_chunks] == [c.source for c in judge.calls[0][1]], (
        "the check must verify the answer against the judged top-5 Sources"
    )


def test_judge_refusal_never_reaches_answer_check(tmp_path: Path) -> None:
    judge = FakeJudge(levels=["low", "low"])
    checker = FakeChecker(verdicts=[CheckVerdict(supported=True)])
    generator = FakeGenerator()
    core, _ = make_core(
        tmp_path,
        generator=generator,
        judge=judge,
        rewriter=FakeRewriter(),
        checker=checker,
    )

    result = core.answer("giải thích bảng băm là gì?", session())

    assert result.refused
    assert checker.calls == [], "the answer check must not run on a judge refusal"
    assert generator.calls == [], "a judge refusal never reaches generation"


def history_with_turns() -> Session:
    s = session()
    s.turns = [
        Turn(role="user", text="giải thích bảng băm là gì?"),
        Turn(role="assistant", text="Bảng băm là một cấu trúc dữ liệu."),
    ]
    return s


def test_follow_up_rewritten_into_standalone_question_before_retrieval(tmp_path: Path) -> None:
    rewriter = FakeSessionRewriter(rewritten="bảng băm cấu trúc dữ liệu")
    generator = FakeGenerator()
    embedder = RecordingEmbedder()
    core, _ = make_core(
        tmp_path, embedder=embedder, generator=generator, session_rewriter=rewriter
    )
    history = history_with_turns()

    result = core.answer("còn ví dụ về nó?", history)

    assert rewriter.calls == [("còn ví dụ về nó?", history.turns)]
    assert embedder.query_calls == 1, "the rewritten query is embedded once, not the raw follow-up"
    assert len(generator.calls) == 1
    question, _ = generator.calls[0]
    assert question == "bảng băm cấu trúc dữ liệu", (
        "the rewritten query must flow through retrieval into generation"
    )
    assert result.answer


def test_first_message_passes_through_unchanged_without_rewrite(tmp_path: Path) -> None:
    rewriter = FakeSessionRewriter()
    generator = FakeGenerator()
    embedder = RecordingEmbedder()
    core, _ = make_core(
        tmp_path, embedder=embedder, generator=generator, session_rewriter=rewriter
    )

    result = core.answer("giải thích bảng băm là gì?", session())

    assert rewriter.calls == [], "the first message has no history to rewrite against"
    assert embedder.query_calls == 1
    question, _ = generator.calls[0]
    assert question == "giải thích bảng băm là gì?"
    assert result.answer


def test_rewritten_query_flows_through_gated_pipeline(tmp_path: Path) -> None:
    session_rewriter = FakeSessionRewriter(rewritten="bảng băm cấu trúc dữ liệu")
    judge = FakeJudge(levels=["high"])
    generator = FakeGenerator()
    core, _ = make_core(
        tmp_path,
        generator=generator,
        judge=judge,
        rewriter=FakeRewriter(),
        session_rewriter=session_rewriter,
    )
    history = history_with_turns()

    result = core.answer("cho ví dụ", history)

    assert not result.refused
    assert judge.calls[0][0] == "bảng băm cấu trúc dữ liệu"
    assert generator.calls[0][0] == "bảng băm cấu trúc dữ liệu"


def test_answer_does_not_mutate_session(tmp_path: Path) -> None:
    rewriter = FakeSessionRewriter()
    core, _ = make_core(tmp_path, session_rewriter=rewriter)
    history = history_with_turns()

    core.answer("còn ví dụ về nó?", history)

    assert [t.text for t in history.turns] == [
        "giải thích bảng băm là gì?",
        "Bảng băm là một cấu trúc dữ liệu.",
    ], "answer() must not append turns; persistence is the caller's job"


def test_rewriter_receives_only_last_six_turns(tmp_path: Path) -> None:
    rewriter = FakeSessionRewriter()
    generator = FakeGenerator()
    core, _ = make_core(tmp_path, generator=generator, session_rewriter=rewriter)
    history = session()
    for i in range(8):
        role = "user" if i % 2 == 0 else "assistant"
        history.turns.append(Turn(role=role, text=f"turn {i}"))

    core.answer("còn ví dụ về nó?", history)

    _, turns = rewriter.calls[0]
    assert len(turns) == 6, "the rewrite must use only the last 6 turns"
    assert [t.text for t in turns] == [f"turn {i}" for i in range(2, 8)]


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