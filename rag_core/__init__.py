from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag_core.answer_check import (
    DEFAULT_UNSUPPORTED_FEEDBACK,
    AnswerChecker,
    CheckVerdict,
    OpenRouterAnswerChecker,
)
from rag_core.chunking import CourseDict, DocumentDict, chunk_dataset
from rag_core.config import Config, load_config
from rag_core.embeddings import Embedder, OpenRouterEmbedder
from rag_core.generator import Generator, OpenRouterGenerator, parse_citations
from rag_core.index import Index
from rag_core.judge import (
    DEFAULT_REPHRASE_SUGGESTION,
    Judge,
    Judgment,
    Level,
    OpenRouterJudge,
    OpenRouterQueryRewriter,
    QueryRewriter,
)
from rag_core.models import MAX_TURNS, AnswerResult, Chunk, Session, Source
from rag_core.rewrite import OpenRouterSessionRewriter, SessionRewriter


class RagCore:
    """The pipeline core.

    The single public entry point is ``answer()``. Everything else is
    internal detail. The UI depends only on this entry point.
    """

    def __init__(
        self,
        data_dir: Path,
        embedder: Embedder,
        generator: Generator,
        index: Index | None = None,
        judge: Judge | None = None,
        rewriter: QueryRewriter | None = None,
        checker: AnswerChecker | None = None,
        session_rewriter: SessionRewriter | None = None,
    ) -> None:
        self._embedder = embedder
        self._generator = generator
        self._judge = judge
        self._rewriter = rewriter
        self._checker = checker
        self._session_rewriter = session_rewriter
        self._index = index or self._build_index(data_dir, embedder)

    @staticmethod
    def _build_index(data_dir: Path, embedder: Embedder) -> Index:
        courses = json.loads((data_dir / "courses.json").read_text(encoding="utf-8"))
        documents = json.loads((data_dir / "documents.json").read_text(encoding="utf-8"))
        chunks = chunk_dataset(list(courses), list(documents))
        return Index(chunks, embedder)

    def _retrieve_sources(self, query_text: str, query_vector: np.ndarray) -> list[Chunk]:
        """Retrieve the chunks behind the answer via hybrid RRF retrieval."""
        return self._index.retrieve(query_text, query_vector)

    def answer(self, user_message: str, session: Session) -> AnswerResult:
        query = self._rewrite_for_session(user_message, session)
        query_vector = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        retrieved = self._retrieve_sources(query, query_vector)

        if self._judge is not None and self._rewriter is not None:
            return self._gated_answer(query, retrieved, self._judge, self._rewriter)
        return self._generate_result(query, retrieved)

    def _rewrite_for_session(self, message: str, session: Session) -> str:
        if self._session_rewriter is None or not session.turns:
            return message
        history = session.turns[-MAX_TURNS:]
        return self._session_rewriter.rewrite(message, history)

    def _gated_answer(
        self,
        question: str,
        chunks: list[Chunk],
        judge: Judge,
        rewriter: QueryRewriter,
    ) -> AnswerResult:
        judgment = judge.assess(question, chunks)
        if judgment.is_high:
            return self._generate_result(question, chunks)
        refined = rewriter.rewrite(question)
        refined_vector = np.asarray(
            self._embedder.embed_query(refined), dtype=np.float32
        )
        refined_chunks = self._retrieve_sources(refined, refined_vector)
        judgment = judge.assess(refined, refined_chunks)
        if judgment.is_high:
            return self._generate_result(refined, refined_chunks)
        return self._refused_result(
            judgment.rephrase_suggestion or DEFAULT_REPHRASE_SUGGESTION
        )

    def _generate_result(self, question: str, chunks: list[Chunk]) -> AnswerResult:
        sources = [chunk.source for chunk in chunks]
        answer_text = self._generator.generate(question, chunks)
        if self._checker is None:
            return self._answer_result(answer_text, sources)
        verdict = self._checker.check(question, answer_text, chunks)
        if verdict.supported:
            return self._answer_result(answer_text, sources)
        feedback = verdict.feedback or DEFAULT_UNSUPPORTED_FEEDBACK
        regenerated = self._generator.generate(question, chunks, feedback=feedback)
        verdict = self._checker.check(question, regenerated, chunks)
        if verdict.supported:
            return self._answer_result(regenerated, sources)
        return self._refused_result(DEFAULT_REPHRASE_SUGGESTION)

    @staticmethod
    def _answer_result(answer_text: str, sources: list[Source]) -> AnswerResult:
        citations = parse_citations(answer_text, sources)
        return AnswerResult(answer=answer_text, citations=citations, sources=sources)

    @staticmethod
    def _refused_result(rephrase_suggestion: str) -> AnswerResult:
        return AnswerResult(
            answer="",
            citations=[],
            sources=[],
            refused=True,
            rephrase_suggestion=rephrase_suggestion,
        )


def build_rag_core(config: Config | None = None) -> RagCore:
    """Build a RagCore wired to the real OpenRouter clients from config."""
    config = config or load_config()
    embedder = OpenRouterEmbedder(
        api_key=config.api_key,
        model=config.embed_model,
        dim=config.embed_dim,
        base_url=config.base_url,
    )
    generator = OpenRouterGenerator(
        api_key=config.api_key,
        model=config.llm_model,
        base_url=config.base_url,
    )
    judge = OpenRouterJudge(
        api_key=config.api_key,
        model=config.llm_model,
        base_url=config.base_url,
    )
    rewriter = OpenRouterQueryRewriter(
        api_key=config.api_key,
        model=config.llm_model,
        base_url=config.base_url,
    )
    checker = OpenRouterAnswerChecker(
        api_key=config.api_key,
        model=config.llm_model,
        base_url=config.base_url,
    )
    session_rewriter = OpenRouterSessionRewriter(
        api_key=config.api_key,
        model=config.llm_model,
        base_url=config.base_url,
    )
    return RagCore(
        config.data_dir,
        embedder,
        generator,
        judge=judge,
        rewriter=rewriter,
        checker=checker,
        session_rewriter=session_rewriter,
    )


__all__ = [
    "RagCore",
    "build_rag_core",
    "load_config",
    "AnswerResult",
    "Session",
    "Chunk",
    "Judge",
    "OpenRouterJudge",
    "QueryRewriter",
    "OpenRouterQueryRewriter",
    "Judgment",
    "Level",
    "AnswerChecker",
    "OpenRouterAnswerChecker",
    "CheckVerdict",
    "SessionRewriter",
    "OpenRouterSessionRewriter",
]