from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag_core.chunking import CourseDict, DocumentDict, chunk_dataset
from rag_core.config import Config, load_config
from rag_core.embeddings import Embedder, OpenRouterEmbedder
from rag_core.generator import Generator, OpenRouterGenerator, parse_citations
from rag_core.index import FINAL_TOP_K, Index, dedupe_by_source
from rag_core.models import AnswerResult, Chunk, Session
from rag_core.reranker import (
    RERANK_INPUT_TOP_K,
    RERANK_KEEP_TOP_K,
    LocalBgeReranker,
    Reranker,
)


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
        reranker: Reranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._generator = generator
        self._reranker = reranker
        self._index = index or self._build_index(data_dir, embedder)

    @staticmethod
    def _build_index(data_dir: Path, embedder: Embedder) -> Index:
        courses = json.loads((data_dir / "courses.json").read_text(encoding="utf-8"))
        documents = json.loads((data_dir / "documents.json").read_text(encoding="utf-8"))
        chunks = chunk_dataset(list(courses), list(documents))
        return Index(chunks, embedder)

    def _retrieve_sources(self, query_text: str, query_vector: np.ndarray) -> list[Chunk]:
        """Retrieve, re-rank (when wired) and dedupe the chunks behind the answer."""
        if self._reranker is None:
            return self._index.retrieve(query_text, query_vector)
        candidates = self._index.fused_candidates(query_text, query_vector, RERANK_INPUT_TOP_K)
        reranked = self._reranker.rerank(query_text, candidates)
        return dedupe_by_source(reranked[:RERANK_KEEP_TOP_K], FINAL_TOP_K)

    def answer(self, user_message: str, session: Session) -> AnswerResult:
        query_vector = np.asarray(self._embedder.embed_query(user_message), dtype=np.float32)
        retrieved = self._retrieve_sources(user_message, query_vector)

        sources = [chunk.source for chunk in retrieved]
        answer_text = self._generator.generate(user_message, retrieved)
        citations = parse_citations(answer_text, sources)
        return AnswerResult(answer=answer_text, citations=citations, sources=sources)


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
    reranker = LocalBgeReranker(model_name=config.rerank_model)
    return RagCore(config.data_dir, embedder, generator, reranker=reranker)


__all__ = [
    "RagCore",
    "build_rag_core",
    "load_config",
    "AnswerResult",
    "Session",
    "Chunk",
    "Reranker",
    "LocalBgeReranker",
]