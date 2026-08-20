from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rag_core.chunking import CourseDict, DocumentDict, chunk_dataset
from rag_core.config import Config, load_config
from rag_core.embeddings import Embedder, OpenRouterEmbedder
from rag_core.generator import Generator, OpenRouterGenerator, parse_citations
from rag_core.index import Index
from rag_core.models import AnswerResult, Session


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
    ) -> None:
        self._embedder = embedder
        self._generator = generator
        self._index = index or self._build_index(data_dir, embedder)

    @staticmethod
    def _build_index(data_dir: Path, embedder: Embedder) -> Index:
        courses = json.loads((data_dir / "courses.json").read_text(encoding="utf-8"))
        documents = json.loads((data_dir / "documents.json").read_text(encoding="utf-8"))
        chunks = chunk_dataset(list(courses), list(documents))
        return Index(chunks, embedder)

    def answer(self, user_message: str, session: Session) -> AnswerResult:
        query_vector = np.asarray(self._embedder.embed_query(user_message), dtype=np.float32)
        retrieved = self._index.retrieve(user_message, query_vector)

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
    return RagCore(config.data_dir, embedder, generator)


__all__ = ["RagCore", "build_rag_core", "load_config", "AnswerResult", "Session"]