from __future__ import annotations

from typing import Protocol, TypeVar

import numpy as np

from rag_core.embeddings import Embedder
from rag_core.models import Chunk

T = TypeVar("T")

DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60
FINAL_TOP_K = 5


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def dedupe_by_source(chunks: list[Chunk], limit: int) -> list[Chunk]:
    """Keep the highest-ranked chunk per (document, chapter), up to ``limit``."""
    seen: set[tuple[str, str]] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        key = (chunk.source.document_id, chunk.source.chapter)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
        if len(unique) >= limit:
            break
    return unique


def rrf_merge(rankings: list[list[T]]) -> list[T]:
    """Fuse rankings of arbitrary items via Reciprocal Rank Fusion (K=60).

    Membership compares items with ``is`` so unhashable mutable dataclasses
    work. Every input item appears exactly once in the output.
    """
    unique: list[T] = []

    def slot(item: T) -> int:
        for i, candidate in enumerate(unique):
            if candidate is item:
                return i
        unique.append(item)
        return len(unique) - 1

    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            idx = slot(item)
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank)
    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    return [unique[i] for i in ordered]


class LexicalScorer(Protocol):
    def get_scores(self, query: list[str]) -> list[float]: ...


class DenseIndex(Protocol):
    def search(
        self, query_vector: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]: ...


class Index:
    """Dense (FAISS cosine) + lexical (BM25) hybrid index over chunks."""

    def __init__(
        self, chunks: list[Chunk], embedder: Embedder, vectors: np.ndarray | None = None
    ) -> None:
        self.chunks = chunks
        if vectors is None:
            raw = np.asarray(
                embedder.embed_batch([chunk.text for chunk in chunks]), dtype=np.float32
            )
        else:
            if vectors.shape[0] != len(chunks):
                raise ValueError(
                    f"vectors rows ({vectors.shape[0]}) must match chunks ({len(chunks)})"
                )
            raw = np.asarray(vectors, dtype=np.float32).copy()
        self._dense = self._dense_from(raw)
        self._lexical = self._build_lexical(chunks)

    @staticmethod
    def _dense_from(matrix: np.ndarray) -> DenseIndex:
        import faiss

        vectors = matrix.copy()
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    @staticmethod
    def _build_dense(chunks: list[Chunk], embedder: Embedder) -> DenseIndex:
        raw = np.asarray(
            embedder.embed_batch([chunk.text for chunk in chunks]), dtype=np.float32
        )
        return Index._dense_from(raw)

    @staticmethod
    def _build_lexical(chunks: list[Chunk]) -> LexicalScorer:
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

        tokenized = [_tokenize(chunk.text) for chunk in chunks]
        return BM25Okapi(tokenized)  # type: ignore[no-any-return]

    def _dense_ranking(self, query_vector: np.ndarray) -> list[int]:
        import faiss

        vectors = np.asarray([query_vector], dtype=np.float32)
        faiss.normalize_L2(vectors)
        scores, indices = self._dense.search(vectors, DENSE_TOP_K)
        return [int(i) for i in indices[0] if i >= 0]

    def _lexical_ranking(self, query_text: str) -> list[int]:
        scores = self._lexical.get_scores(_tokenize(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:BM25_TOP_K]

    @staticmethod
    def _rrf_fusion(rankings: list[list[int]]) -> list[int]:
        scores: dict[int, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(scores, key=lambda i: scores[i], reverse=True)

    def _rank(self, query_text: str, query_vector: np.ndarray) -> list[int]:
        dense_rank = self._dense_ranking(query_vector)
        lexical_rank = self._lexical_ranking(query_text)
        return self._rrf_fusion([dense_rank, lexical_rank])

    def retrieve(self, query_text: str, query_vector: np.ndarray) -> list[Chunk]:
        """Fuse dense top-20 and BM25 top-20 via RRF, dedupe to top-5 Sources.

        Deduplication runs over the full fused ranking so that up to five
        distinct Sources are always returned when they exist.
        """
        fused = self._rank(query_text, query_vector)
        return dedupe_by_source([self.chunks[i] for i in fused], FINAL_TOP_K)

    def fused_candidates(self, query_text: str, query_vector: np.ndarray, k: int) -> list[Chunk]:
        """The top-``k`` chunks of the fused ranking (dense + BM25 via RRF)."""
        fused = self._rank(query_text, query_vector)
        return [self.chunks[i] for i in fused[:k]]

    def fused_ranking(self, query_text: str, query_vector: np.ndarray) -> list[int]:
        """Return the fused chunk ranking (dense + BM25 via RRF) for demos and eval."""
        return self._rank(query_text, query_vector)