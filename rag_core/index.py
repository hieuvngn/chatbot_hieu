from __future__ import annotations

from typing import Protocol

import numpy as np

from rag_core.embeddings import Embedder
from rag_core.models import Chunk

DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60
FINAL_TOP_K = 5


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class LexicalScorer(Protocol):
    def get_scores(self, query: list[str]) -> list[float]: ...


class DenseIndex(Protocol):
    def search(
        self, query_vector: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]: ...


class Index:
    """Dense (FAISS cosine) + lexical (BM25) hybrid index over chunks."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = chunks
        self._dense = self._build_dense(chunks, embedder)
        self._lexical = self._build_lexical(chunks)

    @staticmethod
    def _build_dense(chunks: list[Chunk], embedder: Embedder) -> DenseIndex:
        import faiss

        vectors = np.asarray(
            embedder.embed_batch([chunk.text for chunk in chunks]), dtype=np.float32
        )
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

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
        return [int(i) for i in indices[0]]

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
        """Fuse dense top-20 and BM25 top-20 via RRF, dedupe to top-5 Sources."""
        fused = self._rank(query_text, query_vector)

        seen: set[tuple[str, str]] = set()
        sources: list[Chunk] = []
        for chunk_id in fused:
            chunk = self.chunks[chunk_id]
            key = (chunk.source.document_id, chunk.source.chapter)
            if key in seen:
                continue
            seen.add(key)
            sources.append(chunk)
            if len(sources) >= FINAL_TOP_K:
                break
        return sources

    def fused_ranking(self, query_text: str, query_vector: np.ndarray) -> list[int]:
        """Return the fused chunk ranking (dense + BM25 via RRF) for demos and eval."""
        return self._rank(query_text, query_vector)