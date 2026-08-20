"""Demo for the cross-encoder reranker (ticket 03).

Shows the effect of the re-ranking layer: the RRF-fused top-20 candidates
before re-ranking, their order after the local cross-encoder
(BAAI/bge-reranker-v2-m3 on GPU), and the final top-5 Sources that are
passed to generation.

Run from the repo root:
    uv run python -m demo_rerank "giải thích bảng băm là gì?"

Requires OPENROUTER_API_KEY in .env (see README.md) and the local-GPU
dependencies (torch, transformers).
"""

from __future__ import annotations

import argparse
import json
import sys

from rag_core import RagCore, Session
from rag_core.chunking import chunk_dataset
from rag_core.config import load_config
from rag_core.embeddings import OpenRouterEmbedder
from rag_core.generator import OpenRouterGenerator
from rag_core.index import FINAL_TOP_K, Index, dedupe_by_source
from rag_core.reranker import (
    RERANK_INPUT_TOP_K,
    RERANK_KEEP_TOP_K,
    LocalBgeReranker,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Demo the re-ranking layer.")
    parser.add_argument("question", nargs="?", default="giải thích bảng băm là gì?")
    args = parser.parse_args(argv)

    try:
        config = load_config()
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    print("== Data ==")
    courses = json.loads((config.data_dir / "courses.json").read_text(encoding="utf-8"))
    documents = json.loads((config.data_dir / "documents.json").read_text(encoding="utf-8"))
    chunks = chunk_dataset(list(courses), list(documents))
    print(f"  {len(courses)} courses, {len(documents)} documents, {len(chunks)} index units")

    print("\n== Embedding ==")
    embedder = OpenRouterEmbedder(
        api_key=config.api_key,
        model=config.embed_model,
        dim=config.embed_dim,
        base_url=config.base_url,
    )
    index = Index(chunks, embedder)
    print(f"  index chunks embedded in a single batched request ({config.embed_dim}-dim)")

    query = args.question

    print("\n== Re-ranking ==")
    reranker = LocalBgeReranker(model_name=config.rerank_model)
    print(f"  reranker: {reranker.model_name} on {reranker.device}")

    print(f"\n  query: {query}")
    fused = index.fused_candidates(query, embedder.embed_query(query), RERANK_INPUT_TOP_K)
    print("  fused top-20 (pre-rerank order):")
    for i, chunk in enumerate(fused, start=1):
        print(f"    [{i:2d}] {chunk.source.document_id} / {chunk.source.chapter}")

    reranked = reranker.rerank(query, fused)
    print("  post-rerank top-10:")
    for i, chunk in enumerate(reranked[:RERANK_KEEP_TOP_K], start=1):
        print(f"    [{i:2d}] {chunk.source.document_id} / {chunk.source.chapter}")

    top_sources = dedupe_by_source(reranked[:RERANK_KEEP_TOP_K], FINAL_TOP_K)
    print("  top-5 Sources passed to generation:")
    for i, chunk in enumerate(top_sources, start=1):
        print(f"    [{i}] {chunk.source.document_id} / {chunk.source.chapter}")

    print("\n== Generation ==")
    core = RagCore(
        data_dir=config.data_dir,
        embedder=embedder,
        generator=OpenRouterGenerator(
            api_key=config.api_key,
            model=config.llm_model,
            base_url=config.base_url,
        ),
        index=index,
        reranker=reranker,
    )
    result = core.answer(query, Session(id="demo", user_id="demo", turns=[]))
    print(f"  answer: {result.answer}")
    print("  citations:", [(c.marker, c.source.document_id) for c in result.citations])


if __name__ == "__main__":
    main()