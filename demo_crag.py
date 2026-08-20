"""Demo for the CRAG judge, refine and refusal layer (ticket 04).

Shows the quality gate: the judge's verdict (high / medium / low) on the
top-5 Sources, the refine pass (query rewrite + full re-retrieval and
re-ranking) when the verdict is not high, and the final decision — a
grounded answer or a refusal with a rephrase suggestion.

Run from the repo root:
    uv run python -m demo_crag "giải thích bảng băm là gì?"

Requires OPENROUTER_API_KEY in .env (see README.md) and the local-GPU
dependencies (torch, transformers).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from rag_core.chunking import chunk_dataset
from rag_core.config import load_config
from rag_core.embeddings import OpenRouterEmbedder
from rag_core.generator import OpenRouterGenerator, parse_citations
from rag_core.index import FINAL_TOP_K, Index, dedupe_by_source
from rag_core.judge import OpenRouterJudge, OpenRouterQueryRewriter
from rag_core.models import Chunk
from rag_core.reranker import (
    RERANK_INPUT_TOP_K,
    RERANK_KEEP_TOP_K,
    LocalBgeReranker,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Demo the CRAG judge layer.")
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

    print("\n== Re-ranking ==")
    reranker = LocalBgeReranker(model_name=config.rerank_model)
    print(f"  reranker: {reranker.model_name} on {reranker.device}")

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

    query = args.question
    print("\n== CRAG judge ==")
    print(f"  query: {query}")

    def retrieve_top_five(q: str) -> list[Chunk]:
        vector = np.asarray(embedder.embed_query(q), dtype=np.float32)
        fused = index.fused_candidates(q, vector, RERANK_INPUT_TOP_K)
        reranked = reranker.rerank(q, fused)
        return dedupe_by_source(reranked[:RERANK_KEEP_TOP_K], FINAL_TOP_K)

    top_five = retrieve_top_five(query)
    print("  top-5 Sources:")
    for i, chunk in enumerate(top_five, start=1):
        print(f"    [{i}] {chunk.source.document_id} / {chunk.source.chapter}")

    judgment = judge.assess(query, top_five)
    print(f"  verdict: {judgment.level}")
    if not judgment.is_high:
        refined = rewriter.rewrite(query)
        print(f"  refine: {query} -> {refined}")
        top_five = retrieve_top_five(refined)
        print("  re-retrieved top-5 Sources:")
        for i, chunk in enumerate(top_five, start=1):
            print(f"    [{i}] {chunk.source.document_id} / {chunk.source.chapter}")
        judgment = judge.assess(refined, top_five)
        print(f"  verdict after refine: {judgment.level}")
        if judgment.is_high:
            query = refined
        else:
            print(f"  -> refusal: {judgment.rephrase_suggestion}")
            return

    print("\n== Generation ==")
    generator = OpenRouterGenerator(
        api_key=config.api_key,
        model=config.llm_model,
        base_url=config.base_url,
    )
    answer_text = generator.generate(query, top_five)
    citations = parse_citations(answer_text, [c.source for c in top_five])
    print(f"  answer: {answer_text}")
    print("  citations:", [(c.marker, c.source.document_id) for c in citations])


if __name__ == "__main__":
    main()