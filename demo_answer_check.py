"""Demo for the answer check and regenerate layer (ticket 05).

Shows the generation-side quality gate: a draft answer generated from the
top-5 Sources, the verifier's verdict on it (supported / unsupported, with
the concrete unsupported claims named), a single regeneration with that
feedback when the draft fails, and the final decision — a verified answer
with its Citations or a refusal with a rephrase suggestion.

Run from the repo root:
    uv run python -m demo_answer_check "giải thích bảng băm là gì?"

Requires OPENROUTER_API_KEY in .env (see README.md) and the local-GPU
dependencies (torch, transformers).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from rag_core.answer_check import DEFAULT_UNSUPPORTED_FEEDBACK, OpenRouterAnswerChecker
from rag_core.chunking import chunk_dataset
from rag_core.config import load_config
from rag_core.embeddings import OpenRouterEmbedder
from rag_core.generator import OpenRouterGenerator, parse_citations
from rag_core.index import FINAL_TOP_K, Index, dedupe_by_source
from rag_core.judge import DEFAULT_REPHRASE_SUGGESTION
from rag_core.models import Chunk
from rag_core.reranker import (
    RERANK_INPUT_TOP_K,
    RERANK_KEEP_TOP_K,
    LocalBgeReranker,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Demo the answer check layer.")
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

    generator = OpenRouterGenerator(
        api_key=config.api_key,
        model=config.llm_model,
        base_url=config.base_url,
    )
    checker = OpenRouterAnswerChecker(
        api_key=config.api_key,
        model=config.llm_model,
        base_url=config.base_url,
    )

    query = args.question
    print("\n== Retrieval ==")
    print(f"  query: {query}")
    vector = np.asarray(embedder.embed_query(query), dtype=np.float32)
    fused = index.fused_candidates(query, vector, RERANK_INPUT_TOP_K)
    reranked = reranker.rerank(query, fused)
    top_five: list[Chunk] = dedupe_by_source(reranked[:RERANK_KEEP_TOP_K], FINAL_TOP_K)
    print("  top-5 Sources:")
    for i, chunk in enumerate(top_five, start=1):
        print(f"    [{i}] {chunk.source.document_id} / {chunk.source.chapter}")

    print("\n== Generation (draft) ==")
    draft = generator.generate(query, top_five)
    print(f"  draft: {draft}")

    print("\n== Answer check ==")
    verdict = checker.check(query, draft, top_five)
    print(f"  verdict: {'supported' if verdict.supported else 'unsupported'}")
    if verdict.feedback:
        print(f"  feedback: {verdict.feedback}")
    if not verdict.supported:
        print("\n== Regeneration with feedback ==")
        regenerated = generator.generate(
            query, top_five, feedback=verdict.feedback or DEFAULT_UNSUPPORTED_FEEDBACK
        )
        print(f"  regenerated: {regenerated}")
        verdict = checker.check(query, regenerated, top_five)
        print(
            f"  verdict after regeneration: "
            f"{'supported' if verdict.supported else 'unsupported'}"
        )
        if verdict.supported:
            answer_text = regenerated
        else:
            print(f"  -> refusal: {DEFAULT_REPHRASE_SUGGESTION}")
            return
    else:
        answer_text = draft

    citations = parse_citations(answer_text, [c.source for c in top_five])
    print("\n== Final ==")
    print(f"  answer: {answer_text}")
    print("  citations:", [(c.marker, c.source.document_id) for c in citations])


if __name__ == "__main__":
    main()