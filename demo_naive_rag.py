"""Demo for the naive RAG core (ticket 02).

Shows each layer of the pipeline: chunking, single-batched embedding,
hybrid retrieval (BM25 top-20 + dense top-20 fused by RRF), and a sample
answer with its Citations.

Run from the repo root:
    uv run python -m demo_naive_rag "giải thích bảng băm là gì?"

Requires OPENROUTER_API_KEY in .env (see README.md).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from rag_core import RagCore, Session
from rag_core.chunking import chunk_dataset
from rag_core.config import load_config
from rag_core.embeddings import OpenRouterEmbedder
from rag_core.generator import OpenRouterGenerator
from rag_core.index import Index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Demo the naive RAG pipeline.")
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
    print(f"  {len(courses)} courses, {len(documents)} documents, {len(chunks)} chunks")

    print("\n== Embedding ==")
    embedder = OpenRouterEmbedder(
        api_key=config.api_key,
        model=config.embed_model,
        dim=config.embed_dim,
        base_url=config.base_url,
    )
    index = Index(chunks, embedder)
    print(f"  index chunks embedded in a single batched request ({config.embed_dim}-dim)")

    print("\n== Retrieval (hybrid: BM25 top-20 + dense top-20, RRF k=60) ==")
    query = args.question
    query_vector = np.asarray(embedder.embed_query(query), dtype=np.float32)

    fused = index.fused_ranking(query, query_vector)

    print(f"  query: {query}")
    print("  fused top-5 :", [index.chunks[i].source.document_id for i in fused[:5]])

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
    )
    result = core.answer(query, Session(id="demo", user_id="demo", turns=[]))

    print(f"  answer: {result.answer}")
    print("  sources:")
    for i, source in enumerate(result.sources, start=1):
        print(f"    [{i}] {source.document_id} / {source.chapter}")
    print("  citations:", [(c.marker, c.source.document_id) for c in result.citations])


if __name__ == "__main__":
    main()