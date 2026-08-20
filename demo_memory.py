"""Demo for the memory, auth and database layer (ticket 06).

Shows per-user persistence and conversational context: registering/logging in
against SQLite, the one Session per user that survives across runs, and the
session-aware query rewrite that turns a follow-up ("còn ví dụ về nó?") into a
standalone question using the last 6 turns before it enters retrieval.

Run from the repo root:
    uv run python -m demo_memory "còn ví dụ về nó?"

Requires OPENROUTER_API_KEY in .env (see README.md) and the local-GPU
dependencies (torch, transformers).
"""

from __future__ import annotations

import argparse
import json
import sys

from rag_core import RagCore
from rag_core.chunking import chunk_dataset
from rag_core.config import load_config
from rag_core.db import Database
from rag_core.embeddings import OpenRouterEmbedder
from rag_core.generator import OpenRouterGenerator
from rag_core.index import Index
from rag_core.judge import OpenRouterJudge, OpenRouterQueryRewriter
from rag_core.reranker import LocalBgeReranker
from rag_core.rewrite import OpenRouterSessionRewriter

FIRST_QUESTION = "giải thích bảng băm là gì?"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Demo the memory and auth layer.")
    parser.add_argument("follow_up", nargs="?", default="còn ví dụ về nó?")
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

    print("\n== Database ==")
    db = Database(config.data_dir / "demo.db")
    try:
        user = db.register("hieu", "demo-password")
        print(f"  registered new user: {user.username} (id={user.id})")
    except ValueError:
        existing = db.login("hieu", "demo-password")
        assert existing is not None, "the demo user must be able to log in"
        user = existing
        print(f"  existing user logged in: {user.username} (id={user.id})")
    session = db.get_or_create_session(user.id)
    print(f"  session {session.id} resumed with {len(session.turns)} previous turns")

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
        judge=OpenRouterJudge(
            api_key=config.api_key,
            model=config.llm_model,
            base_url=config.base_url,
        ),
        rewriter=OpenRouterQueryRewriter(
            api_key=config.api_key,
            model=config.llm_model,
            base_url=config.base_url,
        ),
    )
    session_rewriter = OpenRouterSessionRewriter(
        api_key=config.api_key,
        model=config.llm_model,
        base_url=config.base_url,
    )

    print("\n== Turn 1: first message (no history, passes through unchanged) ==")
    print(f"  message: {FIRST_QUESTION}")
    result = core.answer(FIRST_QUESTION, session)
    if result.refused:
        print(f"  -> refusal: {result.rephrase_suggestion}")
    else:
        print(f"  answer: {result.answer}")
        print("  citations:", [(c.marker, c.source.document_id) for c in result.citations])
        db.append_exchange(session.id, FIRST_QUESTION, result.answer)

    print(f"\n== Turn 2: follow-up (rewritten with the last 6 turns) ==")
    print(f"  message: {args.follow_up}")
    print(f"  history: {[turn.text for turn in session.turns]}")
    rewritten = session_rewriter.rewrite(args.follow_up, session.turns)
    print(f"  rewritten query: {rewritten}")
    result = core.answer(rewritten, session)
    if result.refused:
        print(f"  -> refusal: {result.rephrase_suggestion}")
    else:
        print(f"  answer: {result.answer}")
        print("  citations:", [(c.marker, c.source.document_id) for c in result.citations])
        db.append_exchange(session.id, args.follow_up, result.answer)

    print("\n== Session after both turns ==")
    resumed = db.get_session(session.id)
    for i, turn in enumerate(resumed.turns, start=1):
        print(f"  {i}. {turn.role}: {turn.text}")

    db.close()
    print(f"\nDatabase written to {config.data_dir / 'demo.db'}; run again to resume this history.")


if __name__ == "__main__":
    main()