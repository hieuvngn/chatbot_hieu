# 02: Naive RAG core

**What to build:** The naive-RAG core of the pipeline: ingest the synthetic data into chunks (400–600 tokens, ~15% overlap, preferring heading boundaries so chunks map to document + chapter), embed all chunks in one batched request via the OpenRouter embedding model, build the FAISS (dense) and BM25 (lexical) indexes, fuse retrieval with Reciprocal Rank Fusion, and generate an answer from the top-5 Sources with gpt-4o-mini. The single public entry point `answer(user_message, session)` returns the answer text plus its Sources carrying citation metadata (document + chapter), so a user asking a question gets a grounded answer with visible sources.

**Blocked by:** 01 (synthetic data generator)

**Status:** done

- [x] Ingesting the synthetic data produces chunks whose metadata points to document + chapter
- [x] All chunks are embedded in a single batched request (no per-chunk calls)
- [x] Retrieval returns top-20 dense and top-20 BM25 results fused by RRF at k=60
- [x] `answer()` returns a useful answer to a knowledge question grounded in the synthetic documents, with its Sources attached
- [x] The query text is embedded once per call and retrieved against the FAISS index built at startup

## Comments

- 2026-08-20: Implemented in commits `b1a642e` and `319cde2`. `rag_core/` package with `RagCore.answer()` as the only public entry point, `build_rag_core()` factory, config via `.env`, `demo_naive_rag.py` per-layer demo. 30 tests pass, mypy strict clean. Code review findings (single-seam testing, 400–600 token chunks exercised by real data, demo double-embed, dead code) all addressed in the follow-up commit.