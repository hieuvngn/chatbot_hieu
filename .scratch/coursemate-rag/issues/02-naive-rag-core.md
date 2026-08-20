# 02: Naive RAG core

**What to build:** The naive-RAG core of the pipeline: ingest the synthetic data into chunks (400–600 tokens, ~15% overlap, preferring heading boundaries so chunks map to document + chapter), embed all chunks in one batched request via the OpenRouter embedding model, build the FAISS (dense) and BM25 (lexical) indexes, fuse retrieval with Reciprocal Rank Fusion, and generate an answer from the top-5 Sources with gpt-4o-mini. The single public entry point `answer(user_message, session)` returns the answer text plus its Sources carrying citation metadata (document + chapter), so a user asking a question gets a grounded answer with visible sources.

**Blocked by:** 01 (synthetic data generator)

**Status:** ready-for-agent

- [ ] Ingesting the synthetic data produces chunks whose metadata points to document + chapter
- [ ] All chunks are embedded in a single batched request (no per-chunk calls)
- [ ] Retrieval returns top-20 dense and top-20 BM25 results fused by RRF at k=60
- [ ] `answer()` returns a useful answer to a knowledge question grounded in the synthetic documents, with its Sources attached
- [ ] The query text is embedded once per call and retrieved against the FAISS index built at startup