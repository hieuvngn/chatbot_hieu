# 03: Cross-encoder reranker

**What to build:** Add the local-GPU re-ranking stage between RRF fusion and generation: the fused top-20 candidates are scored by the cross-encoder reranker and cut to top-10 then top-5 before being passed to generation. A demo script shows the ranking before and after re-ranking so the effect of this layer is visible. The user experience is a measurably better selection of the sources behind each answer.

**Blocked by:** 02 (naive RAG core)

**Status:** done

- [x] Reranker loads from local GPU and scores the RRF-fused top-20 candidates
- [x] Top-5 after re-ranking are the Sources used by generation (replacing the raw RRF top-5)
- [x] A demo script prints the pre-rerank and post-rerank order for a sample query
- [x] Pipeline still answers correctly end-to-end through `answer()` with re-ranked Sources

## Comments

- 2026-08-20: Implemented. `rag_core/reranker.py` with `Reranker` protocol + `LocalBgeReranker` (BAAI/bge-reranker-v2-m3 via transformers, GPU with CPU fallback, lazy import of the heavy stack); `RagCore.answer()` scores the RRF-fused top-20 with the reranker, keeps the top-10, dedupes to the top-5 Sources for generation; `build_rag_core()` wires it in; `demo_rerank.py` prints pre/post-rerank order. 33 tests pass, mypy strict clean. Note: FlagEmbedding was tried first but is broken with current transformers (`prepare_for_model` removed), so the model is loaded directly via `transformers.AutoModelForSequenceClassification`. Code review findings (windowed-dedupe regression in the naive path, stale docstring, naming, `__all__`) addressed.