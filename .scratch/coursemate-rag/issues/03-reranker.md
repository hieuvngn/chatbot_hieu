# 03: Cross-encoder reranker

**What to build:** Add the local-GPU re-ranking stage between RRF fusion and generation: the fused top-20 candidates are scored by the cross-encoder reranker and cut to top-10 then top-5 before being passed to generation. A demo script shows the ranking before and after re-ranking so the effect of this layer is visible. The user experience is a measurably better selection of the sources behind each answer.

**Blocked by:** 02 (naive RAG core)

**Status:** ready-for-agent

- [ ] Reranker loads from local GPU and scores the RRF-fused top-20 candidates
- [ ] Top-5 after re-ranking are the Sources used by generation (replacing the raw RRF top-5)
- [ ] A demo script prints the pre-rerank and post-rerank order for a sample query
- [ ] Pipeline still answers correctly end-to-end through `answer()` with re-ranked Sources