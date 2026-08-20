# 04: CRAG judge, refine, and refusal

**What to build:** The retrieval-side quality gate: after re-ranking, an LLM judge scores the top-5 Sources high / medium / low. High means answer directly from them. Medium or low triggers one Refine — a query rewrite (exact keywords, no pronouns) followed by a full re-retrieval and re-rank. If the judge is still not high after the refine, the pipeline refuses to answer with a rephrase suggestion. Users with an out-of-domain or ambiguous question now get a clean refusal instead of a hallucinated answer, and ambiguous-but-answerable questions get one retry.

**Blocked by:** 03 (cross-encoder reranker)

**Status:** ready-for-agent

- [ ] Judge assigns high / medium / low to the top-5 Sources for any query
- [ ] Medium/low triggers exactly one refine (rewrite + re-retrieve + re-rank) before deciding
- [ ] A query with no relevant material produces a refusal with a rephrase suggestion (no answer, no fabricated citations)
- [ ] An ambiguous query that refines successfully produces a grounded answer after the single retry