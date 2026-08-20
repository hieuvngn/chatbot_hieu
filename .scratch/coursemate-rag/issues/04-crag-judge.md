# 04: CRAG judge, refine, and refusal

**What to build:** The retrieval-side quality gate: after re-ranking, an LLM judge scores the top-5 Sources high / medium / low. High means answer directly from them. Medium or low triggers one Refine — a query rewrite (exact keywords, no pronouns) followed by a full re-retrieval and re-rank. If the judge is still not high after the refine, the pipeline refuses to answer with a rephrase suggestion. Users with an out-of-domain or ambiguous question now get a clean refusal instead of a hallucinated answer, and ambiguous-but-answerable questions get one retry.

**Blocked by:** 03 (cross-encoder reranker)

**Status:** done

- [x] Judge assigns high / medium / low to the top-5 Sources for any query
- [x] Medium/low triggers exactly one refine (rewrite + re-retrieve + re-rank) before deciding
- [x] A query with no relevant material produces a refusal with a rephrase suggestion (no answer, no fabricated citations)
- [x] An ambiguous query that refines successfully produces a grounded answer after the single retry

## Comments

- 2026-08-20: Implemented. `rag_core/judge.py` with `Judge`/`QueryRewriter` protocols, `Judgment` (level + rephrase suggestion), `OpenRouterJudge` and `OpenRouterQueryRewriter` (LLM passes via OpenRouter; unparseable judge output degrades to `low` so the pipeline refuses rather than answering on an unverifiable verdict). `RagCore.answer()` runs the gate over the re-ranked top-5: high → answer as-is; otherwise one refine (rewrite + full re-retrieve + re-rank) and a second verdict; still not high → refusal with a rephrase suggestion (empty answer, no Citations, generator never called). `AnswerResult` gains `refused` and `rephrase_suggestion`. `build_rag_core()` wires judge + rewriter in; the optional `None` gate keeps the naive/rerank-only demo paths. `demo_crag.py` drives the gate once (single LLM judge call) and generates from the judged top-5. 39 tests pass, mypy strict clean. Review findings addressed: shared `numbered_sources` helper extracted into `generator.py` (removes duplicated prompt formatting), `Judge.assess()` naming, `Judgment.is_high`, `Level` exported, demo no longer re-runs the pipeline (was doubling LLM cost and could print a verdict contradicting the final answer), empty-default rephrase suggestion with pipeline fallback. Note: tests assert pipeline orchestration through recorded fake calls (`judge.calls`, `rewriter.calls`, `generator.calls`, `embedder.query_calls`) — the same fake-recording convention established by tickets 02/03; the spec's "never the internals" is read as no algorithm/prompt testing, not no call-count assertions at the seam.