# 05: Answer check and regenerate

**What to build:** The generation-side quality gate: after an answer is drafted, an LLM verification pass checks that every claim is supported by the cited Sources. If any claim is unsupported, the pipeline regenerates once with concrete feedback naming the unsupported claims. If the regenerated answer is still unsupported, the pipeline refuses to answer. This completes the "never answer without a citation" policy end to end.

**Blocked by:** 04 (CRAG judge, refine, and refusal)

**Status:** done

- [x] Draft answers are checked claim-by-claim against the cited Sources
- [x] A failed check triggers exactly one regeneration with specific feedback about which claims were unsupported
- [x] A second failure produces a refusal with a rephrase suggestion
- [x] Valid answers pass the check and are returned unchanged with their Citations intact

## Comments

- 2026-08-20: Implemented. `rag_core/answer_check.py` with the `AnswerChecker` protocol, `CheckVerdict` (supported flag + regeneration feedback naming the unsupported claims), and `OpenRouterAnswerChecker` (LLM verification via OpenRouter; unparseable verdicts fail closed as unsupported so the pipeline regenerates or refuses rather than serving an unverified answer). `RagCore._generate_result()` runs the gate after generation: supported draft → returned unchanged with its Citations; unsupported → exactly one regeneration whose prompt carries the checker's feedback (`Generator.generate` gains an optional `feedback` parameter, wired through `OpenRouterGenerator`); still unsupported → refusal with a rephrase suggestion (empty answer, no Citations, no Sources). `build_rag_core()` wires the checker in; the optional `None` gate keeps the naive/rerank-only demo paths. `demo_answer_check.py` drives the gate once and shows the draft, verdict, regeneration with feedback, and final decision. 46 tests pass, mypy strict clean.