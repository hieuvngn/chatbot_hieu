# 05: Answer check and regenerate

**What to build:** The generation-side quality gate: after an answer is drafted, an LLM verification pass checks that every claim is supported by the cited Sources. If any claim is unsupported, the pipeline regenerates once with concrete feedback naming the unsupported claims. If the regenerated answer is still unsupported, the pipeline refuses to answer. This completes the "never answer without a citation" policy end to end.

**Blocked by:** 04 (CRAG judge, refine, and refusal)

**Status:** ready-for-agent

- [ ] Draft answers are checked claim-by-claim against the cited Sources
- [ ] A failed check triggers exactly one regeneration with specific feedback about which claims were unsupported
- [ ] A second failure produces a refusal with a rephrase suggestion
- [ ] Valid answers pass the check and are returned unchanged with their Citations intact