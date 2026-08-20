# 08: Evaluation script

**What to build:** A runnable evaluation script over a hand-made bilingual set of ~20 question/ground-truth pairs that reports retrieval hit rate (was the ground-truth Source in the top-5) and citation precision (how many returned Citations were actually used), producing numbers for the report and a regression check that the quality layers help.

**Blocked by:** 05 (answer check and regenerate)

**Status:** ready-for-agent

- [ ] The script loads ~20 hand-made question/ground-truth pairs (Vietnamese and English)
- [ ] It reports retrieval hit rate against the top-5 Sources for each question
- [ ] It reports citation precision across the eval set
- [ ] Output is a readable summary table suitable for the report