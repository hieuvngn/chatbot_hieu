# 08: Evaluation script

**What to build:** A runnable evaluation script over a hand-made bilingual set of ~20 question/ground-truth pairs that reports retrieval hit rate (was the ground-truth Source in the top-5) and citation precision (how many returned Citations were actually used), producing numbers for the report and a regression check that the quality layers help.

**Blocked by:** 05 (answer check and regenerate)

**Status:** done

- [x] The script loads ~20 hand-made question/ground-truth pairs (Vietnamese and English)
- [x] It reports retrieval hit rate against the top-5 Sources for each question
- [x] It reports citation precision across the eval set
- [x] Output is a readable summary table suitable for the report

## Comments

- 2026-08-20: Implemented. `eval.py` — runnable from the repo root (`uv run python -m eval`, with `--retrieval-only` for hit@5 without LLM calls and `--eval-set` to point elsewhere). Loads the hand-made bilingual set `eval_set.json` (20 pairs, 10 Vietnamese + 10 English, each ground truth a document + chapter from the seeded dataset). Per question it reports retrieval hit@5 on both the naive path (RRF top-5) and the full path (re-rank + dedupe) — the side-by-side summary is the regression check — plus citation precision per answer (Citations pointing at the ground-truth Source ÷ Citations returned; pipeline refusals carry no Citations and are excluded from the mean, reported separately as a refusal count). Output is two markdown tables (per-question and aggregate) ready to paste into the report. Tests: 17 new `tests/test_eval.py` cases at the `eval.py` pure-function seam (eval-set loading and validation, hit/precision math, refusal → excluded, table rendering, and guards that the hand-made set is 20 bilingual pairs whose ground truths exist in the seeded dataset), 82 tests pass, mypy strict clean (19 files). README section added; `eval.py` added to the mypy file list. Code review findings addressed: refusals no longer score 0.0 precision (they would have penalized the full pipeline for following the refusal policy — now `n/a`-excluded with a separate refusal count), the duplicated document+chapter predicate extracted into `_matches_truth`, magic seed literal replaced with `gd.SEED`. Note: hit@5 is measured on the question as asked (before any CRAG refine), which is the retrieval metric the ticket defines.