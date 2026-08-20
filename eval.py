"""Evaluation script (ticket 08).

Computes the numbers for the report over a hand-made bilingual set of
~20 question/ground-truth pairs (see eval_set.json):

- retrieval hit rate @5: was the ground-truth Source (document + chapter)
  among the top-5 Sources retrieved for the question, on the naive path
  (RRF top-5) and on the full path (re-rank + dedupe)? Hit rate is measured
  on the question as asked, before any CRAG refine.
- citation precision: of the Citations the generated answer returned, how
  many point at the ground-truth Source? Answers the pipeline refused carry
  no Citations and are excluded from the precision mean (reported separately
  as refusals), so policy-correct refusals never penalize precision.

The summary table side by side is the regression check that the quality
layers (re-ranking, judge, answer check) help.

Run from the repo root:
    uv run python -m eval                    # full eval (LLM generation)
    uv run python -m eval --retrieval-only   # hit rate only, no LLM calls

Requires OPENROUTER_API_KEY in .env (see README.md) and the local-GPU
dependencies (torch, transformers) for the re-ranker.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from rag_core import RagCore, Session
from rag_core.answer_check import OpenRouterAnswerChecker
from rag_core.chunking import chunk_dataset
from rag_core.config import load_config
from rag_core.embeddings import OpenRouterEmbedder
from rag_core.generator import OpenRouterGenerator
from rag_core.index import FINAL_TOP_K, Index, dedupe_by_source
from rag_core.judge import OpenRouterJudge, OpenRouterQueryRewriter
from rag_core.models import AnswerResult, Citation, Source
from rag_core.reranker import (
    RERANK_INPUT_TOP_K,
    RERANK_KEEP_TOP_K,
    LocalBgeReranker,
    Reranker,
)
from rag_core.rewrite import OpenRouterSessionRewriter

DEFAULT_EVAL_SET = Path(__file__).parent / "eval_set.json"
MAX_QUESTION_CHARS = 40


@dataclass(frozen=True)
class GroundTruth:
    document_id: str
    chapter: str


@dataclass(frozen=True)
class EvalItem:
    id: str
    question: str
    language: str
    ground_truth: GroundTruth


@dataclass(frozen=True)
class EvalRow:
    item: EvalItem
    hit_naive: bool
    hit_full: bool
    precision_naive: float | None
    precision_full: float | None
    refused_full: bool


@dataclass(frozen=True)
class EvalSummary:
    hit_rate_naive: float
    hit_rate_full: float
    citation_precision_naive: float | None
    citation_precision_full: float | None
    refusals_full: int
    total: int


def load_eval_set(path: Path) -> list[EvalItem]:
    """Parse and structurally validate the eval set file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list of eval items")
    items: list[EvalItem] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"eval item {index} is not an object")
        truth = entry.get("ground_truth")
        if not isinstance(truth, dict):
            raise ValueError(f"eval item {index} is missing ground_truth")
        try:
            item = EvalItem(
                id=entry["id"],
                question=entry["question"],
                language=entry["language"],
                ground_truth=GroundTruth(
                    document_id=truth["document_id"],
                    chapter=truth["chapter"],
                ),
            )
        except KeyError as exc:
            raise ValueError(
                f"eval item {index} is missing required field {exc.args[0]!r}"
            ) from exc
        if item.language not in ("vi", "en"):
            raise ValueError(f"eval item {index} has unsupported language {item.language!r}")
        if not item.id or not item.question:
            raise ValueError(f"eval item {index} has an empty id or question")
        items.append(item)
    return items


def _matches_truth(source: Source, truth: GroundTruth) -> bool:
    return (
        source.document_id == truth.document_id
        and source.chapter == truth.chapter
    )


def is_hit(sources: list[Source], truth: GroundTruth) -> bool:
    """Was the ground-truth Source (document + chapter) among the sources?"""
    return any(_matches_truth(s, truth) for s in sources)


def hit_rate(hits: list[bool]) -> float:
    if not hits:
        return 0.0
    return sum(hits) / len(hits)


def citation_precision(citations: list[Citation], truth: GroundTruth) -> float:
    """Fraction of returned Citations that point at the ground-truth Source."""
    if not citations:
        return 0.0
    used = [c for c in citations if _matches_truth(c.source, truth)]
    return len(used) / len(citations)


def answer_precision(result: AnswerResult, truth: GroundTruth) -> float | None:
    """Citation precision of one answer; None when the pipeline refused."""
    if result.refused:
        return None
    return citation_precision(result.citations, truth)


def mean_precision(precisions: list[float]) -> float:
    if not precisions:
        return 0.0
    return sum(precisions) / len(precisions)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_table(rows: list[EvalRow]) -> str:
    """A per-question markdown table suitable for pasting into the report."""
    header = (
        "| id | lang | question | ground truth | hit@5 naive | hit@5 full "
        "| citation precision naive | citation precision full | refused |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, separator]
    for row in rows:
        item = row.item
        question = (
            item.question
            if len(item.question) <= MAX_QUESTION_CHARS
            else item.question[: MAX_QUESTION_CHARS - 1] + "…"
        )
        truth = item.ground_truth
        lines.append(
            f"| {item.id} | {item.language} | {question} | {truth.document_id} / "
            f"{truth.chapter} | {_pct(float(row.hit_naive))} | "
            f"{_pct(float(row.hit_full))} | {_pct(row.precision_naive)} | "
            f"{_pct(row.precision_full)} | {'yes' if row.refused_full else ''} |"
        )
    return "\n".join(lines)


def format_summary(summary: EvalSummary) -> str:
    """The aggregate metrics table — the regression check at a glance."""
    return (
        "| metric | naive | full |\n"
        "|---|---|---|\n"
        f"| retrieval hit@5 | {_pct(summary.hit_rate_naive)} | "
        f"{_pct(summary.hit_rate_full)} |\n"
        f"| citation precision | {_pct(summary.citation_precision_naive)} | "
        f"{_pct(summary.citation_precision_full)} |\n"
        f"| refusals (full pipeline) | — | {summary.refusals_full} / "
        f"{summary.total} |"
    )


def retrieve_sources(
    index: Index,
    reranker: Reranker | None,
    query: str,
    query_vector: np.ndarray,
) -> list[Source]:
    """The top-5 Sources the pipeline would answer from, naive or re-ranked."""
    if reranker is None:
        return [chunk.source for chunk in index.retrieve(query, query_vector)]
    fused = index.fused_candidates(query, query_vector, RERANK_INPUT_TOP_K)
    reranked = reranker.rerank(query, fused)
    return [
        chunk.source
        for chunk in dedupe_by_source(reranked[:RERANK_KEEP_TOP_K], FINAL_TOP_K)
    ]


def _empty_session() -> Session:
    return Session(id="eval", user_id="eval", turns=[])


def run_eval(
    items: list[EvalItem],
    index: Index,
    reranker: Reranker | None,
    core_naive: RagCore,
    core_full: RagCore,
    embedder: OpenRouterEmbedder,
    retrieval_only: bool,
) -> tuple[list[EvalRow], EvalSummary]:
    rows: list[EvalRow] = []
    for item in items:
        query = item.question
        vector = np.asarray(embedder.embed_query(query), dtype=np.float32)
        hit_naive = is_hit(retrieve_sources(index, None, query, vector), item.ground_truth)
        hit_full = is_hit(retrieve_sources(index, reranker, query, vector), item.ground_truth)
        precision_naive: float | None = None
        precision_full: float | None = None
        refused_full = False
        if not retrieval_only:
            naive_result = core_naive.answer(query, _empty_session())
            full_result = core_full.answer(query, _empty_session())
            precision_naive = answer_precision(naive_result, item.ground_truth)
            precision_full = answer_precision(full_result, item.ground_truth)
            refused_full = full_result.refused
        rows.append(
            EvalRow(
                item=item,
                hit_naive=hit_naive,
                hit_full=hit_full,
                precision_naive=precision_naive,
                precision_full=precision_full,
                refused_full=refused_full,
            )
        )
    precisions_naive = [r.precision_naive for r in rows if r.precision_naive is not None]
    precisions_full = [r.precision_full for r in rows if r.precision_full is not None]
    summary = EvalSummary(
        hit_rate_naive=hit_rate([r.hit_naive for r in rows]),
        hit_rate_full=hit_rate([r.hit_full for r in rows]),
        citation_precision_naive=(
            mean_precision(precisions_naive) if precisions_naive else None
        ),
        citation_precision_full=(
            mean_precision(precisions_full) if precisions_full else None
        ),
        refusals_full=sum(1 for r in rows if r.refused_full),
        total=len(rows),
    )
    return rows, summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the CourseMate RAG pipeline.")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=DEFAULT_EVAL_SET,
        help=f"Path to the eval set (default: {DEFAULT_EVAL_SET}).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only report retrieval hit@5; skip LLM generation and citation precision.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config()
    except ValueError as exc:
        print(str(exc))
        sys.exit(1)

    try:
        items = load_eval_set(args.eval_set)
    except ValueError as exc:
        print(f"Invalid eval set: {exc}")
        sys.exit(1)

    print("== Data ==")
    courses = json.loads((config.data_dir / "courses.json").read_text(encoding="utf-8"))
    documents = json.loads((config.data_dir / "documents.json").read_text(encoding="utf-8"))
    chunks = chunk_dataset(list(courses), list(documents))
    print(f"  {len(courses)} courses, {len(documents)} documents, {len(chunks)} index units")

    print("\n== Embedding ==")
    embedder = OpenRouterEmbedder(
        api_key=config.api_key,
        model=config.embed_model,
        dim=config.embed_dim,
        base_url=config.base_url,
    )
    index = Index(chunks, embedder)
    print(f"  index chunks embedded in a single batched request ({config.embed_dim}-dim)")

    print("\n== Re-ranking ==")
    reranker = LocalBgeReranker(model_name=config.rerank_model)
    print(f"  reranker: {reranker.model_name} on {reranker.device}")

    generator = OpenRouterGenerator(
        api_key=config.api_key,
        model=config.llm_model,
        base_url=config.base_url,
    )
    core_naive = RagCore(
        data_dir=config.data_dir,
        embedder=embedder,
        generator=generator,
        index=index,
    )
    core_full = RagCore(
        data_dir=config.data_dir,
        embedder=embedder,
        generator=generator,
        index=index,
        reranker=reranker,
        judge=OpenRouterJudge(
            api_key=config.api_key,
            model=config.llm_model,
            base_url=config.base_url,
        ),
        rewriter=OpenRouterQueryRewriter(
            api_key=config.api_key,
            model=config.llm_model,
            base_url=config.base_url,
        ),
        checker=OpenRouterAnswerChecker(
            api_key=config.api_key,
            model=config.llm_model,
            base_url=config.base_url,
        ),
        session_rewriter=OpenRouterSessionRewriter(
            api_key=config.api_key,
            model=config.llm_model,
            base_url=config.base_url,
        ),
    )

    mode = "retrieval only" if args.retrieval_only else "full"
    print(f"\n== Eval ({mode}) ==")
    print(f"  {len(items)} questions, "
          f"{sum(i.language == 'vi' for i in items)} Vietnamese, "
          f"{sum(i.language == 'en' for i in items)} English")
    rows, summary = run_eval(
        items,
        index,
        reranker,
        core_naive,
        core_full,
        embedder,
        args.retrieval_only,
    )

    print("\nPer-question results:")
    print(format_table(rows))
    print("\nSummary (regression check: quality layers vs naive):")
    print(format_summary(summary))


if __name__ == "__main__":
    main()