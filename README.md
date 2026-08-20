# CourseMate RAG Chatbot

A demo chatbot for IT students: course advisory and knowledge Q&A grounded in study documents, built on a retrieval-augmented generation pipeline. See `.scratch/coursemate-rag/spec.md` for the full specification.

## Synthetic data generator

The whole demo runs on synthetic data — there is no real institutional data. `generate_data.py` produces the dataset deterministically (seeded), so any machine can reproduce identical output.

### Running

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). From the repo root:

```sh
uv run python -m generate_data --seed 42
```

Options:

- `--seed N` — random seed for reproducibility (default `42`).
- `--out DIR` — output directory (default `./data`).

Running the script twice with the same seed produces byte-identical output files.

### Output files

The script writes two JSON files into the output directory:

- `courses.json` — the course catalog, a list of 50 course objects. Each object has:
  - `code` — unique course code (e.g. `CS101`)
  - `name` / `name_en` — Vietnamese and English course names
  - `credits` — credit count
  - `prerequisites` — list of course codes that must be completed first
  - `semester` — semester in which the course is offered (1–8)
  - `department` — offering department
  - `instructor` — assigned instructor
  - `description` — a one-paragraph syllabus summary
- `documents.json` — the study-material corpus, a list of 40 documents (slides and textbook chapters) split between Vietnamese and English. Each object has:
  - `id` — unique document id (e.g. `DOC-001`)
  - `course_code` — the course this material belongs to
  - `title` — document title
  - `kind` — `slides` or `textbook`
  - `language` — `vi` or `en`
  - `chapters` — list of chapters/sections, each with an `id`, `title`, and `content` referencing the course and its topics

### Data integrity

The prerequisite graph is a DAG by construction: every prerequisite strictly precedes its course in semester order. `generate_data.py` enforces this invariant after generation (it raises an `AssertionError` if a prerequisite is unknown, not strictly earlier, or part of a cycle).

### Tests

```sh
uv run pytest
```

Tests assert the external contract of the generator: course count and fields, DAG acyclicity, document count, bilingual mix, that content references its course and topics, and that output is deterministic for a fixed seed.

## Naive RAG core

The `rag_core` package ingests the synthetic data into chunks, builds a hybrid FAISS (dense) + BM25 (lexical) index, and answers questions through its single public entry point `answer(user_message, session)`. The pipeline order: hybrid retrieval (BM25 top-20 + dense top-20 fused by Reciprocal Rank Fusion at k=60) → re-ranking → generation from the top-5 Sources.

### Setup

Requires an OpenRouter API key. Create a `.env` file in the repo root:

```sh
OPENROUTER_API_KEY=sk-or-...
```

The embedding model (`nvidia/nemotron-3-embed-1b:free`, 2048-dim) and the LLM (`gpt-4o-mini`) both go through the OpenAI-compatible OpenRouter endpoint. Model names and the data directory are configurable via environment variables (`RAG_LLM_MODEL`, `RAG_EMBED_MODEL`, `RAG_EMBED_DIM`, `RAG_RERANK_MODEL`, `RAG_DATA_DIR`, `OPENROUTER_BASE_URL`).

### Usage

```python
from rag_core import build_rag_core, Session

core = build_rag_core()
result = core.answer("giải thích bảng băm là gì?", Session(id="s1", user_id="u1", turns=[]))
print(result.answer)
for citation in result.citations:
    print(citation.marker, citation.source.document_id, citation.source.chapter)
```

`AnswerResult` carries the answer text, its `Citations` (the `[n]` markers in the answer mapped back to their Sources), and the ordered top-5 `Sources` used. All index units are embedded in a single batched request at build time; queries are embedded one per call.

## Re-ranking

Between retrieval and generation, the RRF-fused top-20 candidates are scored by a cross-encoder re-ranker (`BAAI/bge-reranker-v2-m3`) that runs locally on GPU (CPU fallback when CUDA is unavailable). The re-ranked top-10 chunks are deduplicated to the final top-5 Sources passed to generation, replacing the raw RRF top-5. `build_rag_core()` wires the re-ranker in automatically.

### Demo script

```sh
uv run python -m demo_rerank "giải thích bảng băm là gì?"
```

Prints the fused top-20 in pre-rerank order, the post-rerank top-10, the final top-5 Sources, and a generated answer with its citations, so the effect of this layer is visible side by side.

## CRAG judge

Between re-ranking and generation, the quality gate decides whether the retrieved top-5 Sources are trustworthy enough to answer from. An LLM judge scores them high / medium / low:

- **high** — the sources directly answer the question; the pipeline answers from them as-is.
- **medium / low** — the pipeline Refines exactly once: a query rewrite (exact keywords, pronouns dropped) followed by a full re-retrieval and re-ranking. If the judge is still not high after that single refine, the pipeline refuses with a rephrase suggestion.
- A refusal returns an empty answer with no Citations and never calls the generator — no hallucinated or unsupported answer is ever served.

The `AnswerResult` carries two extra fields when it refuses: `refused=True` and a `rephrase_suggestion`. `build_rag_core()` wires the judge and rewriter in automatically.

### Demo script

```sh
uv run python -m demo_crag "giải thích bảng băm là gì?"
```

Prints the top-5 Sources, the judge's verdict, the refine rewrite with its re-retrieved top-5 and second verdict, and the final decision — a generated answer with citations or a refusal with a rephrase suggestion.

## Naive RAG core demo

```sh
uv run python -m demo_naive_rag "giải thích bảng băm là gì?"
```

Prints the ingested unit count, the single-batched embedding step, the fused retrieval ranking, and a generated answer with its citations.