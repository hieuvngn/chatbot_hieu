# CourseMate RAG Chatbot

A demo chatbot for IT students: course advisory and knowledge Q&A grounded in study documents, built on a retrieval-augmented generation pipeline. See `.scratch/coursemate-rag/spec.md` for the full specification.

> **Hướng dẫn cài đặt & chạy chi tiết:** xem [`docs/SETUP.md`](docs/SETUP.md) — bao gồm cài `uv`, tạo `.env`, sinh data, chạy demo/eval/UI, và troubleshooting.

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
- `documents.json` — the study-material corpus, a list of 40 documents (slides and textbook chapters) split between Vietnamese and English. Each object has:
  - `id` — unique document id (e.g. `DOC-001`)
  - `course_code` — the course this material belongs to
  - `title` — document title
  - `kind` — `slides` or `textbook` (or `exam` / `lab_guide` / `cheatsheet` / `faq` for the extra kinds produced by `--extra-documents`)
  - `language` — `vi` or `en`
  - `chapters` — list of chapters/sections, each with an `id`, `title`, and `content` referencing the course and its topics

The generator also emits four **entity tables** alongside the corpus (see `generate_entities.py`):

- `departments.json` — academic departments (`CNTT`, `MATH`, `GE`).
- `instructors.json` — staff roster; each entry lists the courses they teach and a bio.
- `programs.json` — curricula grouping courses into required + elective tracks (`PR-CNTT`, `PR-AI`, `PR-DS`).
- `terms.json` — specific semester offerings (`T-2025-S1`, `T-2025-S2`, `T-2026-S1`) with the courses that term runs and which instructor teaches them.

### Data integrity

The prerequisite graph is a DAG by construction: every prerequisite strictly precedes its course in semester order. `generate_data.py` enforces this invariant after generation (it raises an `AssertionError` if a prerequisite is unknown, not strictly earlier, or part of a cycle).

### Tests

```sh
uv run pytest
```

Tests assert the external contract of the generator: course count and fields, DAG acyclicity, document count, bilingual mix, that content references its course and topics, and that output is deterministic for a fixed seed.

## Naive RAG core

The `rag_core` package ingests the synthetic data into chunks, builds a hybrid FAISS (dense) + BM25 (lexical) index, and answers questions through its single public entry point `answer(user_message, session)`. The pipeline order: hybrid retrieval (BM25 top-20 + dense top-20 fused by Reciprocal Rank Fusion at k=60) → generation from the top-5 Sources.

### Setup

Requires an OpenRouter API key. Create a `.env` file in the repo root:

```sh
OPENROUTER_API_KEY=sk-or-...
```

The embedding model (`nvidia/nemotron-3-embed-1b:free`, 2048-dim) and the LLM (`gpt-4o-mini`) both go through the OpenAI-compatible OpenRouter endpoint. Model names and the data directory are configurable via environment variables (`RAG_LLM_MODEL`, `RAG_EMBED_MODEL`, `RAG_EMBED_DIM`, `RAG_DATA_DIR`, `OPENROUTER_BASE_URL`).

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

## CRAG judge

Between retrieval and generation, the quality gate decides whether the retrieved top-5 Sources are trustworthy enough to answer from:

- **Cheap gate first** — if the query has a strong BM25 match against the corpus (score ≥ `RETRIEVAL_GATE_THRESHOLD` in `rag_core/index.py`), retrieval is trusted and the answer is generated straight away: no judge call at all.
- Otherwise an LLM judge scores the sources high / medium / low:
  - **high** — the sources directly answer the question; the pipeline answers from them as-is.
  - **medium / low** — the pipeline Refines exactly once: a query rewrite (exact keywords, pronouns dropped) followed by a full re-retrieval. If the judge is still not high after that single refine, the pipeline refuses with a rephrase suggestion.
- A refusal returns an empty answer with no Citations and never calls the generator — no hallucinated or unsupported answer is ever served.

The `AnswerResult` carries two extra fields when it refuses: `refused=True` and a `rephrase_suggestion`. `build_rag_core()` wires the judge and rewriter in automatically.

### Demo script

```sh
uv run python -m demo_crag "giải thích bảng băm là gì?"
```

Prints the top-5 Sources, the judge's verdict, the refine rewrite with its re-retrieved top-5 and second verdict, and the final decision — a generated answer with citations or a refusal with a rephrase suggestion.

## Web search (Firecrawl, optional)

When local retrieval is judged insufficient even after its single Refine, the pipeline performs one corrective web search through the [Firecrawl Search API](https://docs.firecrawl.dev/features/search) and answers from the merged sources (CRAG-style correction). A sidebar toggle ("Tìm kiếm web") additionally merges web results into every retrieval.

- Configure with `FIRECRAWL_API_KEY=...` in `.env` (see `.env.example`). Without a key the feature is silently disabled and the bot behaves exactly as before.
- Web results appear as ordinary citations with `kind="web"`; each citation links back to the source URL. At most 2 Firecrawl calls are made per answer (5 results each, page content truncated to 4000 chars).

## Answer check

After generation, the generation-side quality gate verifies the draft answer. First a cheap gate: a draft that already cites valid Sources is grounded evidence — it passes through unchanged with its Citations intact, no second LLM call. Only a draft with no Citations is checked claim-by-claim against the cited Sources by an LLM verifier; if any claim is unsupported, the pipeline regenerates exactly once with concrete feedback naming the unsupported claims. If the regenerated answer still fails the check, the pipeline refuses with a rephrase suggestion — never serving an unverified answer. `build_rag_core()` wires the verifier in automatically.

### Demo script

```sh
uv run python -m demo_answer_check "giải thích bảng băm là gì?"
```

Prints the top-5 Sources, the draft answer, the verifier's verdict with the named unsupported claims, the regeneration with feedback when the draft fails, and the final decision — a verified answer with citations or a refusal.

## Memory, auth, and database

Per-user persistence and conversational context: a SQLite database (`rag_core/db.py`) with `users`, `conversations` (one per user), and `messages` tables. Register and login by username/password (passwords stored in plaintext by explicit demo choice). Each user has a single Session that persists the last 6 turns across visits. Before retrieval, an LLM pass (`rag_core/rewrite.py`) rewrites a follow-up message into a standalone question using the session history; first messages pass through unchanged. `answer(user_message, session)` consumes the Session and the rewritten query flows through the whole pipeline (retrieval → judge → generation → answer check). `build_rag_core()` wires the session rewriter in automatically.

### Demo script

```sh
uv run python -m demo_memory "còn ví dụ về nó?"
```

Registers/logs in a demo user against SQLite, asks a first question (passed through unchanged), then a follow-up whose rewritten query is printed before it enters retrieval, and shows the session history surviving a database reopen.

## Web UI (React + FastAPI)

The user-facing app is a React single-page application served by the FastAPI backend (`server/`): a login/register screen gates access to the chat; chat messages flow through `answer()` with the logged-in user's Session; every answer renders its Citations as clickable cards showing document + chapter; refusals render distinctly with their rephrase suggestion. History — including Citations and refusals — is stored in SQLite and restored on the next login.

- Upload PDF/TXT/MD làm nguồn tri thức tạm thời theo cuộc trò chuyện (citation kèm trang/heading).
- Sidebar toggle ("Tìm kiếm web") merges web results into every retrieval when Firecrawl is configured.

### Running (dev mode)

Two processes with hot reload:

```sh
uv run uvicorn server.main:app --reload   # API on http://127.0.0.1:8000/api
cd web && npm run dev                     # Vite dev server on http://localhost:5173
```

### Running (production mode)

One process serves both the API and the built SPA:

```sh
cd web && npm run build
uv run serve                              # http://127.0.0.1:8000
```

Requires Node 20+ for the `web/` build, `OPENROUTER_API_KEY` in `.env`, and the synthetic data from `generate_data.py`. The app database (`data/app.db`) is created on first run.

## Naive RAG core demo

```sh
uv run python -m demo_naive_rag "giải thích bảng băm là gì?"
```

Prints the ingested unit count, the single-batched embedding step, the fused retrieval ranking, and a generated answer with its citations.

## Evaluation

`eval.py` computes the report numbers over a hand-made bilingual set of 20 question/ground-truth pairs (`eval_set.json`, 10 Vietnamese + 10 English, each pointing at the document + chapter the answer must come from). It measures, per question and in aggregate:

- **Retrieval hit@5** — was the ground-truth Source among the top-5 Sources retrieved (RRF top-5)?
- **Citation precision** — of the Citations the generated answer returned, how many point at the ground-truth Source? Answers the pipeline refused carry no Citations and are excluded from the precision mean (reported separately as refusals), so policy-correct refusals never penalize precision.

The summary table shows naive vs. full side by side — the regression check that the quality layers (CRAG judge, answer check) help.

```sh
uv run python -m eval                      # full eval: retrieval + LLM generation
uv run python -m eval --retrieval-only     # hit@5 only, no LLM calls
```

Options:

- `--eval-set PATH` — path to the eval set (default `./eval_set.json`).
- `--retrieval-only` — skip generation and report only retrieval hit@5.

The eval set targets the seeded dataset: regenerate the data with `uv run python -m generate_data --seed 42` so the ground-truth document/chapter references stay valid (the test suite asserts this).

## CI/CD

Automated tests run on every push and pull request via GitHub Actions:

- **Python tests** — unit + integration on Python 3.11, 3.12, 3.13 (mypy included)
- **System tests** — run with `OPENROUTER_API_KEY` secret (optional)
- **Web build** — TypeScript typecheck, lint, production build

See `.github/workflows/ci.yml` for the full configuration.
