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

The prerequisite graph is a DAG by construction (prerequisites only ever reference strictly-earlier-semester courses). `generate_data.py` runs a topological sort after generation and raises an `AssertionError` if a cycle is ever introduced.

### Tests

```sh
uv run pytest
```

Tests assert the external contract of the generator: course count and fields, DAG acyclicity, document count, bilingual mix, that content references its course and topics, and that output is deterministic for a fixed seed.