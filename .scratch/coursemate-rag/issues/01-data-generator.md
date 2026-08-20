# 01: Synthetic data generator

**What to build:** A script that generates the demo dataset the whole chatbot runs on: ~50 courses (code, name, credits, prerequisites, semester, department, instructor, description) whose prerequisite graph is a DAG with no cycles, plus ~40 study documents (slides, textbook chapters) split between Vietnamese and English. The generated data is written to the project's data directory and is fully reproducible.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] Generating ~50 courses with realistic IT-curriculum fields produces a prerequisite DAG that is provably acyclic (script asserts no cycles)
- [x] Generating ~40 study documents produces content that references the generated courses and topics, mixed Vietnamese and English
- [x] Running the script twice produces identical output (deterministic, seeded)
- [x] A README/short note explains how to run the script and what each output file contains

## Comments

- Implemented in `generate_data.py` with tests in `tests/test_generate_data.py` (18 tests). Data written to `data/` (`courses.json`, `documents.json`). README covers usage and output schema.
- Acyclicity is provable by construction: every prerequisite strictly precedes its course in semester order, and `assert_no_cycles()` enforces that invariant (unknown prereq, non-earlier semester, or cycle → AssertionError).