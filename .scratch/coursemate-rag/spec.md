Status: ready-for-agent

# CourseMate RAG Chatbot

## Problem Statement

An IT student wants to pick courses and find study material but the information is scattered: course syllabi live in one place, prerequisites in another, and study documents (slides, textbooks, chapters) are unsearchable files. They end up asking seniors or rummaging through folders. A demo chatbot should answer both needs in one place: advise on which course to take (with prerequisites, credits, syllabus details) and answer knowledge questions grounded in actual study documents — and every answer must show its sources so the student can verify it.

## Solution

A Streamlit chatbot backed by a single RAG pipeline with quality gates. The pipeline starts from a "naive RAG" core (retrieve relevant chunks, feed them to an LLM, answer) and layers on top: hybrid retrieval (BM25 + dense vectors fused with Reciprocal Rank Fusion), cross-encoder re-ranking, a CRAG-style judge that decides when retrieved material is trustworthy enough to answer, a Self-RAG-style answer check that refuses unsupported claims, and conversational memory with query rewriting. Citations are shown as clickable expanders pointing at document + chapter.

The chatbot serves two tasks off the same core:
- **Course advisory** — recommending and explaining courses from structured data (syllabus, credits, prerequisites).
- **Knowledge Q&A** — answering questions grounded in study documents with visible citations.

There is no real data: a script generates synthetic data (courses and documents, bilingual Vietnamese/English) so the demo is self-contained and reproducible.

## User Stories

1. As a student, I want to log in or register an account, so that my conversation history is saved and resumed across visits.
2. As a student, I want to ask a question about courses (e.g. "môn nào phù hợp cho sinh viên năm 2?"), so that I get course recommendations grounded in the syllabus data.
3. As a student, I want to ask about prerequisites ("môn X cần học môn nào trước?"), so that I can plan my semester sequence.
4. As a student, I want to compare courses ("so sánh môn X và môn Y"), so that I can choose between options.
5. As a student, I want to ask knowledge questions about topics in the study documents ("giải thích đệ quy"), so that I get answers grounded in the actual slides/textbook chapters.
6. As a student, I want each answer to show its Citation as clickable expanders, so that I can open and verify the document and chapter the answer came from.
7. As a student, I want to ask follow-up questions ("và tài liệu về phần đó?"), so that the bot understands the context instead of treating each question in isolation.
8. As a student, I want to ask in both Vietnamese and English, so that I can use whichever language I'm comfortable with.
9. As a student, I want the bot to tell me when it cannot find supporting material, so that I am never served an unsupported or hallucinated answer.
10. As a student, I want the bot to retry once with a better search when my first query is ambiguous, so that I still get an answer when possible.
11. As a student, I want to see which course a study document belongs to, so that I can connect material to the curriculum.
12. As a student, I want my last 6 turns of conversation remembered, so that short references ("môn đó") work without repeating myself.
13. As a demonstrator, I want a script that generates realistic synthetic courses and documents, so that the demo runs on a fresh machine without real institutional data.
14. As a demonstrator, I want per-layer demo scripts, so that I can show the contribution of each stage (retrieval, re-ranking, judge, answer check) in the report.
15. As a demonstrator, I want an evaluation script that measures retrieval hit rate and citation precision on a small bilingual question set, so that the report has numbers.
16. As a student, I want to end a session and come back to it, so that my history is not lost when I close the browser.

## Implementation Decisions

- **One pipeline core** (`rag_core`) with a single public entry point `answer(user_message, session) -> AnswerResult`. The AnswerResult carries the answer text, its Citations, and the ordered Sources used. Everything below this entry point is internal detail. The UI depends only on this entry point.
- **Pipeline order** (fixed): Query rewrite → hybrid retrieval → RRF fusion → re-ranking → CRAG judge → generation → answer check.
- **Query rewrite**: an LLM pass (gpt-4o-mini) rewrites a follow-up into a standalone question using the last 6 turns of the Session. First messages pass through unchanged.
- **Hybrid search**: BM25 (lexical) top-20 and dense vector top-20, fused with Reciprocal Rank Fusion at k=60. Dense embeddings come from the OpenRouter model `nvidia/nemotron-3-embed-1b:free` (2048-dim, multilingual incl. Vietnamese). All index chunks are embedded in a single batched request to respect the free-tier rate limit; queries are embedded one per request.
- **Index**: FAISS (in-memory, cosine) for dense vectors + `rank-bm25` for lexical. Built once at startup from the synthetic data; rebuilt by the ingest step.
- **Re-ranking**: `BAAI/bge-reranker-v2-m3` run locally on GPU. Takes the RRF-fused top-20... top-10 → final top-5 Sources for generation. (Chosen numbers: each branch top-20, RRF k=60, rerank top-10, return top-5.)
- **Chunking**: 400–600 tokens, ~15% overlap, preferred split on headings so chunks map to chapter/section. Every chunk records its Source (document id + chapter).
- **CRAG judge**: an LLM pass that scores the retrieved Sources high / medium / low. High → use as-is. Medium/low → one **Refine**: rewrite the query (exact keywords, drop pronouns) and re-run retrieval + re-ranking once. If still not high, refuse. There is no web-search fallback in this demo.
- **Generation**: gpt-4o-mini, given the top-5 Sources and the (rewritten) question, instructed to answer only from the Sources and to reference each Citation.
- **Answer check**: a Self-RAG-style LLM verification that every claim in the draft answer is supported by the cited Sources. On failure, regenerate once with concrete feedback (which claim is unsupported). If it fails again, refuse with a suggestion to rephrase.
- **Refusal policy**: never answer without a Citation. Refusal offers a rephrase suggestion.
- **Conversational memory**: per-User, last 6 turns, persisted in SQLite. A Session belongs to a User.
- **Auth**: username/password register + login in Streamlit. Passwords stored in plaintext by explicit choice — demo only, no real security posture.
- **Database**: SQLite with tables for users, conversations, and messages.
- **Data**: `generate_data.py` produces ~50 courses (with a prerequisite DAG guaranteed acyclic) and ~40 study documents (slides/textbook chapters), mixed Vietnamese and English. Courses expose: code, name, credits, prerequisites, semester, department, instructor, description.
- **UI**: Streamlit only (no separate backend). Login/register, chat, and Citation expanders per message.
- **Config**: OpenRouter API key from environment (`.env`). Model names configurable.

## Testing Decisions

- **Single test seam**: the `rag_core.answer()` entry point. Tests assert external behavior only — the answer, its Citations, and its Sources — never the internals of retrieval, re-ranking, or the judge. Layer demo scripts (not tests) exist for the report but are not part of the automated suite.
- **Good test**: given a seeded synthetic dataset and a query, the answer (a) contains the expected fact from the ground-truth Source, (b) cites that Source, and (c) behaves per policy on edge cases (no relevant material → refusal; ambiguous query → one refine then a grounded answer or refusal; follow-up question → rewritten standalone).
- **Modules tested**: the `rag_core` package exclusively, through `answer()`. No UI tests, no database tests beyond what the pipeline touches through a Session.
- **Prior art**: none — greenfield repo; tests establish the convention for future features (one high-level seam, behavioral assertions, seeded synthetic fixtures).
- **Evaluation script** (`eval.py`, separate from the test suite): computes retrieval hit rate (was the ground-truth Source in the top-5) and citation precision (how many Citations were actually used vs. returned) over a hand-made bilingual set of ~20 question/ground-truth pairs.

## Out of Scope

- Real institutional data — everything is synthetic.
- Web-search fallback for CRAG — refusal is the only low-confidence exit.
- Long-term semantic memory — only the last 6 turns of the current Session.
- Fine-tuned Self-RAG/CRAG models — both are prompt-based.
- Production hardening — no rate-limit retry backoff, no auth security, no multi-user concurrency.
- Full RAGAS / benchmark evaluation — a small manual eval set suffices for the report.

## Further Notes

- Build order: `generate_data.py` → ingest + retriever (BM25+dense+RRF) → rerank → crag (judge+refine) → generate + answer check → memory + db + auth → UI → eval. Each step has its own runnable demo script so the report can show the effect of every layer.
- The embedding model (`nvidia/nemotron-3-embed-1b:free`, 2048-dim) and the main model (gpt-4o-mini) both go through the OpenAI-compatible OpenRouter endpoint; the reranker is the only local-GPU component.
- Domain glossary lives in `CONTEXT.md` at the repo root.