# 07: Streamlit UI

**What to build:** The user-facing Streamlit app: login and register screens, a chat view that sends messages through `answer()` with the logged-in user's Session, and per-answer Citation expanders that are clickable and show document + chapter. This is the full end-to-end demo a student or demonstrator can run.

**Blocked by:** 06 (memory, auth, and database)

**Status:** done

- [x] Login and register screens work and gate access to the chat
- [x] Chat sends messages through `answer()` and renders replies with the user's session history
- [x] Each answer renders its Citations as clickable expanders showing document + chapter
- [x] Refusals render distinctly (no fabricated answer, rephrase suggestion visible)
- [x] A logged-out user's history is restored on next login

## Comments

- 2026-08-20: Implemented in `cfa0d4b`. `app.py` — the Streamlit app: login/register tabs (`st.tabs` + forms, non-empty credential validation) gate the chat via `st.stop()`; the chat renders the user's Session history (`db.get_or_create_session`) through `RagCore.answer()` with the logged-in user's Session; each answer renders its Citations as clickable `st.expander`s showing document title + chapter (with document id, course, kind inside); refusals render as a distinct warning with the rephrase suggestion; logging out and back in restores the full history. To make restored history render identically to live answers, `rag_core/db.py` and `rag_core/models.py` gained assistant-turn metadata: `Turn` now carries `citations`, `refused` and `rephrase_suggestion`; the `messages` table has three new columns (with an in-place `ALTER TABLE` migration for ticket-06-schema databases); `append_exchange` persists the `AnswerResult` metadata. Tests: 4 new `tests/test_db.py` cases (citation/refusal persistence, plain-turn default, legacy-schema migration), 65 tests pass, mypy strict clean (17 files), app smoke-tested headless (HTTP 200). README section added; `streamlit` added to dependencies; `app.py` added to the mypy file list. Code review findings addressed: live reply was double-wrapped in a nested `st.chat_message` bubble (removed the outer wrapper — `render_turn` owns the bubble), persist-before-render ordering so restored history always equals what was shown, empty credentials rejected in both forms, expander body now shows the chapter explicitly, misnamed `make_source` test helper renamed to `make_citation`, malformed citation rows raise `ValueError` (not `KeyError`), JSON persisted with `ensure_ascii=False`. Note: the spec's "no UI tests" is respected — the UI is verified by the headless smoke run; citation/refusal persistence is tested at the database seam, following the `tests/test_db.py` convention established by ticket 06.