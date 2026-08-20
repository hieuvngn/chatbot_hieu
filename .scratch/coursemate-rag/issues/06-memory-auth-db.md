# 06: Memory, auth, and database

**What to build:** Per-user persistence and conversational context: a SQLite database with users, conversations, and messages; register and login by username/password (passwords stored in plaintext by explicit demo choice); a Session per user holding the last 6 turns; and an LLM query-rewrite pass that turns follow-up questions into standalone ones using the session history before retrieval. A user can log in, ask follow-ups ("môn đó…"), and come back later to a resumed history.

**Blocked by:** 05 (answer check and regenerate)

**Status:** ready-for-agent

- [ ] Registering a new user and logging in works against the SQLite database
- [ ] A Session persists the last 6 turns per user and resumes across visits
- [ ] A follow-up question is rewritten into a standalone question using session history before it enters retrieval
- [ ] `answer()` accepts a Session and the rewritten query flows through the full pipeline