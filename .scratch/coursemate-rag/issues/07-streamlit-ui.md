# 07: Streamlit UI

**What to build:** The user-facing Streamlit app: login and register screens, a chat view that sends messages through `answer()` with the logged-in user's Session, and per-answer Citation expanders that are clickable and show document + chapter. This is the full end-to-end demo a student or demonstrator can run.

**Blocked by:** 06 (memory, auth, and database)

**Status:** ready-for-agent

- [ ] Login and register screens work and gate access to the chat
- [ ] Chat sends messages through `answer()` and renders replies with the user's session history
- [ ] Each answer renders its Citations as clickable expanders showing document + chapter
- [ ] Refusals render distinctly (no fabricated answer, rephrase suggestion visible)
- [ ] A logged-out user's history is restored on next login