"""Streamlit UI for CourseMate (ticket 07).

The user-facing app: login/register screens that gate access to the chat, a
chat view that sends messages through ``RagCore.answer()`` with the
logged-in user's Session, and per-answer Citation expanders that show the
document and chapter behind every source. Refusals render distinctly with
their rephrase suggestion, and a logged-out user's history — including
Citations and refusals — is restored on the next login from SQLite.

Run from the repo root:
    uv run streamlit run app.py

Requires OPENROUTER_API_KEY in .env (see README.md) and the local-GPU
dependencies (torch, transformers) for the re-ranker.
"""

from __future__ import annotations

import streamlit as st

from rag_core import RagCore, build_rag_core
from rag_core.config import load_config
from rag_core.db import Database
from rag_core.models import AnswerResult, Citation, Session, Turn, User

APP_DB_FILENAME = "app.db"


@st.cache_resource
def get_database() -> Database:
    return Database(load_config().data_dir / APP_DB_FILENAME)


@st.cache_resource
def get_core() -> RagCore:
    return build_rag_core()


def get_user() -> User | None:
    return st.session_state.get("user")


def render_auth() -> None:
    st.title("CourseMate")
    st.caption("Course advisory and knowledge Q&A grounded in study documents.")
    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            if not username or not password:
                st.error("Username and password must not be empty.")
            else:
                user = get_database().login(username, password)
                if user is None:
                    st.error("Invalid username or password.")
                else:
                    st.session_state["user"] = user
                    st.rerun()
    with tab_register:
        with st.form("register_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Register")
        if submitted:
            if not username or not password:
                st.error("Username and password must not be empty.")
            else:
                try:
                    user = get_database().register(username, password)
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["user"] = user
                    st.rerun()


def render_citations(citations: list[Citation]) -> None:
    if not citations:
        return
    st.caption("Sources")
    for citation in citations:
        source = citation.source
        label = f"[{citation.marker}] {source.document_title} — {source.chapter}"
        with st.expander(label):
            st.write(f"Document: {source.document_title} ({source.document_id})")
            st.write(f"Chapter: {source.chapter}")
            st.write(f"Course: {source.course_code}")
            st.write(f"Kind: {source.kind} ({source.language})")


def render_turn(turn: Turn) -> None:
    with st.chat_message(turn.role):
        if turn.role == "assistant" and turn.refused:
            st.warning("I could not find enough supporting material to answer.")
            if turn.rephrase_suggestion:
                st.caption(f"Rephrase suggestion: {turn.rephrase_suggestion}")
            return
        st.write(turn.text)
        render_citations(turn.citations)


def result_to_turn(result: AnswerResult) -> Turn:
    return Turn(
        role="assistant",
        text=result.answer,
        citations=result.citations,
        refused=result.refused,
        rephrase_suggestion=result.rephrase_suggestion,
    )


def render_chat(user: User) -> None:
    st.title("CourseMate")
    st.caption(f"Logged in as {user.username}")
    if st.button("Log out"):
        del st.session_state["user"]
        st.rerun()

    db = get_database()
    session = db.get_or_create_session(user.id)

    for turn in session.turns:
        render_turn(turn)

    prompt = st.chat_input("Ask about courses or study material (e.g. 'giải thích bảng băm là gì?')")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner("Retrieving material and composing the answer..."):
            result = get_core().answer(prompt, session)
        db.append_exchange(
            session.id,
            prompt,
            result.answer,
            citations=result.citations,
            refused=result.refused,
            rephrase_suggestion=result.rephrase_suggestion,
        )
        render_turn(result_to_turn(result))


def main() -> None:
    st.set_page_config(page_title="CourseMate", layout="wide")
    try:
        load_config()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    user = get_user()
    if user is None:
        render_auth()
        st.stop()
    render_chat(user)


if __name__ == "__main__":
    main()