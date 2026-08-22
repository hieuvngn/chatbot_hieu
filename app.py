"""Streamlit UI for CourseMate (ticket 07).

The user-facing app: login/register screens that gate access to the chat, a
chat view that sends messages through ``RagCore.answer()`` with the
logged-in user's Session, and per-answer Citation expanders that show the
document and chapter behind every source. Refusals render distinctly with
their rephrase suggestion, and a logged-out user's history — including
Citations and refusals — is restored on the next login from SQLite.

Run from the repo root:
    uv run streamlit run app.py

Requires OPENROUTER_API_KEY in .env (see README.md / docs/SETUP.md).
"""

from __future__ import annotations

import streamlit as st

from rag_core import RagCore, build_rag_core
from rag_core.attachments import MAX_FILES_PER_CONVERSATION, AttachmentStore
from rag_core.config import load_config
from rag_core.db import APP_DB_FILENAME, Database
from rag_core.models import AnswerResult, Citation, Session, Turn, User


@st.cache_resource
def get_database() -> Database:
    return Database(load_config().data_dir / APP_DB_FILENAME)


@st.cache_resource
def get_core() -> RagCore:
    return build_rag_core()


@st.cache_resource
def get_attachment_store() -> AttachmentStore:
    config = load_config()
    return AttachmentStore(config.data_dir / APP_DB_FILENAME, get_core().embedder)


def get_user() -> User | None:
    return st.session_state.get("user")


def get_active_conversation_id(user: User, db: Database) -> str:
    active = st.session_state.get("active_conversation_id")
    if isinstance(active, str) and active:
        try:
            db.get_session(active)
            return active
        except KeyError:
            pass
    metas = db.list_conversations(user.id)
    if metas:
        st.session_state["active_conversation_id"] = metas[0].id
        return metas[0].id
    session = db.create_conversation(user.id, title="New chat")
    st.session_state["active_conversation_id"] = session.id
    return session.id


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
            origin = "tài liệu đính kèm" if source.kind == "upload" else "kho tài liệu môn học"
            st.write(f"Nguồn: {origin}")
            st.write(f"Course: {source.course_code}")
            st.write(f"Kind: {source.kind} ({source.language})")


def _is_fallback(text: str) -> bool:
    return text.startswith("Lưu ý:") or text.startswith("Note:")


def render_turn(turn: Turn) -> None:
    with st.chat_message(turn.role):
        if turn.role == "assistant" and turn.refused:
            st.warning("I could not find enough supporting material to answer.")
            if turn.rephrase_suggestion:
                st.caption(f"Rephrase suggestion: {turn.rephrase_suggestion}")
            return
        if turn.role == "assistant" and _is_fallback(turn.text) and not turn.citations:
            st.info("No relevant material was found in the course corpus — answer based on general knowledge (no citations).")
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


_KIND_ICONS = {"pdf": "📄", "md": "📝", "txt": "🗒️"}


def render_attachments() -> None:
    session_id = st.session_state.get("active_conversation_id")
    if not isinstance(session_id, str) or not session_id:
        return
    store = get_attachment_store()
    metas = store.list_for(session_id)
    label = f"📎 Tài liệu đính kèm ({len(metas)}/{MAX_FILES_PER_CONVERSATION})"
    with st.sidebar.expander(label, expanded=bool(metas)):
        for meta in metas:
            icon = _KIND_ICONS.get(meta.file_kind, "📄")
            cols = st.columns([4, 1])
            cols[0].caption(
                f"{icon} {meta.filename}\n\n{meta.chunk_count} đoạn · "
                f"{max(1, meta.size_bytes // 1024)} KB"
            )
            with cols[1].popover("🗑️", use_container_width=True):
                st.write(f"Xóa {meta.filename}?")
                if st.button("Xóa", key=f"del_att_{meta.id}", type="primary"):
                    try:
                        store.delete(meta.id)
                    except KeyError:
                        pass
                    st.rerun()
        if len(metas) >= MAX_FILES_PER_CONVERSATION:
            st.caption(
                f"Đã đạt giới hạn {MAX_FILES_PER_CONVERSATION} tài liệu cho chat này."
            )
            return
        uploaded = st.file_uploader(
            "Thêm tài liệu (PDF/TXT/MD)", type=["pdf", "txt", "md"], key="att_uploader"
        )
        if uploaded is not None:
            try:
                user = st.session_state.get("user")
                language = user.language if user is not None else "vi"
                with st.spinner("Đang parse và embedding…"):
                    meta = store.add(session_id, uploaded.name, uploaded.getvalue(), language)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.toast(f"Đã xử lý {meta.filename}: {meta.chunk_count} đoạn.")
                st.session_state.pop("att_uploader", None)
                st.rerun()


def render_sidebar(user: User) -> None:
    db = get_database()
    with st.sidebar:
        st.title("CourseMate")
        if st.button("＋ Tạo chat mới", use_container_width=True, key="new_chat"):
            sess = db.create_conversation(user.id, title="New chat")
            st.session_state["active_conversation_id"] = sess.id
            st.session_state["show_profile"] = False
            st.rerun()
        st.divider()
        st.caption("Lịch sử chat")
        metas = db.list_conversations(user.id)
        for meta in metas:
            is_active = meta.id == st.session_state.get("active_conversation_id")
            cols = st.columns([4, 1])
            with cols[0]:
                label = f"{'▶ ' if is_active else ''}{meta.title[:35]}"
                if st.button(label, key=f"chat_{meta.id}", use_container_width=True):
                    st.session_state["active_conversation_id"] = meta.id
                    st.session_state["show_profile"] = False
                    st.rerun()
            with cols[1]:
                with st.popover("⋯", use_container_width=True):
                    new_title = st.text_input("Đổi tên", value=meta.title, key=f"rename_{meta.id}")
                    if st.button("Lưu", key=f"save_{meta.id}"):
                        try:
                            get_database().update_conversation_title(meta.id, new_title)
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))
                    if st.button("Xóa", key=f"del_{meta.id}", type="primary"):
                        try:
                            get_database().delete_conversation(meta.id)
                            if st.session_state.get("active_conversation_id") == meta.id:
                                st.session_state.pop("active_conversation_id", None)
                            st.rerun()
                        except KeyError as e:
                            st.error(str(e))
            st.caption(f"{meta.updated_at[:16]}  {meta.preview[:30]}", help=meta.preview)
        st.divider()
        render_attachments()
        st.divider()
        display = user.display_name or user.username
        st.caption(f"👤 {display} ({user.username})")
        if st.button("Profile / Setting", use_container_width=True, key="open_profile"):
            st.session_state["show_profile"] = True
            st.rerun()
        if st.button("Log out", key="logout_sidebar"):
            del st.session_state["user"]
            st.session_state.pop("active_conversation_id", None)
            st.session_state.pop("show_profile", None)
            st.rerun()


def render_profile(user: User) -> None:
    st.title("Profile / Setting")
    db = get_database()
    with st.form("profile_form"):
        display_name = st.text_input("Tên hiển thị", value=user.display_name, max_chars=50)
        language = st.selectbox("Ngôn ngữ / Language", options=["vi", "en"], index=0 if user.language == "vi" else 1, help="Chọn ngôn ngữ UI và câu trả lời mặc định")
        submitted = st.form_submit_button("Lưu thay đổi")
    if submitted:
        try:
            updated = db.update_user_profile(user.id, display_name=display_name, language=language)
            st.session_state["user"] = updated
            st.success("Đã lưu profile.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    st.divider()
    st.subheader("Setting")
    st.caption(f"Username: {user.username} (không đổi)")
    st.caption(f"Language hiện tại: {user.language}")
    st.divider()
    st.subheader("Xóa lịch sử")
    st.warning("Xóa tất cả conversations và messages của bạn. Không thể khôi phục.")
    confirm = st.text_input("Gõ DELETE để xác nhận", key="confirm_clear")
    if st.button("Xóa tất cả lịch sử", type="primary", disabled=confirm != "DELETE"):
        db.clear_all_conversations(user.id)
        # tạo 1 chat trống để UI không rỗng
        new = db.create_conversation(user.id, title="New chat")
        st.session_state["active_conversation_id"] = new.id
        st.success("Đã xóa tất cả lịch sử.")
        st.rerun()
    st.divider()
    if st.button("← Quay lại chat"):
        st.session_state["show_profile"] = False
        st.rerun()


def render_chat(user: User) -> None:
    db = get_database()
    session_id = get_active_conversation_id(user, db)
    session = db.get_session(session_id)
    st.title("CourseMate")
    for turn in session.turns:
        render_turn(turn)
    prompt = st.chat_input("Ask about courses or study material (e.g. 'giải thích bảng băm là gì?')")
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner("Retrieving material and composing the answer..."):
            result = get_core().answer(prompt, session)
        db.append_exchange(session_id, prompt, result.answer, citations=result.citations, refused=result.refused, rephrase_suggestion=result.rephrase_suggestion)
        st.rerun()


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
    db = get_database()
    refreshed = db.get_user(user.id)
    if refreshed:
        st.session_state["user"] = refreshed
        user = refreshed
    # Resolve the active conversation BEFORE rendering the sidebar so that
    # render_attachments() does not silently skip its first-render pass.
    get_active_conversation_id(user, db)
    render_sidebar(user)
    if st.session_state.get("show_profile"):
        render_profile(user)
    else:
        render_chat(user)


if __name__ == "__main__":
    main()
