"""REST endpoints over rag_core."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.params import Form
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_core import StreamDelta, StreamDone, StreamRefused, StreamStart
from rag_core.models import AnswerResult, ConversationMeta, Turn, User
from server.schemas import (
    AttachmentContentOut,
    AttachmentOut,
    AttachmentSectionOut,
    ChatIn,
    ConversationCreateIn,
    ConversationOut,
    ConversationRenameIn,
    EntitiesOut,
    EntityOut,
    FeaturesOut,
    ProfileIn,
    RegisterIn,
    TokenOut,
    TurnOut,
    UserOut,
    turn_out,
)
from server.state import AppState, get_state

router = APIRouter()
_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    state: Annotated[AppState, Depends(get_state)],
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> User:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    user_id = state.tokens.user_id_for(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
    user = state.db.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Unknown user.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
StateDep = Annotated[AppState, Depends(get_state)]


@router.post("/auth/register", response_model=TokenOut, status_code=201)
def register(body: RegisterIn, state: StateDep) -> TokenOut:
    username = body.username.strip()
    if not (1 <= len(username) <= 50):
        raise HTTPException(status_code=400, detail="Username must be 1..50 chars.")
    if not body.password.strip():
        raise HTTPException(status_code=400, detail="Password must not be empty.")
    try:
        user = state.db.register(username, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return TokenOut(token=state.tokens.issue(user.id), user=UserOut.model_validate(user))


@router.post("/auth/login", response_model=TokenOut)
def login(body: RegisterIn, state: StateDep) -> TokenOut:
    user = state.db.login(body.username.strip(), body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    return TokenOut(token=state.tokens.issue(user.id), user=UserOut.model_validate(user))


@router.get("/auth/me", response_model=UserOut)
def me(current_user: CurrentUser) -> UserOut:
    return UserOut.model_validate(current_user)


def _require_conversation(state: AppState, user: User, conversation_id: str) -> None:
    try:
        session = state.db.get_session(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found.") from None
    if session.user_id != str(user.id):
        raise HTTPException(status_code=404, detail="Conversation not found.")


@router.patch("/users/me", response_model=UserOut)
def update_profile(body: ProfileIn, current_user: CurrentUser, state: StateDep) -> UserOut:
    try:
        updated = state.db.update_user_profile(
            current_user.id, display_name=body.display_name, language=body.language
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return UserOut.model_validate(updated)


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(current_user: CurrentUser, state: StateDep) -> list[ConversationOut]:
    return [
        ConversationOut.model_validate(meta)
        for meta in state.db.list_conversations(current_user.id)
    ]


@router.post("/conversations", response_model=ConversationOut, status_code=201)
def create_conversation(
    body: ConversationCreateIn, current_user: CurrentUser, state: StateDep
) -> ConversationOut:
    title = body.title.strip()
    if not (1 <= len(title) <= 50):
        raise HTTPException(status_code=400, detail="Title must be 1..50 chars.")
    session = state.db.create_conversation(current_user.id, title=title)
    metas: list[ConversationMeta] = state.db.list_conversations(current_user.id)
    meta = next(m for m in metas if m.id == session.id)
    return ConversationOut.model_validate(meta)


@router.patch("/conversations/{conversation_id}", response_model=ConversationOut)
def rename_conversation(
    conversation_id: str, body: ConversationRenameIn, current_user: CurrentUser, state: StateDep
) -> ConversationOut:
    _require_conversation(state, current_user, conversation_id)
    try:
        state.db.update_conversation_title(conversation_id, body.title)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    metas: list[ConversationMeta] = state.db.list_conversations(current_user.id)
    meta = next(m for m in metas if m.id == conversation_id)
    return ConversationOut.model_validate(meta)


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str, current_user: CurrentUser, state: StateDep) -> None:
    _require_conversation(state, current_user, conversation_id)
    try:
        state.db.delete_conversation(conversation_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Conversation not found.") from None


@router.delete("/conversations", status_code=204)
def clear_conversations(current_user: CurrentUser, state: StateDep) -> None:
    state.db.clear_all_conversations(current_user.id)


@router.get("/conversations/{conversation_id}/messages", response_model=list[TurnOut])
def list_messages(
    conversation_id: str, current_user: CurrentUser, state: StateDep
) -> list[TurnOut]:
    _require_conversation(state, current_user, conversation_id)
    session = state.db.get_session(conversation_id)
    return [turn_out(turn) for turn in session.turns]


@router.post("/conversations/{conversation_id}/chat", response_model=TurnOut)
def chat(
    conversation_id: str, body: ChatIn, current_user: CurrentUser, state: StateDep
) -> TurnOut:
    _require_conversation(state, current_user, conversation_id)
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")
    session = state.db.get_session(conversation_id)
    result = state.core.answer(message, session, use_web=body.use_web)
    state.db.append_exchange(
        conversation_id,
        message,
        result.answer,
        citations=result.citations,
        refused=result.refused,
        rephrase_suggestion=result.rephrase_suggestion,
        skills_applied=result.skills_applied,
    )
    return turn_out(
        Turn(
            role="assistant",
            text=result.answer,
            citations=result.citations,
            refused=result.refused,
            rephrase_suggestion=result.rephrase_suggestion,
            skills_applied=result.skills_applied,
        )
    )


def _ndjson_line(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


@router.post("/conversations/{conversation_id}/chat/stream")
def chat_stream(
    conversation_id: str, body: ChatIn, current_user: CurrentUser, state: StateDep
) -> StreamingResponse:
    """NDJSON streaming variant of /chat.

    Emits one JSON object per line. Schema:
      {"event": "start", "skills_applied": [...], "sources": [...]}
      {"event": "token",  "delta": "..."}              # repeated per delta
      {"event": "done",   "answer": "...", "citations": [...],
       "sources": [...], "refused": bool,
       "rephrase_suggestion": "...", "skills_applied": [...]}
      {"event": "refused","rephrase_suggestion": "..."}
      {"event": "error",  "detail": "..."}

    The synchronous pre-stream pipeline (rewrite / classify / retrieve /
    judge / refine / answer-check) runs server-side before the first line
    ships, so the client only sees tokens from the final accepted answer.
    """
    _require_conversation(state, current_user, conversation_id)
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message must not be empty.")
    session = state.db.get_session(conversation_id)

    def stream() -> Iterator[bytes]:
        skills_applied: list[str] = []
        sources_payload: list[dict[str, object]] = []
        accepted_result: AnswerResult | None = None
        try:
            for event in state.core.stream_answer(message, session, use_web=body.use_web):
                if isinstance(event, StreamStart):
                    skills_applied = list(event.skills_applied)
                    sources_payload = [
                        {
                            "document_id": src.document_id,
                            "document_title": src.document_title,
                            "chapter": src.chapter,
                            "course_code": src.course_code,
                            "kind": src.kind,
                            "language": src.language,
                        }
                        for src in event.sources
                    ]
                    yield _ndjson_line(
                        {"event": "start", "skills_applied": skills_applied,
                         "sources": sources_payload}
                    )
                elif isinstance(event, StreamDelta):
                    yield _ndjson_line({"event": "token", "delta": event.delta})
                elif isinstance(event, StreamDone):
                    accepted_result = event.result
                    yield _ndjson_line(
                        {
                            "event": "done",
                            "answer": event.result.answer,
                            "citations": [
                                {"marker": c.marker,
                                 "source": {
                                     "document_id": c.source.document_id,
                                     "document_title": c.source.document_title,
                                     "chapter": c.source.chapter,
                                     "course_code": c.source.course_code,
                                     "kind": c.source.kind,
                                     "language": c.source.language,
                                 }}
                                for c in event.result.citations
                            ],
                            "sources": sources_payload,
                            "refused": event.result.refused,
                            "rephrase_suggestion": event.result.rephrase_suggestion,
                            "skills_applied": skills_applied,
                        }
                    )
                elif isinstance(event, StreamRefused):
                    yield _ndjson_line(
                        {"event": "refused",
                         "rephrase_suggestion": event.rephrase_suggestion}
                    )
        except Exception as exc:  # noqa: BLE001 — translate pipeline error to client
            yield _ndjson_line({"event": "error", "detail": str(exc)})
            return
        if accepted_result is not None and not accepted_result.refused:
            state.db.append_exchange(
                conversation_id,
                message,
                accepted_result.answer,
                citations=accepted_result.citations,
                refused=accepted_result.refused,
                rephrase_suggestion=accepted_result.rephrase_suggestion,
                skills_applied=accepted_result.skills_applied,
            )

    return StreamingResponse(stream(), media_type="application/x-ndjson")



@router.get("/features", response_model=FeaturesOut)
def features(state: StateDep) -> FeaturesOut:
    return FeaturesOut(has_web_search=bool(state.core.has_web_search))


@router.get("/entities", response_model=EntitiesOut)
def entities(state: StateDep) -> EntitiesOut:
    """Return the full entity bundle so the UI can render citation cards."""
    bundle = state.core.entities
    return EntitiesOut(
        departments=[
            EntityOut(id=d.id, type="department", name=d.name, detail={"name_en": d.name_en})
            for d in bundle.departments
        ],
        instructors=[
            EntityOut(
                id=i.id,
                type="instructor",
                name=i.name,
                detail={
                    "title": i.title,
                    "email": i.email,
                    "department_id": i.department_id,
                    "courses": [{"course_code": c, "role": r} for c, r in i.courses],
                },
            )
            for i in bundle.instructors
        ],
        programs=[
            EntityOut(
                id=p.id,
                type="program",
                name=p.name,
                detail={
                    "name_en": p.name_en,
                    "department_id": p.department_id,
                    "total_credits": p.total_credits,
                    "required_courses": list(p.required_courses),
                    "elective_courses": list(p.elective_courses),
                },
            )
            for p in bundle.programs
        ],
        terms=[
            EntityOut(
                id=t.id,
                type="term",
                name=t.name,
                detail={
                    "year": t.year,
                    "season": t.season,
                    "start_date": t.start_date,
                    "end_date": t.end_date,
                    "offered": [
                        {"course_code": c, "instructor_id": i, "schedule": s}
                        for c, i, s in t.offered
                    ],
                },
            )
            for t in bundle.terms
        ],
    )

    with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
        row = conn.execute(
            "SELECT conversation_id FROM attachments WHERE id = ?", (attachment_id,)
        ).fetchone()
    return str(row[0]) if row is not None else None


@router.get("/attachments", response_model=list[AttachmentOut])
def list_attachments(
    conversation_id: str, current_user: CurrentUser, state: StateDep
) -> list[AttachmentOut]:
    _require_conversation(state, current_user, conversation_id)
    return [
        AttachmentOut.model_validate(meta)
        for meta in state.attachments.list_for(conversation_id)
    ]


@router.post("/attachments", response_model=AttachmentOut, status_code=201)
async def upload_attachment(
    current_user: CurrentUser,
    state: StateDep,
    conversation_id: Annotated[str, Form()],
    file: UploadFile,
) -> AttachmentOut:
    _require_conversation(state, current_user, conversation_id)
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Missing file name.")
    data = await file.read()
    try:
        meta = state.attachments.add(
            conversation_id, file.filename, data, language=current_user.language
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return AttachmentOut.model_validate(meta)


@router.delete("/attachments/{attachment_id}", status_code=204)
def delete_attachment(attachment_id: str, current_user: CurrentUser, state: StateDep) -> None:
    conversation_id = _attachment_conversation(state.db_path, attachment_id)
    if conversation_id is None:
        raise HTTPException(status_code=404, detail="Attachment not found.")
    _require_conversation(state, current_user, conversation_id)
    try:
        state.attachments.delete(attachment_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Attachment not found.") from None

@router.get("/attachments/{attachment_id}/content", response_model=AttachmentContentOut)
def get_attachment_content(
    attachment_id: str, current_user: CurrentUser, state: StateDep
) -> AttachmentContentOut:
    conversation_id = _attachment_conversation(state.db_path, attachment_id)
    if conversation_id is None:
        raise HTTPException(status_code=404, detail="Attachment not found.")
    _require_conversation(state, current_user, conversation_id)
    try:
        sections = state.attachments.get_content(attachment_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Attachment not found.") from None
    return AttachmentContentOut(
        sections=[AttachmentSectionOut(chapter=s.chapter, text=s.text) for s in sections]
    )


def _attachment_conversation(db_path: Path, attachment_id: str) -> str | None:
    with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
        row = conn.execute(
            "SELECT conversation_id FROM attachments WHERE id = ?", (attachment_id,)
        ).fetchone()
    return str(row[0]) if row is not None else None
