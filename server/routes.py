"""REST endpoints over rag_core."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_core.models import ConversationMeta, User
from server.schemas import (
    ConversationCreateIn,
    ConversationOut,
    ConversationRenameIn,
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
    if credentials.credentials:
        user_id = state.tokens.user_id_for(credentials.credentials)
    else:
        user_id = None
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
