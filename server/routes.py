"""REST endpoints over rag_core."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_core.models import User
from server.schemas import RegisterIn, TokenOut, UserOut
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
