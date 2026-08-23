# React (FastAPI + Vite SPA) UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit UI with a FastAPI REST layer over the untouched `rag_core` package plus a React/Vite/Tailwind/shadcn SPA implementing the approved "Chat + panel nguồn" design.

**Architecture:** New `server/` package wraps `Database` + `RagCore.answer()` behind `/api/*` endpoints with Bearer-token auth (tokens table in the existing SQLite file). New `web/` directory holds the Vite React SPA; in production FastAPI serves the built SPA so `uv run serve` runs everything with one command.

**Tech Stack:** Python 3.11+/uv, FastAPI, Pydantic v2, SQLite (existing), pytest + httpx TestClient, mypy strict · Node 20+, Vite, React + TypeScript, Tailwind CSS v4, shadcn/ui, TanStack Query, react-router-dom, react-markdown.

**Spec:** `docs/superpowers/specs/2026-08-23-react-fastapi-ui-design.md`

## Global Constraints

- `rag_core/` package: **zero modifications** — the server reads it only.
- No streaming: `POST /conversations/{id}/chat` blocks and returns the complete assistant turn.
- Reuse the existing SQLite file `data/app.db`; the only schema addition is a `tokens` table via `CREATE TABLE IF NOT EXISTS` inside `server/auth.py`.
- Error mapping everywhere: domain `ValueError` → 400, `KeyError`/missing → 404, missing/invalid token → 401. Ownership violation on conversation/attachment → **404** (never 403).
- Auth: opaque token `secrets.token_hex(32)` stored in the `tokens` table; frontend keeps it in `localStorage["cm_token"]`, sends `Authorization: Bearer <token>`.
- UI copy is Vietnamese (parity with current app); answer language stays driven by `user.language` inside `rag_core`.
- Limits come from `rag_core` constants verbatim: `MAX_FILES_PER_CONVERSATION = 3`, `MAX_FILE_BYTES = 5 MB`, allowed extensions pdf/txt/md.
- Backend on `http://127.0.0.1:8000`; Vite dev server proxies `/api` to it. No CORS middleware needed (same-origin via proxy / static mount).
- Every task ends green: `uv run pytest` (backend tasks) or `npm run build` (frontend tasks); `uv run mypy` strict stays green from Task 1 onward.
- `docs/superpowers/` is gitignored by policy — commit plan/spec files with `git add -f`.
- Commit messages follow repo convention: `feat(server): …`, `feat(web): …`, `chore: …`, `docs: …`.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `server/__init__.py` | Create | Package marker |
| `server/auth.py` | Create | `TokenStore`: tokens table migration + issue/lookup |
| `server/state.py` | Create | `CoreLike` protocol, `AppState`, `build_state()`, `get_state()` dependency |
| `server/schemas.py` | Create | Pydantic request/response models |
| `server/routes.py` | Create | All `/api` endpoints + ownership helpers |
| `server/main.py` | Create | `create_app()`, SPA static mount, `main()` entrypoint |
| `tests/test_server_auth.py` | Create | Register/login/me/token tests |
| `tests/test_server_api.py` | Create | Conversations/chat/features tests (+ shared fakes) |
| `tests/test_server_attachments.py` | Create | Attachment upload/list/delete + ownership tests |
| `pyproject.toml` | Modify | fastapi/uvicorn/python-multipart deps, httpx dev dep, `[project.scripts] serve`, mypy `files += server` |
| `web/…` | Create | Vite SPA (structure listed in Task 6) |
| `app.py` | Delete | Streamlit UI removed (final task) |
| `README.md`, `docs/SETUP.md` | Modify | Replace Streamlit instructions with new UI instructions |

---

### Task 1: Server skeleton + token auth

**Files:**
- Create: `server/__init__.py`, `server/auth.py`, `server/state.py`, `server/schemas.py`, `server/main.py`
- Create (partial): `server/routes.py` (auth endpoints only)
- Modify: `pyproject.toml`
- Test: `tests/test_server_auth.py`

**Interfaces:**
- Consumes: `Database.register/login/get_user`, `load_config`, `APP_DB_FILENAME`.
- Produces (used by every later server task):
  - `TokenStore(path)` with `.issue(user_id: int) -> str`, `.user_id_for(token: str) -> int | None`
  - `CoreLike(Protocol)`: attr `has_web_search: bool`, attr `embedder: Embedder`, method `answer(user_message: str, session: Session, use_web: bool = False) -> AnswerResult`
  - `AppState(db, tokens, core, attachments, db_path)`, `build_state(config=None) -> AppState`, `get_state(request) -> AppState` (lazy, cached on `app.state.coursemate`)
  - `create_app() -> FastAPI`, module-level `app`, `main()` entrypoint
  - Schemas: `UserOut`, `TokenOut(token, user)`, `RegisterIn(username, password)`
  - Routes: `POST /api/auth/register` → 201 `TokenOut`; `POST /api/auth/login` → `TokenOut`; `GET /api/auth/me` → `UserOut`

- [ ] **Step 1: Add dependencies**

Edit `pyproject.toml`: add `"fastapi>=0.115",` and `"uvicorn>=0.30",` to `dependencies`; add `"httpx>=0.27"` to the `dev` extra; change mypy `files` to `["app.py", "generate_data.py", "eval.py", "rag_core", "tests", "server"]`.

Run: `uv sync`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_server_auth.py`:

```python
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rag_core.attachments import AttachmentStore
from rag_core.db import Database
from server.auth import TokenStore
from server.main import create_app
from server.state import AppState


class FakeEmbedder:
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


class FakeCore:
    def __init__(self) -> None:
        self.has_web_search = False
        self.embedder = FakeEmbedder()

    def answer(
        self, user_message: str, session: object, use_web: bool = False
    ) -> object:
        raise AssertionError("not used in auth tests")


def make_client(tmp_path: Path) -> tuple[TestClient, Database]:
    db_path = tmp_path / "app.db"
    db = Database(db_path)
    core = FakeCore()
    state = AppState(
        db=db,
        tokens=TokenStore(db_path),
        core=core,
        attachments=AttachmentStore(db_path, core.embedder),
        db_path=db_path,
    )
    app = create_app()
    app.state.coursemate = state
    return TestClient(app), db


def test_register_login_me(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    registered = client.post("/api/auth/register", json={"username": "hieu", "password": "pw"})
    assert registered.status_code == 201
    body = registered.json()
    assert body["token"]
    assert body["user"]["username"] == "hieu"

    logged_in = client.post("/api/auth/login", json={"username": "hieu", "password": "pw"})
    assert logged_in.status_code == 200
    token = logged_in.json()["token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "hieu"


def test_register_validation_and_duplicates(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    assert client.post("/api/auth/register", json={"username": "  ", "password": "pw"}).status_code == 400
    assert client.post("/api/auth/register", json={"username": "a", "password": " "}).status_code == 400
    client.post("/api/auth/register", json={"username": "a", "password": "pw"})
    dup = client.post("/api/auth/register", json={"username": "a", "password": "other"})
    assert dup.status_code == 400


def test_login_wrong_password(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    client.post("/api/auth/register", json={"username": "a", "password": "pw"})
    bad = client.post("/api/auth/login", json={"username": "a", "password": "nope"})
    assert bad.status_code == 401


def test_me_requires_token(tmp_path: Path) -> None:
    client, _ = make_client(tmp_path)
    assert client.get("/api/auth/me").status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer deadbeef"}).status_code == 401
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_server_auth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server'`.

- [ ] **Step 4: Implement the server modules**

Create empty `server/__init__.py`.

Create `server/auth.py`:

```python
"""Bearer-token sessions backed by a ``tokens`` table in the app database."""

from __future__ import annotations

import contextlib
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class TokenStore:
    """Issues opaque tokens and maps them back to user ids.

    Follows the same connection-per-operation pattern as ``rag_core.db.Database``
    so instances are safe to reuse across threads.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tokens (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def issue(self, user_id: int) -> str:
        token = secrets.token_hex(32)
        now = datetime.now(timezone.utc).isoformat()
        with contextlib.closing(self._connect()) as conn:
            conn.execute(
                "INSERT INTO tokens (token, user_id, created_at) VALUES (?, ?, ?)",
                (token, user_id, now),
            )
            conn.commit()
        return token

    def user_id_for(self, token: str) -> int | None:
        if not token:
            return None
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute("SELECT user_id FROM tokens WHERE token = ?", (token,)).fetchone()
        return int(row["user_id"]) if row is not None else None
```

Create `server/state.py`:

```python
"""Shared application state wired from rag_core, injectable for tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import Request

from rag_core import RagCore, build_rag_core
from rag_core.attachments import AttachmentStore
from rag_core.config import Config, load_config
from rag_core.db import APP_DB_FILENAME, Database
from rag_core.embeddings import Embedder
from rag_core.models import AnswerResult, Session
from server.auth import TokenStore


class CoreLike(Protocol):
    """Structural subset of ``RagCore`` the API relies on."""

    has_web_search: bool
    embedder: Embedder

    def answer(
        self, user_message: str, session: Session, use_web: bool = False
    ) -> AnswerResult: ...


@dataclass
class AppState:
    db: Database
    tokens: TokenStore
    core: CoreLike
    attachments: AttachmentStore
    db_path: Path


def build_state(config: Config | None = None) -> AppState:
    resolved = config if config is not None else load_config()
    db_path = resolved.data_dir / APP_DB_FILENAME
    core: CoreLike = build_rag_core()
    return AppState(
        db=Database(db_path),
        tokens=TokenStore(db_path),
        core=core,
        attachments=AttachmentStore(db_path, core.embedder),
        db_path=db_path,
    )


def get_state(request: Request) -> AppState:
    """FastAPI dependency: returns per-app state, building it lazily."""
    state = getattr(request.app.state, "coursemate", None)
    if state is None:
        state = build_state()
        request.app.state.coursemate = state
    return state
```

Note: `RagCore` satisfies `CoreLike` structurally (`has_web_search` is a read-only property, which satisfies a protocol attribute).

Create `server/schemas.py`:

```python
"""Pydantic request/response models mirroring rag_core domain objects."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    display_name: str
    language: str


class TokenOut(BaseModel):
    token: str
    user: UserOut


class RegisterIn(BaseModel):
    username: str
    password: str
```

Create `server/routes.py`:

```python
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
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    state: Annotated[AppState, Depends(get_state)] = None,
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
```

Create `server/main.py`:

```python
"""FastAPI application factory and entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from server.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="CourseMate API")
    app.include_router(router, prefix="/api")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_server_auth.py -v`
Expected: PASS (4 tests).

Run: `uv run mypy server tests/test_server_auth.py`
Expected: no issues.

- [ ] **Step 6: Commit**

```bash
git add server tests/test_server_auth.py pyproject.toml uv.lock
git commit -m "feat(server): FastAPI skeleton with bearer-token auth endpoints"
```

---

### Task 2: Users + conversations endpoints

**Files:**
- Modify: `server/schemas.py`, `server/routes.py`
- Test: `tests/test_server_api.py` (create)

**Interfaces:**
- Consumes: Task 1 `StateDep`, `CurrentUser`; `Database.update_user_profile/create_conversation/list_conversations/update_conversation_title/delete_conversation/clear_all_conversations/get_session`.
- Produces:
  - Schemas: `ProfileIn(display_name: str | None, language: str | None)`, `SourceOut`, `CitationOut(marker, source)`, `TurnOut(role, text, citations, refused, rephrase_suggestion, skills_applied)`, `ConversationOut(id, title, created_at, updated_at, preview)`, `ConversationCreateIn(title="New chat")`, `ConversationRenameIn(title)`
  - Helper functions: `citation_out(citation: Citation) -> CitationOut`, `turn_out(turn: Turn) -> TurnOut` (used again by Task 3)
  - `_require_conversation(state, user, conversation_id) -> None`: `KeyError` → 404, wrong owner → 404
  - Routes: `PATCH /api/users/me` → `UserOut`; `GET /api/conversations` → `list[ConversationOut]`; `POST /api/conversations` → 201; `PATCH /api/conversations/{id}`; `DELETE /api/conversations/{id}` → 204; `DELETE /api/conversations` → 204; `GET /api/conversations/{id}/messages` → `list[TurnOut]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_server_api.py` (self-contained fakes — later test modules import `FakeCore` and `make_client` from here):

```python
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rag_core.attachments import AttachmentStore
from rag_core.db import Database
from rag_core.models import AnswerResult, Citation, Session, Source
from server.auth import TokenStore
from server.main import create_app
from server.state import AppState


class FakeEmbedder:
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


class FakeCore:
    def __init__(self) -> None:
        self.has_web_search = False
        self.embedder = FakeEmbedder()
        self.calls: list[tuple[str, str, bool]] = []

    def answer(
        self, user_message: str, session: Session, use_web: bool = False
    ) -> AnswerResult:
        self.calls.append((user_message, session.id, use_web))
        source = Source(
            document_id="DOC-001",
            document_title="Slides Cấu trúc dữ liệu",
            chapter="Chương 3: Bảng băm",
            course_code="CS101",
            kind="slides",
            language="vi",
        )
        citation = Citation(marker="[1]", source=source)
        return AnswerResult(
            answer=f"echo: {user_message}", citations=[citation], sources=[source]
        )


def make_client(
    tmp_path: Path,
) -> tuple[TestClient, Database, FakeCore, dict[str, dict[str, str]]]:
    db_path = tmp_path / "app.db"
    db = Database(db_path)
    core = FakeCore()
    state = AppState(
        db=db,
        tokens=TokenStore(db_path),
        core=core,
        attachments=AttachmentStore(db_path, core.embedder),
        db_path=db_path,
    )
    app = create_app()
    app.state.coursemate = state
    client = TestClient(app)

    alice = client.post("/api/auth/register", json={"username": "alice", "password": "pw"}).json()
    bob = client.post("/api/auth/register", json={"username": "bob", "password": "pw"}).json()
    headers = {
        "alice": {"Authorization": f"Bearer {alice['token']}"},
        "bob": {"Authorization": f"Bearer {bob['token']}"},
    }
    return client, db, core, headers


def _create_conversation(client: TestClient, headers: dict[str, str]) -> dict:
    res = client.post("/api/conversations", json={}, headers=headers)
    assert res.status_code == 201
    return res.json()


def test_users_me_profile_update(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    patched = client.patch(
        "/api/users/me", json={"display_name": "Hieu", "language": "en"}, headers=h["alice"]
    )
    assert patched.status_code == 200
    assert patched.json()["display_name"] == "Hieu"
    assert patched.json()["language"] == "en"
    assert client.patch("/api/users/me", json={"language": "fr"}, headers=h["alice"]).status_code == 400


def test_conversation_crud(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = _create_conversation(client, h["alice"])
    assert conv["title"] == "New chat"

    listed = client.get("/api/conversations", headers=h["alice"]).json()
    assert [c["id"] for c in listed] == [conv["id"]]

    renamed = client.patch(
        f"/api/conversations/{conv['id']}", json={"title": "Bảng băm"}, headers=h["alice"]
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Bảng băm"

    assert client.patch(f"/api/conversations/{conv['id']}", json={"title": ""}, headers=h["alice"]).status_code == 400

    assert client.delete(f"/api/conversations/{conv['id']}", headers=h["alice"]).status_code == 204
    assert client.delete(f"/api/conversations/{conv['id']}", headers=h["alice"]).status_code == 404


def test_clear_all_conversations(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    _create_conversation(client, h["alice"])
    _create_conversation(client, h["alice"])
    assert client.delete("/api/conversations", headers=h["alice"]).status_code == 204
    assert client.get("/api/conversations", headers=h["alice"]).json() == []


def test_ownership_is_404(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = _create_conversation(client, h["alice"])
    assert (
        client.patch(f"/api/conversations/{conv['id']}", json={"title": "stolen"}, headers=h["bob"]).status_code
        == 404
    )
    assert client.delete(f"/api/conversations/{conv['id']}", headers=h["bob"]).status_code == 404
    assert client.get(f"/api/conversations/{conv['id']}/messages", headers=h["bob"]).status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server_api.py -v`
Expected: FAIL — 404 on every new route.

- [ ] **Step 3: Implement schemas and routes**

Append to `server/schemas.py` (extend the models import to `from rag_core.models import Citation, Turn`):

```python
class ProfileIn(BaseModel):
    display_name: str | None = None
    language: str | None = None


class SourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    document_title: str
    chapter: str
    course_code: str
    kind: str
    language: str


class CitationOut(BaseModel):
    marker: str
    source: SourceOut


class TurnOut(BaseModel):
    role: str
    text: str
    citations: list[CitationOut]
    refused: bool
    rephrase_suggestion: str
    skills_applied: list[str]


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: str
    updated_at: str
    preview: str


class ConversationCreateIn(BaseModel):
    title: str = "New chat"


class ConversationRenameIn(BaseModel):
    title: str


def citation_out(citation: Citation) -> CitationOut:
    return CitationOut(marker=citation.marker, source=SourceOut.model_validate(citation.source))


def turn_out(turn: Turn) -> TurnOut:
    return TurnOut(
        role=turn.role,
        text=turn.text,
        citations=[citation_out(c) for c in turn.citations],
        refused=turn.refused,
        rephrase_suggestion=turn.rephrase_suggestion,
        skills_applied=list(turn.skills_applied),
    )
```

Append to `server/routes.py` (imports: `from rag_core.models import ConversationMeta, Turn` and the new schemas):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server_api.py tests/test_server_auth.py -v`
Expected: PASS.

Run: `uv run mypy server tests/test_server_api.py`
Expected: no issues.

- [ ] **Step 5: Commit**

```bash
git add server tests/test_server_api.py
git commit -m "feat(server): profile and conversation CRUD endpoints with ownership checks"
```

---

### Task 3: Chat + features endpoints

**Files:**
- Modify: `server/schemas.py`, `server/routes.py`
- Test: `tests/test_server_api.py` (extend)

**Interfaces:**
- Consumes: Task 2 `_require_conversation`, `turn_out`; `CoreLike.answer`; `Database.append_exchange(session_id, user_text, assistant_text, citations=, refused=, rephrase_suggestion=, skills_applied=)`.
- Produces: Schemas `ChatIn(message: str, use_web: bool = False)`, `FeaturesOut(has_web_search: bool)`; Routes `POST /api/conversations/{id}/chat` → `TurnOut`, `GET /api/features` → `FeaturesOut`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_server_api.py`:

```python
def test_chat_round_trip_persists(tmp_path: Path) -> None:
    client, _, core, h = make_client(tmp_path)
    conv = _create_conversation(client, h["alice"])

    reply = client.post(
        f"/api/conversations/{conv['id']}/chat",
        json={"message": "giải thích bảng băm"},
        headers=h["alice"],
    )
    assert reply.status_code == 200
    turn = reply.json()
    assert turn["role"] == "assistant"
    assert turn["text"] == "echo: giải thích bảng băm"
    assert turn["citations"][0]["marker"] == "[1]"
    assert turn["citations"][0]["source"]["document_title"] == "Slides Cấu trúc dữ liệu"
    assert turn["refused"] is False
    assert core.calls == [("giải thích bảng băm", conv["id"], False)]

    messages = client.get(f"/api/conversations/{conv['id']}/messages", headers=h["alice"]).json()
    assert [t["role"] for t in messages] == ["user", "assistant"]
    assert messages[0]["text"] == "giải thích bảng băm"


def test_chat_passes_use_web_flag(tmp_path: Path) -> None:
    client, _, core, h = make_client(tmp_path)
    core.has_web_search = True
    conv = _create_conversation(client, h["alice"])
    client.post(
        f"/api/conversations/{conv['id']}/chat",
        json={"message": "xin chào", "use_web": True},
        headers=h["alice"],
    )
    assert core.calls[-1][2] is True


def test_chat_rejects_blank_message(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = _create_conversation(client, h["alice"])
    blank = client.post(
        f"/api/conversations/{conv['id']}/chat", json={"message": "   "}, headers=h["alice"]
    )
    assert blank.status_code == 400


def test_features_endpoint(tmp_path: Path) -> None:
    client, _, core, _ = make_client(tmp_path)
    assert client.get("/api/features").json() == {"has_web_search": False}
    core.has_web_search = True
    assert client.get("/api/features").json() == {"has_web_search": True}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server_api.py -v -k "chat or features"`
Expected: FAIL — routes don't exist.

- [ ] **Step 3: Implement**

Append to `server/schemas.py`:

```python
class ChatIn(BaseModel):
    message: str
    use_web: bool = False


class FeaturesOut(BaseModel):
    has_web_search: bool
```

Append to `server/routes.py` (import `ChatIn`, `FeaturesOut`, and `Turn` from schemas/models):

```python
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


@router.get("/features", response_model=FeaturesOut)
def features(state: StateDep) -> FeaturesOut:
    return FeaturesOut(has_web_search=bool(state.core.has_web_search))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server_api.py tests/test_server_auth.py -v`
Expected: PASS.

Run: `uv run mypy server`
Expected: no issues.

- [ ] **Step 5: Commit**

```bash
git add server tests/test_server_api.py
git commit -m "feat(server): chat and features endpoints"
```

---

### Task 4: Attachments endpoints

**Files:**
- Modify: `pyproject.toml` (python-multipart), `server/routes.py`
- Test: `tests/test_server_attachments.py` (create)

**Interfaces:**
- Consumes: Task 2 `_require_conversation`; `AttachmentStore.list_for(conversation_id)`, `.add(conversation_id, filename, data, language)` (raises `ValueError` on bad type/size/limit/duplicate), `.delete(attachment_id)` (raises `KeyError`); constant `MAX_FILES_PER_CONVERSATION`.
- Produces: Schema `AttachmentOut(id, conversation_id, filename, file_kind, size_bytes, chunk_count, created_at)`; module helper `_attachment_conversation(db_path, attachment_id) -> str | None` (read-only SQL against the documented `attachments` table); Routes `GET /api/attachments?conversation_id=`, `POST /api/attachments` (multipart `file` + `conversation_id`) → 201, `DELETE /api/attachments/{id}` → 204.

- [ ] **Step 1: Add python-multipart**

Add `"python-multipart>=0.0.9",` to `dependencies` in `pyproject.toml`. Run: `uv sync`.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_server_attachments.py`:

```python
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rag_core.attachments import MAX_FILES_PER_CONVERSATION
from tests.test_server_api import make_client


def _upload(client: TestClient, headers: dict[str, str], conv_id: str, name: str, content: bytes):
    return client.post(
        "/api/attachments",
        data={"conversation_id": conv_id},
        files={"file": (name, content)},
        headers=headers,
    )


def test_upload_list_delete(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=h["alice"]).json()

    uploaded = _upload(client, h["alice"], conv["id"], "notes.txt", "# Tieu de\nNoi dung")
    assert uploaded.status_code == 201
    meta = uploaded.json()
    assert meta["filename"] == "notes.txt"
    assert meta["chunk_count"] >= 1

    listed = client.get("/api/attachments", params={"conversation_id": conv["id"]}, headers=h["alice"]).json()
    assert [a["id"] for a in listed] == [meta["id"]]

    assert client.delete(f"/api/attachments/{meta['id']}", headers=h["alice"]).status_code == 204
    assert client.delete(f"/api/attachments/{meta['id']}", headers=h["alice"]).status_code == 404


def test_upload_limit_and_bad_type(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=h["alice"]).json()
    for i in range(MAX_FILES_PER_CONVERSATION):
        assert _upload(client, h["alice"], conv["id"], f"f{i}.txt", b"noi dung").status_code == 201
    assert _upload(client, h["alice"], conv["id"], "extra.txt", b"noi dung").status_code == 400
    assert _upload(client, h["alice"], conv["id"], "x.exe", b"MZ").status_code == 400


def test_upload_ownership(tmp_path: Path) -> None:
    client, _, _, h = make_client(tmp_path)
    conv = client.post("/api/conversations", json={}, headers=h["alice"]).json()
    stolen = _upload(client, h["bob"], conv["id"], "steal.txt", b"x")
    assert stolen.status_code == 404
    listed = client.get("/api/attachments", params={"conversation_id": conv["id"]}, headers=h["bob"])
    assert listed.status_code == 404
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_server_attachments.py -v`
Expected: FAIL — 404 on `/api/attachments`.

- [ ] **Step 4: Implement**

Append to `server/schemas.py`:

```python
class AttachmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    filename: str
    file_kind: str
    size_bytes: int
    chunk_count: int
    created_at: str
```

Append to `server/routes.py` — top-of-file imports gain:

```python
import contextlib
import sqlite3
from pathlib import Path

from fastapi import UploadFile
from fastapi.params import Form

from server.schemas import AttachmentOut
```

and the endpoints:

```python
def _attachment_conversation(db_path: Path, attachment_id: str) -> str | None:
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
```

- [ ] **Step 5: Run full suite + mypy**

Run: `uv run pytest && uv run mypy`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add server pyproject.toml uv.lock tests/test_server_attachments.py
git commit -m "feat(server): attachment upload/list/delete endpoints with ownership"
```

---

### Task 5: Static SPA serving + `uv run serve`

**Files:**
- Modify: `server/main.py`, `pyproject.toml`
- Test: manual smoke check (full end-to-end in final task)

**Interfaces:**
- Consumes: Tasks 1–4 routes.
- Produces: `uv run serve` starts the API on `http://127.0.0.1:8000`; if `web/dist/` exists it is served at `/` with SPA fallback to `index.html` for client-side routes.

- [ ] **Step 1: Implement SPA serving**

Replace `server/main.py` with:

```python
"""FastAPI application factory and entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from server.routes import router

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


class SPAStaticFiles(StaticFiles):
    """StaticFiles that falls back to index.html for client-side routes."""

    async def get_response(self, path: str, scope: dict[str, Any]) -> Any:
        response = await super().get_response(path, scope)
        if response.status_code == 404:
            return await super().get_response("index.html", scope)
        return response


def create_app() -> FastAPI:
    app = FastAPI(title="CourseMate API")
    app.include_router(router, prefix="/api")
    if WEB_DIST.is_dir():
        app.mount("/", SPAStaticFiles(directory=WEB_DIST, html=True), name="web")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
```

Add to `pyproject.toml`:

```toml
[project.scripts]
serve = "server.main:main"
```

Run: `uv sync` (registers the script).

- [ ] **Step 2: Verify**

Run: `uv run pytest && uv run mypy`
Expected: green (`web/dist` doesn't exist yet so the mount is skipped).

Smoke-test the API standalone:

```sh
uv run uvicorn server.main:app --port 8000 &
sleep 3
curl -s -X POST http://127.0.0.1:8000/api/auth/register \
  -H 'Content-Type: application/json' -d '{"username":"smoke","password":"pw"}'
kill %1
```

Expected: JSON with a token.

- [ ] **Step 3: Commit**

```bash
git add server/main.py pyproject.toml uv.lock
git commit -m "feat(server): serve built SPA with fallback and uv run serve script"
```

---

### Task 6: Web scaffold — Vite + Tailwind v4 + shadcn/ui + router + query + api client

**Files:**
- Create: `web/` project via Vite scaffold; then edit `web/vite.config.ts`, `web/tsconfig.app.json`, `web/src/index.css`, `web/src/lib/api.ts`, `web/src/lib/types.ts`, `web/src/lib/theme.ts`, `web/src/main.tsx`, `web/src/App.tsx`; stub pages `web/src/pages/LoginPage.tsx`, `ChatPage.tsx`, `SettingsPage.tsx`
- Modify: root `.gitignore`

**Interfaces:**
- Consumes: all `/api` endpoints from Tasks 1–5.
- Produces (used by Tasks 7–12):
  - `lib/types.ts`: interfaces `User`, `Source`, `Citation`, `Turn`, `ConversationMeta`, `Attachment`, `Features`
  - `lib/api.ts`: `getToken()`, `setToken(t | null)`, `class ApiError(status, message)`, `api.*` methods (exact signatures below)
  - `lib/theme.ts`: `useTheme(): { theme: "light"|"dark"|"system"; setTheme(t): void }`
  - Query keys used by later tasks: `["conversations"]`, `["me"]`, `["messages", id]`, `["attachments", convId]`, `["features"]`

- [ ] **Step 1: Scaffold**

Verify tooling first: `node --version` must print v20+. From the repo root:

```sh
npm create vite@latest web -- --template react-ts
cd web && npm install
npm install tailwindcss @tailwindcss/vite react-router-dom @tanstack/react-query react-markdown remark-gfm rehype-highlight lucide-react sonner @tailwindcss/typography
npx shadcn@latest init -y -d
npx shadcn@latest add -y button input textarea dialog alert-dialog dropdown-menu switch sheet
```

If a shadcn prompt still appears, accept defaults (New York style, zinc base, CSS variables).

- [ ] **Step 2: Configure Vite + TS + Tailwind**

Replace `web/vite.config.ts`:

```ts
import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
```

In `web/tsconfig.app.json` add under `compilerOptions`:

```json
"baseUrl": ".",
"paths": { "@/*": ["./src/*"] }
```

`web/src/index.css` must start with (keep whatever token block `shadcn init` wrote below it):

```css
@import "tailwindcss";
@plugin "@tailwindcss/typography";
@custom-variant dark (&:where(.dark, .dark *));

body {
  font-family: "Inter", ui-sans-serif, system-ui, sans-serif;
}
```

Delete scaffold boilerplate: `src/App.css`, default assets and their imports.

- [ ] **Step 3: Types + api client + theme**

Create `web/src/lib/types.ts`:

```ts
export interface User {
  id: number;
  username: string;
  display_name: string;
  language: string;
}

export interface Source {
  document_id: string;
  document_title: string;
  chapter: string;
  course_code: string;
  kind: string;
  language: string;
}

export interface Citation {
  marker: string;
  source: Source;
}

export interface Turn {
  role: string;
  text: string;
  citations: Citation[];
  refused: boolean;
  rephrase_suggestion: string;
  skills_applied: string[];
}

export interface ConversationMeta {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  preview: string;
}

export interface Attachment {
  id: string;
  conversation_id: string;
  filename: string;
  file_kind: string;
  size_bytes: number;
  chunk_count: number;
  created_at: string;
}

export interface Features {
  has_web_search: boolean;
}
```

Create `web/src/lib/api.ts`:

```ts
import type { Attachment, ConversationMeta, Features, Turn, User } from "./types";

const TOKEN_KEY = "cm_token";

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null): void {
  if (token === null) localStorage.removeItem(TOKEN_KEY);
  else localStorage.setItem(TOKEN_KEY, token);
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  if (!(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const token = getToken();
  if (token !== null) headers.set("Authorization", `Bearer ${token}`);
  const res = await fetch(`/api${path}`, { ...options, headers });
  if (!res.ok) {
    const body: unknown = await res.json().catch(() => ({ detail: res.statusText }));
    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText;
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

export interface AuthResponse {
  token: string;
  user: User;
}

export const api = {
  register: (username: string, password: string) =>
    request<AuthResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  login: (username: string, password: string) =>
    request<AuthResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  me: () => request<User>("/auth/me"),
  updateProfile: (patch: { display_name?: string; language?: string }) =>
    request<User>("/users/me", { method: "PATCH", body: JSON.stringify(patch) }),

  conversations: () => request<ConversationMeta[]>("/conversations"),
  createConversation: () =>
    request<ConversationMeta>("/conversations", { method: "POST", body: JSON.stringify({}) }),
  renameConversation: (id: string, title: string) =>
    request<ConversationMeta>(`/conversations/${id}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: "DELETE" }),
  clearConversations: () => request<void>("/conversations", { method: "DELETE" }),
  messages: (id: string) => request<Turn[]>(`/conversations/${id}/messages`),
  chat: (id: string, message: string, useWeb: boolean) =>
    request<Turn>(`/conversations/${id}/chat`, {
      method: "POST",
      body: JSON.stringify({ message, use_web: useWeb }),
    }),

  features: () => request<Features>("/features"),
  attachments: (conversationId: string) =>
    request<Attachment[]>(`/attachments?conversation_id=${encodeURIComponent(conversationId)}`),
  uploadAttachment: (conversationId: string, file: File) => {
    const fd = new FormData();
    fd.append("conversation_id", conversationId);
    fd.append("file", file);
    return request<Attachment>("/attachments", { method: "POST", body: fd });
  },
  deleteAttachment: (id: string) => request<void>(`/attachments/${id}`, { method: "DELETE" }),
};
```

Create `web/src/lib/theme.ts`:

```ts
import { useCallback, useEffect, useState } from "react";

export type Theme = "light" | "dark" | "system";
const KEY = "cm_theme";

function apply(theme: Theme): void {
  const dark =
    theme === "dark" ||
    (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
}

export function useTheme(): { theme: Theme; setTheme: (t: Theme) => void } {
  const [theme, setThemeState] = useState<Theme>(
    () => (localStorage.getItem(KEY) as Theme | null) ?? "system",
  );

  useEffect(() => {
    apply(theme);
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = () => {
      if ((localStorage.getItem(KEY) as Theme | null) === "system") apply("system");
    };
    media.addEventListener("change", listener);
    return () => media.removeEventListener("change", listener);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => {
    localStorage.setItem(KEY, next);
    setThemeState(next);
  }, []);

  return { theme, setTheme };
}
```

- [ ] **Step 4: App shell + stub pages**

Replace `web/src/main.tsx`:

```tsx
import { MutationCache, QueryCache, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { Toaster } from "sonner";
import App from "./App";
import { ApiError, setToken } from "./lib/api";
import "./index.css";
import "highlight.js/styles/github-dark-dimmed.css";

const on401 = (error: unknown): void => {
  if (error instanceof ApiError && error.status === 401) {
    setToken(null);
    window.location.assign("/login");
  }
};

const queryClient = new QueryClient({
  queryCache: new QueryCache({ onError: on401 }),
  mutationCache: new MutationCache({ onError: on401 }),
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
      <Toaster position="top-center" richColors />
    </QueryClientProvider>
  </StrictMode>,
);
```

Replace `web/src/App.tsx`:

```tsx
import { Navigate, Route, Routes } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import LoginPage from "./pages/LoginPage";
import SettingsPage from "./pages/SettingsPage";

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/" element={<ChatPage />} />
      <Route path="/c/:conversationId" element={<ChatPage />} />
      <Route path="/settings" element={<SettingsPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
```

Stub pages (replaced by later tasks):

```tsx
// web/src/pages/LoginPage.tsx
export default function LoginPage() {
  return <div className="flex h-screen items-center justify-center">Login</div>;
}

// web/src/pages/ChatPage.tsx
export default function ChatPage() {
  return <div className="p-8">Chat</div>;
}

// web/src/pages/SettingsPage.tsx
export default function SettingsPage() {
  return <div className="p-8">Settings</div>;
}
```

- [ ] **Step 5: Ignore build artifacts + verify build**

Add to root `.gitignore`:

```
web/node_modules/
web/dist/
```

Run (in `web/`): `npm run build`
Expected: `tsc -b && vite build` succeeds.

Manual: start backend (`uv run uvicorn server.main:app --reload`) and frontend (`npm run dev`); opening `http://localhost:5173` shows the Chat stub without console/network errors.

- [ ] **Step 6: Commit**

```bash
git add web .gitignore
git commit -m "feat(web): Vite+Tailwind+shadcn scaffold with api client, router, theme"
```

---

### Task 7: Login/Register page + auth guard

**Files:**
- Create: `web/src/components/RequireAuth.tsx`
- Modify: `web/src/App.tsx`, `web/src/pages/LoginPage.tsx`

**Interfaces:**
- Consumes: `api.login/register`, `setToken`, `getToken`.
- Produces: `<RequireAuth>` wrapper route element (redirects to `/login` when no token); functional `LoginPage` (login ↔ register toggle, inline errors, redirect to `/` on success).

- [ ] **Step 1: RequireAuth + routes**

Create `web/src/components/RequireAuth.tsx`:

```tsx
import { Navigate, Outlet } from "react-router-dom";
import { getToken } from "@/lib/api";

export default function RequireAuth() {
  if (getToken() === null) return <Navigate to="/login" replace />;
  return <Outlet />;
}
```

Update `App.tsx` — protected routes move inside `<RequireAuth/>` (`/login` stays outside):

```tsx
<Route element={<RequireAuth />}>
  <Route path="/" element={<ChatPage />} />
  <Route path="/c/:conversationId" element={<ChatPage />} />
  <Route path="/settings" element={<SettingsPage />} />
</Route>
```

- [ ] **Step 2: LoginPage**

Replace `web/src/pages/LoginPage.tsx`:

```tsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, setToken } from "@/lib/api";

export default function LoginPage() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setPending(true);
    try {
      const fn = mode === "login" ? api.login : api.register;
      const { token } = await fn(username.trim(), password);
      setToken(token);
      navigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Đã có lỗi xảy ra.");
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 px-4 dark:bg-zinc-950">
      <div className="w-full max-w-sm rounded-2xl border bg-white p-8 shadow-sm dark:bg-zinc-900">
        <h1 className="text-2xl font-semibold tracking-tight">CourseMate</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Tư vấn môn học và hỏi đáp trên tài liệu học tập.
        </p>

        <div className="mt-6 grid grid-cols-2 gap-1 rounded-lg bg-zinc-100 p-1 dark:bg-zinc-800">
          {(["login", "register"] as const).map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => {
                setMode(tab);
                setError(null);
              }}
              className={`rounded-md py-1.5 text-sm font-medium transition ${
                mode === tab
                  ? "bg-white shadow-sm dark:bg-zinc-700"
                  : "text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
              }`}
            >
              {tab === "login" ? "Đăng nhập" : "Đăng ký"}
            </button>
          ))}
        </div>

        <form onSubmit={submit} className="mt-6 space-y-4">
          <Input placeholder="Tên đăng nhập" value={username} onChange={(e) => setUsername(e.target.value)} autoFocus />
          <Input type="password" placeholder="Mật khẩu" value={password} onChange={(e) => setPassword(e.target.value)} />
          {error !== null && <p className="text-sm text-red-600">{error}</p>}
          <Button type="submit" disabled={pending} className="w-full rounded-xl">
            {pending ? "Đang xử lý…" : mode === "login" ? "Đăng nhập" : "Đăng ký"}
          </Button>
        </form>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify**

Run: `npm run build` → passes.

Manual (backend running): register a fresh account lands on `/`; clearing localStorage then visiting `/` bounces to `/login`; wrong password shows the red error line; duplicate registration shows the server error message.

- [ ] **Step 4: Commit**

```bash
git add web/src
git commit -m "feat(web): login/register page and auth guard"
```

---

### Task 8: App layout — header + sidebar with history CRUD

**Files:**
- Create: `web/src/components/AppLayout.tsx`, `Sidebar.tsx`, `Header.tsx`, `ThemeToggle.tsx`
- Modify: `web/src/App.tsx`, `web/index.html`

**Interfaces:**
- Consumes: `api.conversations/createConversation/renameConversation/deleteConversation`, `useTheme`.
- Produces:
  - Route group `<Route element={<AppLayout />}>` inside `<RequireAuth>` with index route `ChatRedirect` (`/` picks latest conversation or creates one → `/c/:id`)
  - `Sidebar({ onNavigated?: () => void })` — reused by the mobile Sheet in Header

- [ ] **Step 1: ThemeToggle**

Create `web/src/components/ThemeToggle.tsx`:

```tsx
import { Monitor, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useTheme } from "@/lib/theme";

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Đổi giao diện">
          {theme === "dark" ? <Moon size={18} /> : theme === "light" ? <Sun size={18} /> : <Monitor size={18} />}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={() => setTheme("light")}><Sun size={14} className="mr-2" />Sáng</DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("dark")}><Moon size={14} className="mr-2" />Tối</DropdownMenuItem>
        <DropdownMenuItem onClick={() => setTheme("system")}><Monitor size={14} className="mr-2" />Theo hệ thống</DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
```

- [ ] **Step 2: Sidebar**

Create `web/src/components/Sidebar.tsx`. Structure: "＋ Chat mới" button; scrollable history list (each row = open button + ⋯ dropdown with Đổi tên/Xóa; rename opens a Dialog with Input); footer link to `/settings`. Mutations invalidate `["conversations"]`; deleting the active conversation navigates to `/`.

```tsx
import { MoreHorizontal, Plus } from "lucide-react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";

export default function Sidebar({ onNavigated }: { onNavigated?: () => void }) {
  const navigate = useNavigate();
  const { conversationId } = useParams();
  const queryClient = useQueryClient();
  const { data: conversations = [] } = useQuery({
    queryKey: ["conversations"],
    queryFn: api.conversations,
  });

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["conversations"] });

  const createMutation = useMutation({
    mutationFn: api.createConversation,
    onSuccess: (created) => {
      void invalidate();
      onNavigated?.();
      navigate(`/c/${created.id}`);
    },
    onError: (error) => toast.error(String(error)),
  });
  const renameMutation = useMutation({
    mutationFn: ({ id, title }: { id: string; title: string }) => api.renameConversation(id, title),
    onSuccess: invalidate,
    onError: (error) => toast.error(String(error)),
  });
  const deleteMutation = useMutation({
    mutationFn: api.deleteConversation,
    onSuccess: (_data, deletedId) => {
      void invalidate();
      if (deletedId === conversationId) navigate("/");
    },
    onError: (error) => toast.error(String(error)),
  });

  return (
    <aside className="flex h-full w-full flex-col gap-2 p-3">
      <Button className="justify-start gap-2 rounded-xl" onClick={() => createMutation.mutate()} disabled={createMutation.isPending}>
        <Plus size={16} /> Chat mới
      </Button>

      <nav className="min-h-0 flex-1 space-y-0.5 overflow-y-auto pt-2">
        <p className="px-2 pb-1 text-xs font-medium uppercase tracking-wide text-zinc-400">Lịch sử chat</p>
        {conversations.map((conversation) => (
          <ConversationRow
            key={conversation.id}
            id={conversation.id}
            title={conversation.title}
            active={conversation.id === conversationId}
            onOpen={() => {
              onNavigated?.();
              navigate(`/c/${conversation.id}`);
            }}
            onRename={(title) => renameMutation.mutate({ id: conversation.id, title })}
            onDelete={() => deleteMutation.mutate(conversation.id)}
          />
        ))}
        {conversations.length === 0 && <p className="px-2 text-sm text-zinc-400">Chưa có chat nào.</p>}
      </nav>

      <p className="px-2 pb-1 text-[11px] leading-snug text-zinc-400">
        <Link to="/settings" className="underline-offset-2 hover:underline" onClick={onNavigated}>
          Settings
        </Link>
      </p>
    </aside>
  );
}

function ConversationRow(props: {
  id: string;
  title: string;
  active: boolean;
  onOpen: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}) {
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(props.title);
  return (
    <div className={`group flex items-center gap-1 rounded-lg px-2 ${props.active ? "bg-zinc-100 dark:bg-zinc-800" : "hover:bg-zinc-50 dark:hover:bg-zinc-900"}`}>
      <button type="button" onClick={props.onOpen} className="min-w-0 flex-1 truncate py-1.5 text-left text-sm">
        {props.title}
      </button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" className="h-7 w-7 opacity-0 transition group-hover:opacity-100" aria-label="Tùy chọn">
            <MoreHorizontal size={14} />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start">
          <DropdownMenuItem onSelect={() => { setDraft(props.title); setRenaming(true); }}>Đổi tên</DropdownMenuItem>
          <DropdownMenuItem className="text-red-600" onSelect={props.onDelete}>Xóa</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>

      <Dialog open={renaming} onOpenChange={setRenaming}>
        <DialogContent className="max-w-sm rounded-2xl">
          <DialogHeader><DialogTitle>Đổi tên chat</DialogTitle></DialogHeader>
          <Input value={draft} onChange={(e) => setDraft(e.target.value)} maxLength={50} />
          <DialogFooter>
            <Button
              onClick={() => {
                const trimmed = draft.trim();
                if (trimmed.length > 0) props.onRename(trimmed);
                setRenaming(false);
              }}
            >
              Lưu
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
```

(Task 11 will insert `<AttachmentsSection />` and the web toggle above the footer.)

- [ ] **Step 3: Header + AppLayout**

Create `web/src/components/Header.tsx`:

```tsx
import { LogOut, Menu, Settings } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Sheet, SheetContent, SheetTitle, SheetTrigger,
} from "@/components/ui/sheet";
import { api, setToken } from "@/lib/api";
import Sidebar from "./Sidebar";
import ThemeToggle from "./ThemeToggle";

export default function Header() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const { data: conversations } = useQuery({ queryKey: ["conversations"], queryFn: api.conversations });
  const { data: me } = useQuery({ queryKey: ["me"], queryFn: api.me });
  const active = conversations?.find((c) => c.id === conversationId);

  function logout(): void {
    setToken(null);
    navigate("/login");
  }

  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b px-4">
      <div className="flex items-center gap-2">
        <Sheet open={open} onOpenChange={setOpen}>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="md:hidden" aria-label="Menu">
              <Menu size={18} />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-72 p-0">
            <SheetTitle className="sr-only">Menu</SheetTitle>
            <Sidebar onNavigated={() => setOpen(false)} />
          </SheetContent>
        </Sheet>
        <span className="hidden font-semibold tracking-tight md:inline">CourseMate</span>
        <span className="mx-2 hidden truncate text-sm text-zinc-500 md:inline">{active?.title ?? ""}</span>
      </div>
      <div className="flex items-center gap-1">
        <ThemeToggle />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" aria-label="Tài khoản">
              <span className="flex h-7 w-7 items-center justify-center rounded-full bg-indigo-600 text-xs font-semibold text-white">
                {(me?.display_name ?? "?").charAt(0).toUpperCase()}
              </span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem asChild>
              <Link to="/settings"><Settings size={14} className="mr-2" />Settings</Link>
            </DropdownMenuItem>
            <DropdownMenuItem onClick={logout}><LogOut size={14} className="mr-2" />Log out</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
```

Create `web/src/components/AppLayout.tsx`:

```tsx
import { Outlet } from "react-router-dom";
import Header from "./Header";
import Sidebar from "./Sidebar";

export default function AppLayout() {
  return (
    <div className="flex h-screen overflow-hidden">
      <div className="hidden w-72 shrink-0 border-r bg-white md:block dark:bg-zinc-900">
        <Sidebar />
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        <Header />
        <main className="min-h-0 flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Wire routes with ChatRedirect**

`App.tsx` becomes (add imports `useEffect`, `useNavigate`, `useQueryClient` from react-router/@tanstack and `api`):

```tsx
function ChatRedirect() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: conversations, isLoading } = useQuery({
    queryKey: ["conversations"],
    queryFn: api.conversations,
  });

  useEffect(() => {
    if (isLoading || conversations === undefined) return;
    if (conversations.length > 0) {
      navigate(`/c/${conversations[0].id}`, { replace: true });
      return;
    }
    api.createConversation().then((created) => {
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      navigate(`/c/${created.id}`, { replace: true });
    });
  }, [conversations, isLoading, navigate, queryClient]);

  return <div className="p-8 text-sm text-zinc-400">Đang tải…</div>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<RequireAuth />}>
        <Route element={<AppLayout />}>
          <Route index element={<ChatRedirect />} />
          <Route path="/c/:conversationId" element={<ChatPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
```

Add the Inter font to `web/index.html` `<head>`:

```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
```

- [ ] **Step 5: Verify**

Run: `npm run build` → passes.

Manual checklist (backend running):
- Fresh login → redirected into a newly created chat; URL `/c/<id>`
- "＋ Chat mới" creates + navigates; rename via ⋯ updates list; deleting active chat navigates away
- Theme toggle switches dark/light; survives reload; avatar letter shows display_name initial
- Mobile width (<768px): hamburger opens Sheet sidebar

- [ ] **Step 6: Commit**

```bash
git add web/src web/index.html
git commit -m "feat(web): app shell with header, collapsible sidebar, history CRUD"
```

---

### Task 9: Chat view — messages, send, markdown, empty state

**Files:**
- Create: `web/src/components/Markdown.tsx`, `web/src/components/ChatMessage.tsx`
- Modify: `web/src/pages/ChatPage.tsx`

**Interfaces:**
- Consumes: `api.messages/chat`, query key `["messages", id]`, type `Turn`.
- Produces:
  - `Markdown({ text }: { text: string })` — react-markdown + remark-gfm + rehype-highlight, copy button on fenced code blocks
  - `ChatMessage({ turn }: { turn: Turn })` — user bubble right / assistant full-width (citations/refusal land in Task 10)
  - Pending-send UX: optimistic user bubble + typing indicator; failure → error card, input preserved

- [ ] **Step 1: Markdown component**

Create `web/src/components/Markdown.tsx`:

```tsx
import { Check, Copy } from "lucide-react";
import { memo, useState } from "react";
import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

function extractText(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  const element = node as { props?: { children?: ReactNode } };
  return element.props ? extractText(element.props.children) : "";
}

function CopyButton({ getText }: { getText: () => string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      aria-label="Copy code"
      className="absolute right-2 top-2 rounded-md border bg-zinc-900 p-1 text-zinc-400 opacity-0 transition group-hover:opacity-100"
      onClick={() => {
        void navigator.clipboard.writeText(getText()).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <div className="prose prose-sm max-w-none break-words dark:prose-invert">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          pre({ children, ...props }) {
            return (
              <div className="group relative my-3 overflow-hidden rounded-xl border bg-zinc-950">
                <CopyButton getText={() => extractText(children)} />
                <pre className="overflow-x-auto p-4 text-[13px]" {...props}>
                  {children}
                </pre>
              </div>
            );
          },
          code({ children, className }) {
            if (className?.startsWith("language-")) {
              return (
                <code className={`${className} font-mono`} style={{ whiteSpace: "pre" }}>
                  {children}
                </code>
              );
            }
            return (
              <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-[13px] dark:bg-zinc-800">
                {String(children)}
              </code>
            );
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});

export default Markdown;
```

- [ ] **Step 2: ChatMessage (text only this task)**

Create `web/src/components/ChatMessage.tsx`:

```tsx
import Markdown from "./Markdown";
import type { Turn } from "@/lib/types";

export default function ChatMessage({ turn }: { turn: Turn }) {
  if (turn.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-indigo-100 px-4 py-2.5 text-sm text-indigo-950 dark:bg-indigo-900/60 dark:text-indigo-50">
          {turn.text}
        </div>
      </div>
    );
  }
  return (
    <div className="text-sm leading-relaxed text-zinc-800 dark:text-zinc-100">
      <Markdown text={turn.text} />
    </div>
  );
}
```

- [ ] **Step 3: ChatPage**

Replace `web/src/pages/ChatPage.tsx`:

```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUp } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import ChatMessage from "@/components/ChatMessage";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

const SUGGESTIONS = [
  "Giải thích bảng băm là gì?",
  "Nên học môn nào trước khi học Trí tuệ nhân tạo?",
  "So sánh danh sách liên kết và mảng.",
];

function TypingIndicator() {
  return (
    <div className="flex gap-1 py-2">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="h-2 w-2 animate-bounce rounded-full bg-zinc-400"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </div>
  );
}

export default function ChatPage() {
  const { conversationId } = useParams();
  const queryClient = useQueryClient();
  const [input, setInput] = useState("");
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);
  const [sendError, setSendError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const { data: turns = [], isLoading } = useQuery({
    queryKey: ["messages", conversationId],
    queryFn: () => api.messages(conversationId!),
    enabled: conversationId !== undefined,
  });

  const sendMutation = useMutation({
    mutationFn: (message: string) => api.chat(conversationId!, message, false),
    onMutate: (message) => {
      setPendingQuestion(message);
      setSendError(null);
    },
    onSuccess: () => {
      setPendingQuestion(null);
      setInput("");
      if (textareaRef.current !== null) textareaRef.current.style.height = "auto";
      void queryClient.invalidateQueries({ queryKey: ["messages", conversationId] });
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
    onError: (error) => {
      setPendingQuestion(null);
      setSendError(error instanceof Error ? error.message : "Lỗi không xác định.");
    },
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns.length, pendingQuestion]);

  function submit(text: string): void {
    const trimmed = text.trim();
    if (trimmed.length === 0 || sendMutation.isPending) return;
    sendMutation.mutate(trimmed);
  }

  function resize(event: React.ChangeEvent<HTMLTextAreaElement>): void {
    setInput(event.target.value);
    event.target.style.height = "auto";
    event.target.style.height = `${Math.min(event.target.scrollHeight, 160)}px`;
  }

  const empty = !isLoading && turns.length === 0 && pendingQuestion === null;

  return (
    <div className="mx-auto flex h-full w-full max-w-[720px] flex-col px-4">
      <div className="min-h-0 flex-1 space-y-6 py-6">
        {empty && (
          <div className="flex h-full flex-col items-center justify-center gap-6 text-center">
            <h1 className="text-2xl font-semibold tracking-tight">CourseMate</h1>
            <div className="grid w-full max-w-md gap-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => submit(suggestion)}
                  className="rounded-xl border bg-white px-4 py-3 text-left text-sm shadow-sm transition hover:border-indigo-300 dark:bg-zinc-900"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn, index) => (
          <ChatMessage key={index} turn={turn} />
        ))}

        {pendingQuestion !== null && (
          <>
            <div className="flex justify-end">
              <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-indigo-100 px-4 py-2.5 text-sm dark:bg-indigo-900/60">
                {pendingQuestion}
              </div>
            </div>
            <TypingIndicator />
          </>
        )}

        {sendError !== null && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950">
            Không gửi được câu trả lời: {sendError}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="sticky bottom-0 bg-gradient-to-t from-white via-white pb-4 pt-2 dark:from-zinc-950 dark:via-zinc-950">
        <div className="relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={resize}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit(input);
              }
            }}
            rows={1}
            placeholder="Hỏi về môn học hoặc tài liệu…"
            className="w-full resize-none rounded-2xl border py-3 pl-4 pr-12 text-sm shadow-sm outline-none focus:ring-2 focus:ring-indigo-500/40 dark:bg-zinc-900"
          />
          <Button
            size="icon"
            disabled={input.trim().length === 0 || sendMutation.isPending}
            onClick={() => submit(input)}
            className="absolute bottom-2.5 right-2.5 h-8 w-8 rounded-full"
            aria-label="Gửi"
          >
            <ArrowUp size={16} />
          </Button>
        </div>
        <p className="mt-1.5 text-center text-[11px] text-zinc-400">
          Enter để gửi · Shift+Enter để xuống dòng
        </p>
      </div>
    </div>
  );
}
```

Note the hardcoded third argument `false` in `api.chat(...)` — Task 11 replaces it with the web toggle.

- [ ] **Step 4: Verify**

Run: `npm run build` → passes.

Manual (backend running): suggestion chips send and render replies with markdown; Enter sends / Shift+Enter newlines; typing indicator shows while waiting; stopping the backend and sending shows the red error card with input preserved; view pins to newest message.

- [ ] **Step 5: Commit**

```bash
git add web/src
git commit -m "feat(web): chat view with markdown rendering, optimistic send, empty state"
```

---

### Task 10: Citation cards, refusal card, fallback notice, skills badges

**Files:**
- Create: `web/src/components/CitationCard.tsx`
- Modify: `web/src/components/ChatMessage.tsx`

**Interfaces:**
- Consumes: `Citation`, `Turn` types; shadcn Dialog.
- Produces: `CitationCard({ citation }: { citation: Citation })` — compact card + details Dialog (web citations get an external link); upgraded `ChatMessage` with citations stack right of the answer (below on narrow screens), amber refusal card, blue fallback notice (`text` starts with `Lưu ý:`/`Note:` and no citations), skills badge line.

- [ ] **Step 1: CitationCard**

Create `web/src/components/CitationCard.tsx`:

```tsx
import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import type { Citation } from "@/lib/types";

const KIND_LABEL: Record<string, string> = {
  slides: "slides",
  textbook: "giáo trình",
  upload: "đính kèm",
  web: "web",
};

export default function CitationCard({ citation }: { citation: Citation }) {
  const { source } = citation;
  const isWeb = source.kind === "web";
  return (
    <Dialog>
      <DialogTrigger asChild>
        <button
          type="button"
          className="w-full rounded-xl border bg-white px-3 py-2 text-left text-xs shadow-sm transition hover:border-indigo-300 dark:bg-zinc-900"
        >
          <span className="mr-1 font-semibold text-indigo-600 dark:text-indigo-400">{citation.marker}</span>
          <span className="line-clamp-2">{source.document_title}</span>
          <span className="mt-0.5 flex items-center gap-1 text-zinc-400">
            <span className="rounded bg-zinc-100 px-1 dark:bg-zinc-800">{KIND_LABEL[source.kind] ?? source.kind}</span>
            <span className="truncate">{isWeb ? "liên kết" : source.chapter}</span>
          </span>
        </button>
      </DialogTrigger>
      <DialogContent className="max-w-md rounded-2xl">
        <DialogHeader>
          <DialogTitle>
            {citation.marker} {source.document_title}
          </DialogTitle>
          <DialogDescription>{source.chapter}</DialogDescription>
        </DialogHeader>
        <dl className="space-y-1.5 text-sm">
          <Row label="Mã tài liệu" value={source.document_id} />
          {!isWeb && <Row label="Khóa học" value={source.course_code} />}
          <Row label="Loại" value={KIND_LABEL[source.kind] ?? source.kind} />
          <Row label="Ngôn ngữ" value={source.language} />
        </dl>
        {isWeb && (
          <Button asChild variant="outline" className="gap-2 rounded-xl">
            <a href={source.chapter} target="_blank" rel="noreferrer">
              <ExternalLink size={14} /> Mở nguồn
            </a>
          </Button>
        )}
      </DialogContent>
    </Dialog>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  if (!value) return null;
  return (
    <div className="flex gap-2">
      <dt className="w-24 shrink-0 text-zinc-400">{label}</dt>
      <dd className="min-w-0 break-words">{value}</dd>
    </div>
  );
}
```

- [ ] **Step 2: Upgrade ChatMessage**

Replace `web/src/components/ChatMessage.tsx`:

```tsx
import { AlertTriangle, Info, Puzzle } from "lucide-react";
import CitationCard from "./CitationCard";
import Markdown from "./Markdown";
import type { Turn } from "@/lib/types";

function isFallbackNotice(text: string): boolean {
  return text.startsWith("Lưu ý:") || text.startsWith("Note:");
}

export default function ChatMessage({ turn }: { turn: Turn }) {
  if (turn.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl bg-indigo-100 px-4 py-2.5 text-sm text-indigo-950 dark:bg-indigo-900/60 dark:text-indigo-50">
          {turn.text}
        </div>
      </div>
    );
  }

  if (turn.refused) {
    return (
      <div className="rounded-2xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
        <p className="flex items-center gap-2 font-medium">
          <AlertTriangle size={15} />
          Không tìm đủ tài liệu hỗ trợ để trả lời.
        </p>
        {turn.rephrase_suggestion !== "" && (
          <p className="mt-1 text-xs opacity-80">Gợi ý viết lại: {turn.rephrase_suggestion}</p>
        )}
      </div>
    );
  }

  const fallback = isFallbackNotice(turn.text) && turn.citations.length === 0;

  return (
    <div className="space-y-2">
      {fallback && (
        <p className="flex items-start gap-1.5 text-xs text-blue-700 dark:text-blue-300">
          <Info size={13} className="mt-0.5 shrink-0" />
          Không tìm thấy tài liệu liên quan trong kho — trả lời theo hiểu biết chung (không có trích dẫn).
        </p>
      )}
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start">
        <div className="min-w-0 flex-1 text-sm leading-relaxed text-zinc-800 dark:text-zinc-100">
          <Markdown text={turn.text} />
          {turn.skills_applied.length > 0 && (
            <p className="mt-2 flex flex-wrap items-center gap-1 text-[11px] text-zinc-400">
              <Puzzle size={11} />
              {turn.skills_applied.join(", ")}
            </p>
          )}
        </div>
        {turn.citations.length > 0 && (
          <div className="flex w-full shrink-0 flex-row gap-2 overflow-x-auto lg:w-56 lg:flex-col lg:overflow-visible">
            {turn.citations.map((citation) => (
              <div key={citation.marker} className="w-48 shrink-0 lg:w-full">
                <CitationCard citation={citation} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify**

Run: `npm run build` → passes.

Manual: a grounded question shows markdown + citation cards on the right (wrapping below under `lg`); clicking opens the dialog; web citations open their link; a refusal shows the amber card with suggestion; a fallback answer shows the blue notice.

- [ ] **Step 4: Commit**

```bash
git add web/src
git commit -m "feat(web): citation cards, refusal/fallback cards, skills badges"
```

---

### Task 11: Sidebar attachments + web-search toggle

**Files:**
- Create: `web/src/components/AttachmentsSection.tsx`, `web/src/lib/web-toggle-context.tsx`
- Modify: `web/src/components/Sidebar.tsx`, `web/src/App.tsx`, `web/src/pages/ChatPage.tsx`

**Interfaces:**
- Consumes: `api.attachments/uploadAttachment/deleteAttachment`, `api.features`, `MAX_FILES_PER_CONVERSATION = 3`, query keys `["attachments", convId]`, `["features"]`.
- Produces:
  - `WebToggleProvider` + `useWebToggle(): { useWeb: boolean; setUseWeb(v): void }` persisted in `localStorage["cm_use_web"]`
  - Sidebar sections between history and footer: 📎 attachments (list, upload, delete, limit caption) and 🌐 web toggle (hidden when `has_web_search=false`)
  - `ChatPage` passes the toggle value to `api.chat` instead of hardcoded `false`

- [ ] **Step 1: Web toggle context**

Create `web/src/lib/web-toggle-context.tsx`:

```tsx
import { createContext, useContext, useEffect, useState } from "react";
import type { ReactNode } from "react";

const KEY = "cm_use_web";

interface WebToggleValue {
  useWeb: boolean;
  setUseWeb: (value: boolean) => void;
}

const WebToggleContext = createContext<WebToggleValue>({
  useWeb: false,
  setUseWeb: () => undefined,
});

export function WebToggleProvider({ children }: { children: ReactNode }) {
  const [useWeb, setUseWeb] = useState<boolean>(() => localStorage.getItem(KEY) === "1");
  useEffect(() => {
    localStorage.setItem(KEY, useWeb ? "1" : "0");
  }, [useWeb]);
  return (
    <WebToggleContext.Provider value={{ useWeb, setUseWeb }}>
      {children}
    </WebToggleContext.Provider>
  );
}

export function useWebToggle(): WebToggleValue {
  return useContext(WebToggleContext);
}
```

Wrap in `App.tsx`:

```tsx
<Route element={<RequireAuth />}>
  <Route
    element={
      <WebToggleProvider>
        <AppLayout />
      </WebToggleProvider>
    }
  >
    ...
  </Route>
</Route>
```

- [ ] **Step 2: AttachmentsSection**

Create `web/src/components/AttachmentsSection.tsx`:

```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Paperclip, Trash2, Upload } from "lucide-react";
import { useRef } from "react";
import { useParams } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

const KIND_ICONS: Record<string, string> = { pdf: "📄", md: "📝", txt: "🗒️" };
const MAX_FILES = 3; // MAX_FILES_PER_CONVERSATION in rag_core.attachments

export default function AttachmentsSection() {
  const { conversationId } = useParams();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { data: attachments = [] } = useQuery({
    queryKey: ["attachments", conversationId],
    queryFn: () => api.attachments(conversationId!),
    enabled: conversationId !== undefined,
  });

  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: ["attachments", conversationId] });

  const uploadMutation = useMutation({
    mutationFn: (file: File) => api.uploadAttachment(conversationId!, file),
    onSuccess: (meta) => {
      toast.success(`Đã xử lý ${meta.filename}: ${meta.chunk_count} đoạn.`);
      void invalidate();
    },
    onError: (error) => toast.error(error instanceof Error ? error.message : String(error)),
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteAttachment,
    onSuccess: invalidate,
    onError: (error) => toast.error(String(error)),
  });

  const atLimit = attachments.length >= MAX_FILES;

  return (
    <section className="border-t pt-3">
      <p className="flex items-center gap-1.5 px-2 pb-1 text-xs font-medium uppercase tracking-wide text-zinc-400">
        <Paperclip size={12} /> Tài liệu đính kèm ({attachments.length}/{MAX_FILES})
      </p>
      <ul className="space-y-1 px-2">
        {attachments.map((attachment) => (
          <li key={attachment.id} className="group flex items-center gap-1.5 rounded-lg px-1 py-1 hover:bg-zinc-50 dark:hover:bg-zinc-900">
            <span className="text-sm">{KIND_ICONS[attachment.file_kind] ?? "📄"}</span>
            <span className="min-w-0 flex-1">
              <span className="block truncate text-xs">{attachment.filename}</span>
              <span className="block text-[10px] text-zinc-400">
                {attachment.chunk_count} đoạn · {Math.max(1, Math.floor(attachment.size_bytes / 1024))} KB
              </span>
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 opacity-0 transition group-hover:opacity-100"
              aria-label="Xóa tài liệu"
              onClick={() => deleteMutation.mutate(attachment.id)}
            >
              <Trash2 size={12} />
            </Button>
          </li>
        ))}
      </ul>
      {atLimit ? (
        <p className="px-2 pt-1 text-[11px] text-zinc-400">Đã đạt giới hạn {MAX_FILES} tài liệu cho chat này.</p>
      ) : (
        <>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md"
            className="hidden"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file !== undefined) uploadMutation.mutate(file);
              event.target.value = "";
            }}
          />
          <Button
            variant="outline"
            size="sm"
            className="mx-2 mt-1 w-[calc(100%-1rem)] justify-start gap-2 rounded-xl"
            disabled={uploadMutation.isPending}
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={13} /> {uploadMutation.isPending ? "Đang xử lý…" : "Thêm tài liệu"}
          </Button>
        </>
      )}
    </section>
  );
}
```

Insert `<AttachmentsSection />` into `Sidebar` between the history nav and the footer link.

- [ ] **Step 3: Web toggle in sidebar footer**

In `Sidebar.tsx` add:

```tsx
const { useWeb, setUseWeb } = useWebToggle();
const { data: features } = useQuery({ queryKey: ["features"], queryFn: api.features });
```

and render above the Settings footer link (imports: `Switch` from `@/components/ui/switch`, `Globe` from lucide, `useWebToggle`):

```tsx
{features?.has_web_search && (
  <label className="flex cursor-pointer items-center justify-between rounded-xl px-2 py-1.5 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-900">
    <span className="flex items-center gap-2"><Globe size={14} /> Tìm kiếm web</span>
    <Switch checked={useWeb} onCheckedChange={setUseWeb} />
  </label>
)}
```

- [ ] **Step 4: Wire ChatPage to the toggle**

In `ChatPage.tsx`: `const { useWeb } = useWebToggle();` and change the mutation to:

```tsx
mutationFn: (message: string) => api.chat(conversationId!, message, useWeb),
```

- [ ] **Step 5: Verify**

Run: `npm run build` → passes.

Manual (backend running; without `FIRECRAWL_API_KEY` the toggle stays hidden): upload a small `.txt`/`.md` → row appears with đoạn/KB counts; a 4th upload shows the limit caption and hides the button; delete removes the row; with a Firecrawl key set, the toggle appears, persists across reloads, and reaches the backend (`"use_web": true/false` visible in the network tab).

- [ ] **Step 6: Commit**

```bash
git add web/src
git commit -m "feat(web): sidebar attachments panel and persistent web-search toggle"
```

---

### Task 12: Settings page

**Files:**
- Modify: `web/src/pages/SettingsPage.tsx`

**Interfaces:**
- Consumes: `api.me/updateProfile/clearConversations`, query keys `["me"]`, `["conversations"]`.
- Produces: profile form (display name + language select) and danger zone (typed-DELETE confirm clears all conversations, navigates home).

- [ ] **Step 1: Implement**

Replace `web/src/pages/SettingsPage.tsx`:

```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";

export default function SettingsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { data: me } = useQuery({ queryKey: ["me"], queryFn: api.me });

  const [displayName, setDisplayName] = useState("");
  const [language, setLanguage] = useState("vi");
  const [confirmText, setConfirmText] = useState("");

  useEffect(() => {
    if (me !== undefined) {
      setDisplayName(me.display_name);
      setLanguage(me.language);
    }
  }, [me]);

  const saveMutation = useMutation({
    mutationFn: () => api.updateProfile({ display_name: displayName, language }),
    onSuccess: (updated) => {
      queryClient.setQueryData(["me"], updated);
      toast.success("Đã lưu thay đổi.");
    },
    onError: (error) => toast.error(error instanceof Error ? error.message : String(error)),
  });

  const clearMutation = useMutation({
    mutationFn: api.clearConversations,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      navigate("/");
    },
    onError: (error) => toast.error(String(error)),
  });

  return (
    <div className="mx-auto w-full max-w-xl space-y-8 p-6">
      <div className="flex items-center gap-3">
        <Button asChild variant="ghost" size="icon" aria-label="Quay lại">
          <Link to="/"><ArrowLeft size={18} /></Link>
        </Button>
        <h1 className="text-xl font-semibold tracking-tight">Cài đặt</h1>
      </div>

      <section className="space-y-4 rounded-2xl border bg-white p-6 shadow-sm dark:bg-zinc-900">
        <h2 className="font-medium">Hồ sơ</h2>
        <label className="block space-y-1.5">
          <span className="text-sm text-zinc-500">Tên hiển thị</span>
          <Input value={displayName} maxLength={50} onChange={(event) => setDisplayName(event.target.value)} />
        </label>
        <label className="block space-y-1.5">
          <span className="text-sm text-zinc-500">Ngôn ngữ trả lời</span>
          <select
            value={language}
            onChange={(event) => setLanguage(event.target.value)}
            className="w-full rounded-xl border bg-transparent px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-indigo-500/40 dark:bg-zinc-900"
          >
            <option value="vi">Tiếng Việt</option>
            <option value="en">English</option>
          </select>
        </label>
        <Button
          className="rounded-xl"
          disabled={saveMutation.isPending || displayName.trim().length === 0}
          onClick={() => saveMutation.mutate()}
        >
          Lưu thay đổi
        </Button>
        <p className="text-xs text-zinc-400">Tên đăng nhập: {me?.username}</p>
      </section>

      <section className="space-y-3 rounded-2xl border border-red-200 bg-red-50 p-6 dark:border-red-900 dark:bg-red-950">
        <h2 className="font-medium text-red-800 dark:text-red-200">Vùng nguy hiểm</h2>
        <p className="text-sm text-red-700 dark:text-red-300">
          Xóa tất cả hội thoại và tin nhắn của bạn. Không thể khôi phục.
        </p>
        <Input
          placeholder="Gõ DELETE để xác nhận"
          value={confirmText}
          onChange={(event) => setConfirmText(event.target.value)}
          className="max-w-xs border-red-300 dark:border-red-800"
        />
        <Button
          variant="destructive"
          className="rounded-xl"
          disabled={confirmText !== "DELETE" || clearMutation.isPending}
          onClick={() => clearMutation.mutate()}
        >
          Xóa tất cả lịch sử
        </Button>
      </section>
    </div>
  );
}
```

- [ ] **Step 2: Verify**

Run: `npm run build` → passes.

Manual: rename display name → avatar letter updates after save; switching language affects subsequent answer language; anything other than `DELETE` keeps the destructive button disabled; confirming wipes history and lands on a fresh chat.

- [ ] **Step 3: Commit**

```bash
git add web/src
git commit -m "feat(web): settings page with profile editing and danger zone"
```

---

### Task 13: Cleanup — remove Streamlit, docs, end-to-end verification

**Files:**
- Delete: `app.py`
- Modify: `pyproject.toml` (drop streamlit; mypy files drop `app.py`), `README.md`, `docs/SETUP.md`

**Interfaces:**
- Consumes: everything built in Tasks 1–12.
- Produces: repo with no Streamlit dependency; `uv run serve` serves the full app at one URL.

- [ ] **Step 1: Build the SPA**

Run (in `web/`): `npm run build` → produces `web/dist/`.

- [ ] **Step 2: End-to-end verification before cleanup**

Run: `uv run serve` then open `http://127.0.0.1:8000`.

Checklist:
- Login page renders at `/` (SPA fallback works — try hard-refresh on `/settings`)
- Register/login works against the real API
- Chat round-trip returns a grounded answer with citation cards
- History CRUD + attachments + web toggle + settings all function
- Old data intact: conversations created under Streamlit appear for their users

- [ ] **Step 3: Remove Streamlit**

```bash
git rm app.py
```

Edit `pyproject.toml`: delete the `"streamlit>=1.62.0",` line; change mypy `files` to `["generate_data.py", "eval.py", "rag_core", "tests", "server"]`.

Run: `uv sync && uv run pytest && uv run mypy`
Expected: all green.

- [ ] **Step 4: Update docs**

`grep -rn -i streamlit README.md docs/SETUP.md` and update every hit:

- `README.md`: replace the `## Streamlit UI` section with `## Web UI (React + FastAPI)` describing: dev mode (`uv run uvicorn server.main:app --reload` + `cd web && npm run dev`) and production mode (`cd web && npm run build` then `uv run serve` → http://127.0.0.1:8000); keep the feature description (login gate, chat with citations, attachments, web toggle, history).
- `docs/SETUP.md`: replace `uv run streamlit run app.py` instructions with the two modes above; mention Node 20+ as a requirement.

- [ ] **Step 5: Final verification**

Run: `uv run pytest && uv run mypy`
Expected: green.

Manual: `uv run serve` once more after cleanup; repeat the Step 2 checklist quickly.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock README.md docs/SETUP.md
git commit -m "chore: replace Streamlit UI with FastAPI + React web app"
```

---

## Self-review notes

- Spec coverage: auth (T1), profile (T2/T12), conversations CRUD + messages (T2/T8), chat (T3/T9), citations/refusal/skills (T10), attachments (T4/T11), features/web toggle (T3/T11), static serving + `uv run serve` (T5/T13), ownership 404 rule (T2/T4 tests), old-data reuse (T13 checklist), Vietnamese UI copy throughout.
- Type consistency: `turn_out`/`citation_out` defined once (Task 2) and reused (Task 3); query keys fixed across tasks; `api.chat(id, message, useWeb)` signature consistent between Tasks 6/9/11.











