# Profile + Chat History — Design

**Date:** 2026-08-21
**Status:** Approved (brainstorming 3/3 sections)
**Scope:** Phương án A — Minimal migration (DB + sidebar ChatGPT-like)
**Related:** `CONTEXT.md`, `rag_core/db.py:12`, `rag_core/models.py:40`, `app.py:120`

## 1. Goal

Thêm 2 nhóm tính năng tối giản mà không chạm pipeline RAG:

- **Profile:** mỗi `User` có `display_name` (mặc định = `username`, cho phép đổi) và `language` (`vi`/`en`) lưu DB. Setting chỉ gồm ngôn ngữ UI + "Xóa tất cả lịch sử" (confirm). Không đổi password/username, không theme.
- **Lịch sử chat & Tạo chat mới:** chuyển từ 1 conversation/user sang N conversations/user, sidebar trái kiểu ChatGPT: list chats sort `updated_at DESC`, nút "＋ Tạo chat mới", click chuyển chat, đổi tên/xóa per-chat. Mỗi chat vẫn là `Session` với `MAX_TURNS=6` như cũ.

Quyết định brainstorming: **A (display_name mới) — A (Ngôn ngữ + Xóa lịch sử) — A (Nhiều chats sidebar)**.

## 2. Architecture

Giữ nguyên single seam `RagCore.answer(user_message, session)` `rag_core/__init__.py:92`. Chỉ thay `db` và `app.py`.

```
Streamlit app.py
 ├─ st.sidebar (mới)
 │   ├─ "＋ Tạo chat mới" → create_conversation()
 │   ├─ list_conversations() → buttons (title, updated_at, preview)
 │   │       └─ popover: Đổi tên / Xóa
 │   └─ Profile card (display_name) → toggle show_profile
 ├─ main:
 │   ├─ if show_profile: render_profile() (display_name, language, clear all)
 │   └─ else: render_chat(active_session) (như cũ, dùng get_session + append_exchange)
 └─ st.session_state: user, active_conversation_id, show_profile

Database (SQLite) rag_core/db.py
 ├─ users(id, username UNIQUE, password, display_name TEXT, language TEXT, created_at)
 └─ conversations(id PK, user_id FK → users.id, title TEXT, created_at, updated_at)
        └─ messages(id, conversation_id FK, role, text, created_at, citations, refused, rephrase_suggestion)

RagCore / Index / Generator — không đổi
```

Thay đổi file:

- Sửa: `rag_core/models.py` — `User` thêm `display_name, language`, mới `ConversationMeta`.
- Sửa: `rag_core/db.py` — schema + migrate + 6 API mới.
- Sửa: `app.py` — sidebar, active conversation, profile page, title auto.
- Không đụng: `rag_core/__init__.py`, `rag_core/index.py`, `rag_core/generator.py`.

## 3. Components

### 3.1 Models (`rag_core/models.py`)

```python
@dataclass(frozen=True)
class User:
    id: int
    username: str
    display_name: str  # mới, default username
    language: str      # mới, "vi" | "en", default "vi"

@dataclass(frozen=True)
class ConversationMeta:
    id: str
    user_id: int
    title: str
    created_at: str
    updated_at: str
    preview: str  # 40 chars đầu của user message đầu

# Session giữ nguyên: id, user_id (str), turns
```

`display_name` 1–50 chars, `language` chỉ `vi`/`en`.

### 3.2 Database (`rag_core/db.py`)

**Schema mới (`_create_schema`):**

```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  display_name TEXT NOT NULL DEFAULT '',
  language TEXT NOT NULL DEFAULT 'vi',
  created_at TEXT NOT NULL
);
CREATE TABLE conversations (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id),
  title TEXT NOT NULL DEFAULT 'New chat',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE messages (... như cũ);
CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);
```

Bỏ `UNIQUE` trên `conversations.user_id`. `title` default `"New chat"`, `updated_at` = `created_at` lúc tạo.

**Migrate (`_migrate`):**

1. Thêm cột nếu thiếu: `users.display_name`, `users.language`, `conversations.title`, `conversations.updated_at` (ALTER TABLE).
2. Backfill: `display_name = username WHERE display_name IS NULL OR ''`; `title = 'New chat' WHERE NULL`; `updated_at = created_at`.
3. Nếu `conversations` còn UNIQUE(user_id): detect qua `PRAGMA index_list('conversations')` + `PRAGMA index_info`, nếu có unique index trên `user_id` → recreate: `CREATE TABLE conversations_new(...)` không UNIQUE → `INSERT INTO conversations_new SELECT id, user_id, COALESCE(title,'New chat'), created_at, COALESCE(updated_at, created_at) FROM conversations` → `DROP TABLE conversations` → `ALTER TABLE conversations_new RENAME TO conversations` → tạo lại index + FK.
4. Tạo index nếu chưa có.

Thread-safety giữ `check_same_thread=False` + per-op `_connect()` như hiện tại `rag_core/db.py:42`.

**API mới:**

```python
def create_conversation(self, user_id: int, title: str = "New chat") -> Session
def list_conversations(self, user_id: int) -> list[ConversationMeta]  # ORDER BY updated_at DESC
def get_session(self, session_id: str) -> Session  # giữ nguyên, thêm check exists
def get_user(self, user_id: int) -> User | None
def update_user_profile(self, user_id: int, display_name: str | None = None, language: str | None = None) -> User
def update_conversation_title(self, session_id: str, title: str) -> None
def delete_conversation(self, session_id: str) -> None
def clear_all_conversations(self, user_id: int) -> None
def append_exchange(...) -> None  # bổ sung: UPDATE updated_at, auto title
```

Chi tiết `append_exchange`: sau `executemany` insert 2 messages, chạy `UPDATE conversations SET updated_at=? WHERE id=?`. Nếu `title == "New chat"` và đây là exchange đầu (count messages ==2 sau insert), set `title = user_text.strip()[:40]`. Dùng `SELECT COUNT(*)` trước/sau để xác định.

`list_conversations`: `SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at, (SELECT text FROM messages WHERE conversation_id=c.id AND role='user' ORDER BY id LIMIT 1) AS preview FROM conversations c WHERE c.user_id=? ORDER BY c.updated_at DESC`.

`register`: `INSERT users(username, password, display_name, language, created_at) VALUES (?, ?, ?, 'vi', ?)` với `display_name=username`.

`login`: `SELECT id, username, display_name, language FROM users WHERE ...` → `User(...)` với fallback `display_name or username`.

Giữ `get_or_create_session(user_id)` làm legacy wrapper: nếu `list_conversations` non-empty → `get_session(metas[0].id)`, else `create_conversation`.

### 3.3 UI (`app.py`)

**State:** `st.session_state["user"]`, `st.session_state["active_conversation_id"]`, `st.session_state["show_profile"]` (bool, default False).

**Sidebar (`render_sidebar(user)`):**

- `if st.sidebar.button("＋ Tạo chat mới", use_container_width=True): db.create_conversation(user.id); set active_id = new.id; rerun`
- `metas = db.list_conversations(user.id)` — nếu rỗng, auto `create_conversation`.
- Loop metas: `cols = st.sidebar.columns([4,1])` hoặc `st.sidebar.button(title, key=id)`; active chat in đậm (`**title**`). Dùng `st.sidebar.caption(updated_at[:16])` + preview nếu title còn "New chat".
- Per-chat actions: `with st.sidebar.popover("⋯"):` → `new_title = st.text_input("Đổi tên", value=meta.title)` + `Save` → `update_conversation_title`; `Delete` (red, confirm `st.warning`) → `delete_conversation` → nếu xóa active → chuyển sang metas[0] hoặc tạo mới.
- Bottom: `st.sidebar.divider(); st.sidebar.caption(f"👤 {user.display_name} ({user.username})")`; `if st.sidebar.button("Profile / Setting"):` toggle `show_profile`.

**Main:**

- `if show_profile: render_profile(user)` else `render_chat(user, active_id)`.
- `render_profile(user)`: form `display_name = st.text_input("Tên hiển thị", value=user.display_name)` + `language = st.selectbox("Ngôn ngữ", ["vi","en"], index=0 if user.language=="vi" else 1)` → `Save` → `db.update_user_profile` → update `st.session_state["user"]` → `st.success`. Nút `Xóa tất cả lịch sử` (type primary, red) → cần `st.text_input("Gõ DELETE để xác nhận")` → `db.clear_all_conversations(user.id)` → `create_conversation` → `show_profile=False` → rerun. Nút `Quay lại chat`.
- `render_chat(user, active_id)`: `session = db.get_session(active_id)` (try/except KeyError → tạo mới). Loop `session.turns` → `render_turn` như cũ `app.py:97`. `prompt = st.chat_input(...)` → `result = get_core().answer(prompt, session)` → `db.append_exchange(active_id, prompt, result.answer, ...)` → `st.rerun()` để sidebar cập nhật `updated_at`/`title`. Giữ `_is_fallback` banner.

Không dùng `st.cache_resource` cho active_id; `get_database()` và `get_core()` giữ như cũ.

## 4. Data Flow

**Đăng ký:** `register("hieu","pw")` → `User(id, username="hieu", display_name="hieu", language="vi")` → `create_conversation` lazy khi vào chat.

**Tạo chat mới:** click sidebar → `create_conversation(user.id, "New chat")` → `active_id = new` → sidebar list cập nhật (new lên đầu).

**Gửi tin đầu trong chat mới:** `append_exchange(sid, "giải thích bảng băm?", ...)` → detect first exchange → `UPDATE conversations SET title="giải thích bảng băm?" , updated_at=now()` → list sidebar show title mới.

**Chuyển chat:** click chat B → `active_id = B` → `get_session(B)` → render 6 turns gần nhất.

**Đổi profile:** `update_user_profile(id, display_name="Hiếu Nguyễn", language="en")` → `User` mới lưu → sidebar caption đổi ngay.

**Xóa tất cả:** Setting → `clear_all_conversations(user.id)` → `DELETE messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id=?)` + `DELETE conversations WHERE user_id=?` → tạo 1 chat trống.

## 5. Error Handling

| Case | Behavior |
|------|----------|
| `display_name` rỗng / >50 | `ValueError("display_name must be 1..50 chars")`, UI show `st.error` |
| `language` khác vi/en | `ValueError`, UI fallback vi |
| `active_conversation_id` không tồn tại (bị xóa) | `get_session` raise `KeyError` → `render_chat` tự tạo conversation mới và set active |
| `delete_conversation` id không thuộc user | `PermissionError` hoặc `KeyError`, UI không show nút xóa của user khác vì list đã filter |
| DB legacy với UNIQUE(user_id) | `_migrate` recreate table, giữ data, không mất messages |
| `title` rỗng sau strip | giữ "New chat" |
| `list_conversations` rỗng | auto tạo 1 chat để UI không trống |

## 6. Testing

**Unit `tests/test_db.py` bổ sung (offline, no LLM):**

- `test_register_sets_display_name_and_language`
- `test_update_user_profile_changes_display_name`
- `test_update_user_profile_invalid_language_rejected`
- `test_create_multiple_conversations_per_user`
- `test_list_conversations_ordered_by_updated_at`
- `test_append_exchange_auto_titles_first_message`
- `test_append_exchange_updates_updated_at`
- `test_delete_conversation_removes_messages`
- `test_clear_all_conversations`
- `test_legacy_migration_bỏ_unique` (tạo DB cũ với UNIQUE rồi mở Database mới)
- `test_get_or_create_session_backward_compat`

Giữ các test cũ pass (backward compat `get_or_create_session`).

**Manual QA Streamlit:**

- Register → display_name mặc định = username → đổi tên → sidebar cập nhật
- Tạo 3 chats, gửi tin khác nhau → sidebar sort đúng, title auto
- Chuyển chat → lịch sử khôi phục sau reload + sau `Database` reopen
- Đổi language vi→en → persist sau logout/login
- Xóa 1 chat → active chuyển đúng, xóa tất cả → về 1 chat trống

Không đụng `eval.py`, `tests/test_rag_core.py`.

## 7. Out of Scope

- Đổi `username`/`password`, đổi theme sáng/tối, avatar, phân trang lịch sử, search, export chat, share link.
- Không thêm bảng mới ngoài migrate `users`/`conversations`.
- Không thay đổi `RagCore`, `Index`, `Generator`, `SessionRewrite`.

## 8. Build Order

1. `rag_core/models.py` + `rag_core/db.py` migrate & API mới + unit tests
2. `app.py` sidebar + active conversation + title auto
3. `app.py` profile page (display_name, language, clear all)
4. QA migration từ DB cũ (`test_old_schema_database_migrates_in_place` mở rộng)

## 9. Alternatives Considered

- **Bảng mới `conversations_v2`:** an toàn nhưng duplicate, phức tạp join, không YAGNI.
- **Chỉ `st.session_state` không persist:** nhanh nhưng mất lịch sử sau reload, sai yêu cầu.

## 10. Risks

- SQLite recreate table khi bỏ UNIQUE cần `INSERT SELECT` đúng thứ tự, test với DB có data cũ.
- Streamlit rerun có thể mất `active_conversation_id` nếu không lưu session_state — mitigated bằng set ngay sau create và fallback auto-create.
- Race tạo 2 conversations cùng lúc: `id` là uuid nên không clash, `updated_at` dùng `now()` có thể trùng ms nhưng order vẫn ổn vì `id` ngẫu nhiên không dùng để sort.
