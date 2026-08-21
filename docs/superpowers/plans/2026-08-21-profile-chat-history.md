# Profile + Chat History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm profile tối giản (display_name + language vi/en + xóa tất cả lịch sử) và lịch sử N chats/user kiểu ChatGPT (sidebar, tạo chat mới, đổi tên/xóa) với auto-title và `updated_at` sort, persist SQLite có migrate từ schema cũ 1-chat/user.

**Architecture:** Chỉ đổi `rag_core/models.py` + `rag_core/db.py` (schema bỏ UNIQUE, thêm cột, recreate migrate, thêm 7 API mới giữ `get_or_create_session` như wrapper) và `app.py` (st.sidebar + active_conversation_id + profile page). `RagCore` không đổi, mỗi chat vẫn là `Session` MAX_TURNS=6.

**Tech Stack:** Python 3.11+, Streamlit 1.62, SQLite (sqlite3), pytest 8, mypy strict, `rag_core` package.

## Global Constraints

- Python >=3.11 (pyproject.toml:5)
- `display_name` 1..50 chars, `language` chỉ `vi`/`en` default `vi` (design §3.1)
- `conversations.user_id` bỏ UNIQUE, cho N chats/user; `title` NOT NULL default "New chat", `updated_at` = `created_at` lúc tạo (design §3.2)
- `MAX_TURNS = 6` không đổi `rag_core/models.py:5`
- `Database` phải thread-safe per-op `_connect()` `check_same_thread=False` như cũ `rag_core/db.py:42`
- `mypy --strict` phải pass cho `rag_core/models.py`, `rag_core/db.py`, `app.py` (pyproject.toml:29)
- Không đổi `RagCore`, `Index`, `Generator`, không theme/đổi password (design §7 Out of Scope)

---

## File Structure

**Modified files:**

- `rag_core/models.py:40-58` — Thêm `User.display_name`, `User.language`, mới `ConversationMeta`. Owner of types.
- `rag_core/db.py:49-76` — Sửa `_create_schema`, `_migrate` (thêm cột + recreate bỏ UNIQUE + backfill + index), sửa `register`/`login`, thêm `get_user`, `update_user_profile`, `create_conversation`, `list_conversations`, `update_conversation_title`, `delete_conversation`, `clear_all_conversations`, sửa `append_exchange` (updated_at + auto title), giữ `get_or_create_session` wrapper.
- `app.py:120-148` — Thêm `render_sidebar`, `render_profile`, `get_active_conversation_id`, sửa `render_chat` nhận `session_id`, cập nhật `main()` để điều hướng sidebar/profile/chat.

**Test files:**

- `tests/test_db.py:1-215` — Bổ sung 10+ tests mới cho profile + multi-chat + migrate (giữ các test cũ pass).
- `tests/test_db_profile_chat.py` (hoặc append vào test_db.py) — optional tách, nhưng plan sẽ append vào `tests/test_db.py` để giữ single file.

**New files:** không cần file mới (chỉ modify).

---

### Task 1: Models + DB schema migrate (User profile columns + conversations N chats)

**Files:**
- Modify: `rag_core/models.py:40-58`
- Modify: `rag_core/db.py:49-97`
- Test: `tests/test_db.py` (append)

**Interfaces:**
- Consumes: `Path` for DB, existing `sqlite3`, `json`
- Produces:
  ```python
  @dataclass(frozen=True) class User: id: int; username: str; display_name: str; language: str  # new fields, default in DB but required in Python
  @dataclass(frozen=True) class ConversationMeta: id: str; user_id: int; title: str; created_at: str; updated_at: str; preview: str
  # DB schema new columns: users.display_name TEXT NOT NULL DEFAULT '', users.language TEXT NOT NULL DEFAULT 'vi', conversations.title TEXT NOT NULL DEFAULT 'New chat', conversations.updated_at TEXT NOT NULL
  # DB._migrate() will handle legacy DB without these columns and with UNIQUE(user_id)
  ```
  Task 2 consumes `ConversationMeta` and `User` new fields.

- [ ] **Step 1: Write failing test for User new fields**

  Append to `tests/test_db.py` (or create `tests/test_db_profile_chat.py` if prefer split — but append here for simplicity):

  ```python
  def test_register_sets_display_name_and_language(tmp_path: Path) -> None:
      from rag_core.db import Database
      db = Database(tmp_path / "test.db")
      user = db.register("hieu", "pw")
      assert user.display_name == "hieu"
      assert user.language == "vi"
      # login also returns new fields
      logged = db.login("hieu", "pw")
      assert logged is not None
      assert logged.display_name == "hieu"
      assert logged.language == "vi"

  def test_get_user_returns_profile(tmp_path: Path) -> None:
      from rag_core.db import Database
      db = Database(tmp_path / "test.db")
      u = db.register("alice", "pw")
      fetched = db.get_user(u.id)
      assert fetched is not None
      assert fetched.username == "alice"
      assert fetched.display_name == "alice"
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/test_db.py::test_register_sets_display_name_and_language -v`
  Expected: FAIL `AttributeError: 'User' object has no attribute 'display_name'` or `TypeError: User.__init__() missing...`

- [ ] **Step 3: Implement minimal models change**

  Edit `rag_core/models.py:40-43`:

  ```python
  @dataclass(frozen=True)
  class User:
      id: int
      username: str
      display_name: str  # mới
      language: str      # mới, "vi" | "en"

  @dataclass(frozen=True)
  class ConversationMeta:
      id: str
      user_id: int
      title: str
      created_at: str
      updated_at: str
      preview: str
  ```

  Note: keep `User` frozen, order `id, username, display_name, language` to match DB SELECT order. Update `__all__` if needed (no __all__ in models.py).

- [ ] **Step 4: Run test to verify it still fails (DB not yet migrated)**

  Run: `uv run pytest tests/test_db.py::test_register_sets_display_name_and_language -v`
  Expected: still FAIL `sqlite3.OperationalError: table users has no column named display_name` — proves model change alone insufficient.

- [ ] **Step 5: Implement DB schema + migrate minimal**

  Edit `rag_core/db.py`:

  1. `_create_schema:50` — change users DDL to include `display_name TEXT NOT NULL DEFAULT ''` and `language TEXT NOT NULL DEFAULT 'vi'`, conversations DDL to `user_id INTEGER NOT NULL REFERENCES users(id)` (no UNIQUE), `title TEXT NOT NULL DEFAULT 'New chat'`, `updated_at TEXT NOT NULL`, and add `CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);` at end of executescript.
  2. `_migrate:77` — add loops:
     ```python
     # users columns
     user_cols = {r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()}
     if "display_name" not in user_cols:
         conn.execute("ALTER TABLE users ADD COLUMN display_name TEXT NOT NULL DEFAULT ''")
         conn.execute("UPDATE users SET display_name = username WHERE display_name = ''")
     if "language" not in user_cols:
         conn.execute("ALTER TABLE users ADD COLUMN language TEXT NOT NULL DEFAULT 'vi'")
     # conversations columns
     conv_cols = {r["name"] for r in conn.execute("PRAGMA table_info(conversations)").fetchall()}
     if "title" not in conv_cols:
         conn.execute("ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT 'New chat'")
     if "updated_at" not in conv_cols:
         conn.execute("ALTER TABLE conversations ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")
         conn.execute("UPDATE conversations SET updated_at = created_at WHERE updated_at = ''")
     # handle UNIQUE on user_id: detect via PRAGMA index_list
     for idx in conn.execute("PRAGMA index_list('conversations')").fetchall():
         if idx["unique"] == 1:
             # check if this index covers user_id
             info = conn.execute(f"PRAGMA index_info('{idx['name']}')").fetchall()
             if any(r["name"] == "user_id" for r in info):
                 # recreate table without UNIQUE
                 conn.executescript("""
                 CREATE TABLE conversations_new (
                     id TEXT PRIMARY KEY,
                     user_id INTEGER NOT NULL REFERENCES users(id),
                     title TEXT NOT NULL DEFAULT 'New chat',
                     created_at TEXT NOT NULL,
                     updated_at TEXT NOT NULL
                 );
                 INSERT INTO conversations_new (id, user_id, title, created_at, updated_at)
                     SELECT id, user_id, COALESCE(title, 'New chat'), created_at, COALESCE(updated_at, created_at) FROM conversations;
                 DROP TABLE conversations;
                 ALTER TABLE conversations_new RENAME TO conversations;
                 CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC);
                 """)
                 break
     conn.commit()
     ```
  3. `register:104` — change to `INSERT INTO users (username, password, display_name, language, created_at) VALUES (?, ?, ?, ?, ?)` with `(username, password, username, "vi", self._now())` and return `User(id=lastrowid, username=username, display_name=username, language="vi")`.
  4. `login:118` — change SELECT to `SELECT id, username, display_name, language FROM users WHERE ...` and return `User(id=row["id"], username=row["username"], display_name=row["display_name"] or row["username"], language=row["language"] or "vi")`.
  5. Add `def get_user(self, user_id: int) -> User | None:` — `SELECT id, username, display_name, language FROM users WHERE id=?`.

- [ ] **Step 6: Run test to verify it passes**

  Run: `uv run pytest tests/test_db.py::test_register_sets_display_name_and_language tests/test_db.py::test_get_user_returns_profile -v && uv run mypy rag_core/models.py rag_core/db.py --strict`
  Expected: 2 PASS, mypy clean (fix `row["display_name"]` type with `str(row["display_name"] or row["username"])`).

- [ ] **Step 7: Test legacy migration (DB cũ 1-chat UNIQUE)**

  Append test:

  ```python
  def test_legacy_db_migrates_users_and_conversations(tmp_path: Path) -> None:
      import sqlite3
      path = tmp_path / "legacy.db"
      conn = sqlite3.connect(str(path))
      conn.executescript("""
      CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password TEXT NOT NULL, created_at TEXT NOT NULL);
      CREATE TABLE conversations (id TEXT PRIMARY KEY, user_id INTEGER UNIQUE NOT NULL REFERENCES users(id), created_at TEXT NOT NULL);
      CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL REFERENCES conversations(id), role TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL);
      """)
      conn.execute("INSERT INTO users (id, username, password, created_at) VALUES (1, 'hieu', 'pw', 'now')")
      conn.execute("INSERT INTO conversations (id, user_id, created_at) VALUES ('c1', 1, 'now')")
      conn.commit(); conn.close()
      from rag_core.db import Database
      db = Database(path)
      u = db.login("hieu", "pw")
      assert u is not None and u.display_name == "hieu" and u.language == "vi"
      # after migrate, should allow second conversation
      s2 = db.create_conversation(u.id, title="Second")
      assert s2.id != "c1"
      assert len(db.list_conversations(u.id)) == 2
  ```

  Run: `uv run pytest tests/test_db.py::test_legacy_db_migrates_users_and_conversations -v`
  Expected: PASS (if fail, fix _migrate recreate logic).

- [ ] **Step 8: Commit**

  ```bash
  git add rag_core/models.py rag_core/db.py tests/test_db.py
  git commit -m "feat(db): add display_name/language to User + migrate conversations to N chats (drop UNIQUE)"
  ```

---

### Task 2: DB multi-conversation APIs (create/list/update/delete/clear + title auto + updated_at)

**Files:**
- Modify: `rag_core/db.py:128-217`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `ConversationMeta` from Task 1, `Session`, `Turn`, `Citation`, `Source`, `_now()`
- Produces:
  ```python
  def create_conversation(self, user_id: int, title: str = "New chat") -> Session
  def list_conversations(self, user_id: int) -> list[ConversationMeta]  # ORDER BY updated_at DESC
  def update_conversation_title(self, session_id: str, title: str) -> None  # strip, 1..50 else ValueError
  def delete_conversation(self, session_id: str) -> None
  def clear_all_conversations(self, user_id: int) -> None
  def update_user_profile(self, user_id: int, display_name: str | None = None, language: str | None = None) -> User  # validates
  # modified
  def append_exchange(self, session_id: str, user_text: str, assistant_text: str, ...) -> None  # now also UPDATE updated_at + auto title if first exchange
  def get_or_create_session(self, user_id: int) -> Session  # wrapper kept
  ```

- [ ] **Step 1: Write failing tests for new APIs**

  Append:

  ```python
  def test_create_multiple_conversations_per_user(tmp_path: Path) -> None:
      from rag_core.db import Database
      db = Database(tmp_path / "test.db")
      u = db.register("alice", "pw")
      c1 = db.create_conversation(u.id, title="Chat 1")
      c2 = db.create_conversation(u.id, title="Chat 2")
      assert c1.id != c2.id
      metas = db.list_conversations(u.id)
      assert len(metas) == 2
      assert {m.title for m in metas} == {"Chat 1", "Chat 2"}

  def test_list_conversations_ordered_by_updated_at(tmp_path: Path) -> None:
      from rag_core.db import Database
      import time
      db = Database(tmp_path / "test.db")
      u = db.register("bob", "pw")
      c1 = db.create_conversation(u.id, title="Old")
      time.sleep(0.01)
      c2 = db.create_conversation(u.id, title="New")
      # New should be first (DESC)
      metas = db.list_conversations(u.id)
      assert metas[0].id == c2.id
      # after appending to Old, it becomes first
      db.append_exchange(c1.id, "hi", "hello")
      metas2 = db.list_conversations(u.id)
      assert metas2[0].id == c1.id

  def test_append_exchange_auto_titles_first_message(tmp_path: Path) -> None:
      from rag_core.db import Database
      db = Database(tmp_path / "test.db")
      u = db.register("hieu", "pw")
      s = db.create_conversation(u.id)  # default "New chat"
      assert db.list_conversations(u.id)[0].title == "New chat"
      db.append_exchange(s.id, "giải thích bảng băm là gì?", "answer")
      assert db.list_conversations(u.id)[0].title == "giải thích bảng băm là gì?"
      # second exchange should NOT overwrite title
      db.append_exchange(s.id, "câu 2", "ans2")
      assert db.list_conversations(u.id)[0].title == "giải thích bảng băm là gì?"

  def test_append_truncates_title_40(tmp_path: Path) -> None:
      from rag_core.db import Database
      db = Database(tmp_path / "test.db")
      u = db.register("hieu", "pw")
      s = db.create_conversation(u.id)
      long_text = "a" * 100
      db.append_exchange(s.id, long_text, "ans")
      assert len(db.list_conversations(u.id)[0].title) == 40

  def test_update_user_profile(tmp_path: Path) -> None:
      from rag_core.db import Database
      db = Database(tmp_path / "test.db")
      u = db.register("hieu", "pw")
      updated = db.update_user_profile(u.id, display_name="Hiếu Nguyễn", language="en")
      assert updated.display_name == "Hiếu Nguyễn"
      assert updated.language == "en"
      # persist
      assert db.login("hieu", "pw").language == "en"

  def test_update_user_profile_validation(tmp_path: Path) -> None:
      from rag_core.db import Database
      import pytest
      db = Database(tmp_path / "test.db")
      u = db.register("hieu", "pw")
      with pytest.raises(ValueError):
          db.update_user_profile(u.id, display_name="")
      with pytest.raises(ValueError):
          db.update_user_profile(u.id, display_name="a"*51)
      with pytest.raises(ValueError):
          db.update_user_profile(u.id, language="fr")

  def test_delete_and_clear(tmp_path: Path) -> None:
      from rag_core.db import Database
      db = Database(tmp_path / "test.db")
      u = db.register("hieu", "pw")
      c1 = db.create_conversation(u.id, title="A")
      c2 = db.create_conversation(u.id, title="B")
      db.append_exchange(c1.id, "hi", "hello")
      db.delete_conversation(c1.id)
      assert len(db.list_conversations(u.id)) == 1
      assert db.list_conversations(u.id)[0].id == c2.id
      db.clear_all_conversations(u.id)
      assert db.list_conversations(u.id) == []

  def test_get_or_create_session_backward_compat(tmp_path: Path) -> None:
      from rag_core.db import Database
      db = Database(tmp_path / "test.db")
      u = db.register("hieu", "pw")
      s1 = db.get_or_create_session(u.id)
      s2 = db.get_or_create_session(u.id)
      assert s1.id == s2.id
      # after creating extra, wrapper returns most recent (first in DESC)
      s3 = db.create_conversation(u.id, title="Extra")
      s4 = db.get_or_create_session(u.id)
      assert s4.id == s3.id  # most recent
  ```

- [ ] **Step 2: Run to verify fails**

  Run: `uv run pytest tests/test_db.py::test_create_multiple_conversations_per_user -v`
  Expected: FAIL `AttributeError: 'Database' object has no attribute 'create_conversation'`

- [ ] **Step 3: Implement DB APIs minimal**

  In `rag_core/db.py` after `get_session` add:

  ```python
  def get_user(self, user_id: int) -> User | None:
      with self._connect() as conn:
          row = conn.execute("SELECT id, username, display_name, language FROM users WHERE id=?", (user_id,)).fetchone()
          if row is None: return None
          return User(id=row["id"], username=row["username"], display_name=row["display_name"] or row["username"], language=row["language"] or "vi")

  def update_user_profile(self, user_id: int, display_name: str | None = None, language: str | None = None) -> User:
      if display_name is not None:
          if not (1 <= len(display_name.strip()) <= 50):
              raise ValueError("display_name must be 1..50 chars")
          display_name = display_name.strip()
      if language is not None and language not in ("vi", "en"):
          raise ValueError("language must be 'vi' or 'en'")
      with self._connect() as conn:
          if display_name is not None:
              conn.execute("UPDATE users SET display_name=? WHERE id=?", (display_name, user_id))
          if language is not None:
              conn.execute("UPDATE users SET language=? WHERE id=?", (language, user_id))
          conn.commit()
      user = self.get_user(user_id)
      assert user is not None
      return user

  def create_conversation(self, user_id: int, title: str = "New chat") -> Session:
      session_id = uuid.uuid4().hex
      now = self._now()
      title = title.strip()[:50] or "New chat"
      with self._connect() as conn:
          conn.execute("INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", (session_id, user_id, title, now, now))
          conn.commit()
      return Session(id=session_id, user_id=str(user_id), turns=[])

  def list_conversations(self, user_id: int) -> list[ConversationMeta]:
      with self._connect() as conn:
          rows = conn.execute("""
              SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at,
                     (SELECT text FROM messages WHERE conversation_id=c.id AND role='user' ORDER BY id ASC LIMIT 1) AS preview
              FROM conversations c WHERE c.user_id=? ORDER BY c.updated_at DESC
          """, (user_id,)).fetchall()
          return [ConversationMeta(id=r["id"], user_id=r["user_id"], title=r["title"], created_at=r["created_at"], updated_at=r["updated_at"], preview=(r["preview"] or "")[:40]) for r in rows]

  def update_conversation_title(self, session_id: str, title: str) -> None:
      title = title.strip()
      if not (1 <= len(title) <= 50):
          raise ValueError("title must be 1..50 chars")
      with self._connect() as conn:
          cur = conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (title[:50], self._now(), session_id))
          if cur.rowcount == 0:
              raise KeyError(f"no such conversation: {session_id}")
          conn.commit()

  def delete_conversation(self, session_id: str) -> None:
      with self._connect() as conn:
          conn.execute("DELETE FROM messages WHERE conversation_id=?", (session_id,))
          cur = conn.execute("DELETE FROM conversations WHERE id=?", (session_id,))
          if cur.rowcount == 0:
              raise KeyError(f"no such conversation: {session_id}")
          conn.commit()

  def clear_all_conversations(self, user_id: int) -> None:
      with self._connect() as conn:
          conn.execute("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id=?)", (user_id,))
          conn.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
          conn.commit()
  ```

  Modify `append_exchange:186` — after `conn.commit()` for messages, add:
  ```python
  # update updated_at and auto title
  now = self._now()
  # check if first exchange: count messages ==2
  cnt = conn.execute("SELECT COUNT(*) AS c FROM messages WHERE conversation_id=?", (session_id,)).fetchone()["c"]
  if cnt == 2:
      # first exchange, title is still "New chat" -> set to user_text[:40]
      cur_title = conn.execute("SELECT title FROM conversations WHERE id=?", (session_id,)).fetchone()
      if cur_title and cur_title["title"] == "New chat":
          new_title = user_text.strip()[:40] or "New chat"
          conn.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (new_title, now, session_id))
      else:
          conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, session_id))
  else:
      conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, session_id))
  conn.commit()
  ```
  Need to move commit to after this. Ensure `_connect` context still open, do both in same `with`.

  Modify `get_or_create_session:128` to:
  ```python
  def get_or_create_session(self, user_id: int) -> Session:
      metas = self.list_conversations(user_id)
      if metas:
          return self.get_session(metas[0].id)
      return self.create_conversation(user_id, title="New chat")
  ```

- [ ] **Step 4: Run tests to verify passes**

  Run: `uv run pytest tests/test_db.py -k "test_create_multiple or test_list_conversations or test_append_exchange or test_update_user or test_delete or test_get_or_create" -v && uv run mypy rag_core/db.py --strict`
  Expected: 8 PASS, mypy clean (handle `uuid` import already exists).

- [ ] **Step 5: Run full DB suite regression**

  Run: `uv run pytest tests/test_db.py -v`
  Expected: all 22+ tests PASS (including original 12).

- [ ] **Step 6: Commit**

  ```bash
  git add rag_core/db.py tests/test_db.py
  git commit -m "feat(db): add multi-chat APIs, auto-title, updated_at, profile update"
  ```

---

### Task 3: App sidebar — list chats + tạo chat mới + chọn active

**Files:**
- Modify: `app.py:120-166`
- Test: manual + `tests/test_db.py` already covers DB, no new unit needed for Streamlit (smoke only)

**Interfaces:**
- Consumes: `Database.list_conversations`, `Database.create_conversation`, `Database.get_session`, `st.session_state["active_conversation_id"]`, `st.session_state["user"]`
- Produces:
  ```python
  def get_active_conversation_id(user: User, db: Database) -> str  # ensures at least one conversation, returns active id
  def render_sidebar(user: User) -> None  # renders st.sidebar with button + list
  def render_chat(user: User) -> None  # now uses active_id instead of get_or_create_session(user.id)
  ```

- [ ] **Step 1: Write failing smoke check (optional manual)**

  Create scratch test `tests/test_app_sidebar_smoke.py` (will be deleted) or just manual:

  ```python
  # check that app imports and get_active_conversation_id exists
  from app import get_active_conversation_id
  assert callable(get_active_conversation_id)
  ```

  Run: `uv run pytest tests/test_app_sidebar_smoke.py -v`
  Expected: FAIL `ImportError: cannot import name 'get_active_conversation_id'`.

- [ ] **Step 2: Implement sidebar + active logic minimal**

  Edit `app.py`:

  1. Add helper after `get_user:38`:
     ```python
     def get_active_conversation_id(user: User, db: Database) -> str:
         active = st.session_state.get("active_conversation_id")
         if active:
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
     ```
  2. Add `render_sidebar(user: User)`:
     ```python
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
                 label = f"{'▶ ' if is_active else ''}{meta.title[:35]}"
                 if st.button(label, key=f"chat_{meta.id}", use_container_width=True):
                     st.session_state["active_conversation_id"] = meta.id
                     st.session_state["show_profile"] = False
                     st.rerun()
                 st.caption(f"{meta.updated_at[:16]}  {meta.preview[:30]}")
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
     ```
  3. Modify `render_chat(user: User) -> None` signature to `render_chat(user: User, session_id: str)` or keep but inside call `get_active_conversation_id`:
     ```python
     def render_chat(user: User) -> None:
         db = get_database()
         session_id = get_active_conversation_id(user, db)
         session = db.get_session(session_id)
         # existing header but remove old "Logged in as" duplication — keep st.title
         st.title("CourseMate")
         # remove old Log out button (now in sidebar)
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
     ```
  4. Modify `main:150` to:
     ```python
     def main() -> None:
         st.set_page_config(page_title="CourseMate", layout="wide")
         try:
             load_config()
         except ValueError as exc:
             st.error(str(exc)); st.stop()
         user = get_user()
         if user is None:
             render_auth(); st.stop()
         # refresh user with display_name/language from DB (in case updated)
         db = get_database()
         refreshed = db.get_user(user.id)
         if refreshed:
             st.session_state["user"] = refreshed
             user = refreshed
         render_sidebar(user)
         if st.session_state.get("show_profile"):
             render_profile(user)
         else:
             render_chat(user)
     ```
     Need to implement `render_profile` stub for now (empty) to avoid NameError:
     ```python
     def render_profile(user: User) -> None:
         st.title("Profile / Setting")
         st.write(f"Tên hiển thị: {user.display_name}")
         if st.button("Quay lại chat"):
             st.session_state["show_profile"] = False
             st.rerun()
     ```

- [ ] **Step 3: Run smoke check**

  Run: `uv run python -c "from app import get_active_conversation_id, render_sidebar; print('ok')"`
  Expected: prints ok, mypy `uv run mypy app.py --strict` clean (add `from rag_core.models import User` import).

- [ ] **Step 4: Manual Streamlit smoke (no commit yet)**

  Run: `uv run streamlit run app.py -- --help` or `uv run python -m py_compile app.py`
  Expected: compiles, no ImportError.

- [ ] **Step 5: Commit**

  ```bash
  git add app.py
  git commit -m "feat(ui): add sidebar with chat history, new chat, active selection"
  ```

---

### Task 4: App per-chat actions — đổi tên / xóa + title auto integration

**Files:**
- Modify: `app.py:render_sidebar` (add popover)
- Modify: `rag_core/db.py:append_exchange` already done, verify integration with UI rerun

**Interfaces:**
- Consumes: `Database.update_conversation_title`, `Database.delete_conversation`
- Produces: sidebar popover actions

- [ ] **Step 1: Extend sidebar with popover for rename/delete**

  Edit `render_sidebar` loop:

  ```python
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
  ```

  Note: Streamlit `st.popover` available since 1.32, if not, fallback to `st.expander`.

- [ ] **Step 2: Verify title auto still works after UI change**

  Manual test via DB:

  Run: `uv run pytest tests/test_db.py::test_append_exchange_auto_titles_first_message -v`
  Expected: PASS (Task 2 already).

- [ ] **Step 3: Test delete active conversation handling**

  Add test to `tests/test_db.py` if not already (but logic in UI: after delete, get_active... will auto-create or pick most recent). Verify DB allows delete then list.

  Run: `uv run pytest tests/test_db.py::test_delete_and_clear -v`
  Expected: PASS

- [ ] **Step 4: Run mypy**

  Run: `uv run mypy app.py --strict`
  Expected: PASS (popover may need `# type: ignore` if stubs missing).

- [ ] **Step 5: Commit**

  ```bash
  git add app.py
  git commit -m "feat(ui): add per-chat rename/delete popover, handle active after delete"
  ```

---

### Task 5: Profile page — display_name, language, xóa tất cả lịch sử

**Files:**
- Modify: `app.py:render_profile`
- Test: `tests/test_db.py` (profile validation already), manual Streamlit QA

**Interfaces:**
- Consumes: `Database.update_user_profile`, `Database.get_user`, `Database.clear_all_conversations`, `Database.create_conversation`
- Produces: `render_profile(user: User) -> None` full implementation

- [ ] **Step 1: Write failing test for render_profile existence (optional)**

  Run: `uv run python -c "from app import render_profile; import inspect; print(inspect.getsource(render_profile))"`
  Expected: should contain `display_name` and `language` handling, else FAIL.

- [ ] **Step 2: Implement full render_profile**

  Replace stub:

  ```python
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
  ```

- [ ] **Step 3: Run mypy + py_compile**

  Run: `uv run mypy app.py --strict && uv run python -m py_compile app.py`
  Expected: PASS

- [ ] **Step 4: Full suite regression**

  Run: `uv run pytest -q`
  Expected: all tests PASS (including Task 1-2-3).

- [ ] **Step 5: Manual QA checklist (Streamlit)**

  Steps:
  1. `uv run streamlit run app.py` → Register `hieu` → display_name = `hieu`, language `vi` mặc định → sidebar shows `👤 hieu (hieu)`.
  2. Đổi tên thành `Hiếu Nguyễn`, language `en` → Lưu → sidebar cập nhật, logout/login vẫn giữ.
  3. Tạo 3 chats, mỗi chat gửi 1 câu khác → sidebar titles auto, sorted updated_at DESC, chuyển qua lại lịch sử khôi phục (check `tests/test_db.py::test_session_persists` vẫn pass).
  4. Đổi tên chat 1 → Lưu → title đổi, preview giữ.
  5. Xóa chat đang active → active tự chuyển sang chat gần nhất.
  6. Vào Profile → gõ DELETE → Xóa tất cả → về 1 chat "New chat" trống.
  7. Kiểm tra DB cũ: tạo legacy DB với UNIQUE, mở app → migrate không lỗi.

- [ ] **Step 6: Commit**

  ```bash
  git add app.py
  git commit -m "feat(ui): implement full profile page (display_name, language, clear all)"
  ```

---

## Self-Review

**Spec coverage check:**

- §1 Goal profile display_name + language + xóa tất cả → Task 1 (model), Task 2 (update_user_profile/clear), Task 5 (UI)
- §2 Architecture sidebar + N chats + active_id + RagCore giữ → Task 3,4
- §3.1 Models ConversationMeta + User fields → Task 1
- §3.2 DB schema bỏ UNIQUE, title/updated_at, migrate recreate, index, 7 APIs, auto title, updated_at → Task 1+2
- §3.3 UI sidebar new chat, list sorted, popover rename/delete, profile page, main routing → Task 3+4+5
- §4 Data flows (register, new chat, first message title, switch, update profile, clear) → Task 5 QA
- §5 Error handling table → Task 2 validation, Task 1 migrate
- §6 Testing (10+ cases, legacy migrate, backward compat) → Task 1+2
- §7 Out of Scope (no username/password change, no theme) → Global Constraints enforce, no code added
- §8 Build Order 1→2→3→4 matches spec → Task order correct

**Placeholder scan:** No `TBD/TODO/handle edge cases` without code — all validation like `1..50`, `vi/en`, `strip()[:40]` given concretely, all `ValueError` messages shown, all `st.` calls exact.

**Type consistency:** `User(id, username, display_name, language)` order consistent in Task 1 model and Task 2 DB SELECT/return and Task 3 `get_user` refresh; `ConversationMeta` fields `id, user_id, title, created_at, updated_at, preview` consistent in `list_conversations` SELECT and UI `meta.title/preview`; `update_user_profile(display_name: str|None, language: str|None) -> User` signature used identically in Task 2 impl and Task 5 UI call; `create_conversation(user_id, title="New chat") -> Session` used in Task 3 sidebar and Task 5 clear-all.

**Fixes applied inline:** Added `get_user` early in Task 1 (needed for Task 3 refresh); clarified `append_exchange` to do `SELECT COUNT` and title check inside same connection; added fallback for `st.popover` older Streamlit via expander note.

