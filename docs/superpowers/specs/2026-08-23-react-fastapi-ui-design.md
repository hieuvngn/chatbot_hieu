# UI React (FastAPI + Vite SPA) — Design

**Date:** 2026-08-23
**Status:** Approved (brainstorming 4/4 sections)
**Scope:** Thay Streamlit (`app.py`) bằng FastAPI backend + React SPA; `rag_core` giữ nguyên
**Related:** `app.py`, `rag_core/db.py`, `rag_core/__init__.py`, `rag_core/attachments.py`

## 1. Goal

Streamlit không còn đáp ứng yêu cầu hình thức: giao diện hiện tại trông như app demo.
Cần một giao diện chuẩn "sản phẩm" theo phong cách chatbot hiện đại (ChatGPT/Claude),
vừa để demo chấm điểm vừa để dùng lâu dài.

Quyết định brainstorming đã chốt:

- **Phương án A: FastAPI + React/Vite + Tailwind v4 + shadcn/ui** (bỏ Next.js, bỏ Chainlit).
- Người dùng thành thạo React; mục đích cả demo lẫn dùng thật.
- **Không cần streaming** — giữ `answer()` trả kết quả hoàn chỉnh như hiện tại.
- Bố cục **"Chat + panel nguồn"**: header trên cùng, sidebar trái, citation dạng card bên phải câu trả lời, bo tròn lớn, nhiều khoảng trắng, màu nhấn indigo/violet, dark/light mode.
- Dữ liệu cũ giữ nguyên: SQLite `data/app.db` dùng lại y nguyên — lịch sử chat và tài khoản không mất.

## 2. Architecture

```
btl_ttnt_hieu/
├── rag_core/            # GIỮ NGUYÊN — không đổi gì
├── server/              # MỚI — FastAPI app
│   ├── main.py          # app instance, CORS, mount static khi build
│   ├── auth.py          # token-based session (Bearer)
│   ├── schemas.py       # Pydantic request/response models
│   └── routes.py        # các endpoint
├── web/                 # MỚI — React SPA (Vite + TS + Tailwind v4 + shadcn/ui)
│   └── src/
│       ├── lib/api.ts   # fetch client, token handling
│       ├── pages/       # Login, Chat, Settings
│       └── components/  # Sidebar, Header, ChatMessage, CitationCard, ...
├── app.py               # Streamlit cũ — giữ trong giai đoạn chuyển tiếp, xóa ở bước cuối
└── pyproject.toml       # thêm fastapi, uvicorn; script `uv run serve`
```

Luồng: React gọi FastAPI qua REST (`/api/*`) → FastAPI dùng trực tiếp `Database`
và `RagCore.answer()` hiện có → SQLite `data/app.db` nguyên trạng.
Khi deploy/demo, FastAPI phục vụ bản build của React — chạy bằng một lệnh duy nhất
`uv run serve`.

Dev mode chạy 2 tiến trình: `uv run uvicorn server.main:app --reload` (cổng 8000)
và `npm run dev` trong `web/` (Vite proxy `/api` về cổng 8000).

### Auth

- Register/login trả về token ngẫu nhiên (`secrets.token_hex(32)`), lưu bảng
  `tokens` mới trong SQLite. Migration nhẹ nằm trong `server/` (CREATE TABLE IF NOT EXISTS),
  **không sửa** `rag_core/db.py`.
- Frontend lưu token ở localStorage, gửi `Authorization: Bearer <token>` mỗi request.
- Restart server không làm mất đăng nhập.
- Mật khẩu vẫn plaintext theo lựa chọn demo hiện có của dự án.

## 3. API

Tất cả endpoint prefix `/api`. Lỗi domain map: `ValueError` → 400, `KeyError` → 404,
sai/thiếu token → 401.

| Method | Path | Chức năng |
|---|---|---|
| POST | `/auth/register` | tạo user → token + user |
| POST | `/auth/login` | → token + user |
| GET | `/auth/me` | user hiện tại |
| PATCH | `/users/me` | đổi display_name / language |
| GET / POST | `/conversations` | danh sách metas / tạo chat mới |
| PATCH / DELETE | `/conversations/{id}` | đổi tên / xóa |
| DELETE | `/conversations` | xóa tất cả lịch sử |
| GET | `/conversations/{id}/messages` | toàn bộ turns (kèm citations, refusal, skills) |
| POST | `/conversations/{id}/chat` | `{message, use_web}` → blocking, trả AnswerResult đầy đủ |
| GET / POST | `/attachments?conversation_id=` | list / upload multipart cho conversation |
| DELETE | `/attachments/{id}` | xóa attachment |
| GET | `/features` | `{has_web_search}` |

Quy tắc sở hữu: mọi endpoint theo conversation/attachment kiểm tra `user_id`
khớp user của token — nếu không phải của mình trả **404** (không lộ sự tồn tại).
POST `/conversations/{id}/chat` gọi `core.answer(message, session, use_web=...)`
rồi `db.append_exchange(...)` như `app.py` đang làm.

## 4. Giao diện

### Bố cục 3 vùng (đã đăng nhập)

- **Header mỏng trên cùng:** trái — logo CourseMate; giữa — tên chat đang mở;
  phải — công tắc dark/light + menu avatar (Settings · Log out).
- **Sidebar trái (~280px, collapsible):**
  - Nút "＋ New chat"
  - Danh sách lịch sử chat (active highlight; hover → ⋯ đổi tên/xóa)
  - Mục 📎 Tài liệu đính kèm của chat đang mở (upload PDF/TXT/MD, xem số đoạn/KB, xóa; giới hạn `MAX_FILES_PER_CONVERSATION`)
  - Toggle 🌐 tìm kiếm web (ẩn hẳn khi `/api/features` trả `has_web_search: false`)
- **Vùng chat chính:** cột nội dung max-width ~720px lệch nhẹ sang trái.
  - Empty state: lời chào kèm 3–4 gợi ý câu hỏi mẫu bấm được
  - User: bong bóng nền indigo nhạt bo lớn bên phải
  - Assistant: full-width, Markdown đầy đủ (remark-gfm: bảng, list), code highlight + nút copy code
  - **Citations: chồng card dọc bên phải từng câu trả lời** — mỗi card: marker `[n]`,
    tên tài liệu, chapter, badge slides/textbook/web/upload; click mở dialog chi tiết
    nguồn; citation web mở link trực tiếp. Màn hẹp (< lg) card rơi xuống dưới câu trả lời.
  - Refusal: card màu amber kèm rephrase suggestion; fallback (no citations) có notice riêng
  - Badge nhỏ liệt kê skills_applied
  - Input: textarea tự giãn đáy trang, Enter gửi / Shift+Enter xuống dòng;
    trạng thái chờ là typing indicator

### Trang Settings

Display name (1–50 ký tự), ngôn ngữ vi/en, danger zone xóa toàn bộ lịch sử
(confirm bắt gõ `DELETE`), quay lại chat.

### Phong cách & kỹ thuật

- Bo tròn lớn (`rounded-2xl`), nhiều khoảng trắng, nền neutral zinc,
  màu nhấn **indigo/violet**, font Inter, shadow rất nhẹ.
- Dark/light: theo hệ thống + công tắc thủ công (lưu localStorage).
- `react-router-dom`: `/login`, `/`, `/settings`; route bảo vệ redirect về `/login` khi chưa có token.
- TanStack Query quản lý server state; invalidate danh sách conversations sau send/rename/delete.
- `react-markdown` + `remark-gfm` + highlight code; icons `lucide-react`.

## 5. Error handling

- FastAPI handler chuyển `ValueError` → 400 với message gốc, `KeyError` → 404,
  thiếu/sai token → 401; lỗi không lường trước → 500 JSON thống nhất `{detail}`.
- Frontend: toast/lỗi inline cho form; khi POST chat lỗi mạng/500 → hiển thị
  thông báo trong vùng chat, không mất tin nhắn người dùng đang gõ.

## 6. Testing

- **FastAPI (pytest + TestClient):** luồng register/login/me; ownership
  (conversation người khác → 404); CRUD conversations; POST chat với `RagCore`
  bị mock (không gọi LLM); upload/xóa attachments. Test `rag_core` hiện có chạy nguyên trạng.
- **React:** `tsc --noEmit` + eslint + `vite build` sạch. Không viết unit test UI (YAGNI);
  nghiệm thu bằng checklist thao tác tay theo từng tính năng port từ Streamlit.

## 7. Lộ trình triển khai

Mỗi bước đều để app ở trạng thái chạy được:

1. FastAPI server + auth + tests → verify: pytest xanh
2. Skeleton React: login/register + layout header/sidebar → verify: build sạch, đăng nhập được
3. Vùng chat: gửi/nhận, markdown, card nguồn, refusal → verify: trả lời có citation hiển thị đúng
4. Sidebar đầy đủ: lịch sử CRUD, đính kèm, toggle web + trang Settings → verify: checklist tay
5. Dọn dẹp: xóa `app.py`, gỡ streamlit khỏi pyproject, thêm `uv run serve`
   (build SPA + phục vụ static từ FastAPI) → verify: một lệnh chạy được toàn app

## 8. Ngoài phạm vi

Streaming token-by-token · i18n đa ngôn ngữ cho UI (giữ vi/en qua setting language
chỉ ảnh hưởng câu trả lời) · hash mật khẩu · multi-conversation attachments · PWA/offline.
