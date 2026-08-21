# Document Upload (Attachment RAG) — Design

**Date:** 2026-08-21
**Status:** Approved (brainstorming 3/3 sections)
**Scope:** Phương án A — Attachment index riêng, tái dùng hybrid `Index`
**Related:** `CONTEXT.md`, `rag_core/__init__.py:92`, `rag_core/index.py:45`, `rag_core/chunking.py:39`, `rag_core/db.py:13`, `app.py:137`

## 1. Goal

Cho phép user tải lên tài liệu **PDF / TXT / MD** làm nguồn tri thức tạm thời cho
conversation hiện tại. Nội dung được tự động parse, chia đoạn, embedding và lưu
theo `conversation_id`. Khi chat, hệ thống truy xuất song song từ CourseMate KB
và attachment store, merge bằng RRF; câu trả lời hiển thị citation tới tài liệu
đính kèm kèm vị trí nguồn (`Trang N` cho PDF, heading cho MD, `Nội dung` cho TXT).

Giới hạn chốt: **3 file/conversation, ≤ 5 MB/file**, chỉ `pdf/txt/md`,
không thêm rerank (RRF fusion sẵn có thay thế).

Quyết định brainstorming: **SQLite — Vừa (3 file/5MB) — Bỏ qua rerank — Sidebar uploader — Approach A**.

## 2. Architecture

Giữ nguyên seam `RagCore.answer(user_message, session)`. Thêm một index thứ cấp
chỉ gồm chunk của conversation đang active, fuse ranking với KB gốc.

```
UPLOAD (app.py sidebar)
  st.file_uploader(pdf/txt/md)
    → validate (loại, ≤5MB, <3 file/conversation, không trùng filename)
    → AttachmentStore.add()
        ├─ Parser: PDF(pypdf, theo page) | MD(tách heading #–####) | TXT(nguyên văn)
        ├─ _chunk_text() mỗi section (600 token, overlap 15% — tái dùng chunking.py)
        ├─ embedder.embed_batch()          # 1 lần lúc upload
        └─ INSERT attachments + attachment_chunks   (1 transaction)

CHAT (RagCore.answer)
  query rewrite → intent classify (KNOWLEDGE_QA branch)
    → kb_ranking  = Index.fused_ranking(query, q_vec)            # KB tĩnh, như cũ
    → att_chunks  = AttachmentStore.load_chunks(session.id)
    → att_index   = cache.get_or_build(session.id, chunks + vectors từ DB)  # Index hiện có, không re-embed
    → merged      = rrf_merge(kb_ranking, att_index.fused_ranking(query, q_vec))
    → dedupe_by_source(merged, FINAL_TOP_K=5)
    → Judge/Refine → Generator → Answer-check → Citations         # không đổi

XÓA conversation → cascade xóa attachments + attachment_chunks
```

Chunk của attachment mang `Source` đầy đủ nên toàn bộ downstream
(Judge, Generator, Answer-check, `parse_citations`, persistence citations)
chạy **nguyên trạng, không sửa gì**.

## 3. Data model (SQLite)

Thêm vào schema `Database._create_schema`:

```sql
CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,              -- uuid hex
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    filename TEXT NOT NULL,
    file_kind TEXT NOT NULL,          -- 'pdf' | 'txt' | 'md'
    size_bytes INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attachment_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    attachment_id TEXT NOT NULL REFERENCES attachments(id),
    conversation_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    chapter TEXT NOT NULL,            -- 'Trang N' | heading | 'Nội dung'
    text TEXT NOT NULL,
    embedding BLOB NOT NULL           -- float32 little-endian
);
CREATE INDEX IF NOT EXISTS idx_att_chunks_conv ON attachment_chunks(conversation_id);
```

- Upload là **1 transaction**: lỗi giữa chừng (embedding fail…) → rollback, DB giữ nguyên.
- `delete_conversation()` và `clear_all_conversations()` xóa thêm 2 bảng này trong cùng transaction.
- Vector lưu BLOB để restart app **không phải re-embed**.

## 4. Components

### 4.1 `rag_core/attachments.py` — `AttachmentStore`

| API | Hành vi |
|---|---|
| `add(conversation_id, filename, data_bytes)` | validate → parse → chunk → embed → persist. Trả `AttachmentMeta`. Raise `ValueError` khi vi phạm giới hạn/trùng tên, `RuntimeError` khi parse ra text rỗng. |
| `list_for(conversation_id)` | danh sách `AttachmentMeta(filename, file_kind, size_bytes, chunk_count, created_at)` |
| `delete(attachment_id)` | xóa metadata + chunks |
| `load_chunks(conversation_id)` | `list[Chunk]` với `Source(kind="upload", chapter=…)` |
| `load_vectors(conversation_id)` | `np.ndarray` float32 (n, dim) — vector BLOB đọc từ DB, thứ tự khớp `load_chunks` |

Parser nội bộ trong cùng module:

| Loại | Parse | Chapter |
|---|---|---|
| PDF | `pypdf` (dependency mới, thuần Python), extract từng page | `"Trang {n}"` |
| MD | decode UTF-8, tách section theo heading `#`–`####` (regex) | tiêu đề heading; không có heading → `"Nội dung"` |
| TXT | decode UTF-8 nguyên văn | `"Nội dung"` |

Mỗi section/page đi qua `_chunk_text()` (`chunking.py:39`) — không viết chunker mới.

### 4.2 `Source` mapping (citation chạy nguyên trạng)

```python
Source(
    document_id=f"upload_{uuid4().hex[:8]}",
    document_title=filename,
    chapter="Trang 3" | heading_title | "Nội dung",
    course_code="",
    kind="upload",
    language="vi",   # theo user.language của conversation owner
)
```

### 4.3 `RagCore` thay đổi tối thiểu

- Constructor nhận thêm `attachment_store: AttachmentStore | None = None`.
- `_rrf_fusion` nâng từ method tĩnh thành hàm module-level để tái dùng cho merge 2 tầng.
- `Index.__init__(chunks, embedder)` nhận thêm tham số tùy chọn `vectors: np.ndarray | None`
  — khi cung cấp, bỏ qua `embed_batch` và dùng vector có sẵn (BM25 vẫn build từ text).
  Đây là đường để attachment index rebuild sau restart **không phải re-embed**.
- Trong `answer()` nhánh KNOWLEDGE_QA: nếu có attachment chunks cho `session.id`
  → build/lấy `Index` từ cache `{conversation_id: Index}` (invalidate khi số chunk đổi),
    truyền `load_chunks()` + `load_vectors()` vào constructor
  → hai ranking (KB là list chỉ số; attachment tương tự) được quy đổi về chunk objects,
    RRF-fuse với nhau trước `dedupe_by_source`.
- Cache sống cùng vòng đời `RagCore` (không qua Streamlit cache).
- Build attachment index fail lúc chat → log warning, degrade về chỉ search KB (không crash).
- Intent classifier prompt: bổ sung ví dụ để câu hỏi kiểu "tài liệu này/tệp đính kèm"
  route vào KNOWLEDGE_QA thay vì OTHER.

### 4.4 UI (`app.py` sidebar)

Section **"📎 Tài liệu đính kèm (n/3)"** dưới lịch sử chat, áp dụng cho chat active:

- Danh sách file đã upload: icon theo loại, filename, `N đoạn · size`, nút xóa
  (popover confirm — theo pattern rename/delete chat hiện có).
- `st.file_uploader(type=["pdf","txt","md"])`; spinner "Đang parse và embedding…";
  success hiện số chunk; lỗi hiện `st.error` cụ thể.
- Đủ 3 file → ẩn uploader, caption "Đã đạt giới hạn 3 tài liệu".
- Citation expander của attachment hiển thị thêm dòng "Nguồn: tài liệu đính kèm".

## 5. Error handling

| Trường hợp | Xử lý |
|---|---|
| File > 5MB / sai định dạng | chặn trước khi parse, thông báo rõ |
| PDF scanned (không có text layer) | từ chối, gợi ý dùng bản có text layer |
| Lỗi embedding API giữa chừng | rollback transaction, trạng thái cũ nguyên vẹn |
| Trùng filename trong cùng conversation | từ chối với thông báo |
| Build attachment index fail lúc chat | degrade về chỉ search KB |
| Xóa conversation | cascade xóa trong cùng transaction |

Refusal/fallback hành vi như KB thường: judge low sau Refine → refuse kèm rephrase suggestion.

## 6. Testing

1. **Parsers**: TXT/MD tách heading đúng; MD không heading → 1 section "Nội dung";
   PDF fixture nhỏ (tạo bằng pypdf trong test) → đúng số trang/chapter "Trang N".
2. **Chunking bridge**: section dài → nhiều chunk mang đúng chapter metadata.
3. **DB**: add/list/delete; cascade qua `delete_conversation` + `clear_all_conversations`;
   trùng filename raise `ValueError`; rollback khi embed fail (fake embedder raise);
   `load_vectors` round-trip đúng thứ tự và giá trị với `load_chunks`.
4. **Fusion merge**: ranking giả lập — chunk attachment liên quan lên top khi KB
   không có gì; KB thắng khi attachment không liên quan.
5. **`RagCore.answer()`** với fake embedder/generator + attachment: answer có citation
   trỏ tới `Source(kind="upload", chapter="Trang N")`.
6. **mypy strict** pass toàn bộ file mới/sửa.

## 7. Non-goals

- Không rerank (LLM hay cross-encoder).
- Không lưu file gốc trên disk — chỉ chunks + vector trong SQLite.
- Không OCR cho scanned PDF.
- Không upload cho nhiều conversation cùng lúc hay chia sẻ giữa users.
