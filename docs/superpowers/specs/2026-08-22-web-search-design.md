# Web Search qua Firecrawl — Design

**Date:** 2026-08-22
**Status:** Approved (brainstorming 2/2 sections)
**Scope:** Phương án A — `WebSearcher` component, hai điểm gắn vào pipeline (corrective tự động + toggle thủ công)
**Related:** `rag_core/__init__.py:106`, `rag_core/__init__.py:148`, `rag_core/__init__.py:319`, `rag_core/config.py:26`, `app.py:101`, `rag_core/generator.py:31`

## 1. Goal

Cho phép chatbot trả lời bằng tri thức từ web khi kho tài liệu local không đủ:

- **Corrective tự động** (đúng tinh thần paper CRAG): khi CRAG judge vẫn chấm
  medium/low sau một lần Refine → thay vì fallback/từ chối ngay, pipeline tìm
  web qua **Firecrawl Search API**, hợp nhất kết quả với chunk local, judge chấm
  lần cuối; đạt thì generate, không thì giữ nguyên fallback/refusal hiện có.
- **Toggle thủ công** trong sidebar Streamlit: khi BẬT, *mọi* câu hỏi đều được
  hợp nhất kết quả web vào retrieval trước judge; đồng thời bỏ qua refusal intent
  OTHER (cơ chế giống attachments đang có). Khi TẮT: chỉ còn đường corrective.

Nguồn web được **nhất thể hoá vào `Source`/`Citation`** với `kind="web"` —
citation `[n]`, persistence SQLite và expander UI chạy nguyên trạng.

Giới hạn chốt: **5 kết quả/lần search**, nội dung mỗi trang truncate **4000 ký tự**,
tối đa **2 lần gọi Firecrawl cho một câu trả lời**, timeout **30s**.

Quyết định brainstorming: **Corrective khi retrieval kém + option kích hoạt thủ công — Sidebar toggle, merge kết quả — Nhất thể hoá Source/Citation — Phương án A**.

## 2. Architecture

```
CHAT (RagCore.answer(msg, session, use_web=False))
  session rewrite → intent classify
    ├─ OTHER + use_web=True (+ có searcher) → rơi xuống retrieval (không refuse)   # như attachments
    └─ KNOWLEDGE_QA / COURSE_ADVISOR → như cũ (advisor không bao giờ search web)

  RETRIEVAL (_retrieve_sources)
    kb_ranked   = Index.fused_ranking(...)                       # như cũ
    att_ranked  = attachment ranking theo session                # như cũ
    web_chunks  = web_searcher.search(query)                     # chỉ khi use_web=True
    merged      = rrf_merge([kb, att, web])  → dedupe top-5

  GATED ANSWER (_gated_answer)                                   # judge + refine như cũ
    judge(chunks) high → generate
    else → Refine (rewrite + re-retrieve) → judge lại
      ├─ acceptable → generate
      └─ vẫn medium/low:
           web = web_searcher.search(refined_query)              # corrective, luôn thử nếu có searcher
           merged = dedupe(rrf_merge([refined_chunks, web]))
           judge(merged) lần cuối
             ├─ acceptable → generate (answer check vẫn chạy)
             └─ không → fallback/refusal y như hiện tại
```

Chunk nguồn web mang `Source` đầy đủ nên toàn bộ downstream — Judge,
Generator (`numbered_sources` hiển thị `(title, url)`), Answer-check,
`parse_citations`, persistence citations JSON — chạy **nguyên trạng, không sửa gì**.

Chi phí Firecrawl: toggle ON tối đa 1 call ở retrieval; corrective tối đa 1 call;
worst case (toggle ON + retrieval kém + corrective) = **2 calls/answer**.
Thiếu/không set `FIRECRAWL_API_KEY` → tính năng tự tắt âm thầm, hành vi cũ nguyên vẹn.

## 3. Components

### 3.1 `rag_core/web_search.py` (module mới)

Hằng số: `WEB_TOP_K = 5`, `WEB_MAX_CHARS = 4000`,
`FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v2"`.

```python
class WebSearcher(Protocol):
    def search(self, query: str, k: int = WEB_TOP_K) -> list[Chunk]: ...
```

`FirecrawlWebSearcher(api_key, timeout_s=30)`:

| API | Hành vi |
|---|---|
| `search(query, k)` | POST `/search` body `{"query", "limit": k, "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": true}}` bằng stdlib `urllib.request` (Authorization Bearer). Parse response chấp nhận cả 2 shape: list phẳng (`data: [...]`) lẫn nhóm theo nguồn (`data.web`). Trả `list[Chunk]`; mọi lỗi → `[]`. |
| `_post_json(path, payload)` | helper HTTP mỏng để test monkeypatch, không mock network library. |

Mapping mỗi kết quả → chunk (citation chạy nguyên trạng):

```python
Source(
    document_id=url,                 # URL đầy đủ — duy nhất
    document_title=title or url,     # thiếu title thì dùng URL
    chapter=url,                     # expander hiển thị URL, render thành link bấm được
    course_code="",
    kind="web",
    language="",                     # ngôn ngữ trang web không kiểm soát được
)
text = (markdown or description)[:WEB_MAX_CHARS]   # markdown trống → dùng description
```

### 3.2 `RagCore` (`rag_core/__init__.py`) thay đổi tối thiểu

- Constructor nhận thêm `web_searcher: WebSearcher | None = None`;
  expose property `has_web_search` (UI dùng để ẩn/vô hiệu toggle).
- `answer()` nhận thêm kwarg `use_web=False` — mặc định giữ nguyên hành vi,
  không vỡ caller/tests cũ.
- Điều kiện bỏ qua refusal OTHER mở rộng thành
  `OTHER and not has_attachments(session.id) and not (use_web and self.has_web_search)`.
- `_retrieve_sources(..., use_web=False)`: khi `use_web` → gọi
  `searcher.search(query)` trong try/except (lỗi → `[]`, log warning),
  RRF-fuse `[kb, attachments, web]` trước `dedupe_by_source`.
- `_gated_answer`: sau Refine vẫn không đạt và `self._web_searcher` tồn tại →
  search với **câu đã rewrite (exact keywords)**, merge, judge lần cuối;
  "đạt" dùng lại đúng logic `_is_acceptable` hiện có (high, hoặc medium khi
  `RAG_ALLOW_MEDIUM`); thất bại tiếp → `_fallback_result`/`_refused_result` như cũ.
- COURSE_ADVISOR branch không đổi — không bao giờ gọi web search.

### 3.3 Config & wiring

- `Config` thêm `firecrawl_api_key: str = ""`; `load_config()` đọc env
  `FIRECRAWL_API_KEY` (không bắt buộc, không raise khi thiếu).
- `build_rag_core()`: key khác rỗng mới tạo `FirecrawlWebSearcher(api_key)`
  và truyền vào `RagCore`.
- `.env.example` thêm dòng chú thích `# FIRECRAWL_API_KEY=fc-...`;
  README mục Setup ghi rõ tính năng là optional.

### 3.4 UI (`app.py`)

- Sidebar toggle **"Tìm kiếm web"** (`st.toggle`), lưu `st.session_state["use_web"]`;
  chỉ hiển thị khi `core.has_web_search` (thiếu key → ẩn kèm caption giải thích).
- Chat gọi `core.answer(msg, session, use_web=st.session_state["use_web"])`.
- `render_citations`: với `source.kind == "web"`, bên trong expander render URL
  thành link Markdown bấm được (`[url](url)`); label `[n] title — chapter(url)` giữ nguyên.
- DB **không cần sửa**: citations serialize JSON mọi trường Source tự nhiên,
  lịch sử cũ load lại bình thường.

## 4. Error handling

| Trường hợp | Xử lý |
|---|---|
| HTTP lỗi / timeout 30s / `success=false` / kết quả rỗng | searcher trả `[]`, log warning; pipeline chạy như chưa có web |
| Toggle ON mà searcher fail lúc retrieval | chỉ KB (+attachments), flow judge/refine như cũ |
| Corrective search fail hoặc judge lần cuối vẫn không đạt | fallback/refusal hiện tại nguyên vẹn |
| Kết quả web thiếu `markdown` | dùng `description`; cả hai trống → bỏ kết quả đó |
| Thiếu `FIRECRAWL_API_KEY` | không tạo searcher; `has_web_search=False`; toggle ẩn |

Refusal/fallback hành vi như KB thường — web chỉ là lớp bổ sung phía trước,
không bao giờ làm vỡ câu trả lời.

## 5. Testing (pytest, offline hoàn toàn)

1. **Parser** (`tests/test_web_search.py`): cả 2 shape response; mapping
   `Source(kind="web")` đúng trường; truncate 4000 ký tự; markdown trống → description;
   kết quả trống nội dung → bị bỏ.
2. **Lỗi**: `_post_json` raise / `success=false` / list rỗng → `[]`.
3. **Corrective path** với fake searcher + fake judge: low → low → merge web →
   judge high → generate, citation `kind="web"` xuất hiện; judge vẫn low sau web →
   refusal/fallback như cũ.
4. **Toggle**: `use_web=True` merge web vào retrieval + bypass OTHER;
   `use_web=False` hành vi nguyên vẹn (regression guard cho test cũ);
   searcher=None → `has_web_search=False`, kwarg vô hại.
5. **Config**: đọc env key; thiếu key → `build_rag_core` không tạo searcher.
6. **mypy strict** pass toàn bộ file mới/sửa.

## 6. Non-goals

- Không rerank riêng cho kết quả web (RRF sẵn có thay thế).
- Không cache kết quả web giữa các câu hỏi.
- Không scrape sâu thêm ngoài markdown mà `/search` trả về.
- Không agentic tool-calling (LLM tự quyết khi search) — chọn phương án C bị loại.
- Không áp dụng web search cho nhánh COURSE_ADVISOR.
- Không quota/ngân sách theo user.
