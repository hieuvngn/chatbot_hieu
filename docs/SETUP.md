# Hướng dẫn cài đặt và chạy CourseMate RAG Chatbot

Tài liệu này hướng dẫn chi tiết từ cài đặt môi trường đến chạy toàn bộ pipeline, demo và UI Streamlit. Tài liệu bám sát mã nguồn hiện tại (pipeline: `hybrid retrieval → CRAG judge → generation → answer check`, không còn re-ranker local).

---

## 1. Tổng quan kiến trúc

```
generate_data.py (seed 42) → data/courses.json + data/documents.json
      ↓
rag_core/chunking.py → chunk 400-600 tokens, ~15% overlap
      ↓
rag_core/index.py → FAISS (dense, cosine) + BM25 (lexical) → RRF k=60 → top-5 Sources
      ↓
rag_core/__init__.py:RagCore.answer()
   ├─ rag_core/rewrite.py      : rewrite follow-up → standalone (last 6 turns)
   ├─ rag_core/judge.py        : CRAG judge high/medium/low → 1 lần Refine (rewrite query)
   ├─ rag_core/generator.py    : gpt-4o-mini, cite [1],[2]...
   └─ rag_core/answer_check.py : Self-RAG verifier → 1 lần regenerate hoặc refuse
      ↓
rag_core/db.py (SQLite) + app.py (Streamlit UI)
```

*Điểm vào duy nhất*: `RagCore.answer(user_message, session) -> AnswerResult` (`rag_core/__init__.py:81`). UI, eval và demo đều gọi qua hàm này.

---

## 2. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| OS | Linux / macOS / Windows (WSL khuyến nghị) |
| Python | `>=3.11` (repo khai `pyproject.toml:5`, test trên 3.14) |
| Package manager | [`uv`](https://docs.astral.sh/uv/) `>=0.4` (khuyến nghị) hoặc `pip` |
| API key | Tài khoản OpenRouter (`https://openrouter.ai/keys`) |
| Dung lượng | ~500 MB cho `faiss-cpu`, `torch`, `transformers` (nếu cài full deps) |
| RAM | 4 GB tối thiểu, 8 GB khuyến nghị |

> Lưu ý: Sau khi gỡ `BAAI/bge-reranker-v2-m3`, project **không còn model local bắt buộc**. `torch`/`transformers`/`bitsandbytes`/`accelerate` trong `pyproject.toml:6` hiện là deps thừa, vẫn cài được nhưng không cần để chạy pipeline/ UI/ eval. Chỉ cần `faiss-cpu`, `rank-bm25`, `openai`, `python-dotenv`, `numpy`, `streamlit`.

---

## 3. Cài đặt

### 3.1 Clone repository

```sh
git clone <repo-url>
cd btl_ttnt_hieu
```

### 3.2 Cài `uv` (nếu chưa có)

```sh
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
# Kiểm tra
uv --version
```

Thay thế bằng `pip` nếu không dùng `uv`: `pip install -e .` (xem 3.3b).

### 3.3 Cài dependencies

**Với `uv` (khuyến nghị):**

```sh
# Cài toàn bộ deps theo pyproject.toml + uv.lock
uv sync

# Hoặc cài thêm dev deps để chạy test/mypy
uv sync --extra dev
# hoặc
uv pip install -e ".[dev]"
```

**Với `pip` truyền thống:**

```sh
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
pip install -e ".[dev]"     # để chạy pytest/mypy
```

Kiểm tra cài đặt:

```sh
uv run python -c "import faiss, openai, streamlit; print('deps ok')"
```

### 3.4 Cấu hình `.env`

Tạo file `.env` ở **thư mục gốc** (cùng cấp `pyproject.toml`):

```sh
cp .env.example .env  # nếu có template, hoặc tạo mới
```

Nội dung tối thiểu (`rag_core/config.py:24`):

```ini
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Tùy chọn (ghi đè mặc định trong `rag_core/config.py:7`):

```ini
# Model LLM cho generation / judge / checker / rewrite
RAG_LLM_MODEL=gpt-4o-mini
# Model embedding (OpenRouter, 2048-dim)
RAG_EMBED_MODEL=nvidia/nemotron-3-embed-1b:free
RAG_EMBED_DIM=2048
# Thư mục data (mặc định ./data)
RAG_DATA_DIR=./data
# Base URL OpenRouter (mặc định https://openrouter.ai/api/v1)
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

> `load_config()` sẽ đọc `.env` tự động qua `python-dotenv` nếu file tồn tại, hoặc đọc trực tiếp `os.environ`.

Lấy API key: Đăng ký tại `https://openrouter.ai/`, vào **Keys → Create Key**, copy `sk-or-v1-...`. Free tier đủ cho embedding + `gpt-4o-mini`.

---

## 4. Sinh dữ liệu tổng hợp

Toàn bộ demo chạy trên synthetic data, không có data thật.

```sh
# Sinh với seed mặc định 42 (bắt buộc để khớp eval_set.json)
uv run python -m generate_data --seed 42

# Tùy chọn khác
uv run python -m generate_data --seed 42 --out ./data
uv run python -m generate_data --help
```

Kết quả:

```
Generated 50 courses and 40 documents in data
```

Hai file được tạo:

* `data/courses.json` — 50 môn, mỗi môn: `code`, `name`/`name_en`, `credits`, `prerequisites`, `semester` (1-8), `department`, `instructor`, `description`. Đồ thị prerequisite là DAG (assert trong `generate_data.py:265`).
* `data/documents.json` — 40 tài liệu (`slides`/`textbook`, 20 `vi` + 20 `en`), mỗi tài liệu có `chapters` với `content` chứa topic của môn.

Chạy 2 lần với cùng seed cho output byte-identical. Nếu đổi seed, `eval_set.json` sẽ lệch ground-truth (test `test_hand_made_eval_set_ground_truths_exist_in_dataset` sẽ fail).

Kiểm tra nhanh:

```sh
ls -lh data/
cat data/courses.json | head -50
```

---

## 5. Kiểm tra cài đặt

### 5.1 Chạy tests

```sh
# Toàn bộ suite (79 tests, ~16s)
uv run pytest -v

# Chỉ test pipeline chính
uv run pytest tests/test_rag_core.py -v

# Chỉ test generator
uv run pytest tests/test_generate_data.py -v
```

Kỳ vọng: `79 passed`.

### 5.2 Type check

```sh
uv run mypy rag_core app.py generate_data.py eval.py tests --strict
# Success: no issues found in 18 source files
```

### 5.3 Smoke test pipeline (không cần API key)

```sh
# Embed fake (không gọi OpenRouter) — dùng RecordingEmbedder trong tests
uv run pytest tests/test_rag_core.py::test_answer_returns_answer_with_sources_attached -v
```

---

## 6. Chạy demo CLI

Tất cả demo yêu cầu `OPENROUTER_API_KEY` (trừ khi mock) và `data/` đã được sinh.

### 6.1 Naive RAG

```sh
uv run python -m demo_naive_rag "giải thích bảng băm là gì?"
uv run python -m demo_naive_rag "what is a hash table?"
```

In ra: số chunks, single-batched embedding, fused ranking, answer + citations.

### 6.2 CRAG judge + Refine

```sh
uv run python -m demo_crag "giải thích bảng băm là gì?"
```

In ra: top-5 Sources, verdict `high/medium/low`, refine rewrite (nếu `medium/low`) + re-retrieved top-5 + verdict lần 2, cuối cùng answer hoặc `refusal: ...`.

### 6.3 Answer check (Self-RAG)

```sh
uv run python -m demo_answer_check "giải thích bảng băm là gì?"
```

In ra: draft answer, verdict `supported/unsupported` + feedback, regeneration (nếu fail) + verdict lần 2, final answer hoặc refusal.

### 6.4 Memory + DB + Session rewrite

```sh
uv run python -m demo_memory "còn ví dụ về nó?"
# Mặc định turn 1: "giải thích bảng băm là gì?" → turn 2: follow-up được rewrite
uv run python -m demo_memory "nó có ưu điểm gì?"
```

Demo tạo `data/demo.db` (SQLite), đăng ký user `hieu`, lưu 2 turns, in ra rewritten query (`OpenRouterSessionRewriter` dùng last 6 turns).

---

## 7. Chạy đánh giá (Evaluation)

`eval_set.json` chứa 20 cặp question/ground-truth (10 vi + 10 en) trỏ tới `DOC-xxx / chapter`.

```sh
# Chỉ retrieval, không gọi LLM (nhanh, không tốn key)
uv run python -m eval --retrieval-only

# Full: retrieval + generation + judge + checker (tốn ~20 LLM calls)
uv run python -m eval

# Chỉ định file eval khác
uv run python -m eval --eval-set ./eval_set.json --retrieval-only
```

Output:

* Bảng per-question: `hit@5 naive | hit@5 full | citation precision naive | citation precision full | refused`
* Bảng summary: `retrieval hit@5 | citation precision | refusals`

Sau khi gỡ reranker, `hit@5 naive` và `hit@5 full` bằng nhau (cùng `Index.retrieve`).

---

## 8. Chạy Streamlit UI

```sh
uv run streamlit run app.py
# Mặc định http://localhost:8501
```

**Luồng sử dụng:**

1. Màn hình **Register** → tạo user (username/password lưu plaintext demo, `rag_core/db.py:77`).
2. **Login** → vào chat.
3. Chat: mỗi tin nhắn gọi `RagCore.answer(prompt, session)` (`app.py:133`), hiển thị answer + expanders `Sources [1] title — chapter` (`app.py:80`), hoặc warning `refused` + `rephrase_suggestion`.
4. **Log out** → history vẫn lưu trong `data/app.db`, đăng nhập lại sẽ khôi phục 6 turns gần nhất (`db.get_or_create_session`).

**Tùy chọn chạy:**

```sh
# Đổi port
uv run streamlit run app.py --server.port 8502

# Chạy background
nohup uv run streamlit run app.py &
```

Yêu cầu: `.env` phải có `OPENROUTER_API_KEY`, `data/courses.json` + `documents.json` phải tồn tại (sinh trước ở bước 4). DB `data/app.db` tự tạo lần đầu.

---

## Upload tài liệu (PDF/TXT/MD)

- Trong sidebar mở **📎 Tài liệu đính kèm** của chat đang chọn: tối đa 3 file, mỗi file ≤ 5 MB (pdf/txt/md).
- Nội dung được parse, chia đoạn, embedding và lưu theo chat; câu trả lời trích dẫn kèm vị trí (`Trang N` / heading).
- Dependency mới: `pypdf` (đã có trong pyproject; chạy lại `uv sync` là đủ).

---

## 9. Sử dụng như thư viện

```python
from rag_core import build_rag_core, Session

core = build_rag_core()  # tự đọc .env, build Index (single batched embed)
session = Session(id="s1", user_id="u1", turns=[])

result = core.answer("giải thích bảng băm là gì?", session)
print(result.answer)
print(result.refused, result.rephrase_suggestion)
for c in result.citations:
    print(c.marker, c.source.document_id, c.source.chapter)
for s in result.sources:
    print(s.document_id, s.chapter, s.course_code)
```

*Inject mock để test* (`tests/test_rag_core.py:161`):

```python
from rag_core import RagCore
core = RagCore(data_dir=path, embedder=my_embedder, generator=my_generator,
               judge=None, checker=None)  # không có judge/checker → không refine/verify
```

---

## 10. Biến môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `OPENROUTER_API_KEY` | *(bắt buộc)* | Key OpenRouter |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | Model cho generation/judge/checker/rewrite |
| `RAG_EMBED_MODEL` | `nvidia/nemotron-3-embed-1b:free` | Model embedding |
| `RAG_EMBED_DIM` | `2048` | Chiều vector |
| `RAG_DATA_DIR` | `./data` | Thư mục chứa `courses.json`, `documents.json`, `app.db` |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Endpoint OpenAI-compatible |

---

## 11. Cấu trúc thư mục

```
.
├── app.py                 # Streamlit UI
├── generate_data.py       # Sinh synthetic data
├── eval.py                # Đánh giá hit@5 + citation precision
├── eval_set.json          # 20 Q/A ground-truth
├── pyproject.toml         # deps + tool config
├── data/
│   ├── courses.json       # (sinh ra)
│   ├── documents.json     # (sinh ra)
│   └── app.db             # (tạo khi chạy UI/demo_memory)
├── rag_core/
│   ├── __init__.py        # RagCore + build_rag_core
│   ├── chunking.py        # chunk 400-600 tokens
│   ├── index.py           # FAISS + BM25 + RRF
│   ├── embeddings.py      # OpenRouterEmbedder
│   ├── generator.py       # OpenRouterGenerator + parse_citations
│   ├── judge.py           # OpenRouterJudge + QueryRewriter (CRAG)
│   ├── answer_check.py    # OpenRouterAnswerChecker (Self-RAG)
│   ├── rewrite.py         # OpenRouterSessionRewriter (6 turns)
│   ├── db.py              # SQLite Database
│   ├── models.py          # Source/Chunk/Citation/AnswerResult/Session
│   └── config.py          # load_config
├── demo_naive_rag.py
├── demo_crag.py
├── demo_answer_check.py
├── demo_memory.py
├── tests/
└── docs/
    └── SETUP.md           # (file này)
```

---

## 12. Troubleshooting

| Lỗi | Nguyên nhân | Khắc phục |
|---|---|---|
| `OPENROUTER_API_KEY is not set` | Thiếu `.env` | Tạo `.env` ở root với `OPENROUTER_API_KEY=sk-or-...` (`rag_core/config.py:42`) |
| `FileNotFoundError: data/courses.json` | Chưa sinh data | `uv run python -m generate_data --seed 42` |
| `AssertionError: prerequisite graph contains a cycle` | Data bị sửa tay | Xóa `data/` và sinh lại |
| `No module named 'faiss'` | Chưa `uv sync` | `uv sync` hoặc `pip install faiss-cpu` |
| `401 Unauthorized` từ OpenRouter | Key sai/hết hạn | Kiểm tra key tại `https://openrouter.ai/keys`, thử `curl -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/models` |
| `429 Rate limit` | Free tier giới hạn | Đợi 1 phút, pipeline đã tối ưu single-batched embed (`rag_core/index.py:54`) nên ít request |
| `streamlit not found` | Chưa cài UI deps | `uv sync` + `uv run streamlit run app.py` |
| `eval ground_truth not found` | Đổi seed khác 42 | Sinh lại `uv run python -m generate_data --seed 42` |
| `mypy: cannot find ...` | Thiếu dev deps | `uv sync --extra dev` |

**Dọn dẹp:**

```sh
rm -rf data/courses.json data/documents.json data/app.db data/demo.db
uv run python -m generate_data --seed 42  # sinh lại
```

---

## 13. Gỡ cài đặt

```sh
rm -rf .venv data/app.db data/demo.db
# Nếu dùng pip
pip uninstall coursemate-rag
```

---

## 14. Tài liệu liên quan

* `README.md` — tổng quan ngắn, pipeline, demo command
* `CONTEXT.md` — glossary domain (Source, Citation, CRAG judge, Refine...)
* `.scratch/coursemate-rag/spec.md` — spec đầy đủ, user stories, implementation decisions
* `docs/agents/` — hướng dẫn issue tracker, triage labels

---

*Cập nhật lần cuối: sau khi gỡ reranker (commit gỡ `rag_core/reranker.py`). Nếu cài deps cũ vẫn thấy `torch`/`transformers` trong `pyproject.toml`, có thể bỏ qua — chúng không còn được import.*
