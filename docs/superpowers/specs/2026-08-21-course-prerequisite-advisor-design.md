# Course Prerequisite Advisor — Design

**Date:** 2026-08-21
**Status:** Approved (brainstorming 5/5 sections)
**Scope:** Phương án 1 — Minimal LLM-assisted Advisor (deterministic core + LLM router/extractor)
**Related:** `CONTEXT.md` glossary, `.scratch/coursemate-rag/spec.md`, `rag_core/__init__.py:32 RagCore`

## 1. Goal

Khai thác dữ liệu `prerequisites` đã có trong `courses.json` để chatbot tư vấn môn học chính xác, không xây Knowledge Graph / GNN / GraphRAG / career-profile phức tạp. Trả lời tốt:

- "Tôi đã học OOP và Data Structures, tôi có thể học môn nào tiếp theo?"
- "Tôi có đủ điều kiện học Artificial Intelligence không? Tôi đã học ..."
- "Để học Machine Learning tôi còn thiếu môn gì?"
- "Sau khi học Database, tôi có thể học tiếp những môn nào?"

Quyết định đã chốt (brainstorming Q1–Q5): **A-A-B-A-A**.

- Q1 extractor: **A LLM extractor** (resolve tên VI/EN → mã)
- Q2 router: **A LLM classifier** 3 labels
- Q3 semester: **B extract current_semester nếu có, fallback sort ASC**
- Q4 answer: **A deterministic template, no citations**
- Q5 missing info: **A hỏi lại, không fallback RAG**

## 2. Architecture

Giữ nguyên single entry `RagCore.answer(user_message, session) -> AnswerResult` `rag_core/__init__.py:73`. Chèn nhánh sau `SessionRewrite` (last 6 turns):

```
User
 ↓
SessionRewrite (LLM, last 6 turns → standalone query)  [giữ nguyên]
 ↓
Intent Router — OpenRouterClassifier (LLM, 3 labels)
 ├─ COURSE_ADVISOR ─→ CourseExtractor (LLM) ─→ CourseAdvisor (deterministic) ─→ AnswerResult(template)
 ├─ KNOWLEDGE_QA   ─→ Hybrid Retrieval (BM25 top-20 + dense top-20 RRF k=60) → Judge/Refine → Generation → AnswerCheck [pipeline hiện tại]
 └─ OTHER          ─→ AnswerResult(guidance template, refused=False)
```

Nhánh `COURSE_ADVISOR` **không** qua retrieval/judge/generator/check. Nhánh `OTHER` trả hướng dẫn cố định, không hallucinate.

Thay đổi file:

- Mới: `rag_core/course_advisor.py` (~120 dòng, pure)
- Mới: `rag_core/intent.py` (hoặc `rag_core/course_intent.py`, ~150 dòng, wrap OpenAI client cho classifier + extractor)
- Sửa: `rag_core/__init__.py` — thêm `classifier`, `extractor`, `advisor` vào `RagCore.__init__`, thêm `_advisor_branch`, sửa `answer()` để rẽ nhánh, sửa `build_rag_core()` để wire thật
- Không đụng: `rag_core/index.py`, `rag_core/db.py` (không thêm profile table), `app.py` (render `AnswerResult` như cũ)

## 3. Components

### 3.1 CourseAdvisor (`rag_core/course_advisor.py`) — pure, no LLM, no IO

Load `courses.json` một lần vào `dict[upper_code → Course]` tại `__init__(data_dir: Path)`.

```python
@dataclass(frozen=True)
class Course: code, name, name_en, credits, prerequisites, semester, department, instructor, description

class CourseAdvisor:
    def __init__(self, data_dir: Path): ...
    def get_prerequisites(self, course_code: str) -> list[str] | None: ...  # None nếu không tồn tại
    def get_missing_prerequisites(self, course_code: str, completed: set[str]) -> list[str] | None: ...
    def is_eligible(self, course_code: str, completed: set[str]) -> bool: ...  # False nếu code không tồn tại
    def get_next_courses(self, completed: set[str], current_semester: int | None = None) -> list[Course]: ...
    # helpers
    def resolve_code(self, raw: str) -> str | None: ...
    def get_course(self, code: str) -> Course | None: ...
    def list_courses(self) -> list[Course]: ...
```

Logic `get_next_courses(completed, current_semester)`:

- `eligible = [c for c in all_courses if c.code not in completed and set(c.prerequisites) ⊆ completed]`
- Nếu `current_semester is not None` và `1 <= current_semester <= 8`: `eligible = [c for c in eligible if c.semester >= current_semester]` rồi sort `(semester, code)`; đánh dấu nhóm `semester == current_semester+1` ở template (không filter cứng chỉ `== current+1` để không bỏ sót môn trễ).
- Nếu `current_semester is None` hoặc ngoài 1–8: coi như `None`, sort thuần `(semester, code)`.
- Chuẩn hoá mã: upper + strip.

100% unit-testable offline.

### 3.2 IntentClassifier + CourseExtractor (`rag_core/intent.py`)

Dùng `OpenAI(api_key, base_url)` như `rag_core/rewrite.py:24` và `rag_core/judge.py:79`.

**Classifier:**

```python
Intent = Literal["COURSE_ADVISOR", "KNOWLEDGE_QA", "OTHER"]
class IntentClassifier(Protocol):
    def classify(self, standalone_query: str) -> Intent: ...
class OpenRouterIntentClassifier:
    def __init__(self, api_key, model=DEFAULT_LLM_MODEL, base_url=DEFAULT_BASE_URL): ...
```

Prompt: system few-shot với bảng ví dụ của spec:

- "AI có prerequisite gì?" → COURSE_ADVISOR
- "Tôi còn thiếu gì để học AI?" → COURSE_ADVISOR
- "Tôi nên học môn nào tiếp theo?" → COURSE_ADVISOR
- "Giải thích A* algorithm" → KNOWLEDGE_QA
- "Giải thích overfitting" → KNOWLEDGE_QA
- "Thời tiết hôm nay?" → OTHER

Yêu cầu JSON `{"intent": "..."}`. `temperature=0`. Fail parse/timeout/empty → default `KNOWLEDGE_QA`.

**Extractor:**

```python
@dataclass(frozen=True)
class Extraction:
    completed_courses: list[str]  # mã đã chuẩn hoá upper
    target_course: str | None     # mã upper hoặc None
    current_semester: int | None  # 1..8 hoặc None

class CourseExtractor(Protocol):
    def extract(self, standalone_query: str) -> Extraction: ...

class OpenRouterCourseExtractor: ...
```

Prompt: "Bạn là parser môn học. Cho câu hỏi, trả về JSON `{completed_courses: string[], target_course: string|null, current_semester: int|null}`. Resolve tên tiếng Việt/Anh sang mã (vd 'Học máy'→CS311, 'Trí tuệ nhân tạo'→CS223, 'Cơ sở dữ liệu'→CS211). Chỉ trả mã tồn tại trong catalog (gợi ý list codes). Chuẩn hoá upper. Nếu không nêu kỳ → null. Nếu không nêu completed → []."

Few-shot: `"Tôi đã học CS101 và OOP, còn thiếu gì để học AI?"` → `{"completed_courses":["CS101","CS111"], "target_course":"CS223", "current_semester": null}`.

Fail parse → `Extraction([], None, None)`. Mã không tồn tại vẫn trả về để advisor báo `invalid_codes`.

### 3.3 Wiring trong RagCore (`rag_core/__init__.py`)

```python
class RagCore:
    def __init__(self, data_dir, embedder, generator, index=None,
                 judge=None, rewriter=None, checker=None, session_rewriter=None,
                 classifier: IntentClassifier|None=None,
                 extractor: CourseExtractor|None=None,
                 advisor: CourseAdvisor|None=None, ...):
        ...

    def answer(self, user_message: str, session: Session) -> AnswerResult:
        query = self._rewrite_for_session(user_message, session)  # giữ nguyên
        if self._classifier is not None and self._extractor is not None and self._advisor is not None:
            intent = self._classifier.classify(query)
            if intent == "COURSE_ADVISOR":
                return self._advisor_branch(query)
            if intent == "OTHER":
                return self._other_result()
        # fallback KNOWLEDGE_QA
        query_vector = ...
        retrieved = self._retrieve_sources(query, query_vector)
        ...

    def _advisor_branch(self, query: str) -> AnswerResult: ...
    def _other_result(self) -> AnswerResult: ...
```

`build_rag_core(config)` wire `OpenRouterIntentClassifier`, `OpenRouterCourseExtractor`, `CourseAdvisor(config.data_dir)` khi có `api_key`. Cho phép inject fake trong test.

## 4. Data Flow

**A. "Tôi đã học CS101 và CS112, tôi có thể học môn nào tiếp theo? kỳ 4"**
1. Rewrite → giữ nguyên
2. Classifier → COURSE_ADVISOR
3. Extractor → `{completed:["CS101","CS112"], target:None, current_semester:4}`
4. `get_next_courses({"CS101","CS112"}, 4)` → lọc eligible & semester>=4, sort
5. Template: "Với [CS101, CS112] (kỳ 4), bạn đủ điều kiện: CS211 (kỳ 3 nhưng eligible), CS213... Gợi ý kỳ 4: CS221 thiếu CS214..."

**B. "Tôi có đủ điều kiện học AI không? Tôi đã học CS112, CS102"**
- Extractor: `target=CS223, completed=[CS112,CS102]` → `get_missing(CS223, ...)` → thiếu `STAT101` → template "Chưa đủ, thiếu STAT101"

**C. "Giải thích bảng băm là gì?"**
- Classifier → KNOWLEDGE_QA → pipeline cũ nguyên vẹn (RRF → judge → generate → check)

**D. OTHER "Thời tiết?"**
- Classifier → OTHER → guidance cố định

## 5. Rendering & AnswerResult

Nhánh advisor luôn trả:

```python
AnswerResult(answer=rendered_template, citations=[], sources=[], refused=False, rephrase_suggestion="")
```

- Không gọi `parse_citations`, không tạo `Source`.
- UI `app.py:79 render_citations` tự bỏ qua khi `citations==[]`; `app.py:97 render_turn` render `text` thường.
- OTHER cũng `refused=False` nhưng text là hướng dẫn, không hallucinate.

## 6. Error Handling

| Case | Behavior |
|------|----------|
| `completed` chứa mã không tồn tại (PF101) | Advisor trả `invalid_codes`; template: "Không tìm thấy PF101 trong catalog (50 môn: CS101...). Bỏ qua mã không hợp lệ, tính trên [CS112]." |
| `target_course` không tồn tại | "Không tìm thấy 'XYZ' trong danh mục. Mã gần đúng: CS311..." |
| `completed==[]` + hỏi "có thể học AI không?" | "Để kiểm tra CS223, hãy cho biết môn đã hoàn thành, vd 'Tôi đã học CS101, CS102'. Prereq của CS223 là [CS112, STAT101]." |
| `get_next_courses` rỗng | "Hiện không có môn nào đủ điều kiện với [CS101]. Gợi ý hoàn thành CS102 để mở CS214/CS223... (thiếu 1 prereq)" + liệt kê 2–3 môn gần đủ |
| LLM classifier/extractor fail (timeout, JSON parse) | Classifier fail → default KNOWLEDGE_QA; Extractor fail → Extraction([], None, None) → advisor hỏi lại. Log warning, không crash. |
| Bilingual | Classifier/extractor preserve language; render dùng `name` (vi) hoặc `name_en` (en) tùy ngôn ngữ hỏi |
| Semester ngoài 1–8 | Coi như None, fallback sort ASC |

## 7. Testing

Giữ quy ước single seam `RagCore.answer()` (`spec.md` Testing Decisions).

**Unit deterministic `tests/test_course_advisor.py` (offline, no mocks):**
- `get_prerequisites(CS223)==["CS112","STAT101"]`, `get_prerequisites("FAKE")==None`
- `is_eligible(CS211, {"CS112"})==True`, `is_eligible(CS214, {"CS112"})==False`
- `get_missing(CS223, {"CS112"})==["STAT101"]`, full → `[]`
- `get_next_courses({"CS101"}, None)` chứa CS111/CS112/MATH101, sort semester ASC
- `get_next_courses({"CS101","CS102","CS112"}, 4)` lọc & sort đúng
- Invalid code, semester OOB, empty completed
- ≥15 cases

**Integration `tests/test_advisor_branch.py` (fake LLM):**
- Inject `FakeClassifier`/`FakeExtractor` vào `RagCore(data_dir, embedder=dummy, generator=dummy, classifier=fake, extractor=fake, advisor=real)` — không cần FAISS/OpenRouter.
- Matrix:
  - COURSE_ADVISOR + target + completed → answer chứa "đủ điều kiện" / "thiếu STAT101"
  - COURSE_ADVISOR + empty completed → answer chứa "hãy cho biết các môn đã hoàn thành"
  - KNOWLEDGE_QA → gọi retrieval path (mock)
  - OTHER → guidance
- Assert `result.citations==[]`, `refused==False`, `answer` chứa mã kỳ vọng.

**Không đụng `eval.py`** — advisor không ảnh hưởng hit@5/citation precision.

## 8. Out of Scope

- Không thêm bảng `student_profiles` / `completed_courses` trong `db.py`
- Không Neo4j / KG DB / graph embedding / GraphRAG / GNN / learning path phức tạp
- Không Profile Extraction / Interest Embedding / Multi-factor Ranking / Career Goal
- Không thêm Streamlit UI mới (dùng render hiện tại)

## 9. Build Order

1. `course_advisor.py` + unit tests
2. `intent.py` (classifier + extractor) + prompt tuning
3. Wire vào `RagCore` + integration tests
4. Manual QA với 4 câu mẫu của spec

## 10. Alternatives Considered

- **Gộp router+extractor 1 call:** tiết kiệm 1 call nhưng prompt phức tạp, khó eval riêng.
- **Pure deterministic regex/dict:** 0 LLM cost nhưng giòn với paraphrase VI.

## 11. Risks

- LLM extractor có thể hallucinate mã không tồn tại → mitigated bằng advisor validation + invalid_codes message.
- Classifier misroute câu lai ("Giải thích prerequisite của AI là gì?") → mitigated bằng few-shot + default KNOWLEDGE_QA an toàn.

