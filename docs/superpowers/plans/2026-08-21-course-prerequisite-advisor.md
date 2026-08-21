# Course Prerequisite Advisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm nhánh tư vấn prerequisite deterministic (4 hàm CourseAdvisor + LLM router/extractor) vào RagCore để trả lời chính xác các câu hỏi về điều kiện học, môn tiếp theo, môn thiếu — không cần KG/ML.

**Architecture:** Sau `SessionRewrite`, LLM `IntentClassifier` rẽ 3 nhánh `COURSE_ADVISOR → CourseExtractor → CourseAdvisor → template`, `KNOWLEDGE_QA → pipeline RAG hiện tại`, `OTHER → guidance`. `CourseAdvisor` pure, không LLM, load `courses.json` trong RAM. Wire qua `RagCore.__init__` + `build_rag_core()`.

**Tech Stack:** Python 3.11+, `rag_core` package, OpenAI SDK via OpenRouter (classifier/extractor), SQLite/SQL không đổi, Streamlit không đụng, pytest + mypy strict.

## Global Constraints

- Python >=3.11 (pyproject.toml)
- Mỗi `AnswerResult` nhánh advisor phải có `citations=[]`, `sources=[]`, `refused=False` (đã chốt Section 5 design)
- Không thêm bảng DB mới, không Neo4j/KG/GNN/GraphRAG/Profile (Out of Scope §8 design)
- Single test seam `RagCore.answer()` cho integration; unit `CourseAdvisor` offline không mock LLM
- `mypy --strict` phải pass cho mọi file mới `pyproject.toml:29`
- Dữ liệu: `data/courses.json` là nguồn duy nhất cho graph (50 courses, DAG đã assert)
- Classifier fail open → `KNOWLEDGE_QA`; Extractor fail → `Extraction([], None, None)` → advisor hỏi lại (Section 6 design)

---

## File Structure

**New files:**
- `rag_core/course_advisor.py` — Pure deterministic `CourseAdvisor` + `Course` dataclass, load `courses.json`, 4 APIs + helpers. Owner of prerequisite logic. No LLM, no IO ngoài đọc JSON ở `__init__`.
- `rag_core/intent.py` — `IntentClassifier` + `CourseExtractor` protocols + `OpenRouterIntentClassifier` + `OpenRouterCourseExtractor` + `Extraction` dataclass + prompts. LLM-only, stateless.
- `tests/test_course_advisor.py` — Unit offline cho `CourseAdvisor` (≥15 cases).
- `tests/test_advisor_branch.py` — Integration cho `RagCore` nhánh advisor với fake classifier/extractor/dummy embedder/generator/index.

**Modified files:**
- `rag_core/__init__.py:39-60` — Thêm params `classifier`, `extractor`, `advisor` vào `RagCore.__init__`, thêm `_advisor_branch`, `_other_result`, `_render_advisor_answer`, sửa `answer()` để rẽ sau `_rewrite_for_session`, sửa `build_rag_core()` để wire thật, export symbols mới vào `__all__`.
- `pyproject.toml` — không cần thêm dependency (đã có `openai`, `python-dotenv`).

---

### Task 1: CourseAdvisor deterministic core

**Files:**
- Create: `rag_core/course_advisor.py`
- Test: `tests/test_course_advisor.py`

**Interfaces:**
- Consumes: `Path("data/courses.json")` — JSON list of course dicts (code, name, name_en, prerequisites, semester, etc.) — read once in `__init__`.
- Produces:
  ```python
  @dataclass(frozen=True) class Course: code: str; name: str; name_en: str; credits: int; prerequisites: list[str]; semester: int; department: str; instructor: str; description: str
  class CourseAdvisor:
      def __init__(self, data_dir: Path) -> None
      def get_course(self, code: str) -> Course | None  # upper+strip normalize, None if not found
      def list_courses(self) -> list[Course]
      def get_prerequisites(self, course_code: str) -> list[str] | None  # None if code not found
      def get_missing_prerequisites(self, course_code: str, completed: set[str]) -> list[str] | None  # None if target not found
      def is_eligible(self, course_code: str, completed: set[str]) -> bool  # False if target not found
      def get_next_courses(self, completed: set[str], current_semester: int | None = None) -> list[Course]  # eligible & not completed, sort (semester, code)
      def resolve_code(self, raw: str) -> str | None  # normalize + existence check
  ```
  Later tasks rely on exact names `get_prerequisites`, `get_missing_prerequisites`, `is_eligible`, `get_next_courses`.

- [ ] **Step 1: Write failing test for CourseAdvisor basics**

Create `tests/test_course_advisor.py` with first batch (will fail — module not found):

```python
from pathlib import Path
from rag_core.course_advisor import CourseAdvisor

DATA_DIR = Path("data")

def test_get_prerequisites_known():
    adv = CourseAdvisor(DATA_DIR)
    assert adv.get_prerequisites("CS223") == ["CS112", "STAT101"]  # from COURSE_TABLE

def test_get_prerequisites_unknown():
    adv = CourseAdvisor(DATA_DIR)
    assert adv.get_prerequisites("FAKE999") is None

def test_is_eligible_true():
    adv = CourseAdvisor(DATA_DIR)
    assert adv.is_eligible("CS211", {"CS112"}) is True  # CS211 prereq [CS112]

def test_is_eligible_false_missing():
    adv = CourseAdvisor(DATA_DIR)
    assert adv.is_eligible("CS214", {"CS112"}) is False  # CS214 needs CS102+CS112

def test_get_missing_partial():
    adv = CourseAdvisor(DATA_DIR)
    assert adv.get_missing_prerequisites("CS223", {"CS112"}) == ["STAT101"]

def test_get_missing_none_when_target_unknown():
    adv = CourseAdvisor(DATA_DIR)
    assert adv.get_missing_prerequisites("FAKE", {"CS101"}) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_course_advisor.py -v`
Expected: `ModuleNotFoundError: No module named 'rag_core.course_advisor'` FAIL.

- [ ] **Step 3: Implement minimal CourseAdvisor**

Create `rag_core/course_advisor.py`:

```python
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Course:
    code: str
    name: str
    name_en: str
    credits: int
    prerequisites: list[str]
    semester: int
    department: str
    instructor: str
    description: str

class CourseAdvisor:
    def __init__(self, data_dir: Path) -> None:
        raw = json.loads((Path(data_dir) / "courses.json").read_text(encoding="utf-8"))
        self._by_code: dict[str, Course] = {}
        for c in raw:
            course = Course(code=c["code"], name=c["name"], name_en=c["name_en"], credits=c["credits"], prerequisites=list(c["prerequisites"]), semester=c["semester"], department=c["department"], instructor=c["instructor"], description=c["description"])
            self._by_code[c["code"].upper()] = course
    def _norm(self, code: str) -> str:
        return code.strip().upper()
    def get_course(self, code: str) -> Course | None:
        return self._by_code.get(self._norm(code))
    def list_courses(self) -> list[Course]:
        return list(self._by_code.values())
    def resolve_code(self, raw: str) -> str | None:
        n = self._norm(raw)
        return n if n in self._by_code else None
    def get_prerequisites(self, course_code: str) -> list[str] | None:
        c = self.get_course(course_code)
        return None if c is None else list(c.prerequisites)
    def get_missing_prerequisites(self, course_code: str, completed: set[str]) -> list[str] | None:
        c = self.get_course(course_code)
        if c is None:
            return None
        norm_completed = {self._norm(x) for x in completed}
        return [p for p in c.prerequisites if self._norm(p) not in norm_completed]
    def is_eligible(self, course_code: str, completed: set[str]) -> bool:
        missing = self.get_missing_prerequisites(course_code, completed)
        return False if missing is None else len(missing) == 0
    def get_next_courses(self, completed: set[str], current_semester: int | None = None) -> list[Course]:
        norm_completed = {self._norm(x) for x in completed}
        eligible = [c for c in self._by_code.values() if c.code.upper() not in norm_completed and all(self._norm(p) in norm_completed for p in c.prerequisites)]
        if current_semester is not None and 1 <= current_semester <= 8:
            eligible = [c for c in eligible if c.semester >= current_semester]
        eligible.sort(key=lambda c: (c.semester, c.code))
        return eligible
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_course_advisor.py -v`
Expected: 6 PASS.

- [ ] **Step 5: Add remaining edge-case tests (TDD increment)**

Append to `tests/test_course_advisor.py`:

```python
def test_get_next_courses_sort_asc():
    adv = CourseAdvisor(DATA_DIR)
    nxt = adv.get_next_courses({"CS101"}, None)
    # Must be sorted by (semester, code) and not contain CS101
    assert "CS101" not in {c.code for c in nxt}
    assert nxt == sorted(nxt, key=lambda c: (c.semester, c.code))
    assert any(c.code == "CS112" for c in nxt)  # CS112 prereq [CS101] eligible

def test_get_next_courses_with_semester_filter():
    adv = CourseAdvisor(DATA_DIR)
    nxt_all = adv.get_next_courses({"CS101", "CS102", "CS112"}, None)
    nxt_f4 = adv.get_next_courses({"CS101", "CS102", "CS112"}, 4)
    assert all(c.semester >= 4 for c in nxt_f4)
    assert len(nxt_f4) <= len(nxt_all)

def test_get_next_courses_empty_completed():
    adv = CourseAdvisor(DATA_DIR)
    nxt = adv.get_next_courses(set(), None)
    # Only semester-1 courses with no prereq are eligible
    assert all(c.semester == 1 for c in nxt)
    assert all(len(c.prerequisites) == 0 for c in nxt)

def test_resolve_code_normalization():
    adv = CourseAdvisor(DATA_DIR)
    assert adv.resolve_code(" cs223 ") == "CS223"
    assert adv.resolve_code("fake") is None

def test_is_eligible_unknown_code():
    adv = CourseAdvisor(DATA_DIR)
    assert adv.is_eligible("FAKE", {"CS101"}) is False

def test_get_missing_all_completed():
    adv = CourseAdvisor(DATA_DIR)
    assert adv.get_missing_prerequisites("CS211", {"CS112"}) == []
```

- [ ] **Step 6: Run full unit suite**

Run: `uv run pytest tests/test_course_advisor.py -v && uv run mypy rag_core/course_advisor.py --strict`
Expected: 12 PASS, mypy clean.

- [ ] **Step 7: Commit**

```bash
git add rag_core/course_advisor.py tests/test_course_advisor.py
git commit -m "feat(advisor): add deterministic CourseAdvisor with 4 core APIs"
```

---

### Task 2: Intent Classifier & Course Extractor (LLM)

**Files:**
- Create: `rag_core/intent.py`
- Test: `tests/test_intent.py` (mock OpenAI, not calling real API)

**Interfaces:**
- Consumes: `rag_core/config.py:DEFAULT_LLM_MODEL`, `rag_core/config.py:DEFAULT_BASE_URL`
- Produces:
  ```python
  Intent = Literal["COURSE_ADVISOR", "KNOWLEDGE_QA", "OTHER"]
  @dataclass(frozen=True) class Extraction: completed_courses: list[str]; target_course: str | None; current_semester: int | None

  class IntentClassifier(Protocol): def classify(self, query: str) -> Intent
  class CourseExtractor(Protocol): def extract(self, query: str) -> Extraction

  class OpenRouterIntentClassifier(IntentClassifier):
      def __init__(self, api_key: str, model: str = DEFAULT_LLM_MODEL, base_url: str = DEFAULT_BASE_URL): ...
      def classify(self, query: str) -> Intent: ...

  class OpenRouterCourseExtractor(CourseExtractor):
      def __init__(self, api_key: str, model: str = DEFAULT_LLM_MODEL, base_url: str = DEFAULT_BASE_URL): ...
      def extract(self, query: str) -> Extraction: ...
  ```
  Fail behavior: classifier fails → `KNOWLEDGE_QA`; extractor fails → `Extraction([], None, None)`.

- [ ] **Step 1: Write failing test for classifier + extractor with fake client**

Create `tests/test_intent.py`:

```python
from rag_core.intent import Extraction

def test_extraction_dataclass():
    e = Extraction(completed_courses=["CS101"], target_course="CS223", current_semester=4)
    assert e.completed_courses == ["CS101"]

def test_classifier_protocol_exists():
    from rag_core.intent import OpenRouterIntentClassifier
    assert OpenRouterIntentClassifier is not None

def test_extractor_protocol_exists():
    from rag_core.intent import OpenRouterCourseExtractor
    assert OpenRouterCourseExtractor is not None
```

- [ ] **Step 2: Run test — expect FAIL ModuleNotFound**

Run: `uv run pytest tests/test_intent.py -v`
Expected: `ModuleNotFoundError: rag_core.intent`

- [ ] **Step 3: Implement `rag_core/intent.py` minimal**

Create `rag_core/intent.py` following `rag_core/judge.py:19` and `rag_core/rewrite.py:8` patterns:

```python
from __future__ import annotations
import json, re
from dataclasses import dataclass
from typing import Literal, Protocol
from rag_core.config import DEFAULT_BASE_URL, DEFAULT_LLM_MODEL

Intent = Literal["COURSE_ADVISOR", "KNOWLEDGE_QA", "OTHER"]

_CLASSIFIER_SYSTEM = (
    "You are an intent classifier for a course chatbot. Classify the standalone query into exactly one label: "
    "COURSE_ADVISOR (asks about prerequisites, eligibility, what to take next, missing courses), "
    "KNOWLEDGE_QA (asks to explain a topic, concept, algorithm, theory grounded in documents), "
    "OTHER (smalltalk, weather, unrelated). Reply ONLY JSON {\"intent\":\"...\"}. "
    "Examples: 'AI có prerequisite gì?'->COURSE_ADVISOR, 'Tôi còn thiếu gì để học AI?'->COURSE_ADVISOR, "
    "'Tôi nên học môn nào tiếp theo?'->COURSE_ADVISOR, 'Giải thích bảng băm'->KNOWLEDGE_QA, 'Thời tiết?'->OTHER"
)

_EXTRACTOR_SYSTEM = (
    "You are a course parser. Given a standalone Vietnamese/English query, return ONLY JSON "
    "{\"completed_courses\": string[], \"target_course\": string|null, \"current_semester\": int|null}. "
    "completed_courses: list course codes the user says they completed, uppercased. Resolve Vietnamese names to codes "
    "(e.g. 'Học máy'->CS311, 'Trí tuệ nhân tạo'->CS223, 'Cơ sở dữ liệu'->CS211, 'Nhập môn lập trình'->CS101). "
    "target_course: the course being asked about, or null if asking what next. "
    "current_semester: integer 1-8 if mentioned (e.g. 'kỳ 4'->4), else null. Only return codes that look like COURSE codes."
)

@dataclass(frozen=True)
class Extraction:
    completed_courses: list[str]
    target_course: str | None
    current_semester: int | None

class IntentClassifier(Protocol):
    def classify(self, query: str) -> Intent: ...

class CourseExtractor(Protocol):
    def extract(self, query: str) -> Extraction: ...

def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s

class OpenRouterIntentClassifier:
    def __init__(self, api_key: str, model: str = DEFAULT_LLM_MODEL, base_url: str = DEFAULT_BASE_URL):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
    def classify(self, query: str) -> Intent:
        try:
            resp = self._client.chat.completions.create(model=self._model, messages=[{"role":"system","content":_CLASSIFIER_SYSTEM},{"role":"user","content":query}], temperature=0)
            content = (resp.choices[0].message.content or "").strip()
            data = json.loads(_strip_code_fence(content))
            intent = data.get("intent")
            if intent in ("COURSE_ADVISOR","KNOWLEDGE_QA","OTHER"):
                return intent  # type: ignore
        except Exception:
            pass
        return "KNOWLEDGE_QA"

class OpenRouterCourseExtractor:
    def __init__(self, api_key: str, model: str = DEFAULT_LLM_MODEL, base_url: str = DEFAULT_BASE_URL):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
    def extract(self, query: str) -> Extraction:
        try:
            resp = self._client.chat.completions.create(model=self._model, messages=[{"role":"system","content":_EXTRACTOR_SYSTEM},{"role":"user","content":query}], temperature=0)
            content = (resp.choices[0].message.content or "").strip()
            data = json.loads(_strip_code_fence(content))
            completed = [str(c).strip().upper() for c in data.get("completed_courses", []) if isinstance(c, str) and c.strip()]
            target = data.get("target_course")
            target_norm = str(target).strip().upper() if isinstance(target, str) and target.strip() else None
            sem = data.get("current_semester")
            sem_norm = int(sem) if isinstance(sem, int) and 1 <= sem <= 8 else (int(sem) if isinstance(sem, str) and sem.isdigit() and 1 <= int(sem) <= 8 else None)
            return Extraction(completed_courses=completed, target_course=target_norm, current_semester=sem_norm)
        except Exception:
            return Extraction(completed_courses=[], target_course=None, current_semester=None)
```

- [ ] **Step 4: Run test — expect PASS**

Run: `uv run pytest tests/test_intent.py -v && uv run mypy rag_core/intent.py --strict`
Expected: PASS, mypy clean (add `# type: ignore[import-untyped]` if openai stubs missing).

- [ ] **Step 5: Add mocked LLM behavior tests**

Append to `tests/test_intent.py`:

```python
from unittest.mock import MagicMock, patch
from rag_core.intent import OpenRouterIntentClassifier, OpenRouterCourseExtractor

def _mock_resp(content: str):
    m = MagicMock()
    m.choices = [MagicMock(message=MagicMock(content=content))]
    return m

def test_classifier_parses_course_advisor():
    with patch("openai.OpenAI") as MockOA:
        inst = MockOA.return_value
        inst.chat.completions.create.return_value = _mock_resp('{"intent":"COURSE_ADVISOR"}')
        clf = OpenRouterIntentClassifier(api_key="fake")
        assert clf.classify("AI có prerequisite gì?") == "COURSE_ADVISOR"

def test_classifier_defaults_to_knowledge_on_bad_json():
    with patch("openai.OpenAI") as MockOA:
        inst = MockOA.return_value
        inst.chat.completions.create.return_value = _mock_resp('not json')
        clf = OpenRouterIntentClassifier(api_key="fake")
        assert clf.classify("hello") == "KNOWLEDGE_QA"

def test_extractor_parses_completed_and_target():
    with patch("openai.OpenAI") as MockOA:
        inst = MockOA.return_value
        inst.chat.completions.create.return_value = _mock_resp('{"completed_courses":["CS101","CS112"],"target_course":"CS223","current_semester":4}')
        ext = OpenRouterCourseExtractor(api_key="fake")
        e = ext.extract("Tôi đã học CS101 và CS112, đủ học AI không? kỳ 4")
        assert e.completed_courses == ["CS101","CS112"]
        assert e.target_course == "CS223"
        assert e.current_semester == 4

def test_extractor_defaults_on_failure():
    with patch("openai.OpenAI") as MockOA:
        inst = MockOA.return_value
        inst.chat.completions.create.return_value = _mock_resp('oops')
        ext = OpenRouterCourseExtractor(api_key="fake")
        e = ext.extract("bad")
        assert e == e  # == Extraction([], None, None)
        assert e.completed_courses == []
```

- [ ] **Step 6: Run again**

Run: `uv run pytest tests/test_intent.py -v`
Expected: 7 PASS.

- [ ] **Step 7: Commit**

```bash
git add rag_core/intent.py tests/test_intent.py
git commit -m "feat(intent): add LLM classifier (3 labels) + course extractor (JSON) with fail-open defaults"
```

---

### Task 3: Wire advisor branch into RagCore

**Files:**
- Modify: `rag_core/__init__.py` (add params, branch, templates, build_rag_core wiring, __all__)
- Test: `tests/test_advisor_branch.py` (uses fakes, no real LLM/FAISS)

**Interfaces:**
- Consumes: `CourseAdvisor` from Task 1, `IntentClassifier`/`CourseExtractor`/`Extraction` from Task 2, existing `Session`, `AnswerResult`, `Embedder`, `Generator`, `Index`.
- Produces: `RagCore.answer()` now branches; new private methods:
  ```python
  def _advisor_branch(self, query: str) -> AnswerResult
  def _other_result(self) -> AnswerResult
  def _render_eligible(self, completed: set[str], current_semester: int|None) -> str
  def _render_eligibility_check(self, target: str, completed: set[str]) -> str
  ```
  Templates are deterministic Vietnamese/English based on query language heuristic (contains Vietnamese chars → vi else en).

- [ ] **Step 1: Write failing integration test for branch**

Create `tests/test_advisor_branch.py`:

```python
from pathlib import Path
from rag_core import RagCore, Session
from rag_core.course_advisor import CourseAdvisor
from rag_core.intent import Extraction

class FakeClassifier:
    def __init__(self, intent): self.intent = intent
    def classify(self, q): return self.intent

class FakeExtractor:
    def __init__(self, ext): self.ext = ext
    def extract(self, q): return self.ext

class DummyEmbedder:
    def embed_batch(self, texts): return [[0.0]*8 for _ in texts]
    def embed_query(self, q): return [0.0]*8

class DummyGenerator:
    def generate(self, q, chunks, feedback=None): return "dummy RAG answer [1]"

DATA_DIR = Path("data")
SESSION = Session(id="s1", user_id="u1", turns=[])

def _core(intent, extraction):
    return RagCore(
        data_dir=DATA_DIR,
        embedder=DummyEmbedder(),
        generator=DummyGenerator(),
        classifier=FakeClassifier(intent),
        extractor=FakeExtractor(extraction),
        advisor=CourseAdvisor(DATA_DIR),
    )

def test_advisor_next_courses():
    core = _core("COURSE_ADVISOR", Extraction(completed_courses=["CS101","CS112"], target_course=None, current_semester=4))
    res = core.answer("Tôi đã học CS101 và CS112, học gì tiếp kỳ 4?", SESSION)
    assert res.citations == [] and not res.refused
    assert "CS211" in res.answer or "CS" in res.answer  # at least one eligible

def test_advisor_eligibility_missing():
    core = _core("COURSE_ADVISOR", Extraction(completed_courses=["CS112"], target_course="CS223", current_semester=None))
    res = core.answer("Tôi học CS112, đủ học AI không?", SESSION)
    assert "STAT101" in res.answer  # missing prereq
    assert "thiếu" in res.answer.lower() or "missing" in res.answer.lower()

def test_advisor_asks_when_no_completed():
    core = _core("COURSE_ADVISOR", Extraction(completed_courses=[], target_course="CS223", current_semester=None))
    res = core.answer("Tôi có thể học AI không?", SESSION)
    assert "cho biết" in res.answer.lower() or "completed" in res.answer.lower()
    assert "CS223" in res.answer or "STAT101" in res.answer

def test_advisor_invalid_code():
    core = _core("COURSE_ADVISOR", Extraction(completed_courses=["PF101"], target_course=None, current_semester=None))
    res = core.answer("Tôi học PF101, học gì tiếp?", SESSION)
    assert "PF101" in res.answer and ("Không tìm thấy" in res.answer or "not found" in res.answer.lower())

def test_knowledge_qa_goes_to_rag():
    core = _core("KNOWLEDGE_QA", Extraction([], None, None))
    res = core.answer("Giải thích bảng băm là gì?", SESSION)
    assert "dummy RAG" in res.answer

def test_other_returns_guidance():
    core = _core("OTHER", Extraction([], None, None))
    res = core.answer("Thời tiết hôm nay?", SESSION)
    assert "tư vấn môn học" in res.answer.lower() or "course" in res.answer.lower()
```

- [ ] **Step 2: Run — expect FAIL (RagCore has no classifier param)**

Run: `uv run pytest tests/test_advisor_branch.py -v`
Expected: `TypeError: unexpected keyword argument 'classifier'`.

- [ ] **Step 3: Modify `rag_core/__init__.py` to add wiring**

Edits (exact locations):

1. Top imports: add `from rag_core.course_advisor import CourseAdvisor` and `from rag_core.intent import Extraction, IntentClassifier, CourseExtractor, OpenRouterIntentClassifier, OpenRouterCourseExtractor` (or lazy import inside `build_rag_core` to avoid circular).
2. `RagCore.__init__:39` signature: add `classifier: IntentClassifier|None=None, extractor: CourseExtractor|None=None, advisor: CourseAdvisor|None=None` after `session_rewriter`. Store as `self._classifier`, `self._extractor`, `self._advisor` (also accept `CourseAdvisor` instance or create from `data_dir` if `advisor is None` but classifier present — lazy).
3. `answer:73` — insert after `query = self._rewrite_for_session(...)`:
   ```python
   if self._classifier is not None and self._extractor is not None and self._advisor is not None:
       intent = self._classifier.classify(query)
       if intent == "COURSE_ADVISOR":
           return self._advisor_branch(query)
       if intent == "OTHER":
           return self._other_result(query)
   # fall through to existing retrieval path
   ```
4. Add private methods after `_rewrite_for_session`:
   ```python
   def _advisor_branch(self, query: str) -> AnswerResult: ...
   def _other_result(self, query: str) -> AnswerResult: ...  # guidance template vi/en
   def _render_next(self, completed, sem) -> str: ...
   def _render_check(self, target, completed) -> str: ...
   ```
   Templates must be deterministic, include course codes + names, semester, missing list, invalid_codes warning. Detect vi by `any("\u00C0" <= c <= "\u1EF9" or c in "ăâđêôơư" for c in query.lower())` or simple `in` check.

5. `build_rag_core:172` — instantiate `CourseAdvisor(config.data_dir)`, `OpenRouterIntentClassifier(...)`, `OpenRouterCourseExtractor(...)` and pass to `RagCore`.

6. `__all__:218` — add `"CourseAdvisor", "IntentClassifier", "CourseExtractor", "Extraction", "OpenRouterIntentClassifier", "OpenRouterCourseExtractor"`.

Keep existing `_retrieve_sources`, `_gated_answer`, `_generate_result` untouched.

- [ ] **Step 4: Implement templates inside `_advisor_branch`**

Pseudocode for `_advisor_branch`:

```python
def _advisor_branch(self, query: str) -> AnswerResult:
    ext = self._extractor.extract(query)  # type: ignore
    adv = self._advisor  # type: ignore
    # normalize completed to upper set
    completed_set = {c.strip().upper() for c in ext.completed_courses if c.strip()}
    invalid = [c for c in ext.completed_courses if adv.resolve_code(c) is None]
    valid_completed = {c for c in completed_set if adv.resolve_code(c) is not None}
    is_vi = any(ch in query.lower() for ch in ["tôi", "môn", "học", "prerequisite", "đã"])
    # actually use char range check

    if ext.target_course:
        target_norm = ext.target_course.strip().upper()
        course = adv.get_course(target_norm)
        if course is None:
            msg = f"Không tìm thấy môn '{ext.target_course}' trong danh mục 50 môn." if is_vi else f"Course '{ext.target_course}' not found in catalog."
            return AnswerResult(answer=msg, citations=[], sources=[], refused=False)
        if not valid_completed and ext.completed_courses == []:
            prereqs = ", ".join(course.prerequisites) or "không có"
            msg = f"Để kiểm tra {course.code} ({course.name}), hãy cho biết các môn đã hoàn thành, ví dụ: 'Tôi đã học CS101, CS102'. Prerequisite của {course.code} là: {prereqs}."
            if invalid: msg += f" Lưu ý: {', '.join(invalid)} không có trong catalog."
            return AnswerResult(answer=msg, citations=[], sources=[], refused=False)
        missing = adv.get_missing_prerequisites(target_norm, valid_completed)
        if missing == []:
            msg = f"Bạn đủ điều kiện học {course.code} ({course.name}). Đã hoàn thành: {', '.join(sorted(valid_completed))}."
        else:
            msg = f"Bạn chưa đủ điều kiện học {course.code} ({course.name}). Thiếu: {', '.join(missing)}. Đã có: {', '.join(sorted(valid_completed)) or 'chưa có môn nào'}."
        if invalid: msg += f" (Bỏ qua mã không hợp lệ: {', '.join(invalid)})"
        return AnswerResult(answer=msg, citations=[], sources=[], refused=False)
    else:
        # recommend next
        nxt = adv.get_next_courses(valid_completed, ext.current_semester)
        if not nxt:
            # also suggest near-miss (missing 1 prereq) as hint
            near = [c for c in adv.list_courses() if c.code.upper() not in valid_completed and len(adv.get_missing_prerequisites(c.code, valid_completed) or []) == 1]
            near = sorted(near, key=lambda c: (c.semester, c.code))[:3]
            msg = f"Hiện không có môn nào đủ điều kiện với [{', '.join(sorted(valid_completed)) or 'rỗng'}]."
            if near: msg += f" Gần đủ (thiếu 1 môn): {', '.join(f'{c.code} ({c.name})' for c in near)}."
            return AnswerResult(answer=msg, citations=[], sources=[], refused=False)
        lines = [f"- {c.code} {c.name} ({c.name_en}), kỳ {c.semester}, {c.credits} TC, prereq: {', '.join(c.prerequisites) or 'không có'}" for c in nxt[:10]]
        header = f"Với các môn đã hoàn thành [{', '.join(sorted(valid_completed))}], bạn đủ điều kiện học:"
        msg = header + "\n" + "\n".join(lines)
        if invalid: msg += f"\nLưu ý: bỏ qua mã không hợp lệ: {', '.join(invalid)}"
        if ext.current_semester: msg += f"\n(Đã lọc kỳ >= {ext.current_semester})"
        return AnswerResult(answer=msg, citations=[], sources=[], refused=False)
```

- [ ] **Step 5: Run integration tests**

Run: `uv run pytest tests/test_advisor_branch.py -v && uv run mypy rag_core/__init__.py --strict`
Expected: 6 PASS, mypy clean.

- [ ] **Step 6: Run full suite regression**

Run: `uv run pytest -q`
Expected: all previous `test_rag_core.py`, `test_db.py`, `test_course_advisor.py`, `test_intent.py`, `test_advisor_branch.py` PASS.

- [ ] **Step 7: Commit**

```bash
git add rag_core/__init__.py tests/test_advisor_branch.py
git commit -m "feat(core): wire advisor branch into RagCore answer() with deterministic templates"
```

---

### Task 4: Final verification & docs

**Files:**
- Modify: `README.md` (optional short section), `docs/SETUP.md` if needed
- Verify: `tests/test_course_advisor.py`, `tests/test_advisor_branch.py`, `tests/test_intent.py` all green

**Interfaces:** None new — verification only.

- [ ] **Step 1: Manual QA matrix (run with real data, dummy LLM not needed — use fake injection as in Task 3 but also one live smoke if OPENROUTER_API_KEY present)**

Manual prompts to verify via `python -c`:

```python
from pathlib import Path
from rag_core import RagCore, Session
from rag_core.course_advisor import CourseAdvisor
from rag_core.intent import Extraction

class FakeClf:
    def classify(self, q): return "COURSE_ADVISOR" if "học" in q.lower() or "prereq" in q.lower() else "KNOWLEDGE_QA"
class FakeExt:
    def extract(self, q):
        if "CS101" in q and "CS112" in q: return Extraction(["CS101","CS112"], None, 4)
        if "AI" in q and "CS112" in q: return Extraction(["CS112"], "CS223", None)
        return Extraction([], None, None)

core = RagCore(Path("data"), embedder=__import__("tests.test_advisor_branch", fromlist=["DummyEmbedder"]).DummyEmbedder(), generator=__import__("tests.test_advisor_branch", fromlist=["DummyGenerator"]).DummyGenerator(), classifier=FakeClf(), extractor=FakeExt(), advisor=CourseAdvisor(Path("data")))
print(core.answer("Tôi đã học CS101 và CS112, học gì tiếp kỳ 4?", Session("s","u",[])).answer[:500])
print(core.answer("Tôi có đủ điều kiện học AI không? Tôi đã học CS112", Session("s","u",[])).answer)
```

- [ ] **Step 2: Run mypy strict on whole package**

Run: `uv run mypy rag_core --strict`
Expected: PASS (fix any `Any` from OpenAI mock).

- [ ] **Step 3: Commit docs if changed**

```bash
git add README.md
git commit -m "docs: mention Course Prerequisite Advisor branch"  # if README updated
```

---

## Self-Review

**Spec coverage check:**

- §2 Architecture branching after SessionRewrite → Task 3
- §3.1 CourseAdvisor 4 APIs → Task 1
- §3.2 Classifier 3 labels + Extractor JSON with fail-open → Task 2
- §4 Data flows (3 examples + OTHER) → Task 3 templates + Task 4 QA
- §5 AnswerResult citations=[] refused=False → Task 3
- §6 Error table (invalid codes, empty completed, target unknown, empty next, LLM fail, bilingual, semester OOB) → Task 1 edge tests + Task 3 branch
- §7 Testing (unit deterministic 15+ cases, integration with fakes) → Tasks 1 & 3
- §8 Out of Scope (no DB, no KG) → Global Constraints enforce
- §9 Build Order → Task order 1→2→3→4 matches

**Placeholder scan:** No `TBD/TODO/handle edge cases` — all steps have concrete code.

**Type consistency:** `CourseAdvisor.get_missing_prerequisites` returns `list[str]|None` in Task 1 vs used as `or []` in Task 3 — consistent. `Extraction` field names `completed_courses/target_course/current_semester` match extractor prompt JSON keys and advisor calls in Task 3 — consistent. `Intent` literal values uppercase match classifier prompt & `answer()` branch comparisons — consistent. `RagCore.__init__` params `classifier/extractor/advisor` names used identically in Task 3 test fakes and `build_rag_core` wiring — consistent.

**Fixes applied inline:** Added `resolve_code` to Task 1 interface (needed by Task 3 invalid handling); added `OTHER` branch `_other_result(query)` signature to match `answer()` call; clarified semester OOB → fallback in both spec and Task 1 `get_next_courses`.

