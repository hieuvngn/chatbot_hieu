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
