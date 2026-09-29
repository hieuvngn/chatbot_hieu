from collections.abc import Iterator
from pathlib import Path
from rag_core import RagCore, Session
from rag_core.course_advisor import CourseAdvisor
from rag_core.intent import Extraction, Intent
from rag_core.models import Chunk
from rag_core.skills import Skill

class FakeClassifier:
    def __init__(self, intent: Intent) -> None:
        self.intent = intent
    def classify(self, q: str) -> Intent:
        return self.intent

class FakeExtractor:
    def __init__(self, ext: Extraction) -> None:
        self.ext = ext
    def extract(self, q: str) -> Extraction:
        return self.ext

class DummyEmbedder:
    def embed_batch(self, texts: list[str]) -> list[list[float]]: return [[0.0]*8 for _ in texts]
    def embed_query(self, q: str) -> list[float]: return [0.0]*8

class DummyGenerator:
    def generate(
        self,
        q: str,
        chunks: list[Chunk],
        feedback: str | None = None,
        *,
        skill_instructions: str = "",
    ) -> str:
        return "dummy RAG answer [1]"

    def stream(
        self,
        q: str,
        chunks: list[Chunk],
        feedback: str | None = None,
        *,
        skill_instructions: str = "",
    ) -> Iterator[str]:
        yield self.generate(q, chunks, feedback, skill_instructions=skill_instructions)

DATA_DIR = Path("data")
SESSION = Session(id="s1", user_id="u1", turns=[])

def _core(intent: Intent, extraction: Extraction) -> RagCore:
    return RagCore(
        data_dir=DATA_DIR,
        embedder=DummyEmbedder(),
        generator=DummyGenerator(),
        classifier=FakeClassifier(intent),
        extractor=FakeExtractor(extraction),
        advisor=CourseAdvisor(DATA_DIR),
    )

def test_advisor_next_courses() -> None:
    core = _core("COURSE_ADVISOR", Extraction(completed_courses=["CS101","CS112"], target_course=None, current_semester=4))
    res = core.answer("Tôi đã học CS101 và CS112, học gì tiếp kỳ 4?", SESSION)
    assert res.citations == [] and not res.refused
    assert "CS211" in res.answer or "CS" in res.answer  # at least one eligible

def test_advisor_eligibility_missing() -> None:
    core = _core("COURSE_ADVISOR", Extraction(completed_courses=["CS112"], target_course="CS223", current_semester=None))
    res = core.answer("Tôi học CS112, đủ học AI không?", SESSION)
    assert "STAT101" in res.answer  # missing prereq
    assert "thiếu" in res.answer.lower() or "missing" in res.answer.lower()

def test_advisor_asks_when_no_completed() -> None:
    core = _core("COURSE_ADVISOR", Extraction(completed_courses=[], target_course="CS223", current_semester=None))
    res = core.answer("Tôi có thể học AI không?", SESSION)
    assert "cho biết" in res.answer.lower() or "completed" in res.answer.lower()
    assert "CS223" in res.answer or "STAT101" in res.answer

def test_advisor_invalid_code() -> None:
    core = _core("COURSE_ADVISOR", Extraction(completed_courses=["PF101"], target_course=None, current_semester=None))
    res = core.answer("Tôi học PF101, học gì tiếp?", SESSION)
    assert "PF101" in res.answer and ("Không tìm thấy" in res.answer or "not found" in res.answer.lower())

def test_knowledge_qa_goes_to_rag() -> None:
    core = _core("KNOWLEDGE_QA", Extraction([], None, None))
    res = core.answer("Giải thích bảng băm là gì?", SESSION)
    assert "dummy RAG" in res.answer

def test_other_returns_guidance() -> None:
    core = _core("OTHER", Extraction([], None, None))
    res = core.answer("Thời tiết hôm nay?", SESSION)
    assert "tư vấn môn học" in res.answer.lower() or "course" in res.answer.lower()


SKILLS = [Skill(name="eli5", description="d", instructions="Use simple words.")]


class SpySelector:
    def __init__(self) -> None:
        self.calls = 0

    def select(self, query: str, skills: list[Skill]) -> list[str]:
        self.calls += 1
        return ["eli5"]


def test_advisor_branch_does_not_call_selector() -> None:
    selector = SpySelector()
    core = RagCore(
        data_dir=DATA_DIR,
        embedder=DummyEmbedder(),
        generator=DummyGenerator(),
        classifier=FakeClassifier("COURSE_ADVISOR"),
        extractor=FakeExtractor(Extraction([], None, None)),
        advisor=CourseAdvisor(DATA_DIR),
        skills=SKILLS,
        skill_selector=selector,
    )

    core.answer("Tôi đã học CS101, học gì tiếp?", SESSION)

    assert selector.calls == 0


def test_knowledge_qa_marks_skills_applied() -> None:
    selector = SpySelector()
    core = RagCore(
        data_dir=DATA_DIR,
        embedder=DummyEmbedder(),
        generator=DummyGenerator(),
        classifier=FakeClassifier("KNOWLEDGE_QA"),
        extractor=FakeExtractor(Extraction([], None, None)),
        advisor=CourseAdvisor(DATA_DIR),
        skills=SKILLS,
        skill_selector=selector,
    )

    res = core.answer("Giải thích bảng băm là gì?", SESSION)

    assert selector.calls == 1
    assert res.skills_applied == ["eli5"]
