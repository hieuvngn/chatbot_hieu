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
    def generate(self, q, chunks, feedback=None, *, skill_instructions=""):
        return "dummy RAG answer [1]"

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
