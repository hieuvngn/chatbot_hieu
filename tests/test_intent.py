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
        assert e == Extraction(completed_courses=[], target_course=None, current_semester=None)
        assert e.completed_courses == []
