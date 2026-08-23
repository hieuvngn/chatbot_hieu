from pathlib import Path
from rag_core.course_advisor import CourseAdvisor

DATA_DIR = Path("data")

def test_get_prerequisites_known() -> None:
    adv = CourseAdvisor(DATA_DIR)
    assert adv.get_prerequisites("CS223") == ["CS112", "STAT101"]  # from COURSE_TABLE

def test_get_prerequisites_unknown() -> None:
    adv = CourseAdvisor(DATA_DIR)
    assert adv.get_prerequisites("FAKE999") is None

def test_is_eligible_true() -> None:
    adv = CourseAdvisor(DATA_DIR)
    assert adv.is_eligible("CS211", {"CS112"}) is True  # CS211 prereq [CS112]

def test_is_eligible_false_missing() -> None:
    adv = CourseAdvisor(DATA_DIR)
    assert adv.is_eligible("CS214", {"CS112"}) is False  # CS214 needs CS102+CS112

def test_get_missing_partial() -> None:
    adv = CourseAdvisor(DATA_DIR)
    assert adv.get_missing_prerequisites("CS223", {"CS112"}) == ["STAT101"]

def test_get_missing_none_when_target_unknown() -> None:
    adv = CourseAdvisor(DATA_DIR)
    assert adv.get_missing_prerequisites("FAKE", {"CS101"}) is None

def test_get_next_courses_sort_asc() -> None:
    adv = CourseAdvisor(DATA_DIR)
    nxt = adv.get_next_courses({"CS101"}, None)
    # Must be sorted by (semester, code) and not contain CS101
    assert "CS101" not in {c.code for c in nxt}
    assert nxt == sorted(nxt, key=lambda c: (c.semester, c.code))
    assert any(c.code == "CS112" for c in nxt)  # CS112 prereq [CS101] eligible

def test_get_next_courses_with_semester_filter() -> None:
    adv = CourseAdvisor(DATA_DIR)
    nxt_all = adv.get_next_courses({"CS101", "CS102", "CS112"}, None)
    nxt_f4 = adv.get_next_courses({"CS101", "CS102", "CS112"}, 4)
    assert all(c.semester >= 4 for c in nxt_f4)
    assert len(nxt_f4) <= len(nxt_all)

def test_get_next_courses_empty_completed() -> None:
    adv = CourseAdvisor(DATA_DIR)
    nxt = adv.get_next_courses(set(), None)
    # Only courses with no prereq are eligible (dataset has GE101 sem2, CS416 sem7, CS427 sem8 beyond sem1)
    assert all(len(c.prerequisites) == 0 for c in nxt)
    assert nxt == sorted(nxt, key=lambda c: (c.semester, c.code))
    assert {"CS101", "CS102", "CS103", "CS104", "MATH101", "MATH102"}.issubset({c.code for c in nxt})

def test_resolve_code_normalization() -> None:
    adv = CourseAdvisor(DATA_DIR)
    assert adv.resolve_code(" cs223 ") == "CS223"
    assert adv.resolve_code("fake") is None

def test_is_eligible_unknown_code() -> None:
    adv = CourseAdvisor(DATA_DIR)
    assert adv.is_eligible("FAKE", {"CS101"}) is False

def test_get_missing_all_completed() -> None:
    adv = CourseAdvisor(DATA_DIR)
    assert adv.get_missing_prerequisites("CS211", {"CS112"}) == []
