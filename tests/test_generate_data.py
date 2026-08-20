from __future__ import annotations

import json
from pathlib import Path

import generate_data as gd

SEED = 42


def dataset() -> gd.Dataset:
    return gd.generate(seed=SEED)


def _course_topics(d: gd.Document) -> list[str]:
    """The topic phrases for a document's course, in the document's language."""
    topics = gd.TOPICS[d.course_code]
    return [t[0] if d.language == "vi" else t[1] for t in topics]


def test_generates_50_courses() -> None:
    ds = dataset()
    assert len(ds.courses) == 50


def test_every_course_has_all_required_fields() -> None:
    ds = dataset()
    for c in ds.courses:
        assert c.code
        assert c.name
        assert c.name_en
        assert isinstance(c.credits, int) and 1 <= c.credits <= 6
        assert isinstance(c.prerequisites, list)
        assert isinstance(c.semester, int) and 1 <= c.semester <= 8
        assert c.department
        assert c.instructor
        assert c.description


def test_course_codes_are_unique() -> None:
    ds = dataset()
    codes = [c.code for c in ds.courses]
    assert len(codes) == len(set(codes))


def test_prerequisite_codes_refer_to_known_courses() -> None:
    ds = dataset()
    codes = {c.code for c in ds.courses}
    for c in ds.courses:
        for prereq in c.prerequisites:
            assert prereq in codes, f"{c.code} depends on unknown course {prereq}"


def test_prerequisite_graph_is_acyclic() -> None:
    ds = dataset()
    graph = {c.code: set(c.prerequisites) for c in ds.courses}

    indegree = {node: len(deps) for node, deps in graph.items()}
    queue = [node for node, deg in indegree.items() if deg == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for other, deps in graph.items():
            if node in deps:
                indegree[other] -= 1
                if indegree[other] == 0:
                    queue.append(other)

    assert visited == len(graph), "prerequisite graph contains a cycle"


def test_generate_asserts_acyclicity_without_raising() -> None:
    dataset()


def test_prerequisites_strictly_precede_their_course() -> None:
    ds = dataset()
    by_code = {c.code: c for c in ds.courses}
    for c in ds.courses:
        for prereq in c.prerequisites:
            assert (
                by_code[prereq].semester < c.semester
            ), f"{c.code} (semester {c.semester}) depends on {prereq} (semester {by_code[prereq].semester})"


def test_generates_40_documents() -> None:
    ds = dataset()
    assert len(ds.documents) == 40


def test_documents_reference_known_courses() -> None:
    ds = dataset()
    codes = {c.code for c in ds.courses}
    for d in ds.documents:
        assert d.course_code in codes


def test_documents_are_mixed_vietnamese_and_english() -> None:
    ds = dataset()
    langs = {d.language for d in ds.documents}
    assert langs == {"vi", "en"}


def test_documents_have_chapters_with_content() -> None:
    ds = dataset()
    for d in ds.documents:
        assert d.kind in {"slides", "textbook"}
        assert d.title
        assert len(d.chapters) >= 2
        for ch in d.chapters:
            assert ch.title
            assert ch.content


def test_document_content_references_its_course() -> None:
    ds = dataset()
    courses = {c.code: c for c in ds.courses}
    for d in ds.documents:
        course = courses[d.course_code]
        text = " ".join(ch.content for ch in d.chapters).lower()
        assert (
            course.code.lower() in text
            or course.name.lower() in text
            or course.name_en.lower() in text
        ), f"document {d.id} does not reference its course {course.code}"


def test_document_content_references_topics() -> None:
    ds = dataset()
    for d in ds.documents:
        text = " ".join(ch.content for ch in d.chapters).lower()
        hits = [t for t in _course_topics(d) if t.lower() in text]
        assert hits, (
            f"document {d.id} ({d.language}) references none of its course's "
            f"topics: {_course_topics(d)}"
        )


def test_generation_is_deterministic_for_same_seed() -> None:
    first = gd.generate(seed=7)
    second = gd.generate(seed=7)
    assert first == second


def test_generation_varies_across_seeds() -> None:
    a = gd.generate(seed=7)
    b = gd.generate(seed=8)
    assert a != b


def test_write_creates_json_files(tmp_path: Path) -> None:
    ds = dataset()
    gd.write(ds, tmp_path)
    courses_file = tmp_path / "courses.json"
    documents_file = tmp_path / "documents.json"
    assert courses_file.exists()
    assert documents_file.exists()

    courses = json.loads(courses_file.read_text())
    documents = json.loads(documents_file.read_text())
    assert len(courses) == 50
    assert len(documents) == 40


def test_writing_is_deterministic(tmp_path: Path) -> None:
    gd.write(gd.generate(seed=3), tmp_path)
    files_1 = {
        name: (tmp_path / name).read_bytes()
        for name in ("courses.json", "documents.json")
    }

    other = tmp_path / "other"
    gd.write(gd.generate(seed=3), other)
    files_2 = {
        name: (other / name).read_bytes()
        for name in ("courses.json", "documents.json")
    }

    assert files_1 == files_2


def test_cli_writes_data_dir(tmp_path: Path) -> None:
    from subprocess import run

    out = tmp_path / "out"
    result = run(
        ["python", "-m", "generate_data", "--seed", str(SEED), "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "courses.json").exists()
    assert (out / "documents.json").exists()