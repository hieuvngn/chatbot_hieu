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


def test_generates_at_least_40_documents_by_default() -> None:
    ds = dataset()
    assert len(ds.documents) >= 40



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
        assert d.kind in gd.ALL_DOCUMENT_KINDS
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

def test_default_dataset_has_no_extra_documents() -> None:
    ds = dataset()
    extra_ids = [d for d in ds.documents if int(d.id.split("-")[1]) > 40]
    assert extra_ids == []


def test_extra_documents_only_added_when_requested() -> None:
    base = gd.generate(seed=SEED)
    extended = gd.generate(seed=SEED, extra_documents=20)
    assert len(extended.documents) == len(base.documents) + 20
    extra_kinds = {d.kind for d in extended.documents[len(base.documents):]}
    assert extra_kinds.issubset(set(gd.EXTRA_DOCUMENT_KINDS))


def test_extra_documents_have_unique_ids() -> None:
    ds = gd.generate(seed=SEED, extra_documents=20)
    ids = [d.id for d in ds.documents]
    assert len(ids) == len(set(ids))
    assert ids[-1] == f"DOC-{gd.NUM_DOCUMENTS + 20:03d}"


def test_extra_documents_cover_all_new_kinds() -> None:
    ds = gd.generate(seed=SEED, extra_documents=20)
    kinds = {d.kind for d in ds.documents if int(d.id.split("-")[1]) > 40}
    for kind in gd.EXTRA_DOCUMENT_KINDS:
        assert kind in kinds, f"missing kind {kind} in extra documents"


def test_extra_documents_have_vietnamese_majority() -> None:
    ds = gd.generate(seed=SEED, extra_documents=40)
    extras = [d for d in ds.documents if int(d.id.split("-")[1]) > 40]
    vi = sum(1 for d in extras if d.language == "vi")
    en = sum(1 for d in extras if d.language == "en")
    assert vi > en


def test_extra_documents_reference_their_course() -> None:
    ds = gd.generate(seed=SEED, extra_documents=20)
    courses = {c.code: c for c in ds.courses}
    extras = [d for d in ds.documents if int(d.id.split("-")[1]) > 40]
    for d in extras:
        course = courses[d.course_code]
        text = " ".join(ch.content for ch in d.chapters).lower()
        assert (
            course.code.lower() in text
            or course.name.lower() in text
            or course.name_en.lower() in text
        ), f"document {d.id} does not reference its course {course.code}"


def test_extra_documents_reference_topics() -> None:
    ds = gd.generate(seed=SEED, extra_documents=20)
    extras = [d for d in ds.documents if int(d.id.split("-")[1]) > 40]
    for d in extras:
        text = " ".join(ch.content for ch in d.chapters).lower()
        hits = [t for t in _course_topics(d) if t.lower() in text]
        assert hits, (
            f"document {d.id} ({d.language}) references none of its course's "
            f"topics: {_course_topics(d)}"
        )


def test_extra_documents_are_deterministic() -> None:
    first = gd.generate(seed=SEED, extra_documents=20)
    second = gd.generate(seed=SEED, extra_documents=20)
    extras_1 = [d for d in first.documents if int(d.id.split("-")[1]) > 40]
    extras_2 = [d for d in second.documents if int(d.id.split("-")[1]) > 40]
    assert extras_1 == extras_2


def test_ground_truths_still_present_when_extras_added() -> None:
    ds = gd.generate(seed=SEED, extra_documents=40)
    import eval as ev

    items = ev.load_eval_set(ev.DEFAULT_EVAL_SET)
    chapters = {(d.id, ch.title) for d in ds.documents for ch in d.chapters}
    for item in items:
        truth = (item.ground_truth.document_id, item.ground_truth.chapter)
        assert truth in chapters, f"ground truth {truth} lost in extended dataset"


def test_llm_client_can_rewrite_extra_section() -> None:
    called: list[tuple[str, str, str, str]] = []

    class StubClient(gd._LLMClient):
        def rewrite_section(
            self, content: str, kind: str, language: str, course: gd.Course
        ) -> str | None:
            called.append((course.code, kind, language, content[:20]))
            return content + "\n\n[LLM-rewritten]"

    ds = gd.generate(seed=SEED, extra_documents=8, llm_client=StubClient())
    extras = [d for d in ds.documents if int(d.id.split("-")[1]) > 40]
    assert extras, "expected extras to be generated"
    # Each extra document has intro + summary rewritten (2 calls per doc)
    assert len(called) == 8 * 2
    # Rewrite only applied to extra kinds
    assert all(kind in gd.EXTRA_DOCUMENT_KINDS for _, kind, _, _ in called)
    # LLM tail marker should appear somewhere in the extra doc bodies
    joined = " ".join(ch.content for d in extras for ch in d.chapters)
    assert "[LLM-rewritten]" in joined


def test_llm_client_failure_falls_back_to_template() -> None:
    class FailingClient(gd._LLMClient):
        def rewrite_section(
            self, content: str, kind: str, language: str, course: gd.Course
        ) -> str | None:
            raise RuntimeError("LLM offline")

    ds = gd.generate(seed=SEED, extra_documents=4, llm_client=FailingClient())
    extras = [d for d in ds.documents if int(d.id.split("-")[1]) > 40]
    joined = " ".join(ch.content for d in extras for ch in d.chapters)
    # No marker injected, but content still references its course (template path)
    for d in extras:
        course = next(c for c in ds.courses if c.code == d.course_code)
        assert (
            course.code.lower() in joined.lower()
            or course.name.lower() in joined.lower()
        )
