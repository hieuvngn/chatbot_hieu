from __future__ import annotations

import json
from pathlib import Path

import eval as ev
from rag_core.models import Citation, Source

TRUTH = ev.GroundTruth(document_id="DOC-004", chapter="Phần 1: IoT")


def _source(document_id: str = "DOC-004", chapter: str = "Phần 1: IoT") -> Source:
    return Source(
        document_id=document_id,
        document_title="Bài giảng: IoT và hệ thống nhúng (CS426)",
        chapter=chapter,
        course_code="CS426",
        kind="slides",
        language="vi",
    )


def _citation(document_id: str, chapter: str) -> Citation:
    return Citation(marker="[1]", source=_source(document_id, chapter))


def _write_eval_set(path: Path, entries: object) -> None:
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")


def _expect_value_error(path: Path) -> None:
    import pytest

    with pytest.raises(ValueError):
        ev.load_eval_set(path)


def test_load_eval_set_parses_entries(tmp_path: Path) -> None:
    _write_eval_set(
        tmp_path / "eval.json",
        [
            {
                "id": "e1",
                "question": "IoT là gì?",
                "language": "vi",
                "ground_truth": {"document_id": "DOC-004", "chapter": "Phần 1: IoT"},
            }
        ],
    )

    items = ev.load_eval_set(tmp_path / "eval.json")

    assert len(items) == 1
    assert items[0].id == "e1"
    assert items[0].question == "IoT là gì?"
    assert items[0].language == "vi"
    assert items[0].ground_truth == TRUTH


def test_load_eval_set_rejects_non_list(tmp_path: Path) -> None:
    _write_eval_set(tmp_path / "eval.json", {"id": "e1"})

    _expect_value_error(tmp_path / "eval.json")


def test_load_eval_set_rejects_missing_fields(tmp_path: Path) -> None:
    _write_eval_set(
        tmp_path / "eval.json",
        [
            {
                "id": "e1",
                "question": "IoT là gì?",
                "language": "vi",
            }
        ],
    )

    _expect_value_error(tmp_path / "eval.json")


def test_load_eval_set_rejects_missing_ground_truth_fields(tmp_path: Path) -> None:
    _write_eval_set(
        tmp_path / "eval.json",
        [
            {
                "id": "e1",
                "question": "IoT là gì?",
                "language": "vi",
                "ground_truth": {"document_id": "DOC-004"},
            }
        ],
    )

    _expect_value_error(tmp_path / "eval.json")


def test_load_eval_set_rejects_unsupported_language(tmp_path: Path) -> None:
    _write_eval_set(
        tmp_path / "eval.json",
        [
            {
                "id": "e1",
                "question": "What is IoT?",
                "language": "fr",
                "ground_truth": {"document_id": "DOC-004", "chapter": "Phần 1: IoT"},
            }
        ],
    )

    _expect_value_error(tmp_path / "eval.json")


def test_is_hit_matches_ground_truth_source() -> None:
    assert ev.is_hit([_source()], TRUTH)
    assert ev.is_hit([_source("DOC-005", "Section 2: forecasting"), _source()], TRUTH)


def test_is_hit_false_for_wrong_document_or_chapter() -> None:
    assert not ev.is_hit([_source("DOC-005", "Section 2: forecasting")], TRUTH)
    assert not ev.is_hit([_source(document_id="DOC-004", chapter="Phần 2: vi điều khiển")], TRUTH)
    assert not ev.is_hit([], TRUTH)


def test_hit_rate_is_mean_of_hits() -> None:
    assert ev.hit_rate([True, True, False, True]) == 0.75
    assert ev.hit_rate([]) == 0.0


def test_citation_precision_counts_citations_to_ground_truth() -> None:
    citations = [
        _citation("DOC-004", "Phần 1: IoT"),
        _citation("DOC-005", "Section 2: forecasting"),
        _citation("DOC-004", "Phần 2: vi điều khiển"),
    ]

    assert ev.citation_precision(citations, TRUTH) == 1 / 3


def test_citation_precision_zero_without_citations() -> None:
    assert ev.citation_precision([], TRUTH) == 0.0
    assert ev.citation_precision([_citation("DOC-005", "Section 2: forecasting")], TRUTH) == 0.0
    assert ev.citation_precision([_citation("DOC-004", "Phần 1: IoT")], TRUTH) == 1.0


def test_mean_precision_averages_per_question_precision() -> None:
    assert ev.mean_precision([1.0, 0.5, 0.0]) == 0.5
    assert ev.mean_precision([]) == 0.0


def test_answer_precision_is_none_for_refusals() -> None:
    from rag_core import AnswerResult

    refused = AnswerResult(answer="", citations=[], sources=[], refused=True)
    answered = AnswerResult(
        answer="IoT là gì. [1]",
        citations=[_citation("DOC-004", "Phần 1: IoT")],
        sources=[_source()],
    )

    assert ev.answer_precision(refused, TRUTH) is None
    assert ev.answer_precision(answered, TRUTH) == 1.0


def test_format_table_renders_rows_with_percentages() -> None:
    rows = [
        ev.EvalRow(
            item=ev.EvalItem(id="e1", question="IoT là gì?", language="vi", ground_truth=TRUTH),
            hit_naive=True,
            hit_full=True,
            precision_naive=0.5,
            precision_full=1.0,
            refused_full=False,
        ),
        ev.EvalRow(
            item=ev.EvalItem(
                id="e2", question="What is forecasting?", language="en",
                ground_truth=ev.GroundTruth(document_id="DOC-005", chapter="Section 2: forecasting"),
            ),
            hit_naive=False,
            hit_full=True,
            precision_naive=0.0,
            precision_full=0.0,
            refused_full=True,
        ),
    ]

    table = ev.format_table(rows)

    assert "e1" in table and "e2" in table
    assert "IoT là gì?" in table
    assert "50.0%" in table and "100.0%" in table
    assert "0.0%" in table
    assert "n/a" not in table


def test_format_table_renders_na_marker_for_retrieval_only_rows() -> None:
    rows = [
        ev.EvalRow(
            item=ev.EvalItem(id="e1", question="IoT là gì?", language="vi", ground_truth=TRUTH),
            hit_naive=True,
            hit_full=True,
            precision_naive=None,
            precision_full=None,
            refused_full=False,
        )
    ]

    table = ev.format_table(rows)

    assert "n/a" in table


def test_format_summary_renders_metrics_table() -> None:
    summary = ev.EvalSummary(
        hit_rate_naive=0.5,
        hit_rate_full=0.9,
        citation_precision_naive=0.4,
        citation_precision_full=0.7,
        refusals_full=3,
        total=20,
    )

    text = ev.format_summary(summary)

    assert "hit@5" in text
    assert "citation precision" in text
    assert "50.0%" in text and "90.0%" in text
    assert "40.0%" in text and "70.0%" in text
    assert "3 / 20" in text


def test_hand_made_eval_set_is_loaded_and_bilingual() -> None:
    items = ev.load_eval_set(ev.DEFAULT_EVAL_SET)

    assert len(items) == 20
    assert sum(i.language == "vi" for i in items) >= 8
    assert sum(i.language == "en" for i in items) >= 8


def test_hand_made_eval_set_ground_truths_exist_in_dataset() -> None:
    import generate_data as gd

    ds = gd.generate(seed=gd.SEED)
    chapters = {
        (d.id, ch.title)
        for d in ds.documents
        for ch in d.chapters
    }

    items = ev.load_eval_set(ev.DEFAULT_EVAL_SET)

    for item in items:
        truth = (item.ground_truth.document_id, item.ground_truth.chapter)
        assert truth in chapters, f"ground truth {truth} not in the seeded dataset"