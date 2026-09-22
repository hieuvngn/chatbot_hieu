"""Loader + chunker for the structured entity tables.

Each entity type is read from its own JSON file (``departments.json``,
``instructors.json``, ``programs.json``, ``terms.json``) and converted to a
``Chunk`` whose ``Source`` carries ``entity_type`` + ``entity_id``. The chunks
flow through the same hybrid index as document chapters; only the
``dedupe_by_source`` key changes so an entity never appears twice in the same
top-K result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rag_core.models import (
    Chunk,
    Department,
    EntityBundle,
    Instructor,
    Program,
    Source,
    Term,
)

ENTITY_FILE_NAMES: tuple[str, ...] = (
    "departments.json",
    "instructors.json",
    "programs.json",
    "terms.json",
)


def _read_json(path: Path) -> list[object]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_entity_bundle(data_dir: Path) -> EntityBundle:
    """Read the four entity files from ``data_dir`` into an EntityBundle.

    Missing files yield empty lists rather than raising — the rest of the
    pipeline keeps working with whatever tables exist.
    """
    departments: list[Department] = []
    for raw in _read_json(data_dir / "departments.json"):
        assert isinstance(raw, dict)
        departments.append(
            Department(
                id=str(raw["id"]),
                name=str(raw["name"]),
                name_en=str(raw["name_en"]),
            )
        )

    instructors: list[Instructor] = []
    for raw in _read_json(data_dir / "instructors.json"):
        assert isinstance(raw, dict)
        courses_raw = raw.get("courses", [])
        course_pairs: list[tuple[str, str]] = []
        if isinstance(courses_raw, list):
            for entry in courses_raw:
                if isinstance(entry, dict):
                    course_pairs.append(
                        (str(entry.get("course_code", "")), str(entry.get("role", "")))
                    )
        instructors.append(
            Instructor(
                id=str(raw["id"]),
                name=str(raw["name"]),
                title=str(raw.get("title", "")),
                email=str(raw.get("email", "")),
                department_id=str(raw.get("department_id", "")),
                bio=str(raw.get("bio", "")),
                courses=tuple(course_pairs),
            )
        )

    programs: list[Program] = []
    for raw in _read_json(data_dir / "programs.json"):
        assert isinstance(raw, dict)
        programs.append(
            Program(
                id=str(raw["id"]),
                name=str(raw["name"]),
                name_en=str(raw["name_en"]),
                department_id=str(raw.get("department_id", "")),
                total_credits=int(raw.get("total_credits", 0)),
                required_courses=tuple(str(c) for c in raw.get("required_courses", [])),
                elective_courses=tuple(str(c) for c in raw.get("elective_courses", [])),
            )
        )

    terms: list[Term] = []
    for raw in _read_json(data_dir / "terms.json"):
        assert isinstance(raw, dict)
        offered_raw = raw.get("offered", [])
        offered_pairs: list[tuple[str, str, str]] = []
        if isinstance(offered_raw, list):
            for entry in offered_raw:
                if isinstance(entry, dict):
                    offered_pairs.append(
                        (
                            str(entry.get("course_code", "")),
                            str(entry.get("instructor_id", "")),
                            str(entry.get("schedule", "")),
                        )
                    )
        terms.append(
            Term(
                id=str(raw["id"]),
                name=str(raw["name"]),
                year=int(raw.get("year", 0)),
                season=str(raw.get("season", "")),
                start_date=str(raw.get("start_date", "")),
                end_date=str(raw.get("end_date", "")),
                offered=tuple(offered_pairs),
            )
        )

    return EntityBundle(
        departments=tuple(departments),
        instructors=tuple(instructors),
        programs=tuple(programs),
        terms=tuple(terms),
    )

def _department_text(department: Department) -> str:
    return (
        f"{department.name} ({department.name_en}), mã khoa {department.id}."
    )


def _instructor_text(instructor: Instructor) -> str:
    courses_str = instructor.display_courses() or "chưa phân công"
    title = f"{instructor.title} " if instructor.title else ""
    return (
        f"Giảng viên {title}{instructor.name}, mã {instructor.id}, "
        f"email {instructor.email}, thuộc khoa {instructor.department_id}. "
        f"Phụ trách: {courses_str}. "
        f"Tiểu sử: {instructor.bio}"
    )


def _program_text(program: Program) -> str:
    required = ", ".join(program.required_courses) or "không có"
    elective = ", ".join(program.elective_courses) or "không có"
    return (
        f"Chương trình đào tạo {program.name} ({program.name_en}), mã "
        f"{program.id}, thuộc khoa {program.department_id}, tổng {program.total_credits} "
        f"tín chỉ. Môn bắt buộc: {required}. Môn tự chọn: {elective}."
    )


def _term_text(term: Term) -> str:
    offered_list = []
    for course_code, instructor_id, schedule in term.offered:
        offered_list.append(f"{course_code} ({instructor_id}, {schedule})")
    offered_str = "; ".join(offered_list) or "chưa có lịch mở"
    return (
        f"Học kỳ {term.name} ({term.id}), mùa {term.season} năm {term.year}, "
        f"từ {term.start_date} đến {term.end_date}. Các môn mở: {offered_str}."
    )


def _source_for_entity(entity_type: str, entity_id: str, title: str) -> Source:
    return Source(
        document_id=entity_id,
        document_title=title,
        chapter=entity_type,
        course_code="",
        kind=entity_type,
        language="vi",
        entity_type=entity_type,
        entity_id=entity_id,
    )


def chunk_entity_bundle(bundle: EntityBundle) -> list[Chunk]:
    """Convert each entity into a single Chunk for the hybrid index.

    Entities are short enough to fit on a single index unit; no sliding window
    is applied.
    """
    chunks: list[Chunk] = []
    for department in bundle.departments:
        chunks.append(
            Chunk(
                source=_source_for_entity("department", department.id, department.name),
                text=_department_text(department),
            )
        )
    for instructor in bundle.instructors:
        chunks.append(
            Chunk(
                source=_source_for_entity(
                    "instructor", instructor.id, instructor.name
                ),
                text=_instructor_text(instructor),
            )
        )
    for program in bundle.programs:
        chunks.append(
            Chunk(
                source=_source_for_entity("program", program.id, program.name),
                text=_program_text(program),
            )
        )
    for term in bundle.terms:
        chunks.append(
            Chunk(
                source=_source_for_entity("term", term.id, term.name),
                text=_term_text(term),
            )
        )
    return chunks


__all__ = [
    "ENTITY_FILE_NAMES",
    "chunk_entity_bundle",
    "load_entity_bundle",
]