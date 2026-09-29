"""Generate the structured entity tables (Department, Instructor, Program, Term).

The synthetic data for the RAG demo originally lived in two files
(``courses.json`` + ``documents.json``). This script adds four sibling files
that hold entities which the pipeline can also retrieve and cite:

* ``departments.json`` — academic departments
* ``instructors.json`` — people who teach one or more courses
* ``programs.json``   — curricula grouping courses into required/elective tracks
* ``terms.json``      — specific semester offerings with a per-term instructor override

The four files are deterministic for a given seed and exist alongside the
existing ``generate_data.py`` output; that script is intentionally untouched.

Usage::

    uv run python -m generate_entities --seed 42
    uv run python -m generate_entities --seed 42 --out ./data

The output is intentionally small (a handful of records per file) — the demo's
purpose is to exercise the entity-aware retrieval and citation paths, not to
ship a realistic roster.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

SEED = 42

DATA_DIR = Path(__file__).parent / "data"

# Small fixed pool of Vietnamese full names.
INSTRUCTOR_NAMES: tuple[tuple[str, str], ...] = (
    ("Nguyễn Văn Anh", "Ts."),
    ("Trần Thị Bình", "PGS.Ts"),
    ("Lê Văn Cường", "ThS."),
    ("Phạm Thu Hà", "TS."),
    ("Hoàng Minh Đức", "ThS."),
    ("Vũ Thị Hương", "PGS.Ts"),
    ("Đặng Quốc Huy", "TS."),
    ("Bùi Thanh Hằng", "ThS."),
    ("Đỗ Hồng Nhung", "TS."),
    ("Ngô Đức Thắng", "ThS."),
    ("Trương Quang Vinh", "PGS.Ts"),
    ("Lý Gia Bảo", "TS."),
    ("Võ Thu Trang", "ThS."),
    ("Hà Anh Tuấn", "TS."),
    ("Dương Thúy An", "ThS."),
    ("Phan Công Khánh", "PGS.Ts"),
)

# Departments — keyed by short id; the longer name is the human label.
DEPARTMENTS: tuple[tuple[str, str, str], ...] = (
    ("CNTT", "Khoa Công nghệ thông tin", "Faculty of Information Technology"),
    ("MATH", "Khoa Toán - Thống kê", "Faculty of Mathematics and Statistics"),
    ("GE", "Bộ môn Đại cương", "General Education Department"),
)

# Programs — chosen to exercise the required/elective split.
PROGRAMS: tuple[tuple[str, str, str, str, int], ...] = (
    ("PR-CNTT", "Công nghệ thông tin", "Information Technology", "CNTT", 130),
    ("PR-AI", "Trí tuệ nhân tạo", "Artificial Intelligence", "CNTT", 132),
    ("PR-DS", "Khoa học dữ liệu", "Data Science", "CNTT", 128),
)

# Term roster: keep the calendar tight so the demo has concrete values to cite.
TERMS: tuple[tuple[str, int, str, str, str], ...] = (
    ("T-2025-S1", 2025, "spring", "2025-01-15", "2025-05-15"),
    ("T-2025-S2", 2025, "fall", "2025-08-15", "2025-12-15"),
    ("T-2026-S1", 2026, "spring", "2026-01-15", "2026-05-15"),
)


@dataclass(frozen=True)
class Dataset:
    departments: list[dict[str, str]]
    instructors: list[dict[str, object]]
    programs: list[dict[str, object]]
    terms: list[dict[str, object]]


def _slug_email(name: str, idx: int) -> str:
    base = (
        name.lower()
        .replace(" ", ".")
        .replace("ă", "a")
        .replace("â", "a")
        .replace("ê", "e")
        .replace("ô", "o")
        .replace("ơ", "o")
        .replace("ư", "u")
        .replace("đ", "d")
    )
    # The @example.com domain signals synthetic origin and avoids leaking PII.
    return f"{base}{idx}@example.com"


def _bio_vi(name: str, title: str, dept_name: str) -> str:
    return (
        f"{title} {name} công tác tại {dept_name}. Giảng viên có kinh nghiệm "
        f"giảng dạy và hướng dẫn nhiều học phần trong chương trình, tham gia "
        f"các đề tài nghiên cứu sinh viên và phản biện đồ án."
    )


def _department_lookup() -> dict[str, str]:
    """Map ``Course.department`` (full Vietnamese name) → department id."""
    return {name_vi: d_id for d_id, name_vi, _ in DEPARTMENTS}


def _build_departments() -> list[dict[str, str]]:
    return [
        {"id": d_id, "name": name_vi, "name_en": name_en}
        for d_id, name_vi, name_en in DEPARTMENTS
    ]


def _build_instructors(courses: list[dict[str, object]]) -> list[dict[str, object]]:
    dept_lookup = _department_lookup()
    department_id_by_course: dict[str, str] = {}
    for c in courses:
        department_id_by_course[str(c["code"])] = dept_lookup.get(
            str(c.get("department", "")), "CNTT"
        )

    instructors: list[dict[str, object]] = []
    for idx, (name, title) in enumerate(INSTRUCTOR_NAMES, start=1):
        instructor_id = f"INS-{idx:03d}"
        teaching: list[tuple[str, str]] = []
        for code in sorted(department_id_by_course):
            # Deterministic but varied: distribute the roster evenly.
            if (hash((instructor_id, code)) % len(INSTRUCTOR_NAMES)) == (
                idx % len(INSTRUCTOR_NAMES)
            ):
                teaching.append((code, "lecturer"))
        if not teaching:
            # Guarantee at least one teaching assignment.
            for code in sorted(department_id_by_course):
                if not any(code == t[0] for t in teaching):
                    teaching.append((code, "lecturer"))
                    break
        department_id = (
            department_id_by_course.get(teaching[0][0], "CNTT") if teaching else "CNTT"
        )
        dept_name = next(
            (d[1] for d in DEPARTMENTS if d[0] == department_id),
            "Khoa Công nghệ thông tin",
        )
        instructors.append(
            {
                "id": instructor_id,
                "name": name,
                "title": title,
                "email": _slug_email(name, idx),
                "department_id": department_id,
                "bio": _bio_vi(name, title, dept_name),
                "courses": [
                    {"course_code": code, "role": role} for code, role in teaching
                ],
            }
        )
    return instructors


def _build_programs(courses: list[dict[str, object]]) -> list[dict[str, object]]:
    by_semester: dict[int, list[str]] = {}
    for c in courses:
        by_semester.setdefault(int(str(c["semester"])), []).append(str(c["code"]))
    out: list[dict[str, object]] = []
    for prog_id, name_vi, name_en, dept_id, total in PROGRAMS:
        required: list[str] = []
        elective: list[str] = []
        for semester in sorted(by_semester):
            codes = sorted(by_semester[semester])
            if prog_id == "PR-AI" and semester >= 4:
                elective.extend(c for c in codes if c.startswith(("CS3", "CS4")))
            else:
                mid = max(1, len(codes) // 2)
                required.extend(codes[:mid])
                elective.extend(codes[mid:])
        out.append(
            {
                "id": prog_id,
                "name": name_vi,
                "name_en": name_en,
                "department_id": dept_id,
                "total_credits": total,
                "required_courses": required,
                "elective_courses": elective,
            }
        )
    return out


def _build_terms(
    instructors: list[dict[str, object]],
    courses: list[dict[str, object]],
) -> list[dict[str, object]]:
    sorted_courses = sorted(courses, key=lambda c: str(c["code"]))
    instructor_for_course: dict[str, object] = {
        str(entry["course_code"]): instructor
        for instructor in instructors
        for entry in cast(list[dict[str, object]], instructor["courses"])
    }
    out: list[dict[str, object]] = []
    for term_id, year, season, start, end in TERMS:
        offered: list[dict[str, str]] = []
        for idx, course in enumerate(sorted_courses):
            code = str(course["code"])
            slot = idx % 3
            include = (season == "spring" and slot == 0) or (
                season == "fall" and slot == 1
            )
            if not include:
                continue
            primary = cast(
                dict[str, object], instructor_for_course.get(code, instructors[0])
            )
            # Fall terms pick a different instructor to exercise the override.
            if season == "fall":
                override = instructors[(idx + 1) % len(instructors)]
                instructor_id = (
                    str(override["id"])
                    if override["id"] != primary["id"]
                    else str(primary["id"])
                )
            else:
                instructor_id = str(primary["id"])
            offered.append(
                {
                    "course_code": code,
                    "instructor_id": str(instructor_id),
                    "schedule": f"Thứ {(idx % 6) + 2}, tiết {(idx % 4) + 1}-{(idx % 4) + 3}",
                }
            )
        out.append(
            {
                "id": term_id,
                "name": f"{season.title()} {year}",
                "year": year,
                "season": season,
                "start_date": start,
                "end_date": end,
                "offered": offered,
            }
        )
    return out


def generate(seed: int = SEED, data_dir: Path | None = None) -> Dataset:
    """Build the four entity tables deterministically.

    Reads ``courses.json`` from ``data_dir`` (default ``./data``) so the entity
    tables can be regenerated without touching ``generate_data.py``.
    """
    # seed is currently unused for randomness (all assignments are deterministic
    # hash-based), but kept in the signature so CLI callers don't have to know
    # about the implementation detail.
    del seed
    resolved = data_dir or DATA_DIR
    courses_path = resolved / "courses.json"
    if not courses_path.exists():
        raise FileNotFoundError(
            f"{courses_path} not found — run `uv run python -m generate_data --seed {SEED}` first"
        )
    courses = json.loads(courses_path.read_text(encoding="utf-8"))
    return Dataset(
        departments=_build_departments(),
        instructors=_build_instructors(courses),
        programs=_build_programs(courses),
        terms=_build_terms(_build_instructors(courses), courses),
    )


def write(dataset: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in (
        ("departments", dataset.departments),
        ("instructors", dataset.instructors),
        ("programs", dataset.programs),
        ("terms", dataset.terms),
    ):
        (out_dir / f"{name}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate the CourseMate entity tables (departments, instructors, programs, terms)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_DIR,
        help="Output data directory (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    dataset = generate(seed=args.seed, data_dir=args.out)
    write(dataset, args.out)
    print(
        f"Generated {len(dataset.departments)} departments, "
        f"{len(dataset.instructors)} instructors, "
        f"{len(dataset.programs)} programs, "
        f"{len(dataset.terms)} terms in {args.out}"
    )


__all__ = [
    "Dataset",
    "generate",
    "write",
    "main",
    "DEPARTMENTS",
    "INSTRUCTOR_NAMES",
    "PROGRAMS",
    "TERMS",
]


if __name__ == "__main__":
    main()