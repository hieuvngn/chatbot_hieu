from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Course:
    code: str
    name: str
    name_en: str
    credits: int
    prerequisites: list[str]
    semester: int
    department: str
    instructor: str
    description: str

class CourseAdvisor:
    def __init__(self, data_dir: Path) -> None:
        raw = json.loads((Path(data_dir) / "courses.json").read_text(encoding="utf-8"))
        self._by_code: dict[str, Course] = {}
        for c in raw:
            course = Course(code=c["code"], name=c["name"], name_en=c["name_en"], credits=c["credits"], prerequisites=list(c["prerequisites"]), semester=c["semester"], department=c["department"], instructor=c["instructor"], description=c["description"])
            self._by_code[c["code"].upper()] = course
    def _norm(self, code: str) -> str:
        return code.strip().upper()
    def get_course(self, code: str) -> Course | None:
        return self._by_code.get(self._norm(code))
    def list_courses(self) -> list[Course]:
        return list(self._by_code.values())
    def resolve_code(self, raw: str) -> str | None:
        n = self._norm(raw)
        return n if n in self._by_code else None
    def get_prerequisites(self, course_code: str) -> list[str] | None:
        c = self.get_course(course_code)
        return None if c is None else list(c.prerequisites)
    def get_missing_prerequisites(self, course_code: str, completed: set[str]) -> list[str] | None:
        c = self.get_course(course_code)
        if c is None:
            return None
        norm_completed = {self._norm(x) for x in completed}
        return [p for p in c.prerequisites if self._norm(p) not in norm_completed]
    def is_eligible(self, course_code: str, completed: set[str]) -> bool:
        missing = self.get_missing_prerequisites(course_code, completed)
        return False if missing is None else len(missing) == 0
    def get_next_courses(self, completed: set[str], current_semester: int | None = None) -> list[Course]:
        norm_completed = {self._norm(x) for x in completed}
        eligible = [c for c in self._by_code.values() if c.code.upper() not in norm_completed and all(self._norm(p) in norm_completed for p in c.prerequisites)]
        if current_semester is not None and 1 <= current_semester <= 8:
            eligible = [c for c in eligible if c.semester >= current_semester]
        eligible.sort(key=lambda c: (c.semester, c.code))
        return eligible
