from __future__ import annotations

from dataclasses import dataclass, field

MAX_TURNS = 6

ENTITY_KINDS = ("instructor", "program", "term", "department")


@dataclass(frozen=True)
class Source:
    """The unit a Citation points at.

    Two flavours share one dataclass:

    * Document chapter (entity_type is empty) — ``document_id``/``chapter``
      identify a chunk inside a Document; ``course_code``/``kind``/``language``
      describe the parent Document.
    * Structured entity (entity_type is set) — ``entity_id`` is the entity's
      primary key; the other fields carry a denormalised title + course code
      so the existing UI continues to render a card.

    The kind values for the entity case are ``"instructor"``, ``"program"``,
    ``"term"``, ``"department"``.
    """

    document_id: str
    document_title: str
    chapter: str
    course_code: str
    kind: str
    language: str
    entity_type: str = ""
    entity_id: str = ""


@dataclass
class Chunk:
    source: Source
    text: str


@dataclass(frozen=True)
class Citation:
    marker: str
    source: Source


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    sources: list[Source]
    refused: bool = False
    rephrase_suggestion: str = ""
    skills_applied: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class User:
    id: int
    username: str
    display_name: str
    language: str


@dataclass(frozen=True)
class ConversationMeta:
    id: str
    user_id: int
    title: str
    created_at: str
    updated_at: str
    preview: str


@dataclass
class Turn:
    role: str
    text: str
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    rephrase_suggestion: str = ""
    skills_applied: list[str] = field(default_factory=list)


@dataclass
class Session:
    id: str
    user_id: str
    turns: list[Turn] = field(default_factory=list)


@dataclass(frozen=True)
class Department:
    id: str
    name: str
    name_en: str


@dataclass(frozen=True)
class Instructor:
    id: str
    name: str
    title: str
    email: str
    department_id: str
    bio: str
    courses: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """(course_code, role) pairs the Instructor teaches."""

    def display_courses(self) -> str:
        return ", ".join(
            f"{code} ({role})" for code, role in self.courses
        ) or "không có"


@dataclass(frozen=True)
class Program:
    id: str
    name: str
    name_en: str
    department_id: str
    total_credits: int
    required_courses: tuple[str, ...] = field(default_factory=tuple)
    elective_courses: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Term:
    id: str
    name: str
    year: int
    season: str
    start_date: str
    end_date: str
    offered: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)
    """(course_code, instructor_id, schedule) triples running this Term."""


@dataclass(frozen=True)
class EntityBundle:
    departments: tuple[Department, ...]
    instructors: tuple[Instructor, ...]
    programs: tuple[Program, ...]
    terms: tuple[Term, ...]

    def department_by_id(self, department_id: str) -> Department | None:
        for d in self.departments:
            if d.id == department_id:
                return d
        return None

    def instructor_by_id(self, instructor_id: str) -> Instructor | None:
        for i in self.instructors:
            if i.id == instructor_id:
                return i
        return None

    def program_by_id(self, program_id: str) -> Program | None:
        for p in self.programs:
            if p.id == program_id:
                return p
        return None

    def term_by_id(self, term_id: str) -> Term | None:
        for t in self.terms:
            if t.id == term_id:
                return t
        return None