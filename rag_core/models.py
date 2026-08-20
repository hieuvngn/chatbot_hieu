from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Source:
    document_id: str
    document_title: str
    chapter: str
    course_code: str
    kind: str
    language: str


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


@dataclass
class Turn:
    role: str
    text: str


@dataclass
class Session:
    id: str
    user_id: str
    turns: list[Turn] = field(default_factory=list)