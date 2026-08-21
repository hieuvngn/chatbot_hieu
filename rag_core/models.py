from __future__ import annotations

from dataclasses import dataclass, field

MAX_TURNS = 6


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


@dataclass
class Session:
    id: str
    user_id: str
    turns: list[Turn] = field(default_factory=list)