from __future__ import annotations

from dataclasses import replace
from typing import TypedDict

from rag_core.models import Chunk, Source

MAX_TOKENS = 600
OVERLAP = 0.15


class ChapterDict(TypedDict):
    id: str
    title: str
    content: str


class CourseDict(TypedDict):
    code: str
    name: str
    name_en: str
    credits: int
    prerequisites: list[str]
    semester: int
    department: str
    instructor: str
    description: str


class DocumentDict(TypedDict):
    id: str
    course_code: str
    title: str
    kind: str
    language: str
    chapters: list[ChapterDict]


def _chunk_text(text: str, source: Source) -> list[Chunk]:
    """Split a chapter body into 400-600 token windows with ~15% overlap."""
    tokens = text.split()
    if len(tokens) <= MAX_TOKENS:
        return [Chunk(source=source, text=text)]

    step = int(MAX_TOKENS * (1 - OVERLAP))
    chunks: list[Chunk] = []
    start = 0
    while start < len(tokens):
        end = start + MAX_TOKENS
        window = " ".join(tokens[start:end])
        chunks.append(Chunk(source=source, text=window))
        if end >= len(tokens):
            break
        start += step
    return chunks


def chunk_document(document: DocumentDict, source: Source) -> list[Chunk]:
    """Chunk one document's chapters; each chunk carries document + chapter metadata."""
    chunks: list[Chunk] = []
    for chapter in document["chapters"]:
        chunks.extend(_chunk_text(chapter["content"], replace(source, chapter=chapter["title"])))
    return chunks


def chunk_course(course: CourseDict) -> list[Chunk]:
    """Chunk a course's structured fields into a single retrievable unit."""
    code = course["code"]
    prereqs = ", ".join(course["prerequisites"]) or "không có"
    text = (
        f"Môn học {code} - {course['name']} ({course['name_en']}). "
        f"{course['credits']} tín chỉ, học kỳ {course['semester']}, thuộc {course['department']}. "
        f"Giảng viên: {course['instructor']}. Môn học trước: {prereqs}. "
        f"{course['description']}"
    )
    source = Source(
        document_id=code,
        document_title=course["name"],
        chapter="Thông tin môn học",
        course_code=code,
        kind="course",
        language="vi",
    )
    return [Chunk(source=source, text=text)]


def chunk_dataset(courses: list[CourseDict], documents: list[DocumentDict]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for course in courses:
        chunks.extend(chunk_course(course))
    for document in documents:
        source = Source(
            document_id=document["id"],
            document_title=document["title"],
            chapter="",
            course_code=document["course_code"],
            kind=document["kind"],
            language=document["language"],
        )
        chunks.extend(chunk_document(document, source))
    return chunks