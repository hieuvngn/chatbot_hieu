"""Pydantic request/response models mirroring rag_core domain objects."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rag_core.models import Citation, Turn


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    display_name: str
    language: str


class TokenOut(BaseModel):
    token: str
    user: UserOut


class RegisterIn(BaseModel):
    username: str
    password: str


class ProfileIn(BaseModel):
    display_name: str | None = None
    language: str | None = None


class SourceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: str
    document_title: str
    chapter: str
    course_code: str
    kind: str
    language: str


class CitationOut(BaseModel):
    marker: str
    source: SourceOut


class TurnOut(BaseModel):
    role: str
    text: str
    citations: list[CitationOut]
    refused: bool
    rephrase_suggestion: str
    skills_applied: list[str]


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: str
    updated_at: str
    preview: str


class ConversationCreateIn(BaseModel):
    title: str = "New chat"


class ConversationRenameIn(BaseModel):
    title: str


class ChatIn(BaseModel):
    message: str
    use_web: bool = False


class FeaturesOut(BaseModel):
    has_web_search: bool


class AttachmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    filename: str
    file_kind: str
    size_bytes: int
    chunk_count: int
    created_at: str


def citation_out(citation: Citation) -> CitationOut:
    return CitationOut(marker=citation.marker, source=SourceOut.model_validate(citation.source))


def turn_out(turn: Turn) -> TurnOut:
    return TurnOut(
        role=turn.role,
        text=turn.text,
        citations=[citation_out(c) for c in turn.citations],
        refused=turn.refused,
        rephrase_suggestion=turn.rephrase_suggestion,
        skills_applied=list(turn.skills_applied),
    )
