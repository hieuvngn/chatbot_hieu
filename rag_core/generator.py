from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any, Protocol, cast

from rag_core.config import DEFAULT_BASE_URL, DEFAULT_LLM_MODEL
from rag_core.completion import request_text
from rag_core.models import Chunk, Citation, Source

_SYSTEM_PROMPT = (
    "You are CourseMate, a helpful study assistant for IT students. "
    "Answer the user's question using ONLY the provided sources. "
    "Every claim you make must be supported by one of the sources. "
    'When you use a source, cite it inline as [1], [2], ... matching the '
    "numbered source list. If the sources do not support an answer, say so "
    "instead of guessing."
)

_FALLBACK_SYSTEM_PROMPT = (
    "You are CourseMate, a helpful study assistant for IT students. "
    "No relevant study documents were found for the user's question. "
    "Answer from your general knowledge as helpfully and accurately as possible. "
    "Do NOT invent citations or fake sources. If you are uncertain, say so. "
    "Start the answer with a short disclaimer in the user's language: "
    "'Lưu ý: Không tìm thấy tài liệu phù hợp trong kho tài liệu, câu trả lời dưới đây dựa trên kiến thức chung.' "
    "for Vietnamese, or 'Note: No relevant material was found in the course corpus, the following answer is based on general knowledge.' for English."
)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def numbered_sources(chunks: list[Chunk]) -> str:
    """The numbered source list shared by every prompt that presents chunks."""
    return "\n\n".join(
        f"[{i}] ({chunk.source.document_title}, {chunk.source.chapter}):\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )

class Generator(Protocol):
    def generate(
        self,
        question: str,
        chunks: list[Chunk],
        feedback: str | None = None,
        *,
        skill_instructions: str = "",
    ) -> str: ...

    def stream(
        self,
        question: str,
        chunks: list[Chunk],
        feedback: str | None = None,
        *,
        skill_instructions: str = "",
    ) -> Iterator[str]:
        """Yield token deltas for the final accepted answer; assemble locally
        for the answer-check path."""
        ...


class OpenRouterGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def _build_messages(
        self,
        question: str,
        chunks: list[Chunk],
        feedback: str | None,
        skill_instructions: str,
    ) -> list[dict[str, str]]:
        numbered = numbered_sources(chunks)
        user_prompt = (
            f"Sources:\n{numbered}\n\n"
            f"Question: {question}\n\n"
            f"Answer the question using ONLY these sources, citing them as [1], [2], ..."
        )
        if feedback:
            user_prompt += (
                "\n\nYour previous answer was rejected. "
                f"Feedback: {feedback}"
            )
        system_prompt = _SYSTEM_PROMPT
        if skill_instructions:
            system_prompt += (
                "\n\nAdditional instructions from active skills:\n" + skill_instructions
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate(
        self,
        question: str,
        chunks: list[Chunk],
        feedback: str | None = None,
        *,
        skill_instructions: str = "",
    ) -> str:
        return request_text(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=cast(Any, self._build_messages(question, chunks, feedback, skill_instructions)),
            )
        )

    def stream(
        self,
        question: str,
        chunks: list[Chunk],
        feedback: str | None = None,
        *,
        skill_instructions: str = "",
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=cast(Any, self._build_messages(question, chunks, feedback, skill_instructions)),
            stream=True,
        )
        for chunk in stream:
            # OpenRouter/OpenAI streaming responses include error tuples on
            # failure; only ChatCompletionChunk objects carry the .choices list.
            if not hasattr(chunk, "choices"):
                continue
            for choice in chunk.choices:
                delta = choice.delta.content
                if delta:
                    yield delta

    def generate_fallback(self, question: str) -> str:
        return request_text(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _FALLBACK_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {question}"},
                ],
            )
        )


def parse_citations(answer: str, sources: list[Source]) -> list[Citation]:
    """Extract [n] markers from the answer and map them to the source list."""
    by_index = {i + 1: source for i, source in enumerate(sources)}
    citations: list[Citation] = []
    for match in _CITATION_PATTERN.finditer(answer):
        marker = match.group(0)
        idx = int(match.group(1))
        source = by_index.get(idx)
        if source is not None:
            citations.append(Citation(marker=marker, source=source))
    return citations