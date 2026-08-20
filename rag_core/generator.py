from __future__ import annotations

import re
from typing import Protocol

from rag_core.models import Citation, Source

DEFAULT_LLM_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = (
    "You are CourseMate, a helpful study assistant for IT students. "
    "Answer the user's question using ONLY the provided sources. "
    "Every claim you make must be supported by one of the sources. "
    'When you use a source, cite it inline as [1], [2], ... matching the '
    "numbered source list. If the sources do not support an answer, say so "
    "instead of guessing."
)

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class Generator(Protocol):
    def generate(self, question: str, sources: list[tuple[Source, str]]) -> str: ...


class OpenRouterGenerator:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def generate(self, question: str, sources: list[tuple[Source, str]]) -> str:
        numbered = "\n\n".join(
            f"[{i}] ({s.document_title}, {s.chapter}):\n{text}"
            for i, (s, text) in enumerate(sources, start=1)
        )
        user_prompt = (
            f"Sources:\n{numbered}\n\n"
            f"Question: {question}\n\n"
            f"Answer the question using ONLY these sources, citing them as [1], [2], ..."
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""


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