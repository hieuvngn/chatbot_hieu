from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Protocol

from rag_core.config import DEFAULT_BASE_URL, DEFAULT_LLM_MODEL
from rag_core.generator import numbered_sources
from rag_core.models import Chunk

Level = Literal["high", "medium", "low"]

DEFAULT_REPHRASE_SUGGESTION = (
    "Bạn có thể diễn đạt lại câu hỏi với từ khóa chính xác hơn, "
    "ví dụ nêu rõ tên khái niệm hoặc thuật ngữ cần tìm."
)

_JUDGE_SYSTEM_PROMPT = (
    "You are the CRAG judge of a retrieval pipeline for a study-document "
    "chatbot. Given a question and a numbered list of retrieved sources, "
    "decide whether the sources contain material that answers the question. "
    "Reply with ONLY a JSON object of the form "
    '{"level": "high" | "medium" | "low", "rephrase_suggestion": "..."}. '
    "high = the sources directly and sufficiently answer the question; "
    "medium = the sources are partially relevant or the question is ambiguous; "
    "low = no retrieved source contains relevant material. "
    "For medium or low, rephrase_suggestion must be a rewritten question that "
    "uses exact keywords and no pronouns, phrased so a keyword search would "
    "retrieve better material. For high, rephrase_suggestion must be empty."
)

_REWRITE_SYSTEM_PROMPT = (
    "You rewrite questions for exact-keyword search over study documents. "
    "Given a question, return only the rewritten question: keep the topic and "
    "keywords, drop pronouns, filler words and conversational phrasing. "
    "Do not add anything else."
)


@dataclass(frozen=True)
class Judgment:
    level: Level
    rephrase_suggestion: str = ""

    @property
    def is_high(self) -> bool:
        return self.level == "high"


class Judge(Protocol):
    """Assesses the retrieved chunks for a query: high / medium / low."""

    def assess(self, query: str, chunks: list[Chunk]) -> Judgment: ...


class QueryRewriter(Protocol):
    """Rewrites a query into exact keywords for a refine pass."""

    def rewrite(self, query: str) -> str: ...


def _parse_judgment(content: str) -> Judgment:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("judge response is not a JSON object")
    level = data.get("level")
    if level not in ("high", "medium", "low"):
        raise ValueError(f"unexpected judge level: {level!r}")
    suggestion = data.get("rephrase_suggestion")
    suggestion_text = suggestion if isinstance(suggestion, str) else ""
    return Judgment(level=level, rephrase_suggestion=suggestion_text)


class OpenRouterJudge:
    """LLM judge via the OpenAI-compatible OpenRouter endpoint.

    The LLM is asked to decide whether the retrieved sources can answer the
    question. When the response cannot be parsed, the judge returns ``low`` so
    the pipeline refuses instead of answering on an unverifiable verdict.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def assess(self, query: str, chunks: list[Chunk]) -> Judgment:
        sources = numbered_sources(chunks) or "No sources were retrieved."
        user_prompt = f"Question: {query}\n\nSources:\n{sources}"
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        try:
            return _parse_judgment(content)
        except (json.JSONDecodeError, ValueError):
            return Judgment(level="low")


class OpenRouterQueryRewriter:
    """LLM query rewriter via the OpenAI-compatible OpenRouter endpoint.

    Used by the refine pass to turn a conversational question into exact
    keywords. If the model returns nothing, the original query is kept.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def rewrite(self, query: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or query