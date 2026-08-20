from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from rag_core.config import DEFAULT_BASE_URL, DEFAULT_LLM_MODEL
from rag_core.generator import numbered_sources
from rag_core.models import Chunk

DEFAULT_UNSUPPORTED_FEEDBACK = (
    "The previous answer contained claims not supported by the cited sources. "
    "Rewrite it so every claim is backed by those sources."
)

_CHECKER_SYSTEM_PROMPT = (
    "You are the answer verifier of a study-document chatbot. Given a "
    "question, a numbered list of retrieved sources, and a draft answer, "
    "check whether EVERY claim in the answer is supported by the sources it "
    "cites. Reply with ONLY a JSON object of the form "
    '{"supported": true | false, "unsupported_claims": ["...", ...]}. '
    "A claim is supported if the source it cites contains it or it follows "
    "directly from that source. An answer that makes claims without citing a "
    "source for them is unsupported. If every claim is supported, "
    "supported must be true and unsupported_claims must be empty."
)


@dataclass(frozen=True)
class CheckVerdict:
    supported: bool
    feedback: str = ""


class AnswerChecker(Protocol):
    """Verifies a draft answer claim-by-claim against its cited sources."""

    def check(self, question: str, answer: str, chunks: list[Chunk]) -> CheckVerdict: ...


def _parse_verdict(content: str) -> CheckVerdict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("checker response is not a JSON object")
    supported = data.get("supported")
    if not isinstance(supported, bool):
        raise ValueError(f"unexpected supported value: {supported!r}")
    claims = data.get("unsupported_claims")
    claim_list = [c for c in claims if isinstance(c, str)] if isinstance(claims, list) else []
    if supported or not claim_list:
        return CheckVerdict(supported=supported)
    feedback = (
        "The following claims are not supported by the cited sources: "
        f"{'; '.join(claim_list)}. "
        "Rewrite the answer using only material those sources actually contain."
    )
    return CheckVerdict(supported=False, feedback=feedback)


class OpenRouterAnswerChecker:
    """LLM answer verifier via the OpenAI-compatible OpenRouter endpoint.

    The LLM is asked whether every claim in the draft answer is supported by
    the cited sources. When the response cannot be parsed, the checker fails
    closed (``supported=False``) so the pipeline regenerates or refuses
    instead of serving an unverified answer.
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

    def check(self, question: str, answer: str, chunks: list[Chunk]) -> CheckVerdict:
        sources = numbered_sources(chunks) or "No sources were retrieved."
        user_prompt = (
            f"Question: {question}\n\nSources:\n{sources}\n\n"
            f"Draft answer:\n{answer}"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _CHECKER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        try:
            return _parse_verdict(content)
        except (json.JSONDecodeError, ValueError):
            return CheckVerdict(supported=False)