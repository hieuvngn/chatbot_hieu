from __future__ import annotations

from typing import Protocol

from rag_core.completion import CompletionError, request_text
from rag_core.config import DEFAULT_BASE_URL, DEFAULT_LLM_MODEL
from rag_core.models import Turn

_SESSION_REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's message into a standalone question using the "
    "conversation history. Given the previous turns and the current message, "
    "output ONLY the rewritten question: self-contained, with all pronouns "
    "and references resolved from the history, in the same language as the "
    "current message. If the message is already standalone, output it "
    "unchanged. Do not add anything else."
)


class SessionRewriter(Protocol):
    """Rewrites a message into a standalone question using session history."""

    def rewrite(self, message: str, turns: list[Turn]) -> str: ...


class OpenRouterSessionRewriter:
    """LLM session-aware rewriter via the OpenAI-compatible OpenRouter endpoint.

    Turned the follow-up question into a standalone one using the last turns
    of the Session before retrieval. If the model returns nothing, the
    original message is kept.
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

    def rewrite(self, message: str, turns: list[Turn]) -> str:
        history = "\n".join(f"{turn.role}: {turn.text}" for turn in turns) or "No previous turns."
        user_prompt = f"Previous turns:\n{history}\n\nCurrent message: {message}"
        try:
            rewritten = request_text(
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": _SESSION_REWRITE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
            ).strip()
        except CompletionError:
            return message
        return rewritten or message