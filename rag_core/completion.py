"""Shared handling for OpenAI-compatible chat completions.

OpenRouter reports an upstream provider outage as **HTTP 200** with a null
``choices`` list, carrying the real reason in the body's ``error`` field — for
example ``{"code": 502, "metadata": {"error_type": "provider_unavailable"}}``.
The OpenAI SDK does not treat that as a failure, so indexing ``choices[0]``
raises a bare ``TypeError: 'NoneType' object is not subscriptable`` that hides
the upstream cause. These outcomes are transient, so retry them and only then
fail with a message that names the provider's reason.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

DEFAULT_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 0.5


class CompletionError(RuntimeError):
    """The provider never returned usable choices within the attempt budget."""


def _describe(response: Any) -> str:
    """Render the provider's error field for the exception message."""
    error = getattr(response, "error", None)
    if isinstance(error, dict):
        message = error.get("message") or "provider returned an error"
        code = error.get("code")
        return f"{message} (code {code})" if code is not None else str(message)
    if error is not None:
        return str(error)
    return "empty choices with no error field"


def _attempt_until(
    create: Callable[[], Any],
    read: Callable[[Any], T | None],
    what: str,
    max_attempts: int,
    backoff: float,
    sleep: Callable[[float], None],
) -> T:
    """Retry `create` until `read` yields a value, else raise naming the cause."""
    detail = f"empty {what} with no error field"
    for attempt in range(1, max_attempts + 1):
        response = create()
        value = read(response)
        if value is not None:
            return value
        detail = _describe(response)
        if attempt < max_attempts:
            sleep(backoff * attempt)
    raise CompletionError(
        f"provider returned no {what} after {max_attempts} attempts: {detail}"
    )


def request_text(
    create: Callable[[], Any],
    *,
    max_attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Call ``create`` and return the assistant message text.

    A response with a non-empty ``choices`` list is returned as-is, including a
    legitimately empty content string, so callers keep the same semantics they
    had when they inlined ``response.choices[0].message.content or ""``.

    A null or empty ``choices`` list is treated as a transient provider
    failure: retry up to ``max_attempts`` with a linear backoff, then raise
    ``CompletionError`` naming what the provider reported.
    """

    def read(response: Any) -> str | None:
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        # `or ""` keeps a legitimately empty content string distinct from the
        # None sentinel that means "this attempt failed".
        return choices[0].message.content or ""

    return _attempt_until(create, read, "choices", max_attempts, backoff, sleep)


def request_data(
    create: Callable[[], Any],
    *,
    max_attempts: int = DEFAULT_ATTEMPTS,
    backoff: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Any]:
    """Call ``create`` and return the embeddings payload items.

    The embeddings endpoint has the same null-payload failure mode as chat
    completions, so a null or empty ``data`` list is retried the same way.
    """

    def read(response: Any) -> list[Any] | None:
        data = getattr(response, "data", None)
        return data or None

    return _attempt_until(create, read, "embeddings data", max_attempts, backoff, sleep)
