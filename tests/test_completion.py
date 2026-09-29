"""Regression tests for the empty-choices completion failure mode.

OpenRouter answers an upstream provider outage with HTTP 200 and a null
``choices`` list, carrying the real reason in the body's ``error`` field. The
OpenAI SDK does not treat that as a failure, so unguarded ``choices[0]``
indexing surfaced a bare ``TypeError: 'NoneType' object is not subscriptable``
to API clients instead of retrying or naming the provider's error.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import rag_core.answer_check as answer_check_mod
import rag_core.completion as completion
import rag_core.generator as generator_mod
import rag_core.intent as intent_mod
import rag_core.judge as judge_mod
import rag_core.rewrite as rewrite_mod
from rag_core.answer_check import OpenRouterAnswerChecker
from rag_core.completion import CompletionError, request_text
from rag_core.generator import OpenRouterGenerator
from rag_core.intent import OpenRouterCourseExtractor, OpenRouterIntentClassifier
from rag_core.judge import OpenRouterJudge, OpenRouterQueryRewriter
from rag_core.models import Chunk, Source, Turn
from rag_core.rewrite import OpenRouterSessionRewriter

# Captured verbatim from OpenRouter during a failing system-test run.
EMPTY_RESPONSE = SimpleNamespace(
    id="gen-1",
    choices=None,
    created=None,
    model=None,
    object=None,
    metadata=None,
    usage=None,
    error={
        "message": "Provider returned an empty response",
        "code": 502,
        "metadata": {"error_type": "provider_unavailable"},
    },
)


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        error=None,
    )


class FakeCompletions:
    """Replays a scripted list of responses, then repeats the last one."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.calls = 0

    def create(self, **_: object) -> Any:
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[index]


def wire(client: Any, responses: list[Any]) -> FakeCompletions:
    """Point a client at `responses` instead of the network."""
    completions = FakeCompletions(responses)
    sdk = client._client
    sdk.__dict__.pop("chat", None)
    sdk.__dict__["chat"] = SimpleNamespace(completions=completions)
    return completions


CHUNKS = [
    Chunk(
        source=Source("D1", "Bài giảng", "chương 1", "CS101", "slide", "vi"),
        text="Nội dung bài giảng về cấu trúc dữ liệu.",
    )
]


def no_sleep(_: float) -> None:
    """Skip the real backoff so tests stay fast."""


# ---------------------------------------------------------------------------
# request_text semantics
# ---------------------------------------------------------------------------


def test_returns_content_when_choices_present() -> None:
    assert request_text(lambda: _response("xin chào")) == "xin chào"


def test_none_content_becomes_empty_string_not_error() -> None:
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    assert request_text(lambda: response) == ""


def test_retries_empty_choices_until_it_recovers() -> None:
    calls = {"n": 0}

    def create() -> Any:
        calls["n"] += 1
        return EMPTY_RESPONSE if calls["n"] < 3 else _response("ok")

    assert request_text(create, sleep=no_sleep) == "ok"
    assert calls["n"] == 3, "should stop as soon as a response carries choices"


def test_backs_off_between_attempts() -> None:
    slept: list[float] = []
    calls = {"n": 0}

    def create() -> Any:
        calls["n"] += 1
        return EMPTY_RESPONSE if calls["n"] < 3 else _response("ok")

    request_text(create, sleep=slept.append)
    assert slept == [0.5, 1.0], "linear backoff, no sleep after the last attempt"


_CLIENT_MODULE = {
    "generate": generator_mod,
    "generate_fallback": generator_mod,
    "assess": judge_mod,
    "rewrite": judge_mod,
    "check": answer_check_mod,
}


def _call(client: Any, method: str) -> Any:
    if method in ("generate_fallback", "rewrite"):
        return getattr(client, method)("câu hỏi")
    if method == "check":
        return client.check("câu hỏi", "bài làm", CHUNKS)
    return getattr(client, method)("câu hỏi", CHUNKS)


@pytest.mark.parametrize(
    "method, factory",
    [
        ("generate", OpenRouterGenerator),
        ("generate_fallback", OpenRouterGenerator),
        ("assess", OpenRouterJudge),
        ("rewrite", OpenRouterQueryRewriter),
        ("check", OpenRouterAnswerChecker),
    ],
)
def test_client_raises_completion_error_not_type_error(
    method: str,
    factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _CLIENT_MODULE[method],
        "request_text",
        lambda create, **kw: completion.request_text(create, sleep=no_sleep),
    )
    client = factory("k", "m", "http://x")
    wire(client, [EMPTY_RESPONSE])
    with pytest.raises(CompletionError):
        _call(client, method)


def test_session_rewriter_keeps_original_message_on_empty_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its docstring promises the original message survives an empty reply."""
    monkeypatch.setattr(
        rewrite_mod,
        "request_text",
        lambda create, **kw: completion.request_text(create, sleep=no_sleep),
    )
    rewriter = OpenRouterSessionRewriter("k", "m", "http://x")
    wire(rewriter, [EMPTY_RESPONSE])
    turns = [Turn(role="user", text="câu trước")]
    assert rewriter.rewrite("câu hỏi gốc", turns) == "câu hỏi gốc"


def test_intent_classifier_degrades_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """classify/extract swallow failures by contract, so they must not leak TypeError."""
    monkeypatch.setattr(
        intent_mod,
        "request_text",
        lambda create, **kw: completion.request_text(create, sleep=no_sleep),
    )
    classifier = OpenRouterIntentClassifier("k", "m", "http://x")
    wire(classifier, [EMPTY_RESPONSE])
    assert classifier.classify("hỏi gì đó") == "KNOWLEDGE_QA"

    extractor = OpenRouterCourseExtractor("k", "m", "http://x")
    wire(extractor, [EMPTY_RESPONSE])
    extraction = extractor.extract("tôi đã học CS101")
    assert extraction.completed_courses == []
    assert extraction.target_course is None
    assert extraction.current_semester is None


def test_judge_recovers_when_retry_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single transient empty response must not fail the whole request."""
    monkeypatch.setattr(
        judge_mod,
        "request_text",
        lambda create, **kw: completion.request_text(create, sleep=no_sleep),
    )
    judge = OpenRouterJudge("k", "m", "http://x")
    completions = wire(judge, [EMPTY_RESPONSE, _response(json.dumps({"level": "high"}))])
    judgment = judge.assess("câu hỏi", CHUNKS)
    assert judgment.level == "high"
    assert completions.calls == 2
