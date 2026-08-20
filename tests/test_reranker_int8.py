from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from rag_core.reranker import LocalBgeReranker


def test_int8_load_used_when_cuda_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    captured: dict[str, Any] = {}

    def fake_from_pretrained(model_name: str, **kwargs: Any) -> SimpleNamespace:
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(eval=lambda: None)

    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda name: SimpleNamespace(),
    )
    reranker = LocalBgeReranker("fake/model")
    assert reranker.device == "cuda"
    assert "quantization_config" in captured["kwargs"], (
        "must pass quantization_config on CUDA"
    )
    assert captured["kwargs"]["quantization_config"].load_in_8bit is True
    assert captured["kwargs"]["quantization_config"].llm_int8_skip_modules == [
        "classifier"
    ], "the classifier (scoring head) must stay FP32 or logits collapse to a constant"
    assert captured["kwargs"]["device_map"] == "auto"


def test_fp32_fallback_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    captured: dict[str, Any] = {}

    def fake_from_pretrained(model_name: str, **kwargs: Any) -> SimpleNamespace:
        captured["kwargs"] = kwargs
        return SimpleNamespace(to=lambda device: SimpleNamespace(eval=lambda: None))

    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda name: SimpleNamespace(),
    )
    reranker = LocalBgeReranker("fake/model")
    assert reranker.device == "cpu"
    assert "quantization_config" not in captured["kwargs"]


def test_fp32_fallback_when_int8_load_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    captured: list[dict[str, Any]] = []
    evals: list[bool] = []

    def fake_from_pretrained(model_name: str, **kwargs: Any) -> SimpleNamespace:
        captured.append(kwargs)
        if len(captured) == 1:
            raise RuntimeError("simulated INT8 load failure")
        return SimpleNamespace(
            to=lambda device: SimpleNamespace(eval=lambda: evals.append(True)),
        )

    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda name: SimpleNamespace(),
    )
    with caplog.at_level(logging.WARNING, logger="rag_core.reranker"):
        reranker = LocalBgeReranker("fake/model")
    assert "quantization_config" in captured[0], "INT8 attempt must pass it"
    assert "quantization_config" not in captured[1], "FP32 fallback must not pass it"
    assert reranker.device == "cpu"
    assert evals == [True], "eval() must run on the FP32 fallback model"
    assert "falling back to FP32 on cpu" in caplog.text
