from __future__ import annotations

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
        return SimpleNamespace(
            to=lambda device: SimpleNamespace(eval=lambda: None),
            eval=lambda: None,
        )

    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda name: SimpleNamespace(),
    )
    LocalBgeReranker("fake/model")
    assert captured["kwargs"]["quantization_config"].load_in_8bit is True
    assert captured["kwargs"]["device_map"] == "auto"
    assert "quantization_config" in captured["kwargs"], (
        "must pass quantization_config on CUDA"
    )


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
