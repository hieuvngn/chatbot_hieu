from __future__ import annotations

import logging
from typing import Any, Protocol

from rag_core.config import DEFAULT_RERANK_MODEL
from rag_core.models import Chunk

logger = logging.getLogger(__name__)

RERANK_INPUT_TOP_K = 20
RERANK_KEEP_TOP_K = 10


class Reranker(Protocol):
    """Scores a list of candidate chunks for a query, ordered best first."""

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]: ...


class LocalBgeReranker:
    """Cross-encoder re-ranker (BAAI/bge-reranker-v2-m3) run locally.

    Loads on the local GPU when CUDA is available and falls back to CPU.
    The heavy local-GPU stack (torch, transformers) is imported lazily so
    the rest of the pipeline works without it installed.
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        import torch

        def load_fp32() -> Any:
            return AutoModelForSequenceClassification.from_pretrained(model_name).to(
                "cpu"
            )

        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig

            try:
                quantization_config = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                    load_in_8bit=True,
                    llm_int8_skip_modules=["classifier"],
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                self._device = "cuda"
            except Exception as exc:
                self._model = load_fp32()
                self._device = "cpu"
                logger.warning(
                    "INT8 load failed (%s); falling back to FP32 on %s",
                    exc,
                    self._device,
                )
        else:
            self._model = load_fp32()
            self._device = "cpu"
        self._model.eval()

    @property
    def device(self) -> str:
        return self._device

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        import torch

        pairs = [[query, chunk.text] for chunk in chunks]
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(-1).float()
        scores = logits.tolist()
        ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
        return [chunk for chunk, _ in ranked]