from __future__ import annotations

from typing import Protocol

from rag_core.completion import request_data
from rag_core.config import DEFAULT_BASE_URL, DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL

class Embedder(Protocol):
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenRouterEmbedder:
    """Embeds via the OpenAI-compatible OpenRouter endpoint.

    Index chunks are embedded in one batched request; queries are embedded one
    per request (the pipeline guarantees this at the call sites).
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_EMBED_MODEL,
        dim: int = DEFAULT_EMBED_DIM,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self.dim = dim

    # OpenRouter's free-tier embedding models cap a single batch at ~256 inputs.
    # Chunk transparently so larger indexes can be embedded without callers
    # having to know the upstream limit.
    MAX_BATCH_SIZE = 256

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH_SIZE):
            chunk = texts[start : start + self.MAX_BATCH_SIZE]
            response = request_data(
                lambda: self._client.embeddings.create(
                    model=self._model, input=chunk, encoding_format="float"
                )
            )
            vectors.extend(item.embedding for item in response)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]