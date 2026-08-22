"""Web search via the Firecrawl Search API.

``WebSearcher`` is the pipeline seam: given a query it returns web results as
regular Chunks (Source kind="web"), so citations flow through the pipeline
unchanged. Any failure degrades to an empty list — web search is strictly
additive and never breaks an answer.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Protocol

from rag_core.models import Chunk, Source

WEB_TOP_K = 5
WEB_MAX_CHARS = 4000
FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v2"
DEFAULT_TIMEOUT_S = 30.0


class WebSearcher(Protocol):
    """Searches the web and returns results as pipeline-ready Chunks."""

    def search(self, query: str, k: int = WEB_TOP_K) -> list[Chunk]: ...


def _web_chunk(
    url: str, title: str | None, description: str | None, markdown: str | None
) -> Chunk | None:
    text = (markdown or "").strip() or (description or "").strip()
    if not url or not text:
        return None
    return Chunk(
        source=Source(
            document_id=url,
            document_title=(title or "").strip() or url,
            chapter=url,
            course_code="",
            kind="web",
            language="",
        ),
        text=text[:WEB_MAX_CHARS],
    )


def parse_search_payload(payload: dict[str, Any]) -> list[Chunk]:
    """Flatten both known /search response shapes into Chunks.

    v1-style responses carry a flat ``data`` list; v2 groups entries under
    ``data.web``. Anything unexpected yields an empty list.
    """
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return []
    data = payload.get("data")
    if isinstance(data, dict):
        entries: list[Any] = list(data.get("web") or [])
    elif isinstance(data, list):
        entries = list(data)
    else:
        entries = []
    chunks: list[Chunk] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        chunk = _web_chunk(
            url=str(entry.get("url") or ""),
            title=entry.get("title"),
            description=entry.get("description"),
            markdown=entry.get("markdown"),
        )
        if chunk is not None:
            chunks.append(chunk)
    return chunks


class FirecrawlWebSearcher:
    """Concrete WebSearcher backed by POST {base_url}/search."""

    def __init__(
        self,
        api_key: str,
        base_url: str = FIRECRAWL_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    def search(self, query: str, k: int = WEB_TOP_K) -> list[Chunk]:
        try:
            payload = self._post_json(
                "/search",
                {
                    "query": query,
                    "limit": k,
                    "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
                },
            )
            return parse_search_payload(payload)
        except Exception:
            logging.getLogger(__name__).warning(
                "firecrawl web search failed", exc_info=True
            )
            return []

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return result
