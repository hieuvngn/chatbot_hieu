from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "nvidia/nemotron-3-embed-1b:free"
DEFAULT_EMBED_DIM = 2048
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class Config:
    api_key: str
    data_dir: Path = DEFAULT_DATA_DIR
    llm_model: str = DEFAULT_LLM_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dim: int = DEFAULT_EMBED_DIM
    rerank_model: str = DEFAULT_RERANK_MODEL
    base_url: str = DEFAULT_BASE_URL


def load_config(
    api_key: str | None = None,
    env_file: str | Path = ".env",
    data_dir: str | Path | None = None,
) -> Config:
    """Read configuration from the environment, optionally loading a .env file."""
    if env_file:
        path = Path(env_file)
        if path.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(path)
            except ImportError:
                pass

    resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Create a .env file or set the "
            "environment variable (see README.md)."
        )

    resolved_dir = Path(data_dir) if data_dir else Path(os.environ.get("RAG_DATA_DIR", DEFAULT_DATA_DIR))
    return Config(
        api_key=resolved_key,
        data_dir=resolved_dir,
        llm_model=os.environ.get("RAG_LLM_MODEL", DEFAULT_LLM_MODEL),
        embed_model=os.environ.get("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        embed_dim=int(os.environ.get("RAG_EMBED_DIM", str(DEFAULT_EMBED_DIM))),
        rerank_model=os.environ.get("RAG_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
    )