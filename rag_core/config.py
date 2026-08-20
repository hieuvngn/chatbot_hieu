from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from rag_core.embeddings import DEFAULT_EMBED_DIM, DEFAULT_EMBED_MODEL
from rag_core.generator import DEFAULT_LLM_MODEL

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Config:
    api_key: str
    data_dir: Path = DEFAULT_DATA_DIR
    llm_model: str = DEFAULT_LLM_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dim: int = DEFAULT_EMBED_DIM
    base_url: str = "https://openrouter.ai/api/v1"
    env_files: tuple[Path, ...] = field(default_factory=tuple)


def load_config(
    api_key: str | None = None,
    env_file: str | Path = ".env",
    data_dir: str | Path | None = None,
) -> Config:
    """Read configuration from the environment, optionally loading a .env file."""
    env_files: tuple[Path, ...] = ()
    if env_file:
        path = Path(env_file)
        if path.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(path)
                env_files = (path,)
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
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        env_files=env_files,
    )