from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "nvidia/nemotron-3-embed-1b:free"
DEFAULT_EMBED_DIM = 2048


@dataclass(frozen=True)
class Config:
    api_key: str
    data_dir: Path = DEFAULT_DATA_DIR
    llm_model: str = DEFAULT_LLM_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    embed_dim: int = DEFAULT_EMBED_DIM
    base_url: str = DEFAULT_BASE_URL
    allow_medium: bool = True
    enable_fallback: bool = True


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

    def _parse_bool(key: str, default: bool) -> bool:
        raw = os.environ.get(key)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes", "on")

    return Config(
        api_key=resolved_key,
        data_dir=resolved_dir,
        llm_model=os.environ.get("RAG_LLM_MODEL", DEFAULT_LLM_MODEL),
        embed_model=os.environ.get("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        embed_dim=int(os.environ.get("RAG_EMBED_DIM", str(DEFAULT_EMBED_DIM))),
        base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        allow_medium=_parse_bool("RAG_ALLOW_MEDIUM", True),
        enable_fallback=_parse_bool("RAG_ENABLE_FALLBACK", True),
    )