"""Pytest configuration.

Load .env into os.environ before any test collects, so system tests
(which check os.environ.get("OPENROUTER_API_KEY")) can find the key.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure pytest can import project modules (pyproject.toml also sets this).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env into os.environ so skipif checks pass.
ENV_FILE = ROOT / ".env"
if ENV_FILE.exists() and not os.environ.get("OPENROUTER_API_KEY"):
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE, override=True)
    except ImportError:
        pass
