"""Loads config.yaml and expands ${VAR} and ${VAR:-default} references from the environment."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand(value):
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    if load_dotenv:
        load_dotenv(PROJECT_ROOT / ".env")
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(config_path, encoding="utf-8") as f:
        return _expand(yaml.safe_load(f))


def backend() -> str:
    """Returns 'vertex' (real Gemini on Vertex AI) or 'mock' (offline, for tests and demos)."""
    return os.getenv("CAREAI_BACKEND", "vertex").strip().lower()
