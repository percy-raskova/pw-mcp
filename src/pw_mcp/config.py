"""Project configuration loaded from pyproject.toml."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class PWMCPConfig(BaseModel):
    """Configuration for pw-mcp project."""

    # Chunking parameters (tiktoken-based)
    chunk_target_tokens: int = 350
    chunk_min_tokens: int = 100
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # Embedding configuration
    embedding_provider: str = "ollama"  # "ollama" or "sentence-transformers"
    embedding_model: str = "embeddinggemma"
    ollama_base_url: str = "http://localhost:11434"


@lru_cache(maxsize=1)
def load_config() -> PWMCPConfig:
    """Load configuration from pyproject.toml.

    Returns:
        PWMCPConfig with settings from [tool.pw-mcp] section,
        falling back to defaults if not found.
    """
    pyproject_path = _find_pyproject()
    if pyproject_path is None:
        return PWMCPConfig()

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    tool_config: dict[str, Any] = data.get("tool", {}).get("pw-mcp", {})
    return PWMCPConfig(**tool_config)


def _find_pyproject() -> Path | None:
    """Find pyproject.toml by walking up from current file."""
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Max 10 levels up
        candidate = current / "pyproject.toml"
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


# Convenience accessor
config = load_config()
