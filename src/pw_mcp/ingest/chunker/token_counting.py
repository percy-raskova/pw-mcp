"""Token counting utilities using tiktoken.

This module provides accurate token counting for OpenAI-compatible
embeddings using the cl100k_base encoding (used by text-embedding-3-large).
"""

from __future__ import annotations

import tiktoken

# Global tiktoken encoder (cached for performance)
_TIKTOKEN_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Get or create the tiktoken encoder (cached for performance).

    Uses cl100k_base encoding which is compatible with OpenAI's
    text-embedding-3-large and other modern embedding models.

    Returns:
        tiktoken.Encoding instance
    """
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_ENCODER is None:
        _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
    return _TIKTOKEN_ENCODER


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for

    Returns:
        Number of tokens (0 for empty string)
    """
    if not text:
        return 0
    encoder = _get_encoder()
    return len(encoder.encode(text))


def estimate_tokens(text: str, factor: float) -> int:
    """Estimate token count from text using word count heuristic.

    Args:
        text: The text to estimate tokens for
        factor: Multiplication factor (typically 1.3 for English)

    Returns:
        Estimated token count (0 for empty string)
    """
    if not text or not text.strip():
        return 0
    word_count = len(text.split())
    return int(word_count * factor)
