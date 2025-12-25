"""Factory functions for creating Chunk objects.

This module provides helper functions for constructing Chunk objects
with computed metadata (word counts, token estimates).
"""

from __future__ import annotations

from pw_mcp.ingest.chunker.models import Chunk
from pw_mcp.ingest.chunker.token_counting import count_tokens


def _make_chunk(
    lines: list[str],
    section: str | None,
    line_start: int,
    line_end: int,
    chunk_index: int,
    factor: float,
) -> Chunk:
    """Create a Chunk from accumulated lines.

    Internal helper that joins lines, calculates word count and token estimate.

    Args:
        lines: List of text lines to join
        section: Current section name (None if before first header)
        line_start: Starting line number (1-indexed)
        line_end: Ending line number (1-indexed, inclusive)
        chunk_index: Index of this chunk in the article
        factor: Token estimation factor

    Returns:
        A Chunk object with computed metadata
    """
    text = "\n".join(lines)
    word_count = len(text.split())
    estimated_tokens = int(word_count * factor)

    return Chunk(
        text=text,
        chunk_index=chunk_index,
        section=section,
        line_start=line_start,
        line_end=line_end,
        word_count=word_count,
        estimated_tokens=estimated_tokens,
    )


def _make_chunk_tiktoken(
    lines: list[str],
    section: str | None,
    line_start: int,
    line_end: int,
    chunk_index: int,
) -> Chunk:
    """Create a Chunk using tiktoken for token counting.

    Args:
        lines: List of text lines to join
        section: Current section name (None if before first header)
        line_start: Starting line number (1-indexed)
        line_end: Ending line number (1-indexed, inclusive)
        chunk_index: Index of this chunk in the article

    Returns:
        A Chunk object with accurate token count
    """
    text = "\n".join(lines)
    word_count = len(text.split())
    token_count = count_tokens(text)

    return Chunk(
        text=text,
        chunk_index=chunk_index,
        section=section,
        line_start=line_start,
        line_end=line_end,
        word_count=word_count,
        estimated_tokens=token_count,  # Using actual count, not estimate
    )
