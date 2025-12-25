"""Character-to-line mapping for accurate citation tracking.

This module provides utilities for mapping character positions in joined text
back to original line numbers, enabling accurate line range metadata in chunks.
"""

from __future__ import annotations

from pw_mcp.ingest.chunker.chunk_factory import _make_chunk_tiktoken
from pw_mcp.ingest.chunker.models import Chunk, ChunkConfig
from pw_mcp.ingest.chunker.text_splitting import _split_oversized_text


def _build_char_to_line_map(lines: list[str]) -> list[tuple[int, int, int]]:
    """Build mapping of (char_start, char_end, line_index) for each line.

    Used to map character positions in joined text back to original line numbers.

    Args:
        lines: List of original lines

    Returns:
        List of tuples (char_start, char_end, line_index) where positions
        are relative to the joined text with newlines.
    """
    mapping: list[tuple[int, int, int]] = []
    char_pos = 0
    for line_idx, line in enumerate(lines):
        line_start = char_pos
        line_end = char_pos + len(line)
        mapping.append((line_start, line_end, line_idx))
        char_pos = line_end + 1  # +1 for the newline character
    return mapping


def _find_line_range_for_segment(
    segment_start: int,
    segment_end: int,
    char_to_line_map: list[tuple[int, int, int]],
) -> tuple[int, int]:
    """Find the line indices that a character range spans.

    Args:
        segment_start: Starting character position in joined text
        segment_end: Ending character position in joined text
        char_to_line_map: Mapping from _build_char_to_line_map

    Returns:
        Tuple of (first_line_idx, last_line_idx) - 0-indexed
    """
    first_line: int | None = None
    last_line: int | None = None

    for char_start, char_end, line_idx in char_to_line_map:
        # Check if this line overlaps with segment
        if char_start <= segment_end and char_end >= segment_start:
            if first_line is None:
                first_line = line_idx
            last_line = line_idx

    # Fallback if no overlap found (shouldn't happen)
    if first_line is None:
        first_line = 0
    if last_line is None:
        last_line = len(char_to_line_map) - 1 if char_to_line_map else 0

    return first_line, last_line


def _create_chunks_from_oversized(
    lines: list[str],
    section: str | None,
    line_start: int,
    start_index: int,
    config: ChunkConfig,
) -> list[Chunk]:
    """Create multiple chunks from oversized text with accurate line ranges.

    Args:
        lines: Lines that together exceed max_tokens
        section: Current section name
        line_start: Starting line number (1-indexed, in source document)
        start_index: Starting chunk index
        config: Chunking configuration

    Returns:
        List of Chunk objects, each under max_tokens, with accurate line_range
    """
    # Build character-to-line mapping before joining
    char_to_line_map = _build_char_to_line_map(lines)

    text = "\n".join(lines)
    segments = _split_oversized_text(text, config.max_tokens)

    chunks: list[Chunk] = []
    chunk_index = start_index
    current_pos = 0  # Track position in joined text

    for seg in segments:
        if seg.strip():
            # Find where this segment starts in the joined text
            seg_start = text.find(seg, current_pos)
            if seg_start == -1:
                # Fallback: use current position
                seg_start = current_pos
            seg_end = seg_start + len(seg) - 1

            # Find which lines this segment spans
            first_line_idx, last_line_idx = _find_line_range_for_segment(
                seg_start, seg_end, char_to_line_map
            )

            # Convert to 1-indexed line numbers
            chunk_line_start = line_start + first_line_idx
            chunk_line_end = line_start + last_line_idx

            chunk = _make_chunk_tiktoken(
                [seg],
                section,
                chunk_line_start,
                chunk_line_end,
                chunk_index,
            )
            chunks.append(chunk)
            chunk_index += 1

            # Update position for next search
            current_pos = seg_start + len(seg)

    return chunks
