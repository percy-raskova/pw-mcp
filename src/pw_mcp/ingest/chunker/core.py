"""Core text chunking logic for embedding-ready segments.

This module contains the main chunking algorithms and functions.
Data models are in the separate models.py module.

Design rationale:
- Section headers (== ... ==) always start new chunks (hard breaks)
- Paragraph breaks (blank lines) are preferred split points (soft breaks)
- Sentence boundaries are the next preferred split points
- Chunks aim for target_tokens but respect min/max bounds
- Overlap ensures context continuity at chunk boundaries
- Line numbers enable precise citation back to source
"""

from __future__ import annotations

from pw_mcp.ingest.chunker.chunk_factory import _make_chunk, _make_chunk_tiktoken
from pw_mcp.ingest.chunker.line_mapping import (
    _build_char_to_line_map,
    _create_chunks_from_oversized,
    _find_line_range_for_segment,
)
from pw_mcp.ingest.chunker.models import (
    Chunk,
    ChunkConfig,
)
from pw_mcp.ingest.chunker.section_detection import (
    extract_section_title,
    is_section_header,
)
from pw_mcp.ingest.chunker.text_splitting import (
    _split_oversized_text,
)
from pw_mcp.ingest.chunker.token_counting import count_tokens, estimate_tokens


def _strip_trailing_blank_lines(lines: list[str]) -> list[str]:
    """Remove trailing blank lines from a list of lines.

    Args:
        lines: List of lines to process

    Returns:
        List with trailing blank lines removed
    """
    result = lines.copy()
    while result and not result[-1].strip():
        result.pop()
    return result


def chunk_text(lines: list[str], config: ChunkConfig) -> list[Chunk]:
    """Chunk text into embedding-ready segments (backward-compatible).

    NOTE: This function uses word-based token estimation. For accurate
    token counting, use chunk_text_tiktoken() instead.

    This function parses text line by line, accumulating into chunks.
    It respects section boundaries (hard breaks) and paragraph boundaries
    (soft breaks), while tracking line numbers for citation.

    Algorithm:
    1. Section headers (== ... ==) always start a new chunk
    2. When approaching max_tokens, split at last paragraph boundary
    3. Track line numbers for precise citation back to source

    Args:
        lines: List of lines from text file
        config: Chunking configuration

    Returns:
        List of Chunk objects in document order
    """
    if not lines:
        return []

    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_section: str | None = None
    line_start: int = 1
    last_para_boundary: int = 0
    chunk_index: int = 0

    for line_num, line in enumerate(lines, 1):
        # Check for section header - always starts new chunk
        if is_section_header(line):
            # Flush current chunk if non-empty (has actual content)
            if current_lines:
                cleaned = _strip_trailing_blank_lines(current_lines)
                if cleaned:
                    chunk = _make_chunk(
                        cleaned,
                        current_section,
                        line_start,
                        line_start + len(cleaned) - 1,
                        chunk_index,
                        config.token_estimation_factor,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

            # Update section and start new chunk with header
            current_section = extract_section_title(line)
            current_lines = [line]
            line_start = line_num
            last_para_boundary = 0
            continue

        # Check for paragraph boundary (blank line)
        if not line.strip():
            last_para_boundary = len(current_lines)
            current_lines.append(line)
            continue

        # Add line to current chunk
        current_lines.append(line)

        # Check if approaching max tokens - need to split
        current_text = "\n".join(current_lines)
        estimated = estimate_tokens(current_text, config.token_estimation_factor)

        if estimated >= config.max_tokens:
            # Determine split point - prefer paragraph boundary, else split at current line
            split_point = last_para_boundary if last_para_boundary > 0 else len(current_lines) - 1

            # Don't create empty chunks
            if split_point > 0:
                # Create chunk up to split point
                chunk_lines = current_lines[:split_point]
                cleaned = _strip_trailing_blank_lines(chunk_lines)
                if cleaned:
                    chunk = _make_chunk(
                        cleaned,
                        current_section,
                        line_start,
                        line_start + len(cleaned) - 1,
                        chunk_index,
                        config.token_estimation_factor,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with remaining lines
                remaining = current_lines[split_point:]
                # Skip leading blank lines in remaining
                while remaining and not remaining[0].strip():
                    split_point += 1
                    remaining = remaining[1:]

                current_lines = remaining
                line_start = line_start + split_point
                last_para_boundary = 0

    # Flush final chunk if non-empty
    if current_lines:
        cleaned = _strip_trailing_blank_lines(current_lines)
        if cleaned:
            chunk = _make_chunk(
                cleaned,
                current_section,
                line_start,
                line_start + len(cleaned) - 1,
                chunk_index,
                config.token_estimation_factor,
            )
            chunks.append(chunk)

    return chunks


def _chunk_without_headers(
    text: str,
    lines: list[str],
    config: ChunkConfig,
) -> list[Chunk]:
    """Fast path for documents without section headers.

    Uses semantic-text-splitter for O(n) performance instead of O(n^2)
    line-by-line accumulation. This is critical for large Library works
    that can be 8,000+ lines with no section structure.

    Line numbers are recovered via character-to-line mapping to maintain
    accurate citation support.

    Args:
        text: The full text content (already stripped)
        lines: Pre-split lines (to avoid re-splitting)
        config: Chunking configuration

    Returns:
        List of Chunk objects with accurate line numbers
    """
    # Use semantic splitter (already integrated, Rust/SIMD optimized)
    segments = _split_oversized_text(text, config.max_tokens)

    if not segments:
        return []

    # Build char-to-line mapping for accurate line numbers
    char_to_line_map = _build_char_to_line_map(lines)

    chunks: list[Chunk] = []
    current_pos = 0

    for idx, segment in enumerate(segments):
        if not segment.strip():
            continue

        # Find segment position in original text
        seg_start = text.find(segment, current_pos)
        if seg_start == -1:
            # Fallback: use current position
            seg_start = current_pos
        seg_end = seg_start + len(segment) - 1

        # Map character positions to line numbers
        first_line_idx, last_line_idx = _find_line_range_for_segment(
            seg_start, seg_end, char_to_line_map
        )

        # Create chunk with accurate line numbers (1-indexed)
        chunk = _make_chunk_tiktoken(
            [segment],
            None,  # No section (headerless document)
            first_line_idx + 1,
            last_line_idx + 1,
            idx,
        )
        chunks.append(chunk)
        current_pos = seg_start + len(segment)

    return chunks


def chunk_text_tiktoken(text: str, config: ChunkConfig) -> list[Chunk]:
    """Chunk text using tiktoken for accurate token counting.

    This is the primary chunking function for the RAG pipeline. It uses
    tiktoken for accurate token counting (instead of word-based estimation)
    and supports chunk overlap for context continuity.

    Split hierarchy (in priority order):
    1. Section headers (== ... ==) - hard break, always starts new chunk
    2. Paragraph boundaries (blank lines) - preferred soft break
    3. Sentence boundaries (. ! ?) - secondary soft break
    4. Word boundaries (spaces) - last resort for oversized lines

    For documents without section headers (e.g., long Library works), a fast
    path using semantic-text-splitter is used to avoid O(n^2) performance.

    Args:
        text: Raw text content to chunk
        config: Chunking configuration with token limits and overlap

    Returns:
        List of Chunk objects with accurate token counts
    """
    if not text or not text.strip():
        return []

    lines = text.strip().split("\n")

    # Fast path: headerless documents use semantic splitter directly
    # This avoids O(n^2) line-by-line processing for monolithic Library works
    has_headers = any(is_section_header(line) for line in lines)
    if not has_headers:
        return _chunk_without_headers(text, lines, config)
    chunks: list[Chunk] = []
    current_lines: list[str] = []
    current_section: str | None = None
    line_start: int = 1
    chunk_index: int = 0
    overlap_lines: list[str] = []  # Lines to prepend for overlap

    # Process line by line
    max_iterations = len(lines) * 10 + 100  # Allow for line splitting
    line_idx = 0
    iteration_count = 0

    while line_idx < len(lines):
        iteration_count += 1
        if iteration_count >= max_iterations:
            break  # Safety valve

        line = lines[line_idx]
        line_num = line_idx + 1

        # Check for section header - always starts new chunk
        if is_section_header(line):
            # Flush current chunk if non-empty
            if current_lines:
                cleaned = _strip_trailing_blank_lines(current_lines)
                if cleaned:
                    # Check if current chunk is oversized and needs splitting
                    chunk_text = "\n".join(cleaned)
                    if count_tokens(chunk_text) > config.max_tokens:
                        # Split oversized chunk into multiple chunks
                        sub_chunks = _create_chunks_from_oversized(
                            cleaned, current_section, line_start, chunk_index, config
                        )
                        chunks.extend(sub_chunks)
                        chunk_index += len(sub_chunks)
                        if sub_chunks:
                            overlap_lines = _get_overlap_lines(
                                sub_chunks[-1].text.split("\n"), config.overlap_tokens
                            )
                    else:
                        chunk = _make_chunk_tiktoken(
                            cleaned,
                            current_section,
                            line_start,
                            line_start + len(cleaned) - 1,
                            chunk_index,
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                        overlap_lines = _get_overlap_lines(cleaned, config.overlap_tokens)

            # Update section and start new chunk
            current_section = extract_section_title(line)
            # Include overlap from previous chunk
            if overlap_lines and config.overlap_tokens > 0:
                current_lines = [*overlap_lines, line]
                line_start = line_num - len(overlap_lines)
            else:
                current_lines = [line]
                line_start = line_num
            overlap_lines = []
            line_idx += 1
            continue

        # Check if this single line exceeds max_tokens
        # Use character estimation to avoid expensive token counting for short lines
        # (~4 chars per token, use 3 to be conservative)
        line_could_be_oversized = len(line) > config.max_tokens * 3
        if line_could_be_oversized and count_tokens(line) > config.max_tokens:
            # Flush current chunk first if non-empty
            if current_lines:
                cleaned = _strip_trailing_blank_lines(current_lines)
                if cleaned:
                    chunk_text = "\n".join(cleaned)
                    if count_tokens(chunk_text) > config.max_tokens:
                        sub_chunks = _create_chunks_from_oversized(
                            cleaned, current_section, line_start, chunk_index, config
                        )
                        chunks.extend(sub_chunks)
                        chunk_index += len(sub_chunks)
                    else:
                        chunk = _make_chunk_tiktoken(
                            cleaned,
                            current_section,
                            line_start,
                            line_start + len(cleaned) - 1,
                            chunk_index,
                        )
                        chunks.append(chunk)
                        chunk_index += 1

            # Split the oversized line into multiple segments
            segments = _split_oversized_text(line, config.max_tokens)
            for seg in segments:
                if seg.strip():
                    chunk = _make_chunk_tiktoken(
                        [seg],
                        current_section,
                        line_num,
                        line_num,
                        chunk_index,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

            # Reset for next line
            current_lines = []
            line_start = line_num + 1
            overlap_lines = []
            line_idx += 1
            continue

        # Add line to current chunk
        current_lines.append(line)

        # Check if exceeding max tokens (use char estimate to avoid expensive token count)
        # Only do expensive token count when char length suggests we might be close
        current_text = "\n".join(current_lines)
        char_estimate_tokens = len(current_text) / 4  # ~4 chars per token

        if char_estimate_tokens >= config.max_tokens * 0.7:
            current_tokens = count_tokens(current_text)
        else:
            current_tokens = int(char_estimate_tokens)

        if current_tokens >= config.max_tokens:
            # Find best split point
            split_idx = _find_split_point(current_lines, config.max_tokens)

            if split_idx > 0:
                # Create chunk up to split point
                chunk_lines = current_lines[:split_idx]
                cleaned = _strip_trailing_blank_lines(chunk_lines)
                if cleaned:
                    chunk = _make_chunk_tiktoken(
                        cleaned,
                        current_section,
                        line_start,
                        line_start + len(cleaned) - 1,
                        chunk_index,
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    # Save lines for overlap
                    overlap_lines = _get_overlap_lines(cleaned, config.overlap_tokens)

                # Start new chunk with remaining lines + overlap
                remaining = current_lines[split_idx:]
                # Skip leading blank lines
                while remaining and not remaining[0].strip():
                    split_idx += 1
                    remaining = remaining[1:]

                if overlap_lines and config.overlap_tokens > 0:
                    current_lines = overlap_lines + remaining
                    line_start = line_start + split_idx - len(overlap_lines)
                else:
                    current_lines = remaining
                    line_start = line_start + split_idx
                overlap_lines = []

        line_idx += 1

    # Flush final chunk
    if current_lines:
        cleaned = _strip_trailing_blank_lines(current_lines)
        if cleaned:
            chunk_text = "\n".join(cleaned)
            if count_tokens(chunk_text) > config.max_tokens:
                # Split oversized final chunk
                sub_chunks = _create_chunks_from_oversized(
                    cleaned, current_section, line_start, chunk_index, config
                )
                chunks.extend(sub_chunks)
            else:
                chunk = _make_chunk_tiktoken(
                    cleaned,
                    current_section,
                    line_start,
                    line_start + len(cleaned) - 1,
                    chunk_index,
                )
                chunks.append(chunk)

    return chunks


def _find_split_point(lines: list[str], max_tokens: int) -> int:
    """Find the best line index to split at, respecting token limit.

    Prefers splitting at:
    1. Paragraph boundaries (blank lines)
    2. Sentence endings (lines ending with . ! ?)
    3. Any line boundary

    Uses character-based estimation for speed, with token verification
    only when approaching the limit.

    Args:
        lines: Lines to search for split point
        max_tokens: Maximum token count for the chunk

    Returns:
        Index to split at (exclusive), or len(lines)-1 if no good split found
    """
    # Find last paragraph break that keeps us under max_tokens
    last_para_break = 0
    last_sentence_end = 0
    cumulative_chars = 0

    for i, line in enumerate(lines):
        cumulative_chars += len(line) + 1  # +1 for newline

        # Use char estimate to decide if we need expensive token count
        char_estimate_tokens = cumulative_chars / 4

        if char_estimate_tokens > max_tokens * 0.8:
            # Getting close to limit, do actual token count
            partial_text = "\n".join(lines[: i + 1])
            partial_tokens = count_tokens(partial_text)

            if partial_tokens > max_tokens:
                # We've exceeded the limit, use best prior split
                if last_para_break > 0:
                    return last_para_break
                if last_sentence_end > 0:
                    return last_sentence_end
                # No good split found, split at previous line
                return max(1, i)

        # Track potential split points
        if not line.strip():  # Blank line = paragraph break
            last_para_break = i + 1
        elif line.rstrip().endswith((".", "!", "?", '"', "'")):
            last_sentence_end = i + 1

    # Text fits within limit, no split needed
    return len(lines)


def _get_overlap_lines(lines: list[str], overlap_tokens: int) -> list[str]:
    """Get lines from the end that total approximately overlap_tokens.

    Args:
        lines: Source lines to extract overlap from
        overlap_tokens: Target number of tokens for overlap

    Returns:
        List of lines from the end totaling ~overlap_tokens
    """
    if overlap_tokens <= 0 or not lines:
        return []

    # Work backwards to find lines totaling ~overlap_tokens
    overlap_lines: list[str] = []
    total_tokens = 0

    for line in reversed(lines):
        line_tokens = count_tokens(line)
        if total_tokens + line_tokens > overlap_tokens and overlap_lines:
            break
        overlap_lines.insert(0, line)
        total_tokens += line_tokens

    return overlap_lines
