"""Text splitting for oversized content.

This module handles splitting text that exceeds token limits into smaller
segments. It uses semantic-text-splitter (Rust/SIMD optimized) when available,
with fallback to character-based estimation.
"""

from __future__ import annotations

from pw_mcp.ingest.chunker.token_counting import count_tokens

# Minimum segment size to prevent pathologically small splits
# ~12-15 tokens minimum per segment
MIN_SEGMENT_CHARS = 50


def _split_oversized_text(text: str, max_tokens: int) -> list[str]:
    """Split text that exceeds max_tokens into smaller segments.

    Uses semantic-text-splitter (Rust/SIMD optimized) for fast, accurate
    token-aware splitting. Falls back to character-based estimation if needed.

    Args:
        text: Text to split (may exceed max_tokens)
        max_tokens: Maximum tokens per segment

    Returns:
        List of text segments, each under max_tokens
    """
    # Quick check if already small enough
    if len(text) <= max_tokens * 6 and count_tokens(text) <= max_tokens:
        return [text]

    try:
        # Use semantic-text-splitter for fast, accurate splitting
        from semantic_text_splitter import TextSplitter

        # Create splitter with tiktoken (cl100k_base is used by GPT-4/3.5)
        splitter = TextSplitter.from_tiktoken_model("gpt-4", max_tokens)
        segments = list(splitter.chunks(text))

        # Filter empty segments only - the library produces correct, non-overlapping
        # segments. Deduplication happens at chunk level in write_chunks_jsonl.
        if not segments:
            return [text] if text.strip() else []

        result = [seg.strip() for seg in segments if seg.strip()]
        return result if result else [text.strip()]

    except ImportError:
        # Fallback to character-based estimation if library not available
        return _split_oversized_text_fallback(text, max_tokens)


def _split_oversized_text_fallback(text: str, max_tokens: int) -> list[str]:
    """Fallback text splitting using character estimation.

    Used when semantic-text-splitter is not available.
    """
    target_chars = max_tokens * 3  # Conservative estimate (~3 chars per token)
    segments: list[str] = []
    remaining = text
    max_segments = 100
    prev_segment: str | None = None

    for _ in range(max_segments):
        if not remaining.strip():
            break

        if len(remaining) <= target_chars * 1.5:
            segments.append(remaining.strip())
            break

        # Find split at sentence/word boundary
        split_pos = min(target_chars, len(remaining))
        search_text = remaining[: split_pos + 100]

        # Try sentence boundary
        for punct in [". ", "! ", "? "]:
            pos = search_text.rfind(punct, 0, split_pos + 50)
            if pos > MIN_SEGMENT_CHARS:
                split_pos = pos + 2
                break
        else:
            # Try word boundary
            pos = search_text.rfind(" ", 0, split_pos)
            if pos > MIN_SEGMENT_CHARS:
                split_pos = pos + 1

        segment = remaining[:split_pos].rstrip()
        if segment and segment != prev_segment:
            segments.append(segment)
            prev_segment = segment

        remaining = remaining[split_pos:].lstrip()

    return segments if segments else [text.strip()]


def _find_text_split_position(text: str, max_tokens: int) -> int:
    """Find the best character position to split text under max_tokens.

    Searches for split points in priority order:
    1. Sentence endings (. ! ?)
    2. Clause separators (, ; :)
    3. Word boundaries (spaces)

    Args:
        text: Text to find split position in
        max_tokens: Maximum tokens for the first segment

    Returns:
        Character position to split at (exclusive), or 0 if no valid split found
    """
    # Binary search to find approximate character position for max_tokens
    # Start with a rough estimate (4 chars per token for English)
    chars_per_token_estimate = 4
    search_start = max(1, max_tokens * chars_per_token_estimate // 2)
    search_end = min(len(text), max_tokens * chars_per_token_estimate * 2)

    # Find the character position where we hit max_tokens
    # Use binary search for efficiency
    left, right = search_start, search_end
    max_valid_pos = 0

    for _ in range(50):  # Binary search iterations
        if left >= right:
            break
        mid = (left + right) // 2
        tokens = count_tokens(text[:mid])
        if tokens <= max_tokens:
            max_valid_pos = mid
            left = mid + 1
        else:
            right = mid

    if max_valid_pos == 0:
        # Even the start is too big, find minimum viable position
        for pos in range(1, min(len(text), 100)):
            if count_tokens(text[:pos]) <= max_tokens:
                max_valid_pos = pos

    # Now search backwards from max_valid_pos for best split point
    search_text = text[:max_valid_pos]

    # Priority 1: Sentence boundaries
    for punct in [". ", "! ", "? ", '." ', '!" ', '?" ']:
        pos = search_text.rfind(punct)
        if pos > 0:
            return pos + len(punct.rstrip())  # Include the punctuation

    # Priority 2: Clause separators
    for punct in [", ", "; ", ": "]:
        pos = search_text.rfind(punct)
        if pos > 0:
            return pos + len(punct.rstrip())

    # Priority 3: Word boundary (space)
    pos = search_text.rfind(" ")
    if pos > 0:
        return pos + 1  # Split after the space

    # Fallback: use max_valid_pos directly
    return max_valid_pos


def _emergency_split_position(text: str, max_tokens: int) -> int:
    """Emergency fallback to find any valid split position using binary search.

    Used when normal splitting fails. Finds any position that keeps
    the first segment under max_tokens. Uses O(log n) binary search
    instead of O(n) linear scan for efficiency on long texts.

    Args:
        text: Text to split
        max_tokens: Maximum tokens

    Returns:
        Character position, or 1 as absolute minimum
    """
    if not text:
        return 0

    # Binary search for the position where tokens exceed max_tokens
    left, right = 1, len(text)
    result = len(text)  # Default: entire text fits

    while left <= right:
        mid = (left + right) // 2
        tokens = count_tokens(text[:mid])

        if tokens <= max_tokens:
            left = mid + 1
        else:
            result = max(1, mid - 1)
            right = mid - 1

    return result
