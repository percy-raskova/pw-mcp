"""Text chunking module for embedding-ready segments.

This module chunks text into coherent segments suitable for vector embedding.
It uses tiktoken for accurate token counting and supports chunk overlap for
RAG context continuity.

Design rationale:
- Section headers (== ... ==) always start new chunks (hard breaks)
- Paragraph breaks (blank lines) are preferred split points (soft breaks)
- Sentence boundaries are the next preferred split points
- Chunks aim for target_tokens but respect min/max bounds
- Overlap ensures context continuity at chunk boundaries
- Line numbers enable precise citation back to source
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken

# Global tiktoken encoder (cached for performance)
_TIKTOKEN_ENCODER: tiktoken.Encoding | None = None

# Section header pattern: matches == Title ==, === Subsection ===, etc.
# Captures the equals signs to ensure balanced opening/closing
SECTION_HEADER_PATTERN = re.compile(r"^(={2,6})\s*([^=]+?)\s*\1\s*$")


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


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for text chunking.

    Attributes:
        target_tokens: Ideal chunk size to aim for (default: 600)
        min_tokens: Minimum chunk size to avoid tiny fragments (default: 200)
        max_tokens: Maximum chunk size to stay under embedding limit (default: 1000)
        overlap_tokens: Number of tokens to overlap between chunks (default: 50)
        token_estimation_factor: Multiply word_count by this to estimate tokens
            (default: 1.3, English averages ~1.3 tokens/word)
            Note: This is kept for backwards compatibility but tiktoken
            provides accurate counts via count_tokens().
    """

    target_tokens: int = 600
    min_tokens: int = 200
    max_tokens: int = 1000
    overlap_tokens: int = 50
    token_estimation_factor: float = 1.3


@dataclass
class Chunk:
    """Single chunk of text with metadata.

    Attributes:
        text: Clean text content (what gets embedded)
        chunk_index: Order within article (0-indexed)
        section: Section header this chunk falls under (None if before first header)
        line_start: Starting line in source (1-indexed)
        line_end: Ending line in source (inclusive)
        word_count: Actual word count in chunk
        estimated_tokens: word_count * token_estimation_factor
    """

    text: str
    chunk_index: int
    section: str | None
    line_start: int
    line_end: int
    word_count: int
    estimated_tokens: int


@dataclass
class ChunkedArticle:
    """Article split into chunks with propagated metadata.

    Attributes:
        article_title: Article title (from filename or extraction)
        namespace: ProleWiki namespace (Main, Library, Essays, ProleWiki)
        chunks: Ordered list of chunks
        categories: Categories from [[Category:...]]
        internal_links: Articles this article links TO
        infobox: Parsed infobox if present
        library_work: Library work metadata (Library namespace)
        is_stub: Article marked incomplete
        citation_needed_count: Number of {{Citation needed}} markers
        has_blockquote: Contains <blockquote> content
    """

    article_title: str
    namespace: str
    chunks: list[Chunk]
    categories: list[str] = field(default_factory=list)
    internal_links: list[str] = field(default_factory=list)
    infobox: dict[str, Any] | None = None
    library_work: dict[str, Any] | None = None
    is_stub: bool = False
    citation_needed_count: int = 0
    has_blockquote: bool = False


def is_section_header(line: str) -> bool:
    """Check if a line is a MediaWiki section header.

    Section headers have the form: == Title == or === Subsection === etc.
    The number of equals signs must match on both sides (2-6).

    Args:
        line: The line to check

    Returns:
        True if the line is a section header, False otherwise

    Examples:
        >>> is_section_header("== Introduction ==")
        True
        >>> is_section_header("=== Subsection ===")
        True
        >>> is_section_header("a = b + c")
        False
    """
    return SECTION_HEADER_PATTERN.match(line) is not None


def extract_section_title(line: str) -> str:
    """Extract the title from a section header line.

    Args:
        line: A section header line (must pass is_section_header check)

    Returns:
        The section title with whitespace stripped

    Example:
        >>> extract_section_title("== Introduction ==")
        'Introduction'
    """
    match = SECTION_HEADER_PATTERN.match(line)
    if match:
        return match.group(2).strip()
    return ""


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


def generate_chunk_id(namespace: str, title: str, index: int) -> str:
    """Generate a unique, URL-safe chunk ID.

    Format: {namespace}/{title}#{index}
    Spaces in title are replaced with underscores.

    Args:
        namespace: The ProleWiki namespace
        title: The article title
        index: The chunk index (0-based)

    Returns:
        A URL-safe chunk identifier

    Example:
        >>> generate_chunk_id("Main", "Five-Year Plans", 3)
        'Main/Five-Year_Plans#3'
    """
    safe_title = title.replace(" ", "_")
    return f"{namespace}/{safe_title}#{index}"


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

    Args:
        text: Raw text content to chunk
        config: Chunking configuration with token limits and overlap

    Returns:
        List of Chunk objects with accurate token counts
    """
    if not text or not text.strip():
        return []

    lines = text.strip().split("\n")
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
        line_tokens = count_tokens(line)
        if line_tokens > config.max_tokens:
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

        # Check if exceeding max tokens
        current_text = "\n".join(current_lines)
        current_tokens = count_tokens(current_text)

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


def _create_chunks_from_oversized(
    lines: list[str],
    section: str | None,
    line_start: int,
    start_index: int,
    config: ChunkConfig,
) -> list[Chunk]:
    """Create multiple chunks from oversized text.

    Args:
        lines: Lines that together exceed max_tokens
        section: Current section name
        line_start: Starting line number
        start_index: Starting chunk index
        config: Chunking configuration

    Returns:
        List of Chunk objects, each under max_tokens
    """
    text = "\n".join(lines)
    segments = _split_oversized_text(text, config.max_tokens)

    chunks: list[Chunk] = []
    chunk_index = start_index

    for seg in segments:
        if seg.strip():
            chunk = _make_chunk_tiktoken(
                [seg],
                section,
                line_start,
                line_start + len(lines) - 1,  # Approximate line range
                chunk_index,
            )
            chunks.append(chunk)
            chunk_index += 1

    return chunks


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


def _find_split_point(lines: list[str], max_tokens: int) -> int:
    """Find the best line index to split at, respecting token limit.

    Prefers splitting at:
    1. Paragraph boundaries (blank lines)
    2. Sentence endings (lines ending with . ! ?)
    3. Any line boundary

    Args:
        lines: Lines to search for split point
        max_tokens: Maximum token count for the chunk

    Returns:
        Index to split at (exclusive), or len(lines)-1 if no good split found
    """
    # Find last paragraph break that keeps us under max_tokens
    last_para_break = 0
    last_sentence_end = 0

    for i, line in enumerate(lines):
        # Build text up to this point
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


def _split_oversized_text(text: str, max_tokens: int) -> list[str]:
    """Split text that exceeds max_tokens into smaller segments.

    Uses a hierarchy of split points:
    1. Sentence boundaries (. ! ? followed by space)
    2. Clause boundaries (, ; :)
    3. Word boundaries (spaces)

    Args:
        text: Text to split (may exceed max_tokens)
        max_tokens: Maximum tokens per segment

    Returns:
        List of text segments, each under max_tokens
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    segments: list[str] = []
    remaining = text

    # Safety limit to prevent infinite loops
    max_iterations = len(text) // 10 + 100

    for _ in range(max_iterations):
        if not remaining.strip():
            break

        if count_tokens(remaining) <= max_tokens:
            segments.append(remaining)
            break

        # Find best split point under max_tokens
        split_pos = _find_text_split_position(remaining, max_tokens)

        if split_pos <= 0:
            # Emergency fallback: split at max_tokens worth of characters
            # This shouldn't happen with word boundaries, but safety first
            split_pos = _emergency_split_position(remaining, max_tokens)

        segment = remaining[:split_pos].rstrip()
        if segment:
            segments.append(segment)

        remaining = remaining[split_pos:].lstrip()

    return segments


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
    """Emergency fallback to find any valid split position.

    Used when normal splitting fails. Finds any position that keeps
    the first segment under max_tokens.

    Args:
        text: Text to split
        max_tokens: Maximum tokens

    Returns:
        Character position, or 1 as absolute minimum
    """
    # Start from beginning and find where we exceed max_tokens
    for pos in range(1, len(text) + 1):
        if count_tokens(text[:pos]) > max_tokens:
            return max(1, pos - 1)
    return len(text)


def chunk_article(
    text_path: Path,
    metadata: dict[str, Any],
    config: ChunkConfig,
) -> ChunkedArticle:
    """Chunk an extracted text file with associated metadata.

    Reads the text file, extracts namespace and title from the path,
    and combines with metadata to create a ChunkedArticle.
    Uses tiktoken for accurate token counting.

    Args:
        text_path: Path to the extracted text file
        metadata: Dictionary with article metadata (from extraction phase)
        config: Chunking configuration

    Returns:
        ChunkedArticle with chunks and propagated metadata
    """
    # Read text content
    content = text_path.read_text(encoding="utf-8")

    # Extract namespace from path or metadata
    namespace = metadata.get("namespace", text_path.parent.name)

    # Extract title from path stem (replace underscores with spaces)
    title = text_path.stem.replace("_", " ")

    # Chunk the text using tiktoken-based chunker
    chunks = chunk_text_tiktoken(content, config)

    # Extract internal link targets from metadata
    internal_links: list[str] = []
    raw_links = metadata.get("internal_links", [])
    for link in raw_links:
        if isinstance(link, dict) and "target" in link:
            internal_links.append(link["target"])
        elif isinstance(link, str):
            internal_links.append(link)

    # Build ChunkedArticle with propagated metadata
    return ChunkedArticle(
        article_title=title,
        namespace=namespace,
        chunks=chunks,
        categories=metadata.get("categories", []),
        internal_links=internal_links,
        infobox=metadata.get("infobox"),
        library_work=metadata.get("library_work"),
        is_stub=metadata.get("is_stub", False),
        citation_needed_count=metadata.get("citation_needed_count", 0),
        has_blockquote=metadata.get("has_blockquote", False),
    )


def write_chunks_jsonl(article: ChunkedArticle, output_path: Path) -> None:
    """Write chunked article to JSONL file.

    Each chunk is written as one JSON object per line with all metadata
    attached. The output follows the schema specified in chunking.yaml.

    Args:
        article: ChunkedArticle to serialize
        output_path: Path for the output JSONL file
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in article.chunks:
            # Generate chunk ID
            chunk_id = generate_chunk_id(
                article.namespace,
                article.article_title,
                chunk.chunk_index,
            )

            # Build record with all fields
            record: dict[str, Any] = {
                "chunk_id": chunk_id,
                "text": chunk.text,
                "article_title": article.article_title,
                "namespace": article.namespace,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "line_range": f"{chunk.line_start}-{chunk.line_end}",
                "word_count": chunk.word_count,
                "categories": article.categories,
                "internal_links": article.internal_links,
                "is_stub": article.is_stub,
                "citation_needed_count": article.citation_needed_count,
                "has_blockquote": article.has_blockquote,
            }

            # Write as single line JSON
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
