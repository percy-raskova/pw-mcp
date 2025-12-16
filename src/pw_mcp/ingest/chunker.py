"""Text chunking module for embedding-ready segments.

This module chunks sembr'd (semantically line-broken) text into coherent
segments suitable for vector embedding. It respects section boundaries
(hard breaks) and paragraph boundaries (soft breaks) while tracking
line numbers for precise citation back to source.

Design rationale:
- Section headers (== ... ==) always start new chunks
- Paragraph breaks (blank lines) are preferred split points
- Chunks aim for target_tokens but respect min/max bounds
- Line numbers enable precise citation back to source
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Section header pattern: matches == Title ==, === Subsection ===, etc.
# Captures the equals signs to ensure balanced opening/closing
SECTION_HEADER_PATTERN = re.compile(r"^(={2,6})\s*([^=]+?)\s*\1\s*$")


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for text chunking.

    Attributes:
        target_tokens: Ideal chunk size to aim for (default: 600)
        min_tokens: Minimum chunk size to avoid tiny fragments (default: 200)
        max_tokens: Maximum chunk size to stay under embedding limit (default: 1000)
        token_estimation_factor: Multiply word_count by this to estimate tokens
            (default: 1.3, English averages ~1.3 tokens/word)
    """

    target_tokens: int = 600
    min_tokens: int = 200
    max_tokens: int = 1000
    token_estimation_factor: float = 1.3


@dataclass
class Chunk:
    """Single chunk of text with metadata.

    Attributes:
        text: Clean text content (what gets embedded)
        chunk_index: Order within article (0-indexed)
        section: Section header this chunk falls under (None if before first header)
        line_start: Starting line in sembr'd source (1-indexed)
        line_end: Ending line in sembr'd source (inclusive)
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
    """Chunk sembr'd text into embedding-ready segments.

    This function parses sembr output line by line, accumulating into chunks.
    It respects section boundaries (hard breaks) and paragraph boundaries
    (soft breaks), while tracking line numbers for citation.

    Algorithm:
    1. Section headers (== ... ==) always start a new chunk
    2. When approaching max_tokens, split at last paragraph boundary
    3. Track line numbers for precise citation back to source

    Args:
        lines: List of lines from sembr'd text file
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


def chunk_article(
    sembr_path: Path,
    metadata: dict[str, Any],
    config: ChunkConfig,
) -> ChunkedArticle:
    """Chunk a sembr'd article file with associated metadata.

    Reads the sembr'd text file, extracts namespace and title from the path,
    and combines with metadata to create a ChunkedArticle.

    Args:
        sembr_path: Path to the sembr'd text file
        metadata: Dictionary with article metadata (from extraction phase)
        config: Chunking configuration

    Returns:
        ChunkedArticle with chunks and propagated metadata
    """
    # Read sembr'd text
    content = sembr_path.read_text(encoding="utf-8")
    lines = content.strip().split("\n") if content.strip() else []

    # Extract namespace from path or metadata
    namespace = metadata.get("namespace", sembr_path.parent.name)

    # Extract title from path stem (replace underscores with spaces)
    title = sembr_path.stem.replace("_", " ")

    # Chunk the text
    chunks = chunk_text(lines, config)

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
