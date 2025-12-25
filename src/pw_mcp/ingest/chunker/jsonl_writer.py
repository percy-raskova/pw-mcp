"""JSONL output for chunked articles.

This module provides functions for writing chunked articles to JSONL format
with filtering (micro-chunk removal, deduplication) and metadata enrichment.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pw_mcp.ingest.chunker.models import (
    Chunk,
    ChunkConfig,
    ChunkedArticle,
    FilterStats,
)

logger = logging.getLogger(__name__)


def _extract_year(date_str: str | None) -> int | None:
    """Extract year from a date string.

    Handles formats like:
    - "1917" (just year)
    - "1917-10-25" (ISO format)
    - "October 25, 1917" (human readable)

    Args:
        date_str: Date string to parse, or None.

    Returns:
        Year as integer, or None if extraction fails.
    """
    if not date_str:
        return None

    # Try to find a 4-digit year pattern
    match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", date_str)
    if match:
        return int(match.group(1))

    return None


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
    # Import here to avoid circular dependency
    from pw_mcp.ingest.chunker.core import chunk_text_tiktoken

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


def write_chunks_jsonl(
    article: ChunkedArticle,
    output_path: Path,
    config: ChunkConfig,
) -> FilterStats:
    """Write chunked article to JSONL file with filtering.

    Filters out:
    1. Micro-chunks: Chunks with word_count < config.min_words
    2. Consecutive duplicates: Adjacent chunks with identical text

    Each remaining chunk is written as one JSON object per line with all metadata
    attached. The output follows the schema specified in chunking.yaml.

    Args:
        article: ChunkedArticle to serialize
        output_path: Path for the output JSONL file
        config: ChunkConfig with filtering parameters

    Returns:
        FilterStats with counts of filtered/written chunks
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Track statistics
    total_chunks = len(article.chunks)
    micro_chunks_filtered = 0
    consecutive_duplicates_removed = 0

    # Filter and deduplicate
    filtered_chunks: list[Chunk] = []
    prev_text: str | None = None

    for chunk in article.chunks:
        # Skip micro-chunks (empty or below min_words threshold)
        if config.min_words > 0 and chunk.word_count < config.min_words:
            micro_chunks_filtered += 1
            continue

        # Skip consecutive duplicates
        if prev_text is not None and chunk.text == prev_text:
            consecutive_duplicates_removed += 1
            continue

        filtered_chunks.append(chunk)
        prev_text = chunk.text

    # Log warning if high filter rate (>10%)
    if total_chunks > 0:
        filter_rate = (micro_chunks_filtered + consecutive_duplicates_removed) / total_chunks
        if filter_rate > 0.10:
            logger.warning(
                f"High filter rate ({filter_rate:.1%}): {micro_chunks_filtered} micro-chunks, "
                f"{consecutive_duplicates_removed} duplicates removed from {total_chunks} total"
            )

    # Write filtered chunks with reassigned indices
    with output_path.open("w", encoding="utf-8") as f:
        for new_index, chunk in enumerate(filtered_chunks):
            # Generate chunk ID with new index
            chunk_id = generate_chunk_id(
                article.namespace,
                article.article_title,
                new_index,
            )

            # Build record with all fields
            record: dict[str, Any] = {
                "chunk_id": chunk_id,
                "text": chunk.text,
                "article_title": article.article_title,
                "namespace": article.namespace,
                "section": chunk.section,
                "chunk_index": new_index,
                "line_range": f"{chunk.line_start}-{chunk.line_end}",
                "word_count": chunk.word_count,
                "categories": article.categories,
                "internal_links": article.internal_links,
                "is_stub": article.is_stub,
                "citation_needed_count": article.citation_needed_count,
                "has_blockquote": article.has_blockquote,
                # Phase B: Enriched metadata for RAG filtering
                "library_work_author": (
                    article.library_work.get("author") if article.library_work else None
                ),
                "library_work_type": (
                    article.library_work.get("work_type") if article.library_work else None
                ),
                "library_work_published_year": (
                    _extract_year(article.library_work.get("published_date"))
                    if article.library_work
                    else None
                ),
                "infobox_type": (article.infobox.get("type") if article.infobox else None),
                "political_orientation": (
                    article.infobox.get("fields", {}).get("political_orientation")
                    if article.infobox
                    else None
                ),
                "primary_category": (article.categories[0] if article.categories else None),
                "category_count": len(article.categories),
            }

            # Write as single line JSON
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return FilterStats(
        total_chunks=total_chunks,
        micro_chunks_filtered=micro_chunks_filtered,
        consecutive_duplicates_removed=consecutive_duplicates_removed,
        chunks_written=len(filtered_chunks),
    )
