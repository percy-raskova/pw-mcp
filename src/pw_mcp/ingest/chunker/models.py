"""Core data models for the chunking pipeline.

This module contains the dataclasses used throughout the chunking process:
- ChunkConfig: Configuration parameters for chunking
- Chunk: A single text chunk with metadata
- ChunkedArticle: An article split into chunks with propagated metadata
- FilterStats: Statistics from chunk filtering during JSONL writing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for text chunking.

    Attributes:
        target_tokens: Ideal chunk size to aim for (default: 600)
        min_tokens: Minimum chunk size to avoid tiny fragments (default: 200)
        max_tokens: Maximum chunk size to stay under embedding limit (default: 1000)
        overlap_tokens: Number of tokens to overlap between chunks (default: 50)
        min_words: Minimum word count for a chunk to be included (default: 10).
            Chunks below this threshold are filtered out during JSONL writing.
            Set to 0 to disable micro-chunk filtering.
        token_estimation_factor: Multiply word_count by this to estimate tokens
            (default: 1.3, English averages ~1.3 tokens/word)
            Note: This is kept for backwards compatibility but tiktoken
            provides accurate counts via count_tokens().
    """

    target_tokens: int = 600
    min_tokens: int = 200
    max_tokens: int = 1000
    overlap_tokens: int = 50
    min_words: int = 10
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


@dataclass
class FilterStats:
    """Statistics from chunk filtering during JSONL writing.

    Attributes:
        total_chunks: Number of chunks before filtering
        micro_chunks_filtered: Chunks removed due to word_count < min_words
        consecutive_duplicates_removed: Consecutive identical chunks removed
        chunks_written: Final number of chunks written to JSONL
    """

    total_chunks: int
    micro_chunks_filtered: int
    consecutive_duplicates_removed: int
    chunks_written: int
