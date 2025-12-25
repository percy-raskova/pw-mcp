"""Chunker package - text chunking for embedding-ready segments.

This package provides tiktoken-based chunking with section awareness,
overlap support, and accurate line number tracking for citations.

Public API (all backward-compatible re-exports):
- ChunkConfig: Frozen configuration dataclass
- Chunk: Single chunk with metadata
- ChunkedArticle: Article with chunks and metadata
- FilterStats: Filtering statistics from JSONL writing
- chunk_text_tiktoken: Main chunking function (tiktoken-based)
- chunk_text: Backward-compatible wrapper
- chunk_article: File-based chunking with metadata
- write_chunks_jsonl: JSONL output with filtering
- count_tokens: Token counting utility
- is_section_header: Section detection utility
- extract_section_title: Title extraction utility
- estimate_tokens: Word-based token estimation
- generate_chunk_id: URL-safe chunk ID generation
"""

# Models (Phase 1C)
# Core chunking algorithms (Phase 4A)
from pw_mcp.ingest.chunker.core import (
    chunk_text,
    chunk_text_tiktoken,
)

# JSONL output (Phase 3B)
from pw_mcp.ingest.chunker.jsonl_writer import (
    chunk_article,
    generate_chunk_id,
    write_chunks_jsonl,
)

# Line mapping (Phase 3A)
from pw_mcp.ingest.chunker.line_mapping import (
    _build_char_to_line_map,
    _find_line_range_for_segment,
)
from pw_mcp.ingest.chunker.models import (
    Chunk,
    ChunkConfig,
    ChunkedArticle,
    FilterStats,
)

# Section detection (Phase 1B)
from pw_mcp.ingest.chunker.section_detection import (
    SECTION_HEADER_PATTERN,
    extract_section_title,
    is_section_header,
)

# Text splitting (Phase 2B)
from pw_mcp.ingest.chunker.text_splitting import (
    MIN_SEGMENT_CHARS,
    _emergency_split_position,
    _split_oversized_text,
)

# Token counting (Phase 1A)
from pw_mcp.ingest.chunker.token_counting import count_tokens, estimate_tokens

__all__ = [
    # Constants
    "MIN_SEGMENT_CHARS",
    "SECTION_HEADER_PATTERN",
    # Models
    "Chunk",
    "ChunkConfig",
    "ChunkedArticle",
    "FilterStats",
    # Internal (exported for testing)
    "_build_char_to_line_map",
    "_emergency_split_position",
    "_find_line_range_for_segment",
    "_split_oversized_text",
    # Public API - Chunking
    "chunk_article",
    "chunk_text",
    "chunk_text_tiktoken",
    # Public API - Utilities
    "count_tokens",
    "estimate_tokens",
    "extract_section_title",
    "generate_chunk_id",
    "is_section_header",
    "write_chunks_jsonl",
]
