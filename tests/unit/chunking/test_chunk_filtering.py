"""Unit tests for chunk filtering and deduplication (TDD Red Phase).

Tests for:
- min_words configuration in ChunkConfig
- Micro-chunk filtering
- Consecutive duplicate removal
- Filtering statistics tracking
- Logging for high filter rates
"""

import json
import logging
from pathlib import Path

import pytest

from pw_mcp.ingest.chunker import (
    Chunk,
    ChunkConfig,
    ChunkedArticle,
    FilterStats,
    write_chunks_jsonl,
)

# =============================================================================
# CHUNKCONFIG MIN_WORDS TESTS
# =============================================================================


class TestChunkConfigMinWords:
    """Tests for min_words configuration parameter."""

    @pytest.mark.unit
    def test_chunk_config_has_min_words_field(self) -> None:
        """ChunkConfig should have min_words field."""
        config = ChunkConfig()
        assert hasattr(config, "min_words")

    @pytest.mark.unit
    def test_chunk_config_min_words_default_is_10(self) -> None:
        """min_words should default to 10."""
        config = ChunkConfig()
        assert config.min_words == 10

    @pytest.mark.unit
    def test_chunk_config_min_words_custom(self) -> None:
        """min_words should be configurable."""
        config = ChunkConfig(min_words=5)
        assert config.min_words == 5

    @pytest.mark.unit
    def test_chunk_config_min_words_zero_allowed(self) -> None:
        """min_words=0 should be allowed to disable filtering."""
        config = ChunkConfig(min_words=0)
        assert config.min_words == 0


# =============================================================================
# FILTER STATS TESTS
# =============================================================================


class TestFilterStats:
    """Tests for FilterStats dataclass."""

    @pytest.mark.unit
    def test_filter_stats_dataclass_exists(self) -> None:
        """FilterStats dataclass should exist with required fields."""
        stats = FilterStats(
            total_chunks=10,
            micro_chunks_filtered=2,
            consecutive_duplicates_removed=1,
            chunks_written=7,
        )
        assert stats.total_chunks == 10
        assert stats.micro_chunks_filtered == 2
        assert stats.consecutive_duplicates_removed == 1
        assert stats.chunks_written == 7


# =============================================================================
# MICRO-CHUNK FILTERING TESTS
# =============================================================================


def _make_chunk(
    text: str,
    chunk_index: int,
    word_count: int | None = None,
) -> Chunk:
    """Helper to create a Chunk for testing."""
    if word_count is None:
        word_count = len(text.split())
    return Chunk(
        text=text,
        chunk_index=chunk_index,
        section=None,
        line_start=chunk_index + 1,
        line_end=chunk_index + 1,
        word_count=word_count,
        estimated_tokens=int(word_count * 1.3),
    )


def _make_article(chunks: list[Chunk]) -> ChunkedArticle:
    """Helper to create a ChunkedArticle for testing."""
    return ChunkedArticle(
        article_title="Test Article",
        namespace="Main",
        chunks=chunks,
        categories=["Test"],
        internal_links=[],
        infobox=None,
        library_work=None,
        is_stub=False,
        citation_needed_count=0,
        has_blockquote=False,
    )


class TestMicroChunkFiltering:
    """Tests for filtering chunks below min_words threshold."""

    @pytest.mark.unit
    def test_filters_chunks_below_min_words(self, tmp_path: Path) -> None:
        """Chunks with word_count < min_words should be excluded from output."""
        chunks = [
            _make_chunk(
                "This is a good chunk with enough words to pass the filter.",
                chunk_index=0,
            ),
            _make_chunk(
                "Tiny.",  # Only 1 word
                chunk_index=1,
            ),
            _make_chunk(
                "Another valid chunk with sufficient content for embedding.",
                chunk_index=2,
            ),
        ]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=5)
        output_path = tmp_path / "filtered.jsonl"

        stats = write_chunks_jsonl(article, output_path, config)

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2  # Tiny chunk filtered out
        assert stats.micro_chunks_filtered == 1

    @pytest.mark.unit
    def test_min_words_zero_preserves_all_chunks(self, tmp_path: Path) -> None:
        """When min_words=0, all chunks should be preserved."""
        chunks = [
            _make_chunk("One.", chunk_index=0),
            _make_chunk("Two.", chunk_index=1),
        ]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=0)
        output_path = tmp_path / "unfiltered.jsonl"

        stats = write_chunks_jsonl(article, output_path, config)

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert stats.micro_chunks_filtered == 0

    @pytest.mark.unit
    def test_filters_empty_text_chunks(self, tmp_path: Path) -> None:
        """Chunks with empty or whitespace-only text should be filtered."""
        chunks = [
            _make_chunk(
                "Valid content here with enough words for embedding.",
                chunk_index=0,
            ),
            _make_chunk("   ", chunk_index=1, word_count=0),  # Whitespace only
            _make_chunk("", chunk_index=2, word_count=0),  # Empty
        ]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=1)
        output_path = tmp_path / "output.jsonl"

        stats = write_chunks_jsonl(article, output_path, config)

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert stats.micro_chunks_filtered == 2


# =============================================================================
# CONSECUTIVE DEDUPLICATION TESTS
# =============================================================================


class TestConsecutiveDeduplication:
    """Tests for removing consecutive identical chunks."""

    @pytest.mark.unit
    def test_removes_consecutive_duplicates(self, tmp_path: Path) -> None:
        """Consecutive chunks with identical text should be deduplicated."""
        chunks = [
            _make_chunk(
                "Unique first chunk with plenty of words for embedding.",
                chunk_index=0,
            ),
            _make_chunk(
                "This is repeated content that appears multiple times.",
                chunk_index=1,
            ),
            _make_chunk(
                "This is repeated content that appears multiple times.",
                chunk_index=2,
            ),
            _make_chunk(
                "Unique last chunk with plenty of words for embedding.",
                chunk_index=3,
            ),
        ]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=1)
        output_path = tmp_path / "deduped.jsonl"

        stats = write_chunks_jsonl(article, output_path, config)

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3  # One duplicate removed
        assert stats.consecutive_duplicates_removed == 1

    @pytest.mark.unit
    def test_preserves_non_consecutive_duplicates(self, tmp_path: Path) -> None:
        """Non-consecutive duplicates should be preserved (intentional overlap)."""
        chunks = [
            _make_chunk(
                "Same content that appears again later in the article.",
                chunk_index=0,
            ),
            _make_chunk(
                "Different middle chunk with completely different content.",
                chunk_index=1,
            ),
            _make_chunk(
                "Same content that appears again later in the article.",
                chunk_index=2,
            ),
        ]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=1)
        output_path = tmp_path / "output.jsonl"

        stats = write_chunks_jsonl(article, output_path, config)

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3  # All preserved
        assert stats.consecutive_duplicates_removed == 0

    @pytest.mark.unit
    def test_removes_multiple_consecutive_duplicates(self, tmp_path: Path) -> None:
        """Multiple consecutive identical chunks should all be deduplicated to one."""
        text = "Same exact text repeated many times in a row for testing."
        chunks = [_make_chunk(text, chunk_index=i) for i in range(5)]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=1)
        output_path = tmp_path / "output.jsonl"

        stats = write_chunks_jsonl(article, output_path, config)

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 1  # All deduplicated to one
        assert stats.consecutive_duplicates_removed == 4


# =============================================================================
# FILTER STATS TRACKING TESTS
# =============================================================================


class TestFilterStatsTracking:
    """Tests for accurate filtering statistics."""

    @pytest.mark.unit
    def test_write_chunks_jsonl_returns_stats(self, tmp_path: Path) -> None:
        """write_chunks_jsonl should return FilterStats."""
        chunks = [
            _make_chunk(
                "Valid chunk with sufficient word count for embedding.",
                chunk_index=0,
            ),
        ]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=1)
        output_path = tmp_path / "output.jsonl"

        stats = write_chunks_jsonl(article, output_path, config)

        assert isinstance(stats, FilterStats)
        assert stats.total_chunks == 1
        assert stats.chunks_written == 1

    @pytest.mark.unit
    def test_stats_tracks_micro_chunks(self, tmp_path: Path) -> None:
        """FilterStats should accurately count filtered micro-chunks."""
        chunks = [
            _make_chunk(
                "Valid chunk with plenty of words to pass the filter threshold.",
                chunk_index=0,
            ),
            _make_chunk("Tiny.", chunk_index=1),
            _make_chunk("Also tiny.", chunk_index=2),
        ]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=10)
        output_path = tmp_path / "output.jsonl"

        stats = write_chunks_jsonl(article, output_path, config)

        assert stats.micro_chunks_filtered == 2

    @pytest.mark.unit
    def test_stats_tracks_duplicates(self, tmp_path: Path) -> None:
        """FilterStats should accurately count removed duplicates."""
        text = "Same text for all chunks to test deduplication."
        chunks = [_make_chunk(text, chunk_index=i) for i in range(3)]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=1)
        output_path = tmp_path / "output.jsonl"

        stats = write_chunks_jsonl(article, output_path, config)

        assert stats.consecutive_duplicates_removed == 2

    @pytest.mark.unit
    def test_stats_combined_filtering(self, tmp_path: Path) -> None:
        """Stats should track both micro-chunks and duplicates together."""
        chunks = [
            _make_chunk(
                "Valid first chunk with enough words for embedding quality.",
                chunk_index=0,
            ),
            _make_chunk("Tiny.", chunk_index=1),  # Micro-chunk
            _make_chunk(
                "Duplicate content that appears back to back multiple times.",
                chunk_index=2,
            ),
            _make_chunk(
                "Duplicate content that appears back to back multiple times.",
                chunk_index=3,
            ),  # Consecutive dup
        ]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=5)
        output_path = tmp_path / "output.jsonl"

        stats = write_chunks_jsonl(article, output_path, config)

        assert stats.total_chunks == 4
        assert stats.micro_chunks_filtered == 1
        assert stats.consecutive_duplicates_removed == 1
        assert stats.chunks_written == 2


# =============================================================================
# CHUNK INDEX REASSIGNMENT TESTS
# =============================================================================


class TestChunkIndexReassignment:
    """Tests for chunk index continuity after filtering."""

    @pytest.mark.unit
    def test_chunk_indices_remain_sequential(self, tmp_path: Path) -> None:
        """After filtering, chunk_index values in output should be sequential."""
        chunks = [
            _make_chunk(
                "First valid chunk with sufficient words for embedding.",
                chunk_index=0,
            ),
            _make_chunk("Tiny.", chunk_index=1),  # Will be filtered
            _make_chunk(
                "Second valid chunk with sufficient words for embedding.",
                chunk_index=2,
            ),
        ]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=5)
        output_path = tmp_path / "output.jsonl"

        write_chunks_jsonl(article, output_path, config)

        lines = output_path.read_text().strip().split("\n")
        records = [json.loads(line) for line in lines]
        indices = [r["chunk_index"] for r in records]
        assert indices == [0, 1]  # Sequential after filtering

    @pytest.mark.unit
    def test_chunk_ids_use_reassigned_indices(self, tmp_path: Path) -> None:
        """chunk_id should use reassigned indices, not original."""
        chunks = [
            _make_chunk(
                "First valid chunk with sufficient words for embedding.",
                chunk_index=0,
            ),
            _make_chunk("Tiny.", chunk_index=1),  # Will be filtered
            _make_chunk(
                "Second valid chunk with sufficient words for embedding.",
                chunk_index=2,
            ),
        ]
        article = _make_article(chunks)
        config = ChunkConfig(min_words=5)
        output_path = tmp_path / "output.jsonl"

        write_chunks_jsonl(article, output_path, config)

        lines = output_path.read_text().strip().split("\n")
        records = [json.loads(line) for line in lines]

        # chunk_id should end with #0 and #1, not #0 and #2
        assert records[0]["chunk_id"].endswith("#0")
        assert records[1]["chunk_id"].endswith("#1")


# =============================================================================
# LOGGING TESTS
# =============================================================================


class TestFilterLogging:
    """Tests for logging on high filter rates."""

    @pytest.mark.unit
    def test_high_filter_rate_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log warning when >10% of chunks are filtered."""
        # Create 10 chunks where 2 are micro-chunks (20% filter rate)
        chunks = []
        for i in range(8):
            chunks.append(
                _make_chunk(
                    f"Valid chunk number {i} with plenty of words for embedding.",
                    chunk_index=i,
                )
            )
        chunks.append(_make_chunk("Tiny.", chunk_index=8))
        chunks.append(_make_chunk("Also tiny.", chunk_index=9))

        article = _make_article(chunks)
        config = ChunkConfig(min_words=5)
        output_path = tmp_path / "output.jsonl"

        with caplog.at_level(logging.WARNING):
            write_chunks_jsonl(article, output_path, config)

        # Should have logged a warning about high filter rate
        assert any("filter" in record.message.lower() for record in caplog.records)

    @pytest.mark.unit
    def test_low_filter_rate_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should not log warning when <10% of chunks are filtered."""
        # Create 20 chunks where 1 is micro-chunk (5% filter rate)
        chunks = []
        for i in range(19):
            chunks.append(
                _make_chunk(
                    f"Valid chunk number {i} with plenty of words for embedding.",
                    chunk_index=i,
                )
            )
        chunks.append(_make_chunk("Tiny.", chunk_index=19))

        article = _make_article(chunks)
        config = ChunkConfig(min_words=5)
        output_path = tmp_path / "output.jsonl"

        with caplog.at_level(logging.WARNING):
            write_chunks_jsonl(article, output_path, config)

        # Should NOT have logged a warning
        filter_warnings = [r for r in caplog.records if "filter" in r.message.lower()]
        assert len(filter_warnings) == 0
