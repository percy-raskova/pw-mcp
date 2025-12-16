"""Slow tests requiring real sembr server (TDD Red Phase).

These tests are marked as @pytest.mark.slow and are skipped by default.
Run with: pytest -m slow

Prerequisites:
- sembr server running: mise run sembr-server
- Model loaded: admko/sembr2023-distilbert-base-multilingual-cased
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from pw_mcp.ingest.linebreaker import (
    SembrConfig,
    check_server_health,
    process_batch,
    process_text,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def require_sembr_server() -> None:
    """Skip test if sembr server is not running."""
    if not check_server_health():
        pytest.skip("sembr server not running - start with: mise run sembr-server")


class TestRealServerProcessing:
    """Tests against real sembr server."""

    @pytest.mark.slow
    async def test_real_server_processing(self, require_sembr_server: None) -> None:
        """Should process text through real sembr server."""
        input_text = (
            "Stalin implemented the Five-Year Plans. "
            "These transformed the Soviet Union from an agrarian society "
            "into an industrial power."
        )

        result = await process_text(input_text)

        # Should have line breaks
        assert "\n" in result.text

        # Word count should be preserved
        assert result.input_word_count == result.output_word_count

        # Processing time should be recorded
        assert result.processing_time_ms > 0

    @pytest.mark.slow
    async def test_multilingual_content_russian(
        self,
        require_sembr_server: None,
        load_sembr_fixture: "Callable[[str, str], str]",
    ) -> None:
        """Should process Russian text correctly."""
        russian_text = load_sembr_fixture("input", "russian_text.txt")

        result = await process_text(russian_text)

        # Russian characters should be preserved
        assert any(ord(c) > 127 for c in result.text)  # Has non-ASCII

        # Word count preserved (Russian tokenization)
        assert result.input_word_count == result.output_word_count

    @pytest.mark.slow
    async def test_multilingual_content_chinese(
        self,
        require_sembr_server: None,
        load_sembr_fixture: "Callable[[str, str], str]",
    ) -> None:
        """Should process Chinese text correctly."""
        chinese_text = load_sembr_fixture("input", "chinese_text.txt")

        result = await process_text(chinese_text)

        # Chinese characters should be preserved
        assert any("\u4e00" <= c <= "\u9fff" for c in result.text)

        # Should have some line breaks
        assert result.line_count >= 1

    @pytest.mark.slow
    async def test_large_library_document(
        self,
        require_sembr_server: None,
        load_sembr_fixture: "Callable[[str, str], str]",
    ) -> None:
        """Should handle large Library namespace documents."""
        long_text = load_sembr_fixture("input", "long_paragraph.txt")

        config = SembrConfig(timeout_seconds=120.0)  # Longer timeout for large docs

        result = await process_text(long_text, config)

        # Should produce multiple lines
        assert result.line_count > 5

        # Content should be preserved
        assert result.input_word_count == result.output_word_count

    @pytest.mark.slow
    async def test_full_corpus_sample_100(self, require_sembr_server: None, tmp_path: Path) -> None:
        """Should process 100 files from corpus sample."""
        corpus_path = Path("prolewiki-exports/Main")

        if not corpus_path.exists():
            pytest.skip("Corpus not available")

        # Get first 100 txt files
        txt_files = sorted(corpus_path.glob("*.txt"))[:100]

        if len(txt_files) < 100:
            pytest.skip(f"Only {len(txt_files)} files available")

        # Create temp input dir with sample files
        input_dir = tmp_path / "sample"
        input_dir.mkdir()

        for f in txt_files:
            (input_dir / f.name).write_text(f.read_text())

        output_dir = tmp_path / "output"

        # Use max_concurrent=1 because sembr server is single-threaded
        # Concurrent requests cause "Already borrowed" errors
        results = await process_batch(input_dir, output_dir, max_concurrent=1)

        # Should process all 100 (None values indicate failures)
        assert len(results) == 100

        # All successful results should preserve word count
        successful = [r for r in results if r is not None]
        assert len(successful) == 100, f"Expected 100 successes, got {len(successful)}"
        for result in successful:
            assert result.input_word_count == result.output_word_count
