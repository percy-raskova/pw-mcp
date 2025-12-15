"""Integration tests for sembr processing (TDD Red Phase).

These tests verify the full integration between components:
- File I/O operations
- Batch processing with actual files
- Progress callback invocation
- JSON extraction from Phase 2 output

Note: These tests mock the HTTP layer but use real file operations.
For tests with a real sembr server, see tests/slow/test_sembr_slow.py
"""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pw_mcp.ingest.linebreaker import (
    SembrConfig,
    SembrResult,
    _extract_clean_text_from_json,
    process_batch,
    process_file,
    process_text,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestBatchProcessingIntegration:
    """Integration tests for batch processing."""

    @pytest.mark.integration
    async def test_process_batch_small_sample(self, sembr_input_dir: Path, tmp_path: Path) -> None:
        """Should process small batch of files end-to-end."""
        output_dir = tmp_path / "output"

        with patch("pw_mcp.ingest.linebreaker.check_server_health", return_value=True):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value.__aenter__.return_value = mock_client
                mock_client.post.return_value = MagicMock(
                    status_code=200,
                    json=lambda: {
                        "status": "success",
                        "text": "Processed\ntext.",
                    },
                )

                results = await process_batch(sembr_input_dir, output_dir)

        # Should have processed all fixture files
        assert len(results) >= 1

        # Output files should exist
        output_files = list(output_dir.rglob("*.txt"))
        assert len(output_files) >= 1

    @pytest.mark.integration
    async def test_progress_callback_invoked(self, sembr_input_dir: Path, tmp_path: Path) -> None:
        """Should invoke progress callback with correct parameters."""
        output_dir = tmp_path / "output"
        progress_calls: list[tuple[int, int, str]] = []

        def progress_callback(current: int, total: int, filename: str) -> None:
            progress_calls.append((current, total, filename))

        with patch("pw_mcp.ingest.linebreaker.check_server_health", return_value=True):
            with patch("pw_mcp.ingest.linebreaker.process_file") as mock_file:
                mock_file.return_value = SembrResult(
                    text="Output",
                    line_count=1,
                    processing_time_ms=10.0,
                    input_word_count=1,
                    output_word_count=1,
                )

                await process_batch(
                    sembr_input_dir,
                    output_dir,
                    progress_callback=progress_callback,
                )

        # Progress should have been called for each file
        assert len(progress_calls) >= 1

        # Each call should have (current, total, filename)
        for current, total, filename in progress_calls:
            assert 1 <= current <= total
            assert isinstance(filename, str)
            assert len(filename) > 0

    @pytest.mark.integration
    async def test_output_directory_structure_preserved(self, tmp_path: Path) -> None:
        """Should preserve namespace directory structure in output."""
        # Create input with nested namespace structure
        input_dir = tmp_path / "extracted"
        (input_dir / "Main").mkdir(parents=True)
        (input_dir / "Library" / "Marxist").mkdir(parents=True)
        (input_dir / "Essays").mkdir(parents=True)

        (input_dir / "Main" / "article1.txt").write_text("Main article content.")
        (input_dir / "Library" / "Marxist" / "capital.txt").write_text("Capital content.")
        (input_dir / "Essays" / "essay1.txt").write_text("Essay content.")

        output_dir = tmp_path / "sembr"

        with patch("pw_mcp.ingest.linebreaker.check_server_health", return_value=True):
            with patch("pw_mcp.ingest.linebreaker.process_text") as mock_process:
                mock_process.return_value = SembrResult(
                    text="Processed.",
                    line_count=1,
                    processing_time_ms=10.0,
                    input_word_count=1,
                    output_word_count=1,
                )

                await process_batch(input_dir, output_dir)

        # Verify directory structure
        assert (output_dir / "Main" / "article1.txt").exists()
        assert (output_dir / "Library" / "Marxist" / "capital.txt").exists()
        assert (output_dir / "Essays" / "essay1.txt").exists()


class TestJsonExtractionIntegration:
    """Tests for extracting clean_text from Phase 2 JSON output."""

    @pytest.mark.integration
    async def test_json_clean_text_extraction(self, tmp_path: Path) -> None:
        """Should extract clean_text field from extracted article JSON."""
        import json

        input_dir = tmp_path / "extracted"
        input_dir.mkdir()

        json_content = {
            "title": "Test Article",
            "namespace": "Main",
            "clean_text": "The Soviet Union was a socialist state.",
            "categories": ["Soviet Union"],
            "internal_links": ["socialism"],
        }

        json_file = input_dir / "Main" / "Test Article.json"
        json_file.parent.mkdir(parents=True)
        json_file.write_text(json.dumps(json_content))

        extracted = _extract_clean_text_from_json(json_file)
        assert extracted == "The Soviet Union was a socialist state."

    @pytest.mark.integration
    async def test_process_batch_with_json_input(self, tmp_path: Path) -> None:
        """Should process JSON files by extracting clean_text field."""
        import json

        input_dir = tmp_path / "extracted"
        (input_dir / "Main").mkdir(parents=True)

        json_content = {
            "title": "Test",
            "clean_text": "Content to process.",
        }
        (input_dir / "Main" / "test.json").write_text(json.dumps(json_content))

        output_dir = tmp_path / "sembr"

        with patch("pw_mcp.ingest.linebreaker.check_server_health", return_value=True):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value.__aenter__.return_value = mock_client
                mock_client.post.return_value = MagicMock(
                    status_code=200,
                    json=lambda: {
                        "status": "success",
                        "text": "Content\nto process.",
                    },
                )

                results = await process_batch(input_dir, output_dir)

        assert len(results) == 1
        # Output should be .txt, not .json
        assert (output_dir / "Main" / "test.txt").exists()


class TestConcurrencyIntegration:
    """Tests for concurrent processing behavior."""

    @pytest.mark.integration
    async def test_concurrent_processing(self, tmp_path: Path) -> None:
        """Should process multiple files concurrently."""
        import asyncio
        import time

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create 10 files
        for i in range(10):
            (input_dir / f"file_{i}.txt").write_text(f"Content {i}")

        output_dir = tmp_path / "output"

        async def slow_process_text(text: str, config: SembrConfig | None = None) -> SembrResult:
            await asyncio.sleep(0.1)  # 100ms per file
            return SembrResult(
                text=text,
                line_count=1,
                processing_time_ms=100.0,
                input_word_count=len(text.split()),
                output_word_count=len(text.split()),
            )

        start_time = time.time()

        with patch("pw_mcp.ingest.linebreaker.check_server_health", return_value=True):
            with patch(
                "pw_mcp.ingest.linebreaker.process_text",
                side_effect=slow_process_text,
            ):
                await process_batch(input_dir, output_dir, max_concurrent=5)

        elapsed = time.time() - start_time

        # With 10 files, 100ms each, max_concurrent=5:
        # Sequential: 10 * 0.1 = 1.0s
        # Concurrent (5 at a time): ~2 batches * 0.1 = 0.2s
        # Allow some overhead
        assert elapsed < 0.8, f"Expected concurrent execution, took {elapsed:.2f}s"

    @pytest.mark.integration
    async def test_error_in_one_file_continues_batch(self, tmp_path: Path) -> None:
        """Should continue processing other files if one fails."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        (input_dir / "good1.txt").write_text("Good content one.")
        (input_dir / "bad.txt").write_text("This will fail.")
        (input_dir / "good2.txt").write_text("Good content two.")

        output_dir = tmp_path / "output"

        call_count = 0

        async def selective_fail(text: str, config: SembrConfig | None = None) -> SembrResult:
            nonlocal call_count
            call_count += 1
            if "will fail" in text:
                raise ValueError("Simulated failure")
            return SembrResult(
                text=text,
                line_count=1,
                processing_time_ms=10.0,
                input_word_count=len(text.split()),
                output_word_count=len(text.split()),
            )

        with patch("pw_mcp.ingest.linebreaker.check_server_health", return_value=True):
            with patch(
                "pw_mcp.ingest.linebreaker.process_text",
                side_effect=selective_fail,
            ):
                # Should not raise, but return partial results
                results = await process_batch(input_dir, output_dir)

        # Should have processed all 3 files (with one error)
        assert call_count == 3
        # Results should include successes
        assert len([r for r in results if r is not None]) >= 2


class TestContentPreservationIntegration:
    """Integration tests for content preservation invariant."""

    @pytest.mark.integration
    async def test_word_count_invariant_english(
        self, load_sembr_fixture: "Callable[[str, str], str]", tmp_path: Path
    ) -> None:
        """Should preserve word count for English text."""
        input_text = load_sembr_fixture("input", "simple_english.txt")
        expected_output = load_sembr_fixture("expected", "simple_english.txt")

        input_file = tmp_path / "input.txt"
        input_file.write_text(input_text)
        output_file = tmp_path / "output.txt"

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"status": "success", "text": expected_output},
            )

            result = await process_file(input_file, output_file)

        # Critical invariant
        assert result.input_word_count == result.output_word_count

    @pytest.mark.integration
    async def test_word_set_invariant(self) -> None:
        """Should preserve exact set of words (order-independent)."""
        input_text = "One two three four five six seven eight nine ten."

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {
                    "status": "success",
                    "text": "One two three\nfour five six\nseven eight\nnine ten.",
                },
            )

            result = await process_text(input_text)

        input_words = set(input_text.replace(".", "").split())
        output_words = set(result.text.replace(".", "").split())

        assert input_words == output_words
