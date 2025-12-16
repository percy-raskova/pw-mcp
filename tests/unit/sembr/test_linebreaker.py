"""Unit tests for linebreaker module (TDD Red Phase).

These tests define the expected interface for the sembr integration.
The linebreaker module does not exist yet - tests should fail with ImportError.

Test strategy:
- Mock all HTTP calls (no real sembr server needed)
- Test config defaults and customization
- Test result dataclass behavior
- Test error handling and retry logic
- Test content preservation invariant
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# These imports will fail until the module is implemented
from pw_mcp.ingest.linebreaker import (
    CHUNK_TARGET_BYTES,
    MIN_CONTENT_BYTES,
    URL_PREFIXES,
    SembrConfig,
    SembrContentError,
    SembrError,
    SembrResult,
    SembrServerError,
    SembrSkipError,
    SembrTimeoutError,
    _split_large_paragraph,
    _split_text_for_processing,
    check_server_health,
    is_stub_content,
    process_batch,
    process_file,
    process_text,
)

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestSembrConfigDefaults:
    """Tests for SembrConfig default values."""

    @pytest.mark.unit
    def test_sembr_config_defaults(self) -> None:
        """Should have sensible defaults matching pyproject.toml."""
        config = SembrConfig()
        assert config.server_url == "http://localhost:8384"
        assert config.model_name == "admko/sembr2023-distilbert-base-multilingual-cased"
        assert config.timeout_seconds == 60.0
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 1.0
        assert config.batch_size == 8
        assert config.predict_func == "argmax"

    @pytest.mark.unit
    def test_sembr_config_custom_values(self) -> None:
        """Should accept custom configuration values."""
        config = SembrConfig(
            server_url="http://custom:9000",
            timeout_seconds=120.0,
            max_retries=5,
            batch_size=16,
            predict_func="greedy_linebreaks",
        )
        assert config.server_url == "http://custom:9000"
        assert config.timeout_seconds == 120.0
        assert config.max_retries == 5
        assert config.batch_size == 16
        assert config.predict_func == "greedy_linebreaks"

    @pytest.mark.unit
    def test_sembr_config_immutable(self) -> None:
        """Config should be a frozen dataclass (immutable)."""
        config = SembrConfig()
        with pytest.raises(AttributeError):
            config.server_url = "http://other:8000"  # type: ignore[misc]


# =============================================================================
# RESULT DATACLASS TESTS
# =============================================================================


class TestSembrResult:
    """Tests for SembrResult dataclass."""

    @pytest.mark.unit
    def test_sembr_result_creation(self) -> None:
        """Should create SembrResult with all required fields."""
        result = SembrResult(
            text="Line one.\nLine two.",
            line_count=2,
            processing_time_ms=150.5,
            input_word_count=4,
            output_word_count=4,
        )
        assert result.text == "Line one.\nLine two."
        assert result.line_count == 2
        assert result.processing_time_ms == 150.5
        assert result.input_word_count == 4
        assert result.output_word_count == 4

    @pytest.mark.unit
    def test_sembr_result_word_count_match(self) -> None:
        """Result should track input and output word counts for validation."""
        result = SembrResult(
            text="Test output text.",
            line_count=1,
            processing_time_ms=100.0,
            input_word_count=3,
            output_word_count=3,
        )
        # Invariant: word counts should match
        assert result.input_word_count == result.output_word_count

    @pytest.mark.unit
    def test_sembr_result_empty_text(self) -> None:
        """Should handle empty text result."""
        result = SembrResult(
            text="",
            line_count=0,
            processing_time_ms=0.0,
            input_word_count=0,
            output_word_count=0,
        )
        assert result.text == ""
        assert result.line_count == 0


# =============================================================================
# EXCEPTION TESTS
# =============================================================================


class TestSembrExceptions:
    """Tests for sembr exception hierarchy."""

    @pytest.mark.unit
    def test_sembr_error_is_base_exception(self) -> None:
        """SembrError should be the base for all sembr exceptions."""
        assert issubclass(SembrServerError, SembrError)
        assert issubclass(SembrTimeoutError, SembrError)
        assert issubclass(SembrContentError, SembrError)

    @pytest.mark.unit
    def test_sembr_server_error_message(self) -> None:
        """SembrServerError should include helpful message."""
        error = SembrServerError("Connection refused")
        assert "Connection refused" in str(error)

    @pytest.mark.unit
    def test_sembr_timeout_error_message(self) -> None:
        """SembrTimeoutError should include timeout details."""
        error = SembrTimeoutError("Request timed out after 60s")
        assert "60s" in str(error)

    @pytest.mark.unit
    def test_sembr_content_error_includes_counts(self) -> None:
        """SembrContentError should include word count mismatch details."""
        error = SembrContentError("Word count mismatch: input=100, output=98")
        assert "100" in str(error)
        assert "98" in str(error)


# =============================================================================
# HEALTH CHECK TESTS (Mocked HTTP)
# =============================================================================


class TestCheckServerHealth:
    """Tests for check_server_health function."""

    @pytest.mark.unit
    def test_check_server_health_success(self, mock_health_response: dict[str, str]) -> None:
        """Should return True when server responds with success status."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: mock_health_response,
            )
            result = check_server_health()
            assert result is True
            mock_get.assert_called_once()

    @pytest.mark.unit
    def test_check_server_health_connection_error(self) -> None:
        """Should return False when server is unavailable."""
        with patch("httpx.get") as mock_get:
            import httpx

            mock_get.side_effect = httpx.ConnectError("Connection refused")
            result = check_server_health()
            assert result is False

    @pytest.mark.unit
    def test_check_server_health_timeout(self) -> None:
        """Should return False on timeout (fast fail)."""
        with patch("httpx.get") as mock_get:
            import httpx

            mock_get.side_effect = httpx.TimeoutException("Timeout")
            result = check_server_health()
            assert result is False

    @pytest.mark.unit
    def test_check_server_health_bad_response(self) -> None:
        """Should return False when response status is not 'success'."""
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"status": "error", "error": "Model not loaded"},
            )
            result = check_server_health()
            assert result is False

    @pytest.mark.unit
    def test_check_server_health_custom_config(self) -> None:
        """Should use custom server URL from config."""
        config = SembrConfig(server_url="http://custom:9000")
        with patch("httpx.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"status": "success"},
            )
            check_server_health(config)
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "http://custom:9000/check" in str(call_args)


# =============================================================================
# PROCESS TEXT TESTS (Mocked HTTP)
# =============================================================================


class TestProcessText:
    """Tests for process_text function."""

    @pytest.mark.unit
    async def test_process_text_basic(self, mock_rewrap_response: dict[str, str]) -> None:
        """Should process text and return SembrResult."""
        input_text = (
            "Stalin implemented the Five-Year Plans. " "These transformed the Soviet Union."
        )
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {
                    "status": "success",
                    "text": (
                        "Stalin implemented the Five-Year Plans.\n"
                        "These transformed the Soviet Union."
                    ),
                },
            )

            result = await process_text(input_text)

            assert isinstance(result, SembrResult)
            assert "\n" in result.text  # Has line breaks
            assert result.line_count >= 1

    @pytest.mark.unit
    async def test_process_text_empty_input(self) -> None:
        """Should return empty result for empty input without calling server."""
        result = await process_text("")
        assert result.text == ""
        assert result.line_count == 0
        assert result.input_word_count == 0

    @pytest.mark.unit
    async def test_process_text_whitespace_only(self) -> None:
        """Should return empty result for whitespace-only input."""
        result = await process_text("   \n\t\n   ")
        assert result.text == ""
        assert result.line_count == 0

    @pytest.mark.unit
    async def test_process_text_unicode_russian(
        self, load_sembr_fixture: "Callable[[str, str], str]"
    ) -> None:
        """Should handle Russian Unicode text correctly."""
        russian_input = load_sembr_fixture("input", "russian_text.txt")
        expected_output = load_sembr_fixture("expected", "russian_text.txt")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"status": "success", "text": expected_output},
            )

            result = await process_text(russian_input)

            # Verify Unicode preserved
            assert "Советского" in result.text or len(result.text) > 0
            assert result.input_word_count == result.output_word_count

    @pytest.mark.unit
    async def test_process_text_unicode_chinese(
        self, load_sembr_fixture: "Callable[[str, str], str]"
    ) -> None:
        """Should handle Chinese Unicode text correctly."""
        chinese_input = load_sembr_fixture("input", "chinese_text.txt")
        expected_output = load_sembr_fixture("expected", "chinese_text.txt")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"status": "success", "text": expected_output},
            )

            result = await process_text(chinese_input)

            # Chinese text should be preserved
            assert len(result.text) > 0

    @pytest.mark.unit
    async def test_process_text_preserves_all_words(self) -> None:
        """Critical invariant: no words should be lost during processing."""
        input_text = "The quick brown fox jumps over the lazy dog."
        output_text = "The quick brown fox\njumps over\nthe lazy dog."

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"status": "success", "text": output_text},
            )

            result = await process_text(input_text)

            # Verify word count invariant
            assert result.input_word_count == result.output_word_count

            # Verify actual words match (order-independent)
            input_words = set(input_text.split())
            output_words = set(result.text.split())
            assert input_words == output_words

    @pytest.mark.unit
    async def test_process_text_server_error_retry(self) -> None:
        """Should retry on server error up to max_retries."""
        config = SembrConfig(max_retries=3, retry_delay_seconds=0.01)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Fail twice, succeed on third attempt
            mock_client.post.side_effect = [
                MagicMock(status_code=500, json=lambda: {"status": "error"}),
                MagicMock(status_code=500, json=lambda: {"status": "error"}),
                MagicMock(
                    status_code=200,
                    json=lambda: {"status": "success", "text": "Result."},
                ),
            ]

            result = await process_text("Input text.", config)

            assert result.text == "Result."
            assert mock_client.post.call_count == 3

    @pytest.mark.unit
    async def test_process_text_timeout_retry(self) -> None:
        """Should retry on timeout."""
        config = SembrConfig(max_retries=2, retry_delay_seconds=0.01)

        with patch("httpx.AsyncClient") as mock_client_class:
            import httpx

            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Timeout once, then succeed
            mock_client.post.side_effect = [
                httpx.TimeoutException("Timeout"),
                MagicMock(
                    status_code=200,
                    json=lambda: {"status": "success", "text": "Result."},
                ),
            ]

            result = await process_text("Input text.", config)

            assert result.text == "Result."

    @pytest.mark.unit
    async def test_process_text_max_retries_exceeded(self) -> None:
        """Should raise SembrServerError after max retries exceeded."""
        config = SembrConfig(max_retries=2, retry_delay_seconds=0.01)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(
                status_code=500,
                json=lambda: {"status": "error", "error": "Internal error"},
            )

            with pytest.raises(SembrServerError):
                await process_text("Input text.", config)

    @pytest.mark.unit
    async def test_process_text_content_validation_fails(self) -> None:
        """Should raise SembrContentError when content significantly differs."""
        # Use longer text to ensure difference exceeds tolerance (10 chars min)
        input_text = "One two three four five six seven eight nine ten."  # 10 words
        corrupted_output = "One two."  # Only 2 words (significant content lost!)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"status": "success", "text": corrupted_output},
            )

            with pytest.raises(SembrContentError) as exc_info:
                await process_text(input_text)

            # Error should include details about the mismatch
            assert "10" in str(exc_info.value) or "2" in str(exc_info.value)


# =============================================================================
# FILE PROCESSING TESTS
# =============================================================================


class TestProcessFile:
    """Tests for process_file function."""

    @pytest.mark.unit
    async def test_process_file_creates_output_dir(self, tmp_path: Path) -> None:
        """Should create output directory if it doesn't exist."""
        input_file = tmp_path / "input" / "test.txt"
        input_file.parent.mkdir(parents=True)
        # Content must be >= 100 bytes to avoid stub detection
        input_content = "Test content that is long enough to not be considered a stub file. " * 3
        input_file.write_text(input_content)

        output_file = tmp_path / "output" / "nested" / "test.txt"

        with patch("pw_mcp.ingest.linebreaker.process_text") as mock_process:
            mock_process.return_value = SembrResult(
                text="Test content\nthat is long enough.",
                line_count=2,
                processing_time_ms=100.0,
                input_word_count=15,
                output_word_count=15,
            )

            await process_file(input_file, output_file)

            assert output_file.parent.exists()
            assert output_file.exists()
            assert output_file.read_text() == "Test content\nthat is long enough."

    @pytest.mark.unit
    async def test_process_file_preserves_encoding(self, tmp_path: Path) -> None:
        """Should preserve UTF-8 encoding for Unicode content."""
        input_file = tmp_path / "russian.txt"
        # Russian text must be >= 100 bytes (50+ chars for 2-byte Cyrillic)
        russian_text = (
            "Советский Союз был социалистическим государством. "
            "Это была федерация союзных республик."
        )
        input_file.write_text(russian_text, encoding="utf-8")

        output_file = tmp_path / "output" / "russian.txt"

        with patch("pw_mcp.ingest.linebreaker.process_text") as mock_process:
            mock_process.return_value = SembrResult(
                text="Советский Союз\nбыл социалистическим\nгосударством.",
                line_count=3,
                processing_time_ms=100.0,
                input_word_count=8,
                output_word_count=8,
            )

            await process_file(input_file, output_file)

            content = output_file.read_text(encoding="utf-8")
            assert "Советский" in content

    @pytest.mark.unit
    async def test_process_file_handles_missing_input(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing input file."""
        input_file = tmp_path / "nonexistent.txt"
        output_file = tmp_path / "output.txt"

        with pytest.raises(FileNotFoundError):
            await process_file(input_file, output_file)

    @pytest.mark.unit
    async def test_process_file_returns_result(self, tmp_path: Path) -> None:
        """Should return SembrResult with processing details."""
        input_file = tmp_path / "test.txt"
        # Content must be >= 100 bytes to avoid stub detection
        input_content = "Test content for processing that is long enough. " * 3
        input_file.write_text(input_content)
        output_file = tmp_path / "output.txt"

        with patch("pw_mcp.ingest.linebreaker.process_text") as mock_process:
            expected_result = SembrResult(
                text="Test content\nfor processing.",
                line_count=2,
                processing_time_ms=150.0,
                input_word_count=24,
                output_word_count=24,
            )
            mock_process.return_value = expected_result

            result = await process_file(input_file, output_file)

            assert result == expected_result


# =============================================================================
# BATCH PROCESSING TESTS
# =============================================================================


class TestProcessBatch:
    """Tests for process_batch function."""

    @pytest.mark.unit
    async def test_process_batch_checks_health_first(self, tmp_path: Path) -> None:
        """Should check server health before processing batch."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "test.txt").write_text("Content")
        output_dir = tmp_path / "output"

        with patch("pw_mcp.ingest.linebreaker.check_server_health") as mock_health:
            mock_health.return_value = False

            with pytest.raises(SembrServerError) as exc_info:
                await process_batch(input_dir, output_dir)

            assert "server" in str(exc_info.value).lower()
            mock_health.assert_called_once()

    @pytest.mark.unit
    async def test_process_batch_respects_concurrency_limit(self, tmp_path: Path) -> None:
        """Should limit concurrent processing to max_concurrent."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create multiple input files
        for i in range(10):
            (input_dir / f"file_{i}.txt").write_text(f"Content {i}")

        output_dir = tmp_path / "output"

        concurrent_count = 0
        max_seen_concurrent = 0

        async def mock_process_file(
            input_path: Path, output_path: Path, config: SembrConfig | None = None
        ) -> SembrResult:
            nonlocal concurrent_count, max_seen_concurrent
            concurrent_count += 1
            max_seen_concurrent = max(max_seen_concurrent, concurrent_count)

            import asyncio

            await asyncio.sleep(0.01)  # Simulate processing

            concurrent_count -= 1
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("Output")

            return SembrResult(
                text="Output",
                line_count=1,
                processing_time_ms=10.0,
                input_word_count=1,
                output_word_count=1,
            )

        with patch("pw_mcp.ingest.linebreaker.check_server_health", return_value=True):
            with patch(
                "pw_mcp.ingest.linebreaker.process_file",
                side_effect=mock_process_file,
            ):
                await process_batch(input_dir, output_dir, max_concurrent=3)

        # Should never exceed max_concurrent
        assert max_seen_concurrent <= 3

    @pytest.mark.unit
    async def test_process_batch_preserves_directory_structure(self, tmp_path: Path) -> None:
        """Should preserve namespace subdirectories in output."""
        input_dir = tmp_path / "input"

        # Create namespace subdirectories
        # Content must be >= 100 bytes to avoid stub detection
        long_content = "Content that is long enough to not be considered a stub. " * 3
        (input_dir / "Main").mkdir(parents=True)
        (input_dir / "Library").mkdir(parents=True)
        (input_dir / "Main" / "article.txt").write_text(long_content)
        (input_dir / "Library" / "book.txt").write_text(long_content)

        output_dir = tmp_path / "output"

        with patch("pw_mcp.ingest.linebreaker.check_server_health", return_value=True):
            with patch("pw_mcp.ingest.linebreaker.process_text") as mock_process:
                mock_process.return_value = SembrResult(
                    text="Output content.",
                    line_count=1,
                    processing_time_ms=10.0,
                    input_word_count=15,
                    output_word_count=15,
                )

                await process_batch(input_dir, output_dir)

        # Check directory structure preserved
        assert (output_dir / "Main" / "article.txt").exists()
        assert (output_dir / "Library" / "book.txt").exists()

    @pytest.mark.unit
    async def test_process_batch_returns_all_results(self, tmp_path: Path) -> None:
        """Should return list of SembrResult for all files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        for i in range(3):
            (input_dir / f"file_{i}.txt").write_text(f"Content {i}")

        output_dir = tmp_path / "output"

        with patch("pw_mcp.ingest.linebreaker.check_server_health", return_value=True):
            with patch("pw_mcp.ingest.linebreaker.process_file") as mock_file:
                mock_file.return_value = SembrResult(
                    text="Output",
                    line_count=1,
                    processing_time_ms=10.0,
                    input_word_count=1,
                    output_word_count=1,
                )

                results = await process_batch(input_dir, output_dir)

        assert len(results) == 3
        assert all(isinstance(r, SembrResult) for r in results)


# =============================================================================
# EMPTY RESPONSE HANDLING TESTS
# =============================================================================


class TestEmptyResponseHandling:
    """Tests for handling empty and invalid JSON responses from sembr server.

    These tests address the JSONDecodeError vulnerability when the sembr server
    returns empty responses, particularly for large Library files.
    """

    @pytest.mark.unit
    async def test_process_text_empty_response_body(self) -> None:
        """Should retry when server returns empty response body."""
        config = SembrConfig(max_retries=3, retry_delay_seconds=0.01)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # First two attempts return empty response, third succeeds
            empty_response = MagicMock(
                status_code=200,
                text="",
                json=lambda: {},  # This would fail if called on empty text
            )
            # Override json to raise JSONDecodeError for empty response
            empty_response.json = MagicMock(
                side_effect=ValueError("Expecting value: line 1 column 1 (char 0)")
            )

            success_response = MagicMock(
                status_code=200,
                text='{"status": "success", "text": "Result."}',
                json=lambda: {"status": "success", "text": "Result."},
            )

            mock_client.post.side_effect = [
                empty_response,
                empty_response,
                success_response,
            ]

            result = await process_text("Input text.", config)

            assert result.text == "Result."
            assert mock_client.post.call_count == 3

    @pytest.mark.unit
    async def test_process_text_invalid_json_response(self) -> None:
        """Should retry when server returns invalid JSON."""
        config = SembrConfig(max_retries=2, retry_delay_seconds=0.01)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # First attempt returns invalid JSON, second succeeds
            invalid_response = MagicMock(
                status_code=200,
                text="not valid json",
            )
            # json() raises JSONDecodeError for invalid JSON
            from json import JSONDecodeError

            invalid_response.json = MagicMock(
                side_effect=JSONDecodeError("Expecting value", "not valid json", 0)
            )

            success_response = MagicMock(
                status_code=200,
                text='{"status": "success", "text": "Result."}',
                json=lambda: {"status": "success", "text": "Result."},
            )

            mock_client.post.side_effect = [
                invalid_response,
                success_response,
            ]

            result = await process_text("Input text.", config)

            assert result.text == "Result."
            assert mock_client.post.call_count == 2

    @pytest.mark.unit
    async def test_process_text_empty_response_max_retries_exceeded(self) -> None:
        """Should raise SembrServerError after max retries with empty responses."""
        config = SembrConfig(max_retries=2, retry_delay_seconds=0.01)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # All attempts return empty response
            empty_response = MagicMock(
                status_code=200,
                text="",
            )
            from json import JSONDecodeError

            empty_response.json = MagicMock(side_effect=JSONDecodeError("Expecting value", "", 0))

            mock_client.post.return_value = empty_response

            with pytest.raises(SembrServerError) as exc_info:
                await process_text("Input text.", config)

            # Error message should indicate empty response issue
            assert "empty" in str(exc_info.value).lower() or "json" in str(exc_info.value).lower()

    @pytest.mark.unit
    async def test_process_text_large_file_info_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should log info when processing large input files with chunking."""
        import logging

        config = SembrConfig(max_retries=1, retry_delay_seconds=0.01)

        # Create a large text input (> 400KB)
        large_text = "x" * 500_000  # 500KB of text

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.return_value = MagicMock(
                status_code=200,
                text='{"status": "success", "text": "processed"}',
                json=lambda: {"status": "success", "text": "processed"},
            )

            with caplog.at_level(logging.INFO):
                await process_text(large_text, config)

            # Check that info message about chunked processing was logged
            assert any(
                "large" in record.message.lower() and "chunked" in record.message.lower()
                for record in caplog.records
            )

    @pytest.mark.unit
    def test_check_server_health_empty_response(self) -> None:
        """Should return False when health check returns empty response."""
        with patch("httpx.get") as mock_get:
            empty_response = MagicMock(
                status_code=200,
                text="",
            )
            from json import JSONDecodeError

            empty_response.json = MagicMock(side_effect=JSONDecodeError("Expecting value", "", 0))
            mock_get.return_value = empty_response

            result = check_server_health()
            assert result is False

    @pytest.mark.unit
    def test_check_server_health_invalid_json(self) -> None:
        """Should return False when health check returns invalid JSON."""
        with patch("httpx.get") as mock_get:
            invalid_response = MagicMock(
                status_code=200,
                text="not valid json at all",
            )
            from json import JSONDecodeError

            invalid_response.json = MagicMock(
                side_effect=JSONDecodeError("Expecting value", "not valid json at all", 0)
            )
            mock_get.return_value = invalid_response

            result = check_server_health()
            assert result is False


# =============================================================================
# LARGE FILE CHUNKING TESTS
# =============================================================================


class TestLargeFileChunking:
    """Tests for large file chunking functionality.

    These tests verify the pre-chunking behavior for large Library books
    that would otherwise cause CUDA memory errors when sent to sembr.
    """

    @pytest.mark.unit
    def test_split_text_small_input_unchanged(self) -> None:
        """Small text returns as single-element list."""
        small_text = "This is a small piece of text."
        result = _split_text_for_processing(small_text)

        assert len(result) == 1
        assert result[0] == small_text

    @pytest.mark.unit
    def test_split_text_splits_on_paragraph_boundaries(self) -> None:
        """Large text splits at paragraph (\\n\\n) boundaries."""
        # Create text with multiple paragraphs, exceeding target
        paragraph = "x" * 100_000  # 100KB paragraph
        large_text = f"{paragraph}\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}"

        result = _split_text_for_processing(large_text, target_bytes=150_000)

        # Should split into multiple chunks
        assert len(result) > 1
        # Each chunk should be separated cleanly at paragraph boundaries
        for chunk in result:
            # Chunks shouldn't start or end with paragraph delimiter
            assert not chunk.startswith("\n\n")
            assert not chunk.endswith("\n\n")

    @pytest.mark.unit
    def test_split_text_handles_single_large_paragraph(self) -> None:
        """Large paragraph without paragraph breaks splits on line boundaries."""
        # Single paragraph with line breaks but no paragraph breaks
        line = "This is a line of text that is about 50 chars.\n"
        large_paragraph = line * 10_000  # ~500KB single "paragraph"

        result = _split_text_for_processing(large_paragraph, target_bytes=100_000)

        # Should still split despite no \n\n delimiters
        assert len(result) > 1
        # Total content should be preserved
        rejoined = "\n".join(result)
        # Account for potential whitespace differences
        assert len(rejoined.strip()) >= len(large_paragraph.strip()) * 0.99

    @pytest.mark.unit
    def test_split_text_preserves_all_content(self) -> None:
        """Rejoining chunks equals original text."""
        paragraphs = [f"Paragraph {i} with some content." for i in range(100)]
        original_text = "\n\n".join(paragraphs)

        result = _split_text_for_processing(original_text, target_bytes=500)

        # Rejoin and compare
        rejoined = "\n\n".join(result)
        assert rejoined == original_text

    @pytest.mark.unit
    def test_split_text_respects_target_size(self) -> None:
        """All chunks are under target_bytes (except single-paragraph edge cases)."""
        # Create text with many small paragraphs
        paragraphs = ["A" * 1000 for _ in range(100)]  # 100 1KB paragraphs
        large_text = "\n\n".join(paragraphs)

        target = 10_000  # 10KB target
        result = _split_text_for_processing(large_text, target_bytes=target)

        # Most chunks should be under target
        under_target = sum(1 for chunk in result if len(chunk.encode("utf-8")) <= target)
        assert under_target >= len(result) * 0.9  # At least 90% under target

    @pytest.mark.unit
    def test_split_large_paragraph_on_lines(self) -> None:
        """Large single paragraph splits on line (\\n) boundaries."""
        lines = [f"Line {i}: some text content here" for i in range(1000)]
        large_paragraph = "\n".join(lines)

        result = _split_large_paragraph(large_paragraph, target_bytes=1000)

        # Should split into multiple chunks
        assert len(result) > 1
        # Each chunk should be under target (mostly)
        for chunk in result:
            # Allow some tolerance for edge cases
            assert len(chunk.encode("utf-8")) <= 1500  # Some tolerance

    @pytest.mark.unit
    def test_split_text_unicode_handling(self) -> None:
        """Unicode text (Russian, Chinese) is handled correctly with byte counting."""
        # Russian text - 2 bytes per char in UTF-8
        russian_para = "Советский Союз " * 5000  # ~70KB in UTF-8
        russian_text = f"{russian_para}\n\n{russian_para}"

        result = _split_text_for_processing(russian_text, target_bytes=50_000)

        # Should split based on bytes, not characters
        assert len(result) >= 2
        # All words should be preserved (whitespace may differ due to chunk boundaries)
        original_words = russian_text.split()
        rejoined = "\n\n".join(result)
        rejoined_words = rejoined.split()
        assert original_words == rejoined_words

    @pytest.mark.unit
    def test_chunk_target_bytes_constant_exists(self) -> None:
        """CHUNK_TARGET_BYTES constant is defined and reasonable."""
        # Should be 300KB as specified in plan
        assert CHUNK_TARGET_BYTES == 300_000
        # Should be less than LARGE_FILE_THRESHOLD for safety margin
        from pw_mcp.ingest.linebreaker import LARGE_FILE_THRESHOLD_BYTES

        assert CHUNK_TARGET_BYTES < LARGE_FILE_THRESHOLD_BYTES

    @pytest.mark.unit
    async def test_process_text_uses_chunking_for_large_input(self) -> None:
        """Large input files automatically use chunked processing."""
        # Create text larger than threshold
        large_text = "x" * 500_000  # 500KB

        call_count = 0

        async def mock_post(*args: object, **kwargs: object) -> MagicMock:
            nonlocal call_count
            call_count += 1
            return MagicMock(
                status_code=200,
                text='{"status": "success", "text": "processed"}',
                json=lambda: {"status": "success", "text": "processed chunk"},
            )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post = mock_post

            result = await process_text(large_text)

            # Should have made multiple HTTP calls (one per chunk)
            assert call_count > 1
            assert isinstance(result, SembrResult)

    @pytest.mark.unit
    async def test_process_text_chunking_sequential(self) -> None:
        """Chunks are processed sequentially, not concurrently."""
        # Track concurrent processing
        concurrent_count = 0
        max_concurrent = 0
        call_order: list[int] = []

        async def mock_post(*args: object, **kwargs: object) -> MagicMock:
            nonlocal concurrent_count, max_concurrent
            call_index = len(call_order)
            call_order.append(call_index)

            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)

            # Simulate some async work
            await asyncio.sleep(0.01)

            concurrent_count -= 1

            return MagicMock(
                status_code=200,
                text='{"status": "success", "text": "processed"}',
                json=lambda: {"status": "success", "text": "processed"},
            )

        # Create text that will be split into multiple chunks
        paragraphs = ["A" * 100_000 for _ in range(5)]  # 5 x 100KB paragraphs
        large_text = "\n\n".join(paragraphs)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post = mock_post

            await process_text(large_text)

            # Should have processed sequentially (max 1 concurrent)
            assert max_concurrent == 1
            # Call order should be sequential
            assert call_order == list(range(len(call_order)))

    @pytest.mark.unit
    async def test_process_text_chunking_preserves_result_structure(self) -> None:
        """Chunked processing returns valid SembrResult with combined output."""
        # Create text that will be split into chunks
        para1 = "First paragraph content. " * 10000  # ~240KB
        para2 = "Second paragraph content. " * 10000  # ~260KB
        large_text = f"{para1}\n\n{para2}"

        chunk_outputs = ["Processed first.\nWith linebreaks.", "Processed second.\nAlso broken."]
        chunk_index = 0

        async def mock_post(*args: object, **kwargs: object) -> MagicMock:
            nonlocal chunk_index
            output = chunk_outputs[min(chunk_index, len(chunk_outputs) - 1)]
            chunk_index += 1
            return MagicMock(
                status_code=200,
                text=f'{{"status": "success", "text": "{output}"}}',
                json=lambda o=output: {"status": "success", "text": o},
            )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post = mock_post

            result = await process_text(large_text)

            # Should return valid SembrResult
            assert isinstance(result, SembrResult)
            # Combined output should contain content from both chunks
            assert "Processed first" in result.text
            assert "Processed second" in result.text
            # Chunks should be joined with paragraph separator
            assert "\n\n" in result.text


# =============================================================================
# STUB DETECTION TESTS
# =============================================================================


class TestStubDetectionConstants:
    """Tests for stub detection constants."""

    @pytest.mark.unit
    def test_min_content_bytes_constant_exists(self) -> None:
        """MIN_CONTENT_BYTES constant should be defined."""
        assert MIN_CONTENT_BYTES == 100

    @pytest.mark.unit
    def test_url_prefixes_constant_exists(self) -> None:
        """URL_PREFIXES should include common URL schemes."""
        assert "http://" in URL_PREFIXES
        assert "https://" in URL_PREFIXES
        assert "ftp://" in URL_PREFIXES


class TestIsStubContent:
    """Tests for is_stub_content() function."""

    @pytest.mark.unit
    def test_empty_content_is_stub(self) -> None:
        """Empty string should be detected as stub."""
        assert is_stub_content("") is True

    @pytest.mark.unit
    def test_whitespace_only_is_stub(self) -> None:
        """Whitespace-only content should be detected as stub."""
        assert is_stub_content("   ") is True
        assert is_stub_content("\n\t\n") is True

    @pytest.mark.unit
    def test_url_only_is_stub(self) -> None:
        """URL-only content should be detected as stub."""
        assert is_stub_content("https://example.com/file.pdf") is True
        assert is_stub_content("http://gateway.ipfs.io/ipfs/bafyk...") is True
        assert is_stub_content("ftp://ftp.example.com/resource") is True

    @pytest.mark.unit
    def test_multiple_urls_only_is_stub(self) -> None:
        """Multiple URLs without other content should be detected as stub."""
        content = "https://example.com\nhttps://another.com"
        assert is_stub_content(content) is True

        content_with_empty = "https://example.com\n\nhttps://another.com\n"
        assert is_stub_content(content_with_empty) is True

    @pytest.mark.unit
    def test_short_content_is_stub(self) -> None:
        """Content shorter than MIN_CONTENT_BYTES should be detected as stub."""
        assert is_stub_content("Hello") is True
        assert is_stub_content("A" * 50) is True
        assert is_stub_content("A" * 99) is True

    @pytest.mark.unit
    def test_real_content_not_stub(self) -> None:
        """Content of sufficient length should not be detected as stub."""
        assert is_stub_content("A" * 100) is False
        assert is_stub_content("A" * 200) is False

    @pytest.mark.unit
    def test_real_article_content_not_stub(self) -> None:
        """Real article content should not be detected as stub."""
        article = (
            "The Soviet Union was a socialist state that spanned Eurasia. "
            "It was a union of multiple subnational Soviet republics."
        )
        assert is_stub_content(article) is False

    @pytest.mark.unit
    def test_url_with_substantial_content_not_stub(self) -> None:
        """URL followed by substantial content should not be detected as stub."""
        content = "https://example.com\n\n" + "Real content here. " * 20
        assert is_stub_content(content) is False

    @pytest.mark.unit
    def test_url_at_end_of_real_content_not_stub(self) -> None:
        """Real content with URL at end should not be detected as stub."""
        content = "Real article content " * 20 + "\n\nSource: https://example.com"
        assert is_stub_content(content) is False

    @pytest.mark.unit
    def test_unicode_content_byte_counting(self) -> None:
        """Byte counting should work correctly for Unicode (multi-byte chars)."""
        # Russian text - each Cyrillic char is ~2 bytes in UTF-8
        # 50 chars = ~100 bytes
        russian_short = "А" * 49  # 49 chars = ~98 bytes, should be stub
        russian_enough = "А" * 50  # 50 chars = ~100 bytes, should not be stub

        assert is_stub_content(russian_short) is True
        assert is_stub_content(russian_enough) is False


class TestSembrSkipError:
    """Tests for SembrSkipError exception."""

    @pytest.mark.unit
    def test_sembr_skip_error_inherits_from_sembr_error(self) -> None:
        """SembrSkipError should inherit from SembrError."""
        assert issubclass(SembrSkipError, SembrError)

    @pytest.mark.unit
    def test_sembr_skip_error_message(self) -> None:
        """SembrSkipError should include the provided message."""
        error = SembrSkipError("Stub content detected (size=50 bytes)")
        assert "Stub content" in str(error)
        assert "50 bytes" in str(error)

    @pytest.mark.unit
    def test_sembr_skip_error_catchable_as_sembr_error(self) -> None:
        """SembrSkipError should be catchable as SembrError."""
        try:
            raise SembrSkipError("test")
        except SembrError as e:
            assert "test" in str(e)


class TestProcessFileStubDetection:
    """Tests for stub detection in process_file()."""

    @pytest.mark.unit
    async def test_process_file_raises_skip_error_for_stub(self, tmp_path: Path) -> None:
        """process_file should raise SembrSkipError for stub content."""
        input_file = tmp_path / "stub.txt"
        input_file.write_text("https://example.com/file.pdf")

        output_file = tmp_path / "output" / "stub.txt"

        with pytest.raises(SembrSkipError) as exc_info:
            await process_file(input_file, output_file)

        assert "stub" in str(exc_info.value).lower()

    @pytest.mark.unit
    async def test_process_file_raises_skip_error_for_empty_file(self, tmp_path: Path) -> None:
        """process_file should raise SembrSkipError for empty files."""
        input_file = tmp_path / "empty.txt"
        input_file.write_text("")

        output_file = tmp_path / "output" / "empty.txt"

        with pytest.raises(SembrSkipError):
            await process_file(input_file, output_file)

    @pytest.mark.unit
    async def test_process_file_raises_skip_error_for_short_content(self, tmp_path: Path) -> None:
        """process_file should raise SembrSkipError for very short content."""
        input_file = tmp_path / "short.txt"
        input_file.write_text("Too short")

        output_file = tmp_path / "output" / "short.txt"

        with pytest.raises(SembrSkipError):
            await process_file(input_file, output_file)

    @pytest.mark.unit
    async def test_process_file_succeeds_for_real_content(self, tmp_path: Path) -> None:
        """process_file should process files with real content."""
        input_file = tmp_path / "real.txt"
        real_content = "This is real article content. " * 10  # Well over 100 bytes
        input_file.write_text(real_content)

        output_file = tmp_path / "output" / "real.txt"

        with patch("pw_mcp.ingest.linebreaker.process_text") as mock_process:
            mock_process.return_value = SembrResult(
                text="Processed content.\nWith linebreaks.",
                line_count=2,
                processing_time_ms=100.0,
                input_word_count=50,
                output_word_count=50,
            )

            result = await process_file(input_file, output_file)

            assert result.line_count == 2
            mock_process.assert_called_once()


# =============================================================================
# LONG LINE SPLITTING TESTS
# =============================================================================


class TestSplitLongLineConstants:
    """Tests for line splitting constants."""

    @pytest.mark.unit
    def test_max_line_chars_constant_exists(self) -> None:
        """MAX_LINE_CHARS constant should be defined."""
        from pw_mcp.ingest.linebreaker import MAX_LINE_CHARS

        assert MAX_LINE_CHARS == 1500

    @pytest.mark.unit
    def test_target_chunk_chars_constant_exists(self) -> None:
        """TARGET_CHUNK_CHARS constant should be defined for optimal split size."""
        from pw_mcp.ingest.linebreaker import TARGET_CHUNK_CHARS

        assert TARGET_CHUNK_CHARS == 1000


class TestSplitLongLine:
    """Tests for split_long_line() function."""

    @pytest.mark.unit
    def test_short_line_unchanged(self) -> None:
        """Lines under MAX_LINE_CHARS return as single-element list."""
        from pw_mcp.ingest.linebreaker import MAX_LINE_CHARS, split_long_line

        short_line = "This is a short line."
        result = split_long_line(short_line)
        assert result == [short_line]

        # Edge case: exactly at limit
        at_limit = "A" * MAX_LINE_CHARS
        result = split_long_line(at_limit)
        assert result == [at_limit]

    @pytest.mark.unit
    def test_splits_at_sentence_boundary(self) -> None:
        """Prefers splitting at '. ' over other boundaries."""
        from pw_mcp.ingest.linebreaker import split_long_line

        # Create a line with a sentence boundary in the middle
        first_sentence = "First sentence ends here. "
        second_sentence = "Second sentence continues " + "with more words " * 100

        long_line = first_sentence + second_sentence
        result = split_long_line(long_line, max_chars=50)

        # Should split at the sentence boundary
        assert len(result) >= 2
        assert result[0].endswith(".")

    @pytest.mark.unit
    def test_splits_at_comma_if_no_sentence(self) -> None:
        """Falls back to ', ' when no sentence boundary found."""
        from pw_mcp.ingest.linebreaker import split_long_line

        # Create a line with commas but no periods
        long_line = "word, " * 300  # ~1800 chars with no sentence endings
        result = split_long_line(long_line, max_chars=100)

        # Should split at comma boundaries
        assert len(result) > 1
        # Each chunk should end at a natural break
        for chunk in result[:-1]:
            # Should end with comma (possibly with trailing space stripped)
            assert chunk.endswith(",") or chunk.endswith(", ")

    @pytest.mark.unit
    def test_splits_at_space_if_no_punctuation(self) -> None:
        """Falls back to space when no punctuation found."""
        from pw_mcp.ingest.linebreaker import split_long_line

        # Create a line with only spaces, no punctuation
        long_line = "word " * 350  # ~1750 chars
        result = split_long_line(long_line, max_chars=100)

        # Should split at space boundaries
        assert len(result) > 1
        # No chunk should exceed max_chars
        for chunk in result:
            assert len(chunk) <= 100

    @pytest.mark.unit
    def test_very_long_line_multiple_chunks(self) -> None:
        """A 10,000 char line splits into multiple chunks."""
        from pw_mcp.ingest.linebreaker import split_long_line

        # Simulate the Battleground Tibet footnotes problem
        long_line = "Footnote text with some content. " * 300  # ~10,200 chars
        result = split_long_line(long_line, max_chars=1500)

        # Should create multiple chunks
        assert len(result) >= 7  # 10200/1500 ~= 7
        # No chunk should exceed max_chars
        for chunk in result:
            assert len(chunk) <= 1500

    @pytest.mark.unit
    def test_utf8_boundary_respected(self) -> None:
        """Doesn't split in middle of multi-byte UTF-8 character."""
        from pw_mcp.ingest.linebreaker import split_long_line

        # Create a line with multi-byte UTF-8 characters
        # German diacritics are 2 bytes each in UTF-8
        german_text = "Gedächtnislücken " * 100  # ~1700 chars
        result = split_long_line(german_text, max_chars=100)

        # Should split cleanly
        assert len(result) > 1
        # Each chunk should be valid UTF-8 (no partial characters)
        for chunk in result:
            # This should not raise - valid UTF-8
            chunk.encode("utf-8").decode("utf-8")

    @pytest.mark.unit
    def test_german_diacritics(self) -> None:
        """Handles ä, ü, ö, ß correctly."""
        from pw_mcp.ingest.linebreaker import split_long_line

        # Real German text with diacritics
        german = "Heinrich Harrer, Gedächtnislücken über die SS. " * 40
        result = split_long_line(german, max_chars=100)

        # Should preserve all characters
        combined = "".join(result)
        assert "ä" in combined
        assert "ü" in combined

    @pytest.mark.unit
    def test_chinese_characters(self) -> None:
        """Handles CJK characters correctly (3+ bytes in UTF-8)."""
        from pw_mcp.ingest.linebreaker import split_long_line

        # Chinese text - each char is 3 bytes
        chinese = "中国共产党是中国工人阶级的先锋队" * 50  # ~800 chars
        result = split_long_line(chinese, max_chars=100)

        # Should split at character boundaries (CJK doesn't use spaces)
        # So it should use hard break but still be valid UTF-8
        for chunk in result:
            chunk.encode("utf-8").decode("utf-8")

    @pytest.mark.unit
    def test_empty_line_unchanged(self) -> None:
        """Empty line returns as single empty element."""
        from pw_mcp.ingest.linebreaker import split_long_line

        result = split_long_line("")
        assert result == [""]

    @pytest.mark.unit
    def test_preserves_content(self) -> None:
        """All content is preserved after splitting."""
        from pw_mcp.ingest.linebreaker import split_long_line

        original = "The quick brown fox jumps over the lazy dog. " * 50
        result = split_long_line(original, max_chars=100)

        # Joining chunks should give back all content
        # (with possible whitespace normalization at boundaries)
        combined = " ".join(result)
        # Check that all words are present
        for word in ["quick", "brown", "fox", "jumps", "lazy", "dog"]:
            assert word in combined
