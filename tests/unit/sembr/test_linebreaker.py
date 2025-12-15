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

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# These imports will fail until the module is implemented
from pw_mcp.ingest.linebreaker import (
    SembrConfig,
    SembrContentError,
    SembrError,
    SembrResult,
    SembrServerError,
    SembrTimeoutError,
    check_server_health,
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
        """Should raise SembrContentError when word count doesn't match."""
        input_text = "One two three four five."  # 5 words
        corrupted_output = "One two three."  # Only 3 words (content lost!)

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
            assert "5" in str(exc_info.value) or "3" in str(exc_info.value)


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
        input_file.write_text("Test content.")

        output_file = tmp_path / "output" / "nested" / "test.txt"

        with patch("pw_mcp.ingest.linebreaker.process_text") as mock_process:
            mock_process.return_value = SembrResult(
                text="Test\ncontent.",
                line_count=2,
                processing_time_ms=100.0,
                input_word_count=2,
                output_word_count=2,
            )

            await process_file(input_file, output_file)

            assert output_file.parent.exists()
            assert output_file.exists()
            assert output_file.read_text() == "Test\ncontent."

    @pytest.mark.unit
    async def test_process_file_preserves_encoding(self, tmp_path: Path) -> None:
        """Should preserve UTF-8 encoding for Unicode content."""
        input_file = tmp_path / "russian.txt"
        russian_text = "Советский Союз был социалистическим государством."
        input_file.write_text(russian_text, encoding="utf-8")

        output_file = tmp_path / "output" / "russian.txt"

        with patch("pw_mcp.ingest.linebreaker.process_text") as mock_process:
            mock_process.return_value = SembrResult(
                text="Советский Союз\nбыл социалистическим\nгосударством.",
                line_count=3,
                processing_time_ms=100.0,
                input_word_count=4,
                output_word_count=4,
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
        input_file.write_text("Test content for processing.")
        output_file = tmp_path / "output.txt"

        with patch("pw_mcp.ingest.linebreaker.process_text") as mock_process:
            expected_result = SembrResult(
                text="Test content\nfor processing.",
                line_count=2,
                processing_time_ms=150.0,
                input_word_count=4,
                output_word_count=4,
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
        (input_dir / "Main").mkdir(parents=True)
        (input_dir / "Library").mkdir(parents=True)
        (input_dir / "Main" / "article.txt").write_text("Main content")
        (input_dir / "Library" / "book.txt").write_text("Library content")

        output_dir = tmp_path / "output"

        with patch("pw_mcp.ingest.linebreaker.check_server_health", return_value=True):
            with patch("pw_mcp.ingest.linebreaker.process_text") as mock_process:
                mock_process.return_value = SembrResult(
                    text="Output",
                    line_count=1,
                    processing_time_ms=10.0,
                    input_word_count=1,
                    output_word_count=1,
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
