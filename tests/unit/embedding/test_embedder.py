"""Unit tests for embedder module (TDD Green Phase).

These tests verify the embedding module implementation.
Tests use mocking to avoid requiring a running Ollama server.

Test strategy:
- Test EmbedConfig defaults and customization
- Test embed_texts() function behavior
- Test embed_article_chunks() function behavior
- Test write_embeddings_npy() function behavior
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pw_mcp.ingest.embedder import (
    EmbedConfig,
    EmbeddedArticle,
    OllamaConnectionError,
    embed_article_chunks,
    embed_texts,
    write_embeddings_npy,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestEmbedConfigDefaults:
    """Tests for EmbedConfig default values."""

    @pytest.mark.unit
    def test_embed_config_defaults(self) -> None:
        """Should have sensible defaults matching embedding.yaml specification.

        Expected defaults:
        - model: "embeddinggemma"
        - dimensions: 768
        - batch_size: 32
        - ollama_host: "http://localhost:11434"
        - max_retries: 3
        - retry_delay: 1.0
        """
        config = EmbedConfig()

        assert config.model == "embeddinggemma"
        assert config.dimensions == 768
        assert config.batch_size == 32
        assert config.ollama_host == "http://localhost:11434"
        assert config.max_retries == 3
        assert config.retry_delay == 1.0

    @pytest.mark.unit
    def test_embed_config_custom_values(self) -> None:
        """Should accept custom configuration values.

        Test that all fields can be overridden:
        - model, dimensions, batch_size, ollama_host, max_retries, retry_delay
        """
        config = EmbedConfig(
            model="custom-model",
            dimensions=384,
            batch_size=16,
            ollama_host="http://custom:11434",
            max_retries=5,
            retry_delay=2.0,
        )

        assert config.model == "custom-model"
        assert config.dimensions == 384
        assert config.batch_size == 16
        assert config.ollama_host == "http://custom:11434"
        assert config.max_retries == 5
        assert config.retry_delay == 2.0

    @pytest.mark.unit
    def test_embed_config_frozen(self) -> None:
        """Config should be a frozen dataclass (immutable).

        Attempting to modify any field after creation should raise AttributeError.
        """
        config = EmbedConfig()

        with pytest.raises(AttributeError):
            config.model = "new-model"  # type: ignore[misc]

        with pytest.raises(AttributeError):
            config.batch_size = 64  # type: ignore[misc]

    @pytest.mark.unit
    def test_embed_config_validation(self) -> None:
        """Should validate configuration values are within acceptable ranges.

        - batch_size should be positive (> 0)
        - dimensions should be positive (> 0)
        - max_retries should be non-negative (>= 0)
        - retry_delay should be non-negative (>= 0.0)
        """
        # batch_size must be positive
        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbedConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbedConfig(batch_size=-1)

        # dimensions must be positive
        with pytest.raises(ValueError, match="dimensions must be positive"):
            EmbedConfig(dimensions=0)

        with pytest.raises(ValueError, match="dimensions must be positive"):
            EmbedConfig(dimensions=-1)

        # max_retries must be non-negative
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            EmbedConfig(max_retries=-1)

        # retry_delay must be non-negative
        with pytest.raises(ValueError, match="retry_delay must be non-negative"):
            EmbedConfig(retry_delay=-0.1)

        # Zero values for max_retries and retry_delay should be valid
        config = EmbedConfig(max_retries=0, retry_delay=0.0)
        assert config.max_retries == 0
        assert config.retry_delay == 0.0


# =============================================================================
# EMBED_TEXTS() TESTS
# =============================================================================


class TestEmbedTextsSingle:
    """Tests for embed_texts() with single text input."""

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.embed")
    def test_embed_texts_single(
        self,
        mock_embed: MagicMock,
        sample_embed_config: dict[str, Any],
        mock_ollama_embeddings: list[list[float]],
    ) -> None:
        """Single text should produce shape (1, 768).

        embed_texts(["hello world"], config) -> np.ndarray of shape (1, 768)
        """
        # Setup mock to return single embedding
        mock_embed.return_value = MagicMock(embeddings=[mock_ollama_embeddings[0]])

        config = EmbedConfig(**sample_embed_config)
        result = embed_texts(["hello world"], config)

        assert result.shape == (1, 768)
        mock_embed.assert_called_once()


class TestEmbedTextsBatch:
    """Tests for embed_texts() with multiple texts."""

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.embed")
    def test_embed_texts_batch(
        self,
        mock_embed: MagicMock,
        sample_embed_config: dict[str, Any],
        mock_ollama_embeddings: list[list[float]],
    ) -> None:
        """Multiple texts should produce shape (N, 768).

        embed_texts(["text1", "text2", "text3"], config) -> shape (3, 768)
        """
        # Setup mock to return 3 embeddings
        mock_embed.return_value = MagicMock(embeddings=mock_ollama_embeddings[:3])

        config = EmbedConfig(**sample_embed_config)
        result = embed_texts(["text1", "text2", "text3"], config)

        assert result.shape == (3, 768)
        mock_embed.assert_called_once()

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.embed")
    def test_embed_texts_batching_at_limit(
        self,
        mock_embed: MagicMock,
        sample_embed_config: dict[str, Any],
    ) -> None:
        """Should split into batches when exceeding batch_size.

        With batch_size=32, 100 texts should result in 4 batches:
        32 + 32 + 32 + 4 = 100
        """
        # Create config with batch_size=32
        config = EmbedConfig(**{**sample_embed_config, "batch_size": 32})

        # Mock returns embeddings for each batch
        def mock_embed_side_effect(**kwargs: Any) -> MagicMock:
            input_texts = kwargs.get("input", [])
            mock_embeddings = [[0.1] * 768 for _ in input_texts]
            return MagicMock(embeddings=mock_embeddings)

        mock_embed.side_effect = mock_embed_side_effect

        # Embed 100 texts
        texts = [f"text {i}" for i in range(100)]
        result = embed_texts(texts, config)

        assert result.shape == (100, 768)
        # Should be called 4 times: 32 + 32 + 32 + 4
        assert mock_embed.call_count == 4


class TestEmbedTextsEdgeCases:
    """Tests for embed_texts() edge cases."""

    @pytest.mark.unit
    def test_embed_texts_empty_list(
        self,
        sample_embed_config: dict[str, Any],
    ) -> None:
        """Empty input should return shape (0, 768).

        embed_texts([], config) -> np.ndarray of shape (0, 768)
        """
        config = EmbedConfig(**sample_embed_config)
        result = embed_texts([], config)

        assert result.shape == (0, 768)
        assert result.dtype == np.float32


class TestEmbedTextsReturnType:
    """Tests for embed_texts() return type and shape."""

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.embed")
    def test_embed_texts_returns_numpy(
        self,
        mock_embed: MagicMock,
        sample_embed_config: dict[str, Any],
        mock_ollama_embeddings: list[list[float]],
    ) -> None:
        """Should return a numpy ndarray, not a list.

        import numpy as np
        result = embed_texts(["test"], config)
        assert isinstance(result, np.ndarray)
        """
        mock_embed.return_value = MagicMock(embeddings=[mock_ollama_embeddings[0]])

        config = EmbedConfig(**sample_embed_config)
        result = embed_texts(["test"], config)

        assert isinstance(result, np.ndarray)

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.embed")
    def test_embed_texts_correct_shape(
        self,
        mock_embed: MagicMock,
        sample_embed_config: dict[str, Any],
        mock_ollama_embeddings: list[list[float]],
    ) -> None:
        """Shape should be (num_texts, dimensions).

        For N texts with 768-dim model:
        result.shape == (N, 768)
        """
        # Test with 5 texts
        mock_embed.return_value = MagicMock(embeddings=mock_ollama_embeddings)

        config = EmbedConfig(**sample_embed_config)
        texts = ["text1", "text2", "text3", "text4", "text5"]
        result = embed_texts(texts, config)

        assert result.shape == (5, 768)

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.embed")
    def test_embed_texts_correct_dtype(
        self,
        mock_embed: MagicMock,
        sample_embed_config: dict[str, Any],
        mock_ollama_embeddings: list[list[float]],
    ) -> None:
        """Should return float32 dtype for memory efficiency.

        result.dtype == np.float32
        """
        mock_embed.return_value = MagicMock(embeddings=[mock_ollama_embeddings[0]])

        config = EmbedConfig(**sample_embed_config)
        result = embed_texts(["test"], config)

        assert result.dtype == np.float32


class TestEmbedTextsErrorHandling:
    """Tests for embed_texts() error handling."""

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.embed")
    @patch("pw_mcp.ingest.embedder.time.sleep")
    def test_embed_texts_retry_on_failure(
        self,
        mock_sleep: MagicMock,
        mock_embed: MagicMock,
        sample_embed_config: dict[str, Any],
        mock_ollama_embeddings: list[list[float]],
    ) -> None:
        """Should retry up to max_retries times on transient failures.

        Uses exponential backoff starting from retry_delay.
        Should succeed if retry eventually works.
        Should raise OllamaConnectionError after exhausting retries.
        """
        # Test case 1: Succeeds on retry
        call_count = 0

        def succeed_on_third_call(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Server unavailable")
            return MagicMock(embeddings=[mock_ollama_embeddings[0]])

        mock_embed.side_effect = succeed_on_third_call

        config = EmbedConfig(**{**sample_embed_config, "max_retries": 3, "retry_delay": 0.1})
        result = embed_texts(["test"], config)

        assert result.shape == (1, 768)
        assert call_count == 3  # Failed twice, succeeded on third

        # Test case 2: Exhausts retries
        mock_embed.reset_mock()
        mock_embed.side_effect = ConnectionError("Server down")

        config_no_retry = EmbedConfig(
            **{**sample_embed_config, "max_retries": 2, "retry_delay": 0.1}
        )

        with pytest.raises(OllamaConnectionError) as exc_info:
            embed_texts(["test"], config_no_retry)

        assert "Failed to connect" in str(exc_info.value)


# =============================================================================
# EMBED_ARTICLE_CHUNKS() TESTS
# =============================================================================


class TestEmbedArticleChunks:
    """Tests for embed_article_chunks() function."""

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.embed")
    def test_embed_article_reads_jsonl(
        self,
        mock_embed: MagicMock,
        embedding_fixtures_dir: Path,
        sample_embed_config: dict[str, Any],
        mock_ollama_embeddings: list[list[float]],
    ) -> None:
        """Should read and parse JSONL file correctly.

        Each line is a JSON object with at least a "text" field.
        Should handle the sample_chunks.jsonl fixture (5 chunks).
        """
        # Setup mock to return 5 embeddings
        mock_embed.return_value = MagicMock(embeddings=mock_ollama_embeddings)

        config = EmbedConfig(**sample_embed_config)
        jsonl_path = embedding_fixtures_dir / "sample_chunks.jsonl"

        result = embed_article_chunks(jsonl_path, config)

        assert result.num_chunks == 5
        assert result.embeddings.shape == (5, 768)

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.embed")
    def test_embed_article_extracts_text(
        self,
        mock_embed: MagicMock,
        embedding_fixtures_dir: Path,
        sample_embed_config: dict[str, Any],
        mock_ollama_embeddings: list[list[float]],
    ) -> None:
        """Should extract the 'text' field from each JSONL line for embedding.

        The embedding is performed on the 'text' field content only.
        """
        # Setup mock to capture the texts passed
        mock_embed.return_value = MagicMock(embeddings=mock_ollama_embeddings)

        config = EmbedConfig(**sample_embed_config)
        jsonl_path = embedding_fixtures_dir / "sample_chunks.jsonl"

        embed_article_chunks(jsonl_path, config)

        # Verify embed was called with the text content
        mock_embed.assert_called_once()
        call_kwargs = mock_embed.call_args[1]
        texts = call_kwargs["input"]

        # The fixture has 5 chunks with specific text content
        assert len(texts) == 5
        assert "Five-Year Plans" in texts[0]

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.embed")
    def test_embed_article_preserves_order(
        self,
        mock_embed: MagicMock,
        embedding_fixtures_dir: Path,
        sample_embed_config: dict[str, Any],
        mock_ollama_embeddings: list[list[float]],
    ) -> None:
        """CRITICAL: Embedding order must match JSONL line order.

        This is essential for Stage 6 ChromaDB loading:
        - embeddings[0] corresponds to JSONL line 0 (chunk_index 0)
        - embeddings[1] corresponds to JSONL line 1 (chunk_index 1)
        - etc.

        Breaking this alignment would cause incorrect semantic search results.
        """
        # Setup mock with distinct embeddings
        mock_embed.return_value = MagicMock(embeddings=mock_ollama_embeddings)

        config = EmbedConfig(**sample_embed_config)
        jsonl_path = embedding_fixtures_dir / "sample_chunks.jsonl"

        result = embed_article_chunks(jsonl_path, config)

        # Verify embedding order matches (compare first few values)
        for i in range(5):
            np.testing.assert_array_almost_equal(
                result.embeddings[i][:10],
                np.array(mock_ollama_embeddings[i][:10], dtype=np.float32),
            )

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.embed")
    def test_embed_article_extracts_metadata(
        self,
        mock_embed: MagicMock,
        embedding_fixtures_dir: Path,
        sample_embed_config: dict[str, Any],
        mock_ollama_embeddings: list[list[float]],
    ) -> None:
        """Should extract article_title and namespace from JSONL.

        Returns EmbeddedArticle with:
        - article_title from first chunk's "article_title" field
        - namespace from first chunk's "namespace" field
        - num_chunks equal to number of JSONL lines
        """
        mock_embed.return_value = MagicMock(embeddings=mock_ollama_embeddings)

        config = EmbedConfig(**sample_embed_config)
        jsonl_path = embedding_fixtures_dir / "sample_chunks.jsonl"

        result = embed_article_chunks(jsonl_path, config)

        assert result.article_title == "Five-Year Plans"
        assert result.namespace == "Main"
        assert result.num_chunks == 5

    @pytest.mark.unit
    def test_embed_article_handles_empty(
        self,
        tmp_path: Path,
        sample_embed_config: dict[str, Any],
    ) -> None:
        """Should handle empty JSONL files gracefully.

        Returns EmbeddedArticle with:
        - num_chunks = 0
        - embeddings.shape = (0, 768)
        """
        # Create empty JSONL file
        empty_jsonl = tmp_path / "empty.jsonl"
        empty_jsonl.write_text("")

        config = EmbedConfig(**sample_embed_config)
        result = embed_article_chunks(empty_jsonl, config)

        assert result.num_chunks == 0
        assert result.embeddings.shape == (0, 768)
        assert result.article_title == ""
        assert result.namespace == ""


# =============================================================================
# WRITE_EMBEDDINGS_NPY() TESTS
# =============================================================================


class TestWriteEmbeddingsNpy:
    """Tests for write_embeddings_npy() function."""

    @pytest.mark.unit
    def test_write_embeddings_creates_file(
        self,
        tmp_path: Path,
        sample_embedded_article: dict[str, Any],
    ) -> None:
        """Should create a .npy file at the specified path.

        write_embeddings_npy(article, output_path)
        assert output_path.exists()
        """
        output_path = tmp_path / "embeddings.npy"

        # Create EmbeddedArticle from fixture data
        embeddings = np.array(sample_embedded_article["embeddings"], dtype=np.float32)
        article = EmbeddedArticle(
            article_title=sample_embedded_article["article_title"],
            namespace=sample_embedded_article["namespace"],
            num_chunks=sample_embedded_article["num_chunks"],
            embeddings=embeddings,
        )

        write_embeddings_npy(article, output_path)

        assert output_path.exists()

    @pytest.mark.unit
    def test_write_embeddings_correct_shape(
        self,
        tmp_path: Path,
        sample_embedded_article: dict[str, Any],
    ) -> None:
        """Saved .npy should have shape (num_chunks, 768).

        For an article with 5 chunks:
        np.load(path).shape == (5, 768)
        """
        output_path = tmp_path / "embeddings.npy"

        embeddings = np.array(sample_embedded_article["embeddings"], dtype=np.float32)
        article = EmbeddedArticle(
            article_title=sample_embedded_article["article_title"],
            namespace=sample_embedded_article["namespace"],
            num_chunks=sample_embedded_article["num_chunks"],
            embeddings=embeddings,
        )

        write_embeddings_npy(article, output_path)

        loaded = np.load(output_path)
        assert loaded.shape == (5, 768)

    @pytest.mark.unit
    def test_write_embeddings_loadable(
        self,
        tmp_path: Path,
        sample_embedded_article: dict[str, Any],
    ) -> None:
        """Saved file should be loadable with np.load().

        data = np.load(output_path)
        assert data is not None
        assert data.dtype == np.float32
        """
        output_path = tmp_path / "embeddings.npy"

        embeddings = np.array(sample_embedded_article["embeddings"], dtype=np.float32)
        article = EmbeddedArticle(
            article_title=sample_embedded_article["article_title"],
            namespace=sample_embedded_article["namespace"],
            num_chunks=sample_embedded_article["num_chunks"],
            embeddings=embeddings,
        )

        write_embeddings_npy(article, output_path)

        data = np.load(output_path)
        assert data is not None
        assert data.dtype == np.float32

    @pytest.mark.unit
    def test_write_embeddings_creates_dirs(
        self,
        tmp_path: Path,
        sample_embedded_article: dict[str, Any],
    ) -> None:
        """Should create parent directories if they don't exist.

        output_path = tmp_path / "deeply" / "nested" / "dir" / "embeddings.npy"
        write_embeddings_npy(article, output_path)
        assert output_path.exists()
        """
        output_path = tmp_path / "deeply" / "nested" / "dir" / "embeddings.npy"

        embeddings = np.array(sample_embedded_article["embeddings"], dtype=np.float32)
        article = EmbeddedArticle(
            article_title=sample_embedded_article["article_title"],
            namespace=sample_embedded_article["namespace"],
            num_chunks=sample_embedded_article["num_chunks"],
            embeddings=embeddings,
        )

        write_embeddings_npy(article, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()


# =============================================================================
# CHECK_OLLAMA_READY() TESTS - Implicit in other tests
# =============================================================================
# Note: check_ollama_ready() is tested implicitly through CLI tests
# (test_embed_validates_ollama in test_embed_cli.py)
