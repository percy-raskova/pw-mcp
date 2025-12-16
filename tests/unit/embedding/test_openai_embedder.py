"""Unit tests for OpenAI embedding provider (TDD Red Phase).

These tests verify the OpenAI embedding implementation.
Tests use mocking to avoid requiring real API calls.

Test strategy:
- Test EmbedConfig with provider field and OpenAI-specific validation
- Test OpenAI-specific exception types
- Test _get_openai_api_key() environment loading
- Test embed_texts() with OpenAI provider routing
- Test check_openai_ready() function
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pw_mcp.ingest.embedder import EmbedConfig

if TYPE_CHECKING:
    pass


# =============================================================================
# CONFIG TESTS - PROVIDER FIELD
# =============================================================================


class TestEmbedConfigProvider:
    """Tests for EmbedConfig provider field."""

    @pytest.mark.unit
    def test_default_provider_is_ollama(self) -> None:
        """Default provider should be 'ollama' for backward compatibility."""
        config = EmbedConfig()
        assert config.provider == "ollama"

    @pytest.mark.unit
    def test_openai_provider_accepted(self) -> None:
        """Should accept 'openai' as a valid provider."""
        config = EmbedConfig(
            provider="openai",
            model="text-embedding-3-large",
            dimensions=1536,
        )
        assert config.provider == "openai"
        assert config.model == "text-embedding-3-large"
        assert config.dimensions == 1536

    @pytest.mark.unit
    def test_invalid_provider_rejected(self) -> None:
        """Should reject invalid provider values."""
        with pytest.raises(ValueError, match="provider must be"):
            EmbedConfig(provider="invalid-provider")

    @pytest.mark.unit
    def test_openai_dimensions_min_valid(self) -> None:
        """OpenAI supports minimum 256 dimensions."""
        config = EmbedConfig(provider="openai", dimensions=256)
        assert config.dimensions == 256

    @pytest.mark.unit
    def test_openai_dimensions_max_valid(self) -> None:
        """OpenAI supports maximum 3072 dimensions."""
        config = EmbedConfig(provider="openai", dimensions=3072)
        assert config.dimensions == 3072

    @pytest.mark.unit
    def test_openai_dimensions_below_min_rejected(self) -> None:
        """OpenAI should reject dimensions below 256."""
        with pytest.raises(ValueError, match="256-3072"):
            EmbedConfig(provider="openai", dimensions=100)

    @pytest.mark.unit
    def test_openai_dimensions_above_max_rejected(self) -> None:
        """OpenAI should reject dimensions above 3072."""
        with pytest.raises(ValueError, match="256-3072"):
            EmbedConfig(provider="openai", dimensions=4000)

    @pytest.mark.unit
    def test_ollama_dimensions_not_validated_by_openai_rules(self) -> None:
        """Ollama provider should not be constrained by OpenAI dimension limits."""
        # Ollama can have any positive dimension
        config = EmbedConfig(provider="ollama", dimensions=768)
        assert config.dimensions == 768

        # Ollama allows 100 dimensions (would fail OpenAI validation)
        config_small = EmbedConfig(provider="ollama", dimensions=100)
        assert config_small.dimensions == 100


# =============================================================================
# EXCEPTION TESTS
# =============================================================================


class TestOpenAIExceptions:
    """Tests for OpenAI-specific exceptions."""

    @pytest.mark.unit
    def test_openai_auth_error_exists(self) -> None:
        """OpenAIAuthError should be importable."""
        from pw_mcp.ingest.embedder import OpenAIAuthError

        error = OpenAIAuthError("API key missing")
        assert str(error) == "API key missing"

    @pytest.mark.unit
    def test_openai_connection_error_exists(self) -> None:
        """OpenAIConnectionError should be importable."""
        from pw_mcp.ingest.embedder import OpenAIConnectionError

        error = OpenAIConnectionError("Network failure")
        assert str(error) == "Network failure"


# =============================================================================
# API KEY LOADING TESTS
# =============================================================================


class TestGetOpenAIApiKey:
    """Tests for _get_openai_api_key() function."""

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345"}, clear=False)
    def test_loads_from_environment(self) -> None:
        """Should load API key from OPENAI_API_KEY environment variable."""
        from pw_mcp.ingest.embedder import _get_openai_api_key

        key = _get_openai_api_key()
        assert key == "sk-test-key-12345"

    @pytest.mark.unit
    @patch.dict(os.environ, {}, clear=True)
    @patch("dotenv.load_dotenv")
    def test_raises_auth_error_when_missing(self, mock_load_dotenv: MagicMock) -> None:
        """Should raise OpenAIAuthError when API key is not set."""
        from pw_mcp.ingest.embedder import OpenAIAuthError, _get_openai_api_key

        with pytest.raises(OpenAIAuthError, match="OPENAI_API_KEY not found"):
            _get_openai_api_key()

    @pytest.mark.unit
    @patch.dict(os.environ, {}, clear=True)
    @patch("dotenv.load_dotenv")
    def test_calls_dotenv_load(self, mock_load_dotenv: MagicMock) -> None:
        """Should call dotenv.load_dotenv() to load .env file."""
        from pw_mcp.ingest.embedder import OpenAIAuthError, _get_openai_api_key

        # Even though key is missing, load_dotenv should be called
        with pytest.raises(OpenAIAuthError):
            _get_openai_api_key()

        mock_load_dotenv.assert_called_once()


# =============================================================================
# EMBED_TEXTS() WITH OPENAI PROVIDER
# =============================================================================


class TestEmbedTextsOpenAI:
    """Tests for embed_texts() with OpenAI provider."""

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    def test_embed_texts_openai_single(
        self,
        mock_openai_class: MagicMock,
        mock_openai_embeddings_1536: list[list[float]],
    ) -> None:
        """Single text with OpenAI should produce shape (1, 1536)."""
        from pw_mcp.ingest.embedder import embed_texts

        # Setup mock client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Setup mock response
        mock_embedding = MagicMock()
        mock_embedding.embedding = mock_openai_embeddings_1536[0]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        config = EmbedConfig(
            provider="openai",
            model="text-embedding-3-large",
            dimensions=1536,
        )
        result = embed_texts(["hello world"], config)

        assert result.shape == (1, 1536)
        assert result.dtype == np.float32

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    def test_embed_texts_openai_batch(
        self,
        mock_openai_class: MagicMock,
        mock_openai_embeddings_1536: list[list[float]],
    ) -> None:
        """Multiple texts with OpenAI should produce shape (N, 1536)."""
        from pw_mcp.ingest.embedder import embed_texts

        # Setup mock client
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Setup mock response with 3 embeddings
        mock_embeddings = [MagicMock() for _ in range(3)]
        for i, mock_emb in enumerate(mock_embeddings):
            mock_emb.embedding = mock_openai_embeddings_1536[i]
        mock_response = MagicMock()
        mock_response.data = mock_embeddings
        mock_client.embeddings.create.return_value = mock_response

        config = EmbedConfig(
            provider="openai",
            model="text-embedding-3-large",
            dimensions=1536,
        )
        result = embed_texts(["text1", "text2", "text3"], config)

        assert result.shape == (3, 1536)

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    def test_embed_texts_openai_passes_dimensions(
        self,
        mock_openai_class: MagicMock,
        mock_openai_embeddings_1536: list[list[float]],
    ) -> None:
        """Should pass dimensions parameter to OpenAI API."""
        from pw_mcp.ingest.embedder import embed_texts

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_embedding = MagicMock()
        mock_embedding.embedding = mock_openai_embeddings_1536[0]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        config = EmbedConfig(
            provider="openai",
            model="text-embedding-3-large",
            dimensions=1536,
        )
        embed_texts(["test"], config)

        # Verify dimensions was passed
        mock_client.embeddings.create.assert_called_once()
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["dimensions"] == 1536
        assert call_kwargs["model"] == "text-embedding-3-large"

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    @patch("time.sleep")
    def test_embed_texts_openai_retries_on_rate_limit(
        self,
        mock_sleep: MagicMock,
        mock_openai_class: MagicMock,
        mock_openai_embeddings_1536: list[list[float]],
    ) -> None:
        """Should retry with exponential backoff on RateLimitError."""
        from openai import RateLimitError

        from pw_mcp.ingest.embedder import embed_texts

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # First call fails with rate limit, second succeeds
        call_count = 0

        def mock_create(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    message="Rate limit exceeded",
                    response=MagicMock(status_code=429),
                    body=None,
                )
            mock_embedding = MagicMock()
            mock_embedding.embedding = mock_openai_embeddings_1536[0]
            mock_response = MagicMock()
            mock_response.data = [mock_embedding]
            return mock_response

        mock_client.embeddings.create.side_effect = mock_create

        config = EmbedConfig(
            provider="openai",
            model="text-embedding-3-large",
            dimensions=1536,
            max_retries=3,
            retry_delay=0.1,
        )
        result = embed_texts(["test"], config)

        assert result.shape == (1, 1536)
        assert call_count == 2
        mock_sleep.assert_called_once()


# =============================================================================
# CHECK_OPENAI_READY() TESTS
# =============================================================================


class TestCheckOpenAIReady:
    """Tests for check_openai_ready() function."""

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    def test_returns_true_when_api_works(
        self,
        mock_openai_class: MagicMock,
        mock_openai_embeddings_1536: list[list[float]],
    ) -> None:
        """Should return True when API responds correctly."""
        from pw_mcp.ingest.embedder import check_openai_ready

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_embedding = MagicMock()
        mock_embedding.embedding = mock_openai_embeddings_1536[0]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        config = EmbedConfig(
            provider="openai",
            model="text-embedding-3-large",
            dimensions=1536,
        )
        assert check_openai_ready(config) is True

    @pytest.mark.unit
    @patch.dict(os.environ, {}, clear=True)
    @patch("dotenv.load_dotenv")
    def test_returns_false_when_no_api_key(self, mock_load_dotenv: MagicMock) -> None:
        """Should return False when API key is missing."""
        from pw_mcp.ingest.embedder import check_openai_ready

        config = EmbedConfig(
            provider="openai",
            model="text-embedding-3-large",
            dimensions=1536,
        )
        assert check_openai_ready(config) is False

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    def test_returns_false_on_api_error(self, mock_openai_class: MagicMock) -> None:
        """Should return False when API call fails."""
        from pw_mcp.ingest.embedder import check_openai_ready

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.embeddings.create.side_effect = Exception("API error")

        config = EmbedConfig(
            provider="openai",
            model="text-embedding-3-large",
            dimensions=1536,
        )
        assert check_openai_ready(config) is False


# =============================================================================
# CHECK_PROVIDER_READY() TESTS
# =============================================================================


class TestCheckProviderReady:
    """Tests for check_provider_ready() routing function."""

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.check_ollama_ready")
    def test_routes_to_ollama_check(self, mock_ollama_check: MagicMock) -> None:
        """Should call check_ollama_ready for ollama provider."""
        from pw_mcp.ingest.embedder import check_provider_ready

        mock_ollama_check.return_value = True
        config = EmbedConfig(provider="ollama")

        result = check_provider_ready(config)

        assert result is True
        mock_ollama_check.assert_called_once_with(config)

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.check_openai_ready")
    def test_routes_to_openai_check(self, mock_openai_check: MagicMock) -> None:
        """Should call check_openai_ready for openai provider."""
        from pw_mcp.ingest.embedder import check_provider_ready

        mock_openai_check.return_value = True
        config = EmbedConfig(
            provider="openai",
            model="text-embedding-3-large",
            dimensions=1536,
        )

        result = check_provider_ready(config)

        assert result is True
        mock_openai_check.assert_called_once_with(config)


# =============================================================================
# TOKEN-AWARE BATCHING TESTS
# =============================================================================


class TestTokenAwareBatching:
    """Tests for token-aware batching to stay within OpenAI limits.

    OpenAI embedding API limits:
    - 8192 tokens max per individual text input
    - 300,000 tokens max total across all inputs in a single request
    """

    @pytest.mark.unit
    def test_build_token_aware_batches_exists(self) -> None:
        """_build_token_aware_batches function should exist."""
        from pw_mcp.ingest.embedder import _build_token_aware_batches

        assert callable(_build_token_aware_batches)

    @pytest.mark.unit
    def test_single_small_text_returns_one_batch(self) -> None:
        """Single small text should result in one batch."""
        from pw_mcp.ingest.embedder import _build_token_aware_batches

        texts = ["This is a small test."]
        batches = _build_token_aware_batches(texts)

        assert len(batches) == 1
        assert batches[0] == texts

    @pytest.mark.unit
    def test_many_small_texts_fit_in_one_batch(self) -> None:
        """Many small texts should fit in one batch if under token limit."""
        from pw_mcp.ingest.embedder import _build_token_aware_batches

        texts = ["Small text." for _ in range(10)]
        batches = _build_token_aware_batches(texts, max_tokens=1000)

        assert len(batches) == 1
        assert len(batches[0]) == 10

    @pytest.mark.unit
    def test_splits_when_token_limit_exceeded(self) -> None:
        """Should split into multiple batches when token limit is exceeded."""
        from pw_mcp.ingest.embedder import _build_token_aware_batches

        # Create texts that will exceed a small token limit
        # Each "word " is about 1-2 tokens
        texts = ["word " * 100 for _ in range(10)]  # ~100-200 tokens each
        batches = _build_token_aware_batches(texts, max_tokens=300)

        # Should create multiple batches
        assert len(batches) > 1
        # All texts should be present
        total_texts = sum(len(batch) for batch in batches)
        assert total_texts == 10

    @pytest.mark.unit
    def test_preserves_text_order(self) -> None:
        """Batching should preserve the order of texts."""
        from pw_mcp.ingest.embedder import _build_token_aware_batches

        texts = [f"Text number {i}" for i in range(20)]
        batches = _build_token_aware_batches(texts, max_tokens=100)

        # Flatten batches and check order
        flattened = [text for batch in batches for text in batch]
        assert flattened == texts

    @pytest.mark.unit
    def test_respects_max_items_limit(self) -> None:
        """Should respect max_items even if tokens would allow more."""
        from pw_mcp.ingest.embedder import _build_token_aware_batches

        texts = ["tiny" for _ in range(100)]  # Very small texts
        batches = _build_token_aware_batches(texts, max_tokens=100000, max_items=10)

        # Each batch should have at most 10 items
        for batch in batches:
            assert len(batch) <= 10

    @pytest.mark.unit
    def test_oversized_text_gets_own_batch(self) -> None:
        """A single text exceeding max_tokens should get its own batch."""
        from pw_mcp.ingest.embedder import _build_token_aware_batches

        small_text = "small"
        large_text = "word " * 1000  # ~1000+ tokens
        texts = [small_text, large_text, small_text]

        batches = _build_token_aware_batches(texts, max_tokens=500)

        # The large text should be in its own batch
        assert any(len(batch) == 1 and batch[0] == large_text for batch in batches)
        # All texts should be present
        total_texts = sum(len(batch) for batch in batches)
        assert total_texts == 3

    @pytest.mark.unit
    def test_each_batch_under_token_limit(self) -> None:
        """Each batch should have total tokens under the limit."""
        from pw_mcp.ingest.chunker import count_tokens
        from pw_mcp.ingest.embedder import _build_token_aware_batches

        # Create varied-size texts
        texts = ["word " * (i * 10 + 10) for i in range(50)]
        max_tokens = 500
        batches = _build_token_aware_batches(texts, max_tokens=max_tokens)

        for batch in batches:
            batch_tokens = sum(count_tokens(text) for text in batch)
            # Allow single oversized texts in their own batch
            if len(batch) > 1:
                assert batch_tokens <= max_tokens, f"Batch has {batch_tokens} tokens"

    @pytest.mark.unit
    def test_realistic_chunk_scenario(self) -> None:
        """Test with realistic chunk sizes (~500 tokens each)."""
        from pw_mcp.ingest.chunker import count_tokens
        from pw_mcp.ingest.embedder import _build_token_aware_batches

        # Simulate 85 chunks of ~500 tokens (like "State and Revolution")
        texts = ["This is a test chunk with content. " * 50 for _ in range(85)]

        # Use OpenAI-safe limit
        batches = _build_token_aware_batches(texts, max_tokens=7500, max_items=32)

        # Should create multiple batches
        assert len(batches) > 1

        # Verify each batch respects limits
        for i, batch in enumerate(batches):
            assert len(batch) <= 32, f"Batch {i} has {len(batch)} items"
            batch_tokens = sum(count_tokens(text) for text in batch)
            # Each batch should be under 7500 tokens (or be a single oversized item)
            if len(batch) > 1:
                assert batch_tokens <= 7500, f"Batch {i} has {batch_tokens} tokens"

        # All texts should be accounted for
        total_texts = sum(len(batch) for batch in batches)
        assert total_texts == 85


class TestEmbedTextsTokenAwareBatching:
    """Tests that embed_texts uses token-aware batching for OpenAI."""

    @pytest.mark.unit
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False)
    @patch("openai.OpenAI")
    def test_openai_uses_multiple_batches_for_large_input(
        self, mock_openai_class: MagicMock
    ) -> None:
        """embed_texts should make multiple API calls for large inputs."""
        from pw_mcp.ingest.embedder import embed_texts

        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Track API calls
        call_count = 0
        texts_per_call: list[int] = []

        def mock_create(**kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            input_texts = kwargs.get("input", [])
            texts_per_call.append(len(input_texts))

            # Return mock embeddings matching input size
            mock_response = MagicMock()
            mock_response.data = []
            for _ in range(len(input_texts)):
                mock_embedding = MagicMock()
                mock_embedding.embedding = [0.1] * 1536
                mock_response.data.append(mock_embedding)
            return mock_response

        mock_client.embeddings.create.side_effect = mock_create

        # Create 50 chunks of ~500 tokens each (simulating a large document)
        texts = ["This is a test chunk with some content. " * 50 for _ in range(50)]

        config = EmbedConfig(
            provider="openai",
            model="text-embedding-3-large",
            dimensions=1536,
        )
        result = embed_texts(texts, config)

        # Should have made multiple API calls
        assert call_count > 1, f"Expected multiple API calls, got {call_count}"
        # Should return embeddings for all texts
        assert result.shape[0] == 50
        # Each call should have received a subset of texts
        assert all(n > 0 for n in texts_per_call)
        assert sum(texts_per_call) == 50
