"""Slow tests requiring real Ollama server (TDD Phase 5.4).

These tests are marked as @pytest.mark.slow and are skipped by default.
Run with: pytest -m slow tests/slow/test_embedding_slow.py

Prerequisites:
- Ollama server running: ollama serve
- embeddinggemma model pulled: ollama pull embeddinggemma
"""

import numpy as np
import pytest

from pw_mcp.ingest.embedder import EmbedConfig, embed_texts


class TestRealOllamaEmbedding:
    """Tests against real Ollama server with embeddinggemma model."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_embed_real_chunk(self, require_ollama_server: None) -> None:
        """Should process a chunk through live Ollama and return valid embedding."""
        config = EmbedConfig()
        test_text = (
            "The Five-Year Plans transformed the Soviet Union "
            "from an agrarian society into an industrial power."
        )

        result = embed_texts([test_text], config)

        # Shape validation: single text produces (1, 768)
        assert result.shape == (1, 768)

        # Type validation: float32 for memory efficiency
        assert result.dtype == np.float32

        # Values are valid (not NaN, not inf)
        assert np.all(np.isfinite(result))

        # Model actually produced output (not all zeros)
        assert not np.allclose(result, 0)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_embedding_dimensions(self, require_ollama_server: None) -> None:
        """Should verify batch embeddings maintain shape (num_chunks, 768)."""
        config = EmbedConfig()
        test_texts = [
            "First chunk about Soviet industrialization.",
            "Second chunk discussing Five-Year Plans.",
            "Third chunk on economic transformation.",
        ]

        result = embed_texts(test_texts, config)

        # Shape: (num_texts, embedding_dim)
        assert result.shape == (3, 768)

        # Each row is a distinct embedding (semantically different texts)
        assert not np.allclose(result[0], result[1])
        assert not np.allclose(result[1], result[2])

    @pytest.mark.slow
    @pytest.mark.integration
    def test_embedding_normalized(self, require_ollama_server: None) -> None:
        """Should verify L2 norm of embeddings is approximately 1.0."""
        config = EmbedConfig()
        test_texts = [
            "Marxism-Leninism is the foundation of communist thought.",
            "The dialectical method analyzes contradictions in society.",
            "Historical materialism examines societal development.",
        ]

        result = embed_texts(test_texts, config)

        # Compute L2 norm for each embedding
        norms = np.linalg.norm(result, axis=1)

        # All norms should be close to 1.0 (L2 normalized)
        for i, norm in enumerate(norms):
            assert abs(norm - 1.0) < 0.01, f"Embedding {i} has L2 norm {norm:.4f}, expected ~1.0"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_embedding_deterministic(self, require_ollama_server: None) -> None:
        """Should return identical embeddings for same input text."""
        config = EmbedConfig()
        test_text = "Lenin developed the theory of the vanguard party."

        # Embed the same text twice
        result1 = embed_texts([test_text], config)
        result2 = embed_texts([test_text], config)

        # Results should be identical (or very close due to floating point)
        np.testing.assert_array_almost_equal(
            result1,
            result2,
            decimal=6,
            err_msg="Same input should produce identical embeddings",
        )
