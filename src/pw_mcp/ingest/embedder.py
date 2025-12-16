"""Vector embedding module for generating chunk embeddings via Ollama or OpenAI.

This module handles the embedding of text chunks using Ollama or OpenAI embedding APIs.
It provides:
- Batch embedding of texts with automatic batching
- JSONL file processing for article chunks
- NPY file output for efficient storage
- Retry logic with exponential backoff
- Pre-flight checks for embedding provider health
- Support for multiple embedding providers (Ollama, OpenAI)

Design rationale:
- Uses numpy arrays for memory efficiency and ChromaDB compatibility
- Preserves chunk order (critical for Stage 6 ChromaDB loading)
- Stores embeddings separately from metadata for flexibility
- Supports resume/incremental processing via file existence checks
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from ollama import embed

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class OllamaConnectionError(Exception):
    """Raised when unable to connect to Ollama server."""

    pass


class OllamaModelError(Exception):
    """Raised when the embedding model returns unexpected results."""

    pass


class OpenAIConnectionError(Exception):
    """Raised when unable to connect to OpenAI API."""

    pass


class OpenAIAuthError(Exception):
    """Raised when OpenAI API key is missing or invalid."""

    pass


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class EmbedConfig:
    """Configuration for embedding generation.

    Attributes:
        provider: Embedding provider ("ollama" or "openai")
        model: Name of the embedding model (default: "embeddinggemma")
        dimensions: Expected embedding dimensions (default: 768 for embeddinggemma)
        batch_size: Number of texts to embed per API call (default: 32)
        ollama_host: Ollama server URL (default: "http://localhost:11434")
        max_retries: Maximum retry attempts on failure (default: 3)
        retry_delay: Base delay between retries in seconds (default: 1.0)
    """

    provider: Literal["ollama", "openai"] = "ollama"
    model: str = "embeddinggemma"
    dimensions: int = 768
    batch_size: int = 32
    ollama_host: str = "http://localhost:11434"
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if self.provider not in ("ollama", "openai"):
            raise ValueError(f"provider must be 'ollama' or 'openai', got '{self.provider}'")
        # OpenAI text-embedding-3-large supports dimensions 256-3072
        if self.provider == "openai" and not (256 <= self.dimensions <= 3072):
            raise ValueError(f"OpenAI dimensions must be 256-3072, got {self.dimensions}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class EmbeddedArticle:
    """Article with generated embeddings.

    Attributes:
        article_title: Title of the article
        namespace: ProleWiki namespace (Main, Library, Essays, ProleWiki)
        num_chunks: Number of chunks in the article
        embeddings: Numpy array of shape (num_chunks, dimensions), dtype float32
    """

    article_title: str
    namespace: str
    num_chunks: int
    embeddings: NDArray[np.float32]


# =============================================================================
# OPENAI HELPER FUNCTIONS
# =============================================================================


def _get_openai_api_key() -> str:
    """Load OpenAI API key from environment or .env file.

    Returns:
        The OpenAI API key string.

    Raises:
        OpenAIAuthError: If OPENAI_API_KEY is not set.
    """
    # Try to load from .env file if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass  # python-dotenv not installed, rely on environment

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIAuthError("OPENAI_API_KEY not found. Set in .env file or environment.")
    return api_key


def _embed_openai_batch(texts: list[str], config: EmbedConfig) -> list[list[float]]:
    """Embed a batch of texts via OpenAI API with retry logic.

    Args:
        texts: Batch of texts to embed
        config: Embedding configuration

    Returns:
        List of embedding vectors

    Raises:
        OpenAIConnectionError: If unable to connect after all retries
        OpenAIAuthError: If API key is missing or invalid
    """
    from openai import APIConnectionError, OpenAI, RateLimitError

    client = OpenAI(api_key=_get_openai_api_key())
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            response = client.embeddings.create(
                model=config.model,
                input=texts,
                dimensions=config.dimensions,
            )

            # Extract embeddings, preserving order
            return [item.embedding for item in response.data]

        except RateLimitError as e:
            last_exception = e
            if attempt < config.max_retries:
                delay = config.retry_delay * (2**attempt)
                time.sleep(delay)

        except APIConnectionError as e:
            last_exception = e
            if attempt < config.max_retries:
                delay = config.retry_delay * (2**attempt)
                time.sleep(delay)

        except Exception as e:
            last_exception = e
            if attempt < config.max_retries:
                delay = config.retry_delay * (2**attempt)
                time.sleep(delay)

    raise OpenAIConnectionError(
        f"OpenAI failed after {config.max_retries + 1} attempts: {last_exception}"
    )


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def embed_texts(texts: list[str], config: EmbedConfig) -> NDArray[np.float32]:
    """Embed a batch of texts via configured provider.

    Args:
        texts: List of texts to embed
        config: Embedding configuration

    Returns:
        Numpy array of shape (len(texts), dimensions) with dtype float32.
        Returns shape (0, dimensions) for empty input.

    Raises:
        OllamaConnectionError: If unable to connect to Ollama after all retries
        OllamaModelError: If Ollama model returns unexpected dimensions
        OpenAIConnectionError: If unable to connect to OpenAI after all retries
        OpenAIAuthError: If OpenAI API key is missing
    """
    # Handle empty input
    if not texts:
        return np.zeros((0, config.dimensions), dtype=np.float32)

    all_embeddings: list[list[float]] = []

    # Process in batches
    batch_count = (len(texts) + config.batch_size - 1) // config.batch_size

    for batch_idx in range(batch_count):
        start_idx = batch_idx * config.batch_size
        end_idx = min(start_idx + config.batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        # Route to appropriate provider
        if config.provider == "openai":
            batch_embeddings = _embed_openai_batch(batch_texts, config)
        else:  # ollama (default)
            batch_embeddings = _embed_batch_with_retry(batch_texts, config)

        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings, dtype=np.float32)


def _embed_batch_with_retry(texts: list[str], config: EmbedConfig) -> list[list[float]]:
    """Embed a single batch with retry logic.

    Args:
        texts: Batch of texts to embed
        config: Embedding configuration

    Returns:
        List of embedding vectors

    Raises:
        OllamaConnectionError: If unable to connect after all retries
        OllamaModelError: If model returns unexpected dimensions
    """
    last_exception: Exception | None = None

    for attempt in range(config.max_retries + 1):
        try:
            response = embed(
                model=config.model,
                input=texts,
            )

            embeddings = response.embeddings

            # Validate dimensions
            if embeddings and len(embeddings[0]) != config.dimensions:
                raise OllamaModelError(
                    f"Expected {config.dimensions} dimensions, " f"got {len(embeddings[0])}"
                )

            return [list(emb) for emb in embeddings]

        except OllamaModelError:
            # Don't retry on model errors (they won't self-correct)
            raise

        except Exception as e:
            last_exception = e
            if attempt < config.max_retries:
                # Exponential backoff
                delay = config.retry_delay * (2**attempt)
                time.sleep(delay)

    # All retries exhausted
    raise OllamaConnectionError(
        f"Failed to connect to Ollama after {config.max_retries + 1} attempts: " f"{last_exception}"
    )


def embed_article_chunks(jsonl_path: Path, config: EmbedConfig) -> EmbeddedArticle:
    """Embed all chunks from a JSONL file.

    Reads the JSONL file line by line, extracts the "text" field from each
    chunk, and generates embeddings. The order of embeddings matches the
    order of lines in the JSONL file (critical for ChromaDB loading).

    Args:
        jsonl_path: Path to the JSONL file containing chunks
        config: Embedding configuration

    Returns:
        EmbeddedArticle with embeddings and metadata extracted from first chunk

    Raises:
        OllamaConnectionError: If unable to connect to Ollama
        OllamaModelError: If model returns unexpected results
        FileNotFoundError: If JSONL file does not exist
        json.JSONDecodeError: If JSONL contains invalid JSON
    """
    # Read all chunks from JSONL
    chunks: list[dict[str, object]] = []
    content = jsonl_path.read_text(encoding="utf-8")

    for line in content.strip().split("\n"):
        if line.strip():
            chunks.append(json.loads(line))

    # Handle empty file
    if not chunks:
        return EmbeddedArticle(
            article_title="",
            namespace="",
            num_chunks=0,
            embeddings=np.zeros((0, config.dimensions), dtype=np.float32),
        )

    # Extract metadata from first chunk
    first_chunk = chunks[0]
    article_title = str(first_chunk.get("article_title", ""))
    namespace = str(first_chunk.get("namespace", ""))

    # Extract text from all chunks (preserving order)
    texts = [str(chunk.get("text", "")) for chunk in chunks]

    # Generate embeddings
    embeddings = embed_texts(texts, config)

    return EmbeddedArticle(
        article_title=article_title,
        namespace=namespace,
        num_chunks=len(chunks),
        embeddings=embeddings,
    )


def write_embeddings_npy(article: EmbeddedArticle, output_path: Path) -> None:
    """Save embeddings to a .npy file.

    Creates parent directories if they don't exist. Uses np.save with
    allow_pickle=False for security and compatibility.

    Args:
        article: EmbeddedArticle containing embeddings to save
        output_path: Path for the output .npy file
    """
    # Create parent directories
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    np.save(output_path, article.embeddings, allow_pickle=False)


def check_ollama_ready(config: EmbedConfig) -> bool:
    """Pre-flight check for Ollama server health.

    Attempts a test embedding to verify:
    1. Ollama server is reachable
    2. Model is available
    3. Embeddings have expected dimensions

    Args:
        config: Embedding configuration

    Returns:
        True if server is healthy and ready, False otherwise
    """
    try:
        # Try a test embedding
        response = embed(
            model=config.model,
            input=["test"],
        )

        # Verify dimensions
        if response.embeddings:
            actual_dims = len(response.embeddings[0])
            if actual_dims != config.dimensions:
                return False

        return True

    except Exception:
        return False


def check_openai_ready(config: EmbedConfig) -> bool:
    """Pre-flight check for OpenAI API health.

    Attempts a test embedding to verify:
    1. API key is valid
    2. Model is available
    3. Embeddings have expected dimensions

    Args:
        config: Embedding configuration

    Returns:
        True if API is healthy and ready, False otherwise
    """
    try:
        from openai import OpenAI

        client = OpenAI(api_key=_get_openai_api_key())
        response = client.embeddings.create(
            model=config.model,
            input=["test"],
            dimensions=config.dimensions,
        )

        # Verify dimensions
        if response.data:
            actual_dims = len(response.data[0].embedding)
            if actual_dims != config.dimensions:
                return False

        return True

    except Exception:
        return False


def check_provider_ready(config: EmbedConfig) -> bool:
    """Pre-flight check for embedding provider health.

    Routes to the appropriate provider-specific health check.

    Args:
        config: Embedding configuration

    Returns:
        True if provider is healthy and ready, False otherwise
    """
    if config.provider == "openai":
        return check_openai_ready(config)
    return check_ollama_ready(config)
