"""Shared pytest fixtures for pw-mcp tests."""

import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
MEDIAWIKI_DIR = FIXTURES_DIR / "mediawiki"


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to fixtures directory."""
    return FIXTURES_DIR


@pytest.fixture
def mediawiki_dir() -> Path:
    """Return path to MediaWiki fixtures."""
    return MEDIAWIKI_DIR


@pytest.fixture
def load_fixture() -> callable:
    """Factory fixture to load MediaWiki fixture files.

    Usage:
        def test_something(load_fixture):
            content = load_fixture("infoboxes", "politician.txt")
    """

    def _load(category: str, name: str) -> str:
        path = MEDIAWIKI_DIR / category / name
        return path.read_text(encoding="utf-8")

    return _load


@pytest.fixture
def politician_infobox(load_fixture: callable) -> str:
    """Sample politician infobox (Abraham Lincoln)."""
    return load_fixture("infoboxes", "politician.txt")


@pytest.fixture
def country_infobox(load_fixture: callable) -> str:
    """Sample country infobox (Kingdom of Norway)."""
    return load_fixture("infoboxes", "country.txt")


@pytest.fixture
def political_party_infobox(load_fixture: callable) -> str:
    """Sample political party infobox (Communist Party of Germany)."""
    return load_fixture("infoboxes", "political_party.txt")


@pytest.fixture
def person_infobox(load_fixture: callable) -> str:
    """Sample person infobox (Jean-Francois Champollion)."""
    return load_fixture("infoboxes", "person.txt")


@pytest.fixture
def philosopher_infobox(load_fixture: callable) -> str:
    """Sample philosopher infobox (Ludwig Feuerbach)."""
    return load_fixture("infoboxes", "philosopher.txt")


@pytest.fixture
def organization_infobox(load_fixture: callable) -> str:
    """Sample organization infobox (Central Military Commission)."""
    return load_fixture("infoboxes", "organization.txt")


@pytest.fixture
def youtuber_infobox(load_fixture: callable) -> str:
    """Sample youtuber infobox (The Kavernacle)."""
    return load_fixture("infoboxes", "youtuber.txt")


@pytest.fixture
def web_citation(load_fixture: callable) -> str:
    """Sample web citation template."""
    return load_fixture("citations", "web_citation.txt")


@pytest.fixture
def citation_template(load_fixture: callable) -> str:
    """Sample citation template."""
    return load_fixture("citations", "citation_template.txt")


@pytest.fixture
def simple_links(load_fixture: callable) -> str:
    """Sample simple internal links."""
    return load_fixture("links", "simple_links.txt")


@pytest.fixture
def piped_links(load_fixture: callable) -> str:
    """Sample piped internal links [[Target|Display]]."""
    return load_fixture("links", "piped_links.txt")


@pytest.fixture
def category_links(load_fixture: callable) -> str:
    """Sample category links [[Category:Name]]."""
    return load_fixture("links", "category_links.txt")


@pytest.fixture
def unicode_infobox(load_fixture: callable) -> str:
    """Sample infobox with Unicode characters (Suppiluliuma I)."""
    return load_fixture("edge_cases", "unicode_infobox.txt")


# =============================================================================
# CHUNKING FIXTURES
# =============================================================================

CHUNKING_FIXTURES_DIR = FIXTURES_DIR / "chunking"


@pytest.fixture
def chunking_fixtures_dir() -> Path:
    """Return path to chunking fixtures directory."""
    return CHUNKING_FIXTURES_DIR


@pytest.fixture
def chunking_input_dir() -> Path:
    """Return path to chunking input fixtures (extracted text)."""
    return CHUNKING_FIXTURES_DIR / "input"


@pytest.fixture
def chunking_expected_dir() -> Path:
    """Return path to chunking expected output fixtures (JSONL)."""
    return CHUNKING_FIXTURES_DIR / "expected"


@pytest.fixture
def chunking_metadata_dir() -> Path:
    """Return path to chunking metadata fixtures (ArticleData JSON)."""
    return CHUNKING_FIXTURES_DIR / "metadata"


@pytest.fixture
def load_chunking_fixture() -> callable:
    """Factory fixture to load chunking fixture files.

    Usage:
        def test_something(load_chunking_fixture):
            content = load_chunking_fixture("input", "simple_article.txt")
    """

    def _load(category: str, name: str) -> str:
        path = CHUNKING_FIXTURES_DIR / category / name
        return path.read_text(encoding="utf-8")

    return _load


@pytest.fixture
def load_chunking_metadata() -> callable:
    """Factory fixture to load chunking metadata JSON files.

    Usage:
        def test_something(load_chunking_metadata):
            data = load_chunking_metadata("article_main.json")
    """

    def _load(name: str) -> dict[str, Any]:
        path = CHUNKING_FIXTURES_DIR / "metadata" / name
        return json.loads(path.read_text(encoding="utf-8"))

    return _load


@pytest.fixture
def simple_article_text(load_chunking_fixture: callable) -> str:
    """Load simple article fixture (2-3 sections)."""
    return load_chunking_fixture("input", "simple_article.txt")


@pytest.fixture
def long_article_text(load_chunking_fixture: callable) -> str:
    """Load long article fixture (exceeds max_tokens)."""
    return load_chunking_fixture("input", "long_article.txt")


@pytest.fixture
def no_headers_text(load_chunking_fixture: callable) -> str:
    """Load article without section headers."""
    return load_chunking_fixture("input", "no_headers.txt")


@pytest.fixture
def many_headers_text(load_chunking_fixture: callable) -> str:
    """Load article with many small sections."""
    return load_chunking_fixture("input", "many_headers.txt")


@pytest.fixture
def unicode_content_text(load_chunking_fixture: callable) -> str:
    """Load article with Cyrillic and Chinese text."""
    return load_chunking_fixture("input", "unicode_content.txt")


@pytest.fixture
def article_main_metadata(load_chunking_metadata: callable) -> dict[str, Any]:
    """Load Main namespace article metadata with infobox."""
    return load_chunking_metadata("article_main.json")


@pytest.fixture
def article_library_metadata(load_chunking_metadata: callable) -> dict[str, Any]:
    """Load Library namespace article metadata."""
    return load_chunking_metadata("article_library.json")


@pytest.fixture
def article_minimal_metadata(load_chunking_metadata: callable) -> dict[str, Any]:
    """Load minimal article metadata."""
    return load_chunking_metadata("article_minimal.json")


# =============================================================================
# EMBEDDING FIXTURES
# =============================================================================

EMBEDDING_FIXTURES_DIR = FIXTURES_DIR / "embedding"


@pytest.fixture
def embedding_fixtures_dir() -> Path:
    """Return path to embedding fixtures directory."""
    return EMBEDDING_FIXTURES_DIR


@pytest.fixture
def load_embedding_fixture() -> callable:
    """Factory fixture to load embedding fixture files.

    Usage:
        def test_something(load_embedding_fixture):
            content = load_embedding_fixture("sample_chunks.jsonl")
    """

    def _load(name: str) -> str:
        path = EMBEDDING_FIXTURES_DIR / name
        return path.read_text(encoding="utf-8")

    return _load


@pytest.fixture
def sample_chunks_jsonl(load_embedding_fixture: callable) -> str:
    """Load sample chunks JSONL fixture content."""
    return load_embedding_fixture("sample_chunks.jsonl")


@pytest.fixture
def mock_ollama_embeddings() -> list[list[float]]:
    """Pre-computed mock embeddings for 5 chunks (768 dimensions each).

    Returns deterministic mock vectors for testing without Ollama.
    Each vector is normalized to unit length for realism.
    """
    import math

    embeddings: list[list[float]] = []
    for i in range(5):
        # Create a deterministic but unique vector for each chunk
        # Use a simple pattern: different base values scaled by index
        vector = [0.0] * 768
        for j in range(768):
            # Create variation based on chunk index and dimension
            vector[j] = math.sin((i + 1) * (j + 1) * 0.01) * 0.1

        # L2 normalize the vector (as embeddinggemma does)
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        embeddings.append(vector)

    return embeddings


@pytest.fixture
def sample_embed_config() -> dict[str, Any]:
    """Sample EmbedConfig parameters for testing.

    Returns a dict that can be unpacked into EmbedConfig constructor.
    """
    return {
        "model": "embeddinggemma",
        "dimensions": 768,
        "batch_size": 32,
        "ollama_host": "http://localhost:11434",
        "max_retries": 3,
        "retry_delay": 1.0,
    }


@pytest.fixture
def sample_embedded_article(mock_ollama_embeddings: list[list[float]]) -> dict[str, Any]:
    """Sample EmbeddedArticle data for testing.

    Returns a dict with the expected structure of an EmbeddedArticle.
    The embeddings field contains a numpy-compatible nested list.
    """
    return {
        "article_title": "Five-Year Plans",
        "namespace": "Main",
        "num_chunks": 5,
        "embeddings": mock_ollama_embeddings,
    }


@pytest.fixture
def mock_openai_embeddings_1536() -> list[list[float]]:
    """Pre-computed mock embeddings for 5 chunks (1536 dimensions each).

    Returns deterministic mock vectors for testing OpenAI provider without API calls.
    Each vector is normalized to unit length for realism.
    """
    import math

    embeddings: list[list[float]] = []
    for i in range(5):
        vector = [0.0] * 1536
        for j in range(1536):
            vector[j] = math.sin((i + 1) * (j + 1) * 0.01) * 0.1

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        embeddings.append(vector)

    return embeddings


@pytest.fixture
def sample_openai_config() -> dict[str, Any]:
    """Sample EmbedConfig parameters for OpenAI testing.

    Returns a dict that can be unpacked into EmbedConfig constructor.
    """
    return {
        "provider": "openai",
        "model": "text-embedding-3-large",
        "dimensions": 1536,
        "batch_size": 32,
        "max_retries": 3,
        "retry_delay": 1.0,
    }


# =============================================================================
# OLLAMA SERVER FIXTURES (for slow tests)
# =============================================================================


@pytest.fixture
def require_ollama_server() -> None:
    """Skip test if Ollama server with embeddinggemma is not running.

    This fixture checks if the Ollama server is accessible and the
    embeddinggemma model is available by using the embedder's health check.

    Usage:
        @pytest.mark.slow
        def test_real_embedding(require_ollama_server):
            # This test only runs when Ollama is available
            ...
    """
    from pw_mcp.ingest.embedder import EmbedConfig, check_ollama_ready

    config = EmbedConfig()
    if not check_ollama_ready(config):
        pytest.skip(
            "Ollama server not running or embeddinggemma model not available. "
            "Start with: ollama serve && ollama pull embeddinggemma"
        )
