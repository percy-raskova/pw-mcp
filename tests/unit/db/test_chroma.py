"""Unit tests for ChromaDB interface."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pw_mcp.db import (
    ChromaDBConfig,
    ProleWikiDB,
    deserialize_metadata,
    serialize_metadata,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to chromadb fixtures directory."""
    return Path(__file__).parent.parent.parent / "fixtures" / "chromadb"


@pytest.fixture
def sample_chunks_path(fixtures_dir: Path) -> Path:
    """Return path to sample chunks JSONL file."""
    return fixtures_dir / "sample.jsonl"


@pytest.fixture
def sample_embeddings_path(fixtures_dir: Path) -> Path:
    """Return path to sample embeddings NPY file."""
    return fixtures_dir / "sample.npy"


@pytest.fixture
def temp_db_path() -> Path:
    """Create a temporary directory for ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def db_config(temp_db_path: Path) -> ChromaDBConfig:
    """Create a ChromaDB config for testing."""
    return ChromaDBConfig(
        persist_path=temp_db_path,
        collection_name="test_collection",
        embedding_dimensions=1536,
    )


@pytest.fixture
def db(db_config: ChromaDBConfig) -> ProleWikiDB:
    """Create a ProleWikiDB instance for testing."""
    return ProleWikiDB(db_config)


# =============================================================================
# METADATA SERIALIZATION TESTS
# =============================================================================


class TestSerializeMetadata:
    """Tests for serialize_metadata function."""

    def test_basic_serialization(self) -> None:
        """Test basic chunk serialization."""
        chunk = {
            "article_title": "Test Article",
            "namespace": "Main",
            "section": "Introduction",
            "chunk_index": 0,
            "line_range": "1-5",
            "word_count": 100,
            "categories": ["Testing", "Python"],
            "internal_links": ["Unit testing", "PyTest"],
            "is_stub": False,
            "citation_needed_count": 2,
            "has_blockquote": True,
        }

        result = serialize_metadata(chunk)

        assert result["article_title"] == "Test Article"
        assert result["namespace"] == "Main"
        assert result["section"] == "Introduction"
        assert result["chunk_index"] == 0
        assert result["line_range"] == "1-5"
        assert result["word_count"] == 100
        assert result["categories"] == '["Testing", "Python"]'
        assert result["internal_links"] == '["Unit testing", "PyTest"]'
        assert result["is_stub"] is False
        assert result["citation_needed_count"] == 2
        assert result["has_blockquote"] is True

    def test_phase_b_fields_serialization(self) -> None:
        """Test Phase B enriched metadata fields serialization."""
        chunk = {
            "article_title": "State and Revolution",
            "namespace": "Library",
            "section": "Chapter 1",
            "chunk_index": 0,
            "line_range": "1-50",
            "word_count": 350,
            "categories": ["Marxism-Leninism"],
            "internal_links": [],
            "is_stub": False,
            "citation_needed_count": 0,
            "has_blockquote": False,
            "library_work_author": "Vladimir Lenin",
            "library_work_type": "Book",
            "library_work_published_year": 1917,
            "infobox_type": None,
            "political_orientation": None,
            "primary_category": "Marxism-Leninism",
            "category_count": 1,
        }

        result = serialize_metadata(chunk)

        assert result["library_work_author"] == "Vladimir Lenin"
        assert result["library_work_type"] == "Book"
        assert result["library_work_published_year"] == 1917
        assert result["infobox_type"] == ""  # None → ""
        assert result["political_orientation"] == ""  # None → ""
        assert result["primary_category"] == "Marxism-Leninism"
        assert result["category_count"] == 1

    def test_phase_b_fields_none_handling(self) -> None:
        """Test that None Phase B fields are converted to ChromaDB-safe values."""
        chunk = {
            "article_title": "Test",
            "namespace": "Main",
            "section": None,
            "chunk_index": 0,
            "line_range": "1-1",
            "word_count": 10,
            "categories": [],
            "internal_links": [],
            "is_stub": False,
            "citation_needed_count": 0,
            "has_blockquote": False,
            # All Phase B fields are None or missing
        }

        result = serialize_metadata(chunk)

        # String fields should become empty string
        assert result["library_work_author"] == ""
        assert result["library_work_type"] == ""
        assert result["infobox_type"] == ""
        assert result["political_orientation"] == ""
        assert result["primary_category"] == ""
        # Integer field should become -1
        assert result["library_work_published_year"] == -1
        # Count field should default to 0
        assert result["category_count"] == 0

    def test_none_section_becomes_empty_string(self) -> None:
        """Test that None section is converted to empty string."""
        chunk = {
            "article_title": "Test",
            "namespace": "Main",
            "section": None,
            "chunk_index": 0,
            "line_range": "1-1",
            "word_count": 10,
            "categories": [],
            "internal_links": [],
            "is_stub": False,
            "citation_needed_count": 0,
            "has_blockquote": False,
        }

        result = serialize_metadata(chunk)
        assert result["section"] == ""

    def test_empty_lists_serialize_to_json(self) -> None:
        """Test that empty lists are serialized as JSON arrays."""
        chunk = {
            "article_title": "Test",
            "namespace": "Main",
            "section": "Test",
            "chunk_index": 0,
            "line_range": "1-1",
            "word_count": 10,
            "categories": [],
            "internal_links": [],
            "is_stub": False,
            "citation_needed_count": 0,
            "has_blockquote": False,
        }

        result = serialize_metadata(chunk)
        assert result["categories"] == "[]"
        assert result["internal_links"] == "[]"


class TestDeserializeMetadata:
    """Tests for deserialize_metadata function."""

    def test_basic_deserialization(self) -> None:
        """Test basic metadata deserialization."""
        metadata = {
            "article_title": "Test Article",
            "namespace": "Main",
            "section": "Introduction",
            "chunk_index": 0,
            "line_range": "1-5",
            "word_count": 100,
            "categories": '["Testing", "Python"]',
            "internal_links": '["Unit testing", "PyTest"]',
            "is_stub": False,
            "citation_needed_count": 2,
            "has_blockquote": True,
        }

        result = deserialize_metadata(metadata)

        assert result["article_title"] == "Test Article"
        assert result["section"] == "Introduction"
        assert result["categories"] == ["Testing", "Python"]
        assert result["internal_links"] == ["Unit testing", "PyTest"]

    def test_empty_section_becomes_none(self) -> None:
        """Test that empty section is restored to None."""
        metadata = {
            "section": "",
            "categories": "[]",
            "internal_links": "[]",
        }

        result = deserialize_metadata(metadata)
        assert result["section"] is None

    def test_phase_b_fields_deserialization(self) -> None:
        """Test Phase B fields are properly deserialized."""
        metadata = {
            "article_title": "State and Revolution",
            "namespace": "Library",
            "section": "Chapter 1",
            "chunk_index": 0,
            "line_range": "1-50",
            "word_count": 350,
            "categories": '["Marxism-Leninism"]',
            "internal_links": "[]",
            "is_stub": False,
            "citation_needed_count": 0,
            "has_blockquote": False,
            "library_work_author": "Vladimir Lenin",
            "library_work_type": "Book",
            "library_work_published_year": 1917,
            "infobox_type": "",
            "political_orientation": "",
            "primary_category": "Marxism-Leninism",
            "category_count": 1,
        }

        result = deserialize_metadata(metadata)

        assert result["library_work_author"] == "Vladimir Lenin"
        assert result["library_work_type"] == "Book"
        assert result["library_work_published_year"] == 1917
        assert result["infobox_type"] is None  # "" → None
        assert result["political_orientation"] is None  # "" → None
        assert result["primary_category"] == "Marxism-Leninism"
        assert result["category_count"] == 1

    def test_phase_b_fields_empty_to_none(self) -> None:
        """Test that empty Phase B string fields become None."""
        metadata = {
            "library_work_author": "",
            "library_work_type": "",
            "library_work_published_year": -1,
            "infobox_type": "",
            "political_orientation": "",
            "primary_category": "",
            "categories": "[]",
            "internal_links": "[]",
        }

        result = deserialize_metadata(metadata)

        assert result["library_work_author"] is None
        assert result["library_work_type"] is None
        assert result["library_work_published_year"] is None  # -1 → None
        assert result["infobox_type"] is None
        assert result["political_orientation"] is None
        assert result["primary_category"] is None

    def test_roundtrip_serialization(self) -> None:
        """Test that serialize + deserialize preserves data."""
        original = {
            "article_title": "Roundtrip Test",
            "namespace": "Library",
            "section": "Chapter 1",
            "chunk_index": 5,
            "line_range": "100-150",
            "word_count": 500,
            "categories": ["History", "Economics"],
            "internal_links": ["Karl Marx", "Das Kapital"],
            "is_stub": True,
            "citation_needed_count": 3,
            "has_blockquote": False,
        }

        serialized = serialize_metadata(original)
        deserialized = deserialize_metadata(serialized)

        assert deserialized["article_title"] == original["article_title"]
        assert deserialized["namespace"] == original["namespace"]
        assert deserialized["section"] == original["section"]
        assert deserialized["categories"] == original["categories"]
        assert deserialized["internal_links"] == original["internal_links"]

    def test_roundtrip_phase_b_fields(self) -> None:
        """Test that Phase B fields survive serialize + deserialize roundtrip."""
        original = {
            "article_title": "What Is To Be Done?",
            "namespace": "Library",
            "section": "Introduction",
            "chunk_index": 0,
            "line_range": "1-100",
            "word_count": 800,
            "categories": ["Marxism-Leninism", "Party building"],
            "internal_links": ["RSDLP", "Iskra"],
            "is_stub": False,
            "citation_needed_count": 0,
            "has_blockquote": True,
            "library_work_author": "Vladimir Lenin",
            "library_work_type": "Pamphlet",
            "library_work_published_year": 1902,
            "infobox_type": None,
            "political_orientation": None,
            "primary_category": "Marxism-Leninism",
            "category_count": 2,
        }

        serialized = serialize_metadata(original)
        deserialized = deserialize_metadata(serialized)

        assert deserialized["library_work_author"] == original["library_work_author"]
        assert deserialized["library_work_type"] == original["library_work_type"]
        assert (
            deserialized["library_work_published_year"] == original["library_work_published_year"]
        )
        assert deserialized["infobox_type"] is None
        assert deserialized["political_orientation"] is None
        assert deserialized["primary_category"] == original["primary_category"]
        assert deserialized["category_count"] == original["category_count"]


# =============================================================================
# PROLEWIKIDB TESTS
# =============================================================================


class TestProleWikiDBInit:
    """Tests for ProleWikiDB initialization."""

    def test_creates_collection(self, db: ProleWikiDB) -> None:
        """Test that init creates a collection."""
        assert db.collection is not None
        assert db.count() == 0

    def test_uses_cosine_distance(self, db: ProleWikiDB) -> None:
        """Test that collection uses cosine distance metric."""
        metadata = db.collection.metadata
        assert metadata is not None
        assert metadata.get("hnsw:space") == "cosine"


class TestProleWikiDBLoadArticle:
    """Tests for loading articles into ChromaDB."""

    def test_load_sample_article(
        self,
        db: ProleWikiDB,
        sample_chunks_path: Path,
        sample_embeddings_path: Path,
    ) -> None:
        """Test loading the sample article from fixtures."""
        num_chunks = db.load_article(sample_chunks_path, sample_embeddings_path)

        assert num_chunks == 3
        assert db.count() == 3

    def test_load_creates_correct_ids(
        self,
        db: ProleWikiDB,
        sample_chunks_path: Path,
        sample_embeddings_path: Path,
    ) -> None:
        """Test that loaded chunks have correct IDs."""
        db.load_article(sample_chunks_path, sample_embeddings_path)

        # Retrieve all chunks
        results = db.collection.get(include=["metadatas"])
        ids = set(results["ids"])

        assert "Main/Test_Article#0" in ids
        assert "Main/Test_Article#1" in ids
        assert "Main/Test_Article#2" in ids

    def test_load_preserves_metadata(
        self,
        db: ProleWikiDB,
        sample_chunks_path: Path,
        sample_embeddings_path: Path,
    ) -> None:
        """Test that metadata is correctly stored."""
        db.load_article(sample_chunks_path, sample_embeddings_path)

        # Get first chunk
        result = db.collection.get(ids=["Main/Test_Article#0"], include=["metadatas"])
        metadata = result["metadatas"][0] if result["metadatas"] else {}

        assert metadata["article_title"] == "Test Article"
        assert metadata["namespace"] == "Main"
        assert metadata["chunk_index"] == 0

    def test_load_mismatched_counts_raises_error(
        self,
        db: ProleWikiDB,
        sample_chunks_path: Path,
        temp_db_path: Path,
    ) -> None:
        """Test that mismatched chunk/embedding counts raise ValueError."""
        # Create embeddings with wrong count (2 instead of 3)
        wrong_embeddings = np.random.rand(2, 1536).astype(np.float32)
        wrong_path = temp_db_path / "wrong.npy"
        np.save(wrong_path, wrong_embeddings)

        with pytest.raises(ValueError, match="doesn't match"):
            db.load_article(sample_chunks_path, wrong_path)


class TestProleWikiDBSearch:
    """Tests for semantic search functionality."""

    def test_search_returns_results(
        self,
        db: ProleWikiDB,
        sample_chunks_path: Path,
        sample_embeddings_path: Path,
    ) -> None:
        """Test that search returns results."""
        db.load_article(sample_chunks_path, sample_embeddings_path)

        # Use first embedding as query
        embeddings: NDArray[np.float32] = np.load(sample_embeddings_path)
        query = embeddings[0].tolist()

        results = db.search(query, limit=3)

        assert len(results) == 3
        assert results[0].chunk_id == "Main/Test_Article#0"  # Should match itself

    def test_search_with_limit(
        self,
        db: ProleWikiDB,
        sample_chunks_path: Path,
        sample_embeddings_path: Path,
    ) -> None:
        """Test that search respects limit parameter."""
        db.load_article(sample_chunks_path, sample_embeddings_path)

        embeddings: NDArray[np.float32] = np.load(sample_embeddings_path)
        query = embeddings[0].tolist()

        results = db.search(query, limit=1)
        assert len(results) == 1


class TestProleWikiDBGetArticleChunks:
    """Tests for retrieving article chunks."""

    def test_get_article_chunks(
        self,
        db: ProleWikiDB,
        sample_chunks_path: Path,
        sample_embeddings_path: Path,
    ) -> None:
        """Test retrieving all chunks for an article."""
        db.load_article(sample_chunks_path, sample_embeddings_path)

        chunks = db.get_article_chunks("Test Article", "Main")

        assert len(chunks) == 3
        # Should be sorted by chunk_index
        assert chunks[0]["chunk_index"] == 0
        assert chunks[1]["chunk_index"] == 1
        assert chunks[2]["chunk_index"] == 2

    def test_get_nonexistent_article(self, db: ProleWikiDB) -> None:
        """Test retrieving chunks for non-existent article returns empty list."""
        chunks = db.get_article_chunks("Nonexistent Article", "Main")
        assert chunks == []


class TestProleWikiDBDeleteCollection:
    """Tests for collection deletion."""

    def test_delete_collection(
        self,
        db: ProleWikiDB,
        sample_chunks_path: Path,
        sample_embeddings_path: Path,
    ) -> None:
        """Test that delete_collection clears all data."""
        db.load_article(sample_chunks_path, sample_embeddings_path)
        assert db.count() == 3

        db.delete_collection()
        assert db.count() == 0
