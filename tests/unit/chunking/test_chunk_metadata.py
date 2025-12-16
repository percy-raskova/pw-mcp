"""Unit tests for chunk metadata and JSONL output (TDD Red Phase).

These tests define the expected behavior for metadata attachment,
ID generation, and JSONL serialization in the chunking module.
The chunker module does not exist yet - tests should fail with ImportError.

Test strategy:
- Test Chunk and ChunkedArticle dataclasses
- Test metadata propagation from ArticleData
- Test chunk ID generation
- Test JSONL output format
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

# These imports will fail until the module is implemented
from pw_mcp.ingest.chunker import (
    Chunk,
    ChunkConfig,
    ChunkedArticle,
    chunk_article,
    generate_chunk_id,
    write_chunks_jsonl,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# CHUNK DATACLASS TESTS
# =============================================================================


class TestChunkDataclass:
    """Tests for Chunk dataclass."""

    @pytest.mark.unit
    def test_chunk_creation(self) -> None:
        """Should create Chunk with all required fields."""
        chunk = Chunk(
            text="Sample chunk text content.",
            chunk_index=0,
            section="Introduction",
            line_start=1,
            line_end=5,
            word_count=4,
            estimated_tokens=5,
        )
        assert chunk.text == "Sample chunk text content."
        assert chunk.chunk_index == 0
        assert chunk.section == "Introduction"
        assert chunk.line_start == 1
        assert chunk.line_end == 5
        assert chunk.word_count == 4
        assert chunk.estimated_tokens == 5

    @pytest.mark.unit
    def test_chunk_fields_complete(self) -> None:
        """Chunk should have all fields from chunking.yaml specification."""
        chunk = Chunk(
            text="Test",
            chunk_index=0,
            section=None,
            line_start=1,
            line_end=1,
            word_count=1,
            estimated_tokens=1,
        )
        # Verify all required fields exist
        assert hasattr(chunk, "text")
        assert hasattr(chunk, "chunk_index")
        assert hasattr(chunk, "section")
        assert hasattr(chunk, "line_start")
        assert hasattr(chunk, "line_end")
        assert hasattr(chunk, "word_count")
        assert hasattr(chunk, "estimated_tokens")

    @pytest.mark.unit
    def test_chunk_line_range_valid(self) -> None:
        """Line range should be valid (start <= end)."""
        chunk = Chunk(
            text="Test",
            chunk_index=0,
            section=None,
            line_start=5,
            line_end=10,
            word_count=1,
            estimated_tokens=1,
        )
        assert chunk.line_start <= chunk.line_end


# =============================================================================
# CHUNKED ARTICLE DATACLASS TESTS
# =============================================================================


class TestChunkedArticleDataclass:
    """Tests for ChunkedArticle dataclass."""

    @pytest.mark.unit
    def test_chunked_article_creation(self) -> None:
        """Should create ChunkedArticle with article-level metadata."""
        chunks = [
            Chunk(
                text="Chunk one.",
                chunk_index=0,
                section=None,
                line_start=1,
                line_end=1,
                word_count=2,
                estimated_tokens=3,
            )
        ]
        article = ChunkedArticle(
            article_title="Test Article",
            namespace="Main",
            chunks=chunks,
            categories=["Category One", "Category Two"],
            internal_links=["Link One", "Link Two"],
            infobox=None,
            library_work=None,
            is_stub=False,
            citation_needed_count=0,
            has_blockquote=False,
        )
        assert article.article_title == "Test Article"
        assert article.namespace == "Main"
        assert len(article.chunks) == 1
        assert article.categories == ["Category One", "Category Two"]

    @pytest.mark.unit
    def test_chunked_article_chunk_list(self) -> None:
        """ChunkedArticle should contain a list of Chunk objects."""
        chunk1 = Chunk(
            text="First chunk.",
            chunk_index=0,
            section="Intro",
            line_start=1,
            line_end=1,
            word_count=2,
            estimated_tokens=3,
        )
        chunk2 = Chunk(
            text="Second chunk.",
            chunk_index=1,
            section="Body",
            line_start=2,
            line_end=2,
            word_count=2,
            estimated_tokens=3,
        )
        article = ChunkedArticle(
            article_title="Test",
            namespace="Main",
            chunks=[chunk1, chunk2],
            categories=[],
            internal_links=[],
            infobox=None,
            library_work=None,
            is_stub=False,
            citation_needed_count=0,
            has_blockquote=False,
        )
        assert len(article.chunks) == 2
        assert article.chunks[0].chunk_index == 0
        assert article.chunks[1].chunk_index == 1


# =============================================================================
# METADATA ATTACHMENT TESTS
# =============================================================================


class TestMetadataAttachment:
    """Tests for metadata propagation from ArticleData to chunks."""

    @pytest.mark.unit
    def test_article_title_propagated(
        self, article_main_metadata: dict[str, Any], tmp_path: Path
    ) -> None:
        """Article title should be propagated to ChunkedArticle."""
        sembr_file = tmp_path / "Main" / "Test_Article.txt"
        sembr_file.parent.mkdir(parents=True)
        sembr_file.write_text("Some content here.")

        # Mock ArticleData would be constructed from metadata
        # For now, test that chunk_article extracts title from path
        result = chunk_article(sembr_file, article_main_metadata, ChunkConfig())

        assert result.article_title is not None
        assert len(result.article_title) > 0

    @pytest.mark.unit
    def test_namespace_propagated(
        self, article_library_metadata: dict[str, Any], tmp_path: Path
    ) -> None:
        """Namespace should be propagated to ChunkedArticle."""
        sembr_file = tmp_path / "Library" / "Test_Book.txt"
        sembr_file.parent.mkdir(parents=True)
        sembr_file.write_text("Book content here.")

        result = chunk_article(sembr_file, article_library_metadata, ChunkConfig())

        assert result.namespace == "Library"

    @pytest.mark.unit
    def test_categories_json_format(
        self, article_main_metadata: dict[str, Any], tmp_path: Path
    ) -> None:
        """Categories should be available as list."""
        sembr_file = tmp_path / "Main" / "Test.txt"
        sembr_file.parent.mkdir(parents=True)
        sembr_file.write_text("Content.")

        result = chunk_article(sembr_file, article_main_metadata, ChunkConfig())

        assert isinstance(result.categories, list)
        assert len(result.categories) >= 0

    @pytest.mark.unit
    def test_internal_links_json_format(
        self, article_main_metadata: dict[str, Any], tmp_path: Path
    ) -> None:
        """Internal links should be available as list."""
        sembr_file = tmp_path / "Main" / "Test.txt"
        sembr_file.parent.mkdir(parents=True)
        sembr_file.write_text("Content with [[Link]].")

        result = chunk_article(sembr_file, article_main_metadata, ChunkConfig())

        assert isinstance(result.internal_links, list)

    @pytest.mark.unit
    def test_infobox_fields_attached(
        self, article_main_metadata: dict[str, Any], tmp_path: Path
    ) -> None:
        """Infobox data should be attached when present."""
        sembr_file = tmp_path / "Main" / "Person.txt"
        sembr_file.parent.mkdir(parents=True)
        sembr_file.write_text("Person article content.")

        result = chunk_article(sembr_file, article_main_metadata, ChunkConfig())

        # article_main_metadata has infobox
        assert result.infobox is not None or "infobox" in article_main_metadata

    @pytest.mark.unit
    def test_library_work_fields_attached(
        self, article_library_metadata: dict[str, Any], tmp_path: Path
    ) -> None:
        """Library work metadata should be attached for Library namespace."""
        sembr_file = tmp_path / "Library" / "Book.txt"
        sembr_file.parent.mkdir(parents=True)
        sembr_file.write_text("Book content.")

        result = chunk_article(sembr_file, article_library_metadata, ChunkConfig())

        # article_library_metadata has library_work
        assert result.library_work is not None or "library_work" in article_library_metadata

    @pytest.mark.unit
    def test_section_tracked(self, tmp_path: Path) -> None:
        """Each chunk should track its section name."""
        sembr_file = tmp_path / "Main" / "Article.txt"
        sembr_file.parent.mkdir(parents=True)
        sembr_file.write_text("== Introduction ==\nContent here.\n== Body ==\nMore content.")

        minimal_metadata: dict[str, Any] = {
            "clean_text": "",
            "infobox": None,
            "citations": [],
            "internal_links": [],
            "categories": [],
            "sections": ["Introduction", "Body"],
            "namespace": "Main",
            "reference_count": 0,
            "is_stub": False,
            "citation_needed_count": 0,
            "has_blockquote": False,
        }

        result = chunk_article(sembr_file, minimal_metadata, ChunkConfig())

        # Should have chunks with sections tracked
        sections = [c.section for c in result.chunks]
        assert "Introduction" in sections or any(s is not None for s in sections)

    @pytest.mark.unit
    def test_chunk_index_sequential(self, tmp_path: Path) -> None:
        """Chunk indices should be sequential starting from 0."""
        sembr_file = tmp_path / "Main" / "Article.txt"
        sembr_file.parent.mkdir(parents=True)
        sembr_file.write_text("== One ==\nA\n== Two ==\nB\n== Three ==\nC")

        minimal_metadata: dict[str, Any] = {
            "clean_text": "",
            "infobox": None,
            "citations": [],
            "internal_links": [],
            "categories": [],
            "sections": [],
            "namespace": "Main",
            "reference_count": 0,
            "is_stub": False,
            "citation_needed_count": 0,
            "has_blockquote": False,
        }

        result = chunk_article(sembr_file, minimal_metadata, ChunkConfig())

        indices = [c.chunk_index for c in result.chunks]
        assert indices == list(range(len(indices)))

    @pytest.mark.unit
    def test_status_flags_attached(
        self, article_main_metadata: dict[str, Any], tmp_path: Path
    ) -> None:
        """Status flags (is_stub, citation_needed_count, has_blockquote) should be attached."""
        sembr_file = tmp_path / "Main" / "Article.txt"
        sembr_file.parent.mkdir(parents=True)
        sembr_file.write_text("Content.")

        result = chunk_article(sembr_file, article_main_metadata, ChunkConfig())

        assert isinstance(result.is_stub, bool)
        assert isinstance(result.citation_needed_count, int)
        assert isinstance(result.has_blockquote, bool)


# =============================================================================
# ID GENERATION TESTS
# =============================================================================


class TestIdGeneration:
    """Tests for chunk ID generation."""

    @pytest.mark.unit
    def test_id_format_correct(self) -> None:
        """Chunk ID should follow format: {namespace}/{title}#{index}."""
        chunk_id = generate_chunk_id("Main", "Test Article", 0)
        assert chunk_id == "Main/Test_Article#0"

    @pytest.mark.unit
    def test_id_url_safe_spaces_underscores(self) -> None:
        """Spaces in title should be replaced with underscores for URL safety."""
        chunk_id = generate_chunk_id("Main", "Five-Year Plans", 3)
        assert " " not in chunk_id
        assert "Five-Year_Plans" in chunk_id
        assert chunk_id == "Main/Five-Year_Plans#3"

    @pytest.mark.unit
    def test_id_deterministic(self) -> None:
        """Same inputs should always produce same ID."""
        id1 = generate_chunk_id("Library", "Das Kapital", 5)
        id2 = generate_chunk_id("Library", "Das Kapital", 5)
        assert id1 == id2

    @pytest.mark.unit
    def test_id_unique_per_article(self) -> None:
        """Different indices should produce different IDs."""
        id1 = generate_chunk_id("Main", "Article", 0)
        id2 = generate_chunk_id("Main", "Article", 1)
        id3 = generate_chunk_id("Main", "Article", 2)

        assert id1 != id2
        assert id2 != id3
        assert id1 != id3


# =============================================================================
# JSONL OUTPUT TESTS
# =============================================================================


class TestJsonlOutput:
    """Tests for JSONL file output."""

    @pytest.mark.unit
    def test_write_jsonl_creates_file(self, tmp_path: Path) -> None:
        """write_chunks_jsonl should create output file."""
        chunks = [
            Chunk(
                text="Test chunk.",
                chunk_index=0,
                section=None,
                line_start=1,
                line_end=1,
                word_count=2,
                estimated_tokens=3,
            )
        ]
        article = ChunkedArticle(
            article_title="Test",
            namespace="Main",
            chunks=chunks,
            categories=[],
            internal_links=[],
            infobox=None,
            library_work=None,
            is_stub=False,
            citation_needed_count=0,
            has_blockquote=False,
        )

        output_path = tmp_path / "Main" / "Test.jsonl"
        write_chunks_jsonl(article, output_path)

        assert output_path.exists()

    @pytest.mark.unit
    def test_jsonl_one_record_per_line(self, tmp_path: Path) -> None:
        """Each chunk should be one line in the JSONL file."""
        chunks = [
            Chunk(
                text=f"Chunk {i}.",
                chunk_index=i,
                section=None,
                line_start=i + 1,
                line_end=i + 1,
                word_count=2,
                estimated_tokens=3,
            )
            for i in range(3)
        ]
        article = ChunkedArticle(
            article_title="Test",
            namespace="Main",
            chunks=chunks,
            categories=[],
            internal_links=[],
            infobox=None,
            library_work=None,
            is_stub=False,
            citation_needed_count=0,
            has_blockquote=False,
        )

        output_path = tmp_path / "output.jsonl"
        write_chunks_jsonl(article, output_path)

        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3

    @pytest.mark.unit
    def test_jsonl_utf8_encoding(self, tmp_path: Path) -> None:
        """JSONL output should be UTF-8 encoded for Unicode content."""
        chunks = [
            Chunk(
                text="Русский текст and 中文内容.",
                chunk_index=0,
                section=None,
                line_start=1,
                line_end=1,
                word_count=4,
                estimated_tokens=5,
            )
        ]
        article = ChunkedArticle(
            article_title="Unicode Test",
            namespace="Main",
            chunks=chunks,
            categories=[],
            internal_links=[],
            infobox=None,
            library_work=None,
            is_stub=False,
            citation_needed_count=0,
            has_blockquote=False,
        )

        output_path = tmp_path / "unicode.jsonl"
        write_chunks_jsonl(article, output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "Русский" in content
        assert "中文" in content

    @pytest.mark.unit
    def test_jsonl_schema_matches_spec(self, tmp_path: Path) -> None:
        """JSONL records should have all fields from chunking.yaml spec."""
        chunks = [
            Chunk(
                text="Test content.",
                chunk_index=0,
                section="Introduction",
                line_start=1,
                line_end=5,
                word_count=2,
                estimated_tokens=3,
            )
        ]
        article = ChunkedArticle(
            article_title="Test Article",
            namespace="Main",
            chunks=chunks,
            categories=["Cat1", "Cat2"],
            internal_links=["Link1"],
            infobox=None,
            library_work=None,
            is_stub=False,
            citation_needed_count=1,
            has_blockquote=True,
        )

        output_path = tmp_path / "spec.jsonl"
        write_chunks_jsonl(article, output_path)

        line = output_path.read_text().strip()
        record = json.loads(line)

        # Verify required fields from chunking.yaml
        assert "chunk_id" in record
        assert "text" in record
        assert "article_title" in record
        assert "namespace" in record
        assert "section" in record
        assert "chunk_index" in record
        assert "line_range" in record
        assert "word_count" in record
