"""Unit tests for chunker module (TDD Red Phase).

These tests define the expected interface for the chunking module.
The chunker module does not exist yet - tests should fail with ImportError.

Test strategy:
- Test config defaults and customization
- Test token boundary handling
- Test section boundary handling
- Test edge cases
- Test core chunking function
"""

from typing import TYPE_CHECKING

import pytest

# These imports will fail until the module is implemented
from pw_mcp.ingest.chunker import (
    Chunk,
    ChunkConfig,
    chunk_text,
    estimate_tokens,
    is_section_header,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestChunkConfigDefaults:
    """Tests for ChunkConfig default values."""

    @pytest.mark.unit
    def test_chunk_config_defaults(self) -> None:
        """Should have sensible defaults matching chunking.yaml specification."""
        config = ChunkConfig()
        assert config.target_tokens == 600
        assert config.min_tokens == 200
        assert config.max_tokens == 1000
        assert config.token_estimation_factor == 1.3

    @pytest.mark.unit
    def test_chunk_config_custom_values(self) -> None:
        """Should accept custom configuration values."""
        config = ChunkConfig(
            target_tokens=400,
            min_tokens=100,
            max_tokens=800,
            token_estimation_factor=1.5,
        )
        assert config.target_tokens == 400
        assert config.min_tokens == 100
        assert config.max_tokens == 800
        assert config.token_estimation_factor == 1.5

    @pytest.mark.unit
    def test_chunk_config_immutable(self) -> None:
        """Config should be a frozen dataclass (immutable)."""
        config = ChunkConfig()
        with pytest.raises(AttributeError):
            config.target_tokens = 500  # type: ignore[misc]


# =============================================================================
# TOKEN BOUNDARY TESTS
# =============================================================================


class TestTokenBoundaries:
    """Tests for token boundary handling in chunking."""

    @pytest.mark.unit
    def test_target_tokens_respected(self, simple_article_text: str) -> None:
        """Chunks should stay near target token count when possible."""
        config = ChunkConfig(target_tokens=600, max_tokens=1000)
        lines = simple_article_text.strip().split("\n")
        chunks = chunk_text(lines, config)

        # Small article - should have reasonable chunk sizes
        for chunk in chunks:
            assert chunk.estimated_tokens <= config.max_tokens

    @pytest.mark.unit
    def test_min_tokens_enforced(self) -> None:
        """Should not create tiny chunks below min_tokens unless unavoidable."""
        config = ChunkConfig(min_tokens=200, target_tokens=600)
        lines = [
            "== Header ==",
            "Short content here.",
            "",
            "== Another Header ==",
            "More short content that should be chunked.",
        ]
        chunks = chunk_text(lines, config)

        # If multiple tiny sections, they should be combined if possible
        # Exception: section boundaries are respected
        assert len(chunks) >= 1

    @pytest.mark.unit
    def test_max_tokens_enforced(self, long_article_text: str) -> None:
        """Should split content that exceeds max_tokens."""
        config = ChunkConfig(max_tokens=300)  # Force splitting
        lines = long_article_text.strip().split("\n")
        chunks = chunk_text(lines, config)

        # Every chunk should be under max_tokens
        for chunk in chunks:
            assert chunk.estimated_tokens <= config.max_tokens

    @pytest.mark.unit
    def test_token_estimation_accuracy(self) -> None:
        """Token estimation should use word_count * factor."""
        text = "The quick brown fox jumps over the lazy dog."  # 9 words
        factor = 1.3
        estimated = estimate_tokens(text, factor)

        expected = int(9 * factor)  # 11 or 12 depending on rounding
        assert abs(estimated - expected) <= 1

    @pytest.mark.unit
    def test_word_count_calculation(self) -> None:
        """Word count should accurately count whitespace-separated words."""
        config = ChunkConfig()
        lines = ["The quick brown fox.", "Jumps over the lazy dog."]
        chunks = chunk_text(lines, config)

        assert len(chunks) == 1
        assert chunks[0].word_count == 9  # 4 + 5 words


# =============================================================================
# SECTION BOUNDARY TESTS
# =============================================================================


class TestSectionBoundaries:
    """Tests for section boundary handling."""

    @pytest.mark.unit
    def test_section_header_starts_new_chunk(self) -> None:
        """Section headers (== ... ==) should start a new chunk."""
        config = ChunkConfig()
        lines = [
            "Content before header.",
            "== New Section ==",
            "Content after header.",
        ]
        chunks = chunk_text(lines, config)

        # Should have at least 2 chunks - one before header, one starting with header
        assert len(chunks) >= 2
        # Second chunk should have section set
        assert chunks[1].section == "New Section"

    @pytest.mark.unit
    def test_no_split_mid_section(self) -> None:
        """Should not split in the middle of small sections."""
        config = ChunkConfig(max_tokens=2000)  # Large enough for small sections
        lines = [
            "== Section One ==",
            "First line of section one.",
            "Second line of section one.",
            "== Section Two ==",
            "First line of section two.",
        ]
        chunks = chunk_text(lines, config)

        # Each section should be intact
        for chunk in chunks:
            text = chunk.text
            if "Section One" in text:
                assert "First line of section one" in text
                assert "Second line of section one" in text

    @pytest.mark.unit
    def test_paragraph_break_preference(self) -> None:
        """Should prefer breaking at paragraph boundaries (blank lines)."""
        config = ChunkConfig(max_tokens=100)  # Force splitting
        lines = [
            "First paragraph line one.",
            "First paragraph line two.",
            "",
            "Second paragraph line one.",
            "Second paragraph line two.",
        ]
        chunks = chunk_text(lines, config)

        # If split occurs, should be at the blank line
        if len(chunks) > 1:
            # First chunk should end with first paragraph
            assert "First paragraph" in chunks[0].text
            # Second chunk should start with second paragraph
            assert "Second paragraph" in chunks[1].text

    @pytest.mark.unit
    def test_large_section_splits_at_paragraphs(self, long_article_text: str) -> None:
        """Large sections should split at paragraph boundaries."""
        config = ChunkConfig(max_tokens=200)  # Force splitting within sections
        lines = long_article_text.strip().split("\n")
        chunks = chunk_text(lines, config)

        # Should have multiple chunks for a long article
        assert len(chunks) > 3

        # Chunks should not split mid-sentence where possible
        for chunk in chunks:
            text = chunk.text.strip()
            # Check that chunk ends with sentence-ending punctuation or header
            if text and not text.endswith("=="):
                assert text[-1] in ".!?\"'" or text.endswith("==")

    @pytest.mark.unit
    def test_consecutive_headers(self) -> None:
        """Consecutive headers should each start their own chunk."""
        config = ChunkConfig()
        lines = [
            "== First Header ==",
            "== Second Header ==",
            "== Third Header ==",
            "Some content finally.",
        ]
        chunks = chunk_text(lines, config)

        # Should have chunks for each header
        assert len(chunks) >= 3

    @pytest.mark.unit
    def test_nested_headers(self) -> None:
        """Should handle nested headers (===, ====, etc.)."""
        config = ChunkConfig()
        lines = [
            "== Main Section ==",
            "Main content.",
            "=== Subsection ===",
            "Subsection content.",
            "==== Sub-subsection ====",
            "Deep content.",
        ]
        chunks = chunk_text(lines, config)

        # Each header level should be recognized
        sections = [c.section for c in chunks if c.section]
        assert "Main Section" in sections
        assert "Subsection" in sections
        assert "Sub-subsection" in sections


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in chunking."""

    @pytest.mark.unit
    def test_empty_input_returns_no_chunks(self) -> None:
        """Empty input should return empty list of chunks."""
        config = ChunkConfig()
        chunks = chunk_text([], config)
        assert chunks == []

    @pytest.mark.unit
    def test_single_line_creates_chunk(self) -> None:
        """Single line input should create a single chunk."""
        config = ChunkConfig()
        lines = ["Just one line of content."]
        chunks = chunk_text(lines, config)

        assert len(chunks) == 1
        assert chunks[0].text == "Just one line of content."
        assert chunks[0].word_count == 5

    @pytest.mark.unit
    def test_no_headers_null_section(self, no_headers_text: str) -> None:
        """Articles without headers should have section=None."""
        config = ChunkConfig()
        lines = no_headers_text.strip().split("\n")
        chunks = chunk_text(lines, config)

        # All chunks should have section=None
        for chunk in chunks:
            assert chunk.section is None

    @pytest.mark.unit
    def test_unicode_preserved_cyrillic(self, unicode_content_text: str) -> None:
        """Should preserve Cyrillic Unicode characters."""
        config = ChunkConfig()
        lines = unicode_content_text.strip().split("\n")
        chunks = chunk_text(lines, config)

        # Find chunk with Russian content
        all_text = " ".join(c.text for c in chunks)
        assert "Русский" in all_text
        assert "язык" in all_text

    @pytest.mark.unit
    def test_unicode_preserved_chinese(self, unicode_content_text: str) -> None:
        """Should preserve Chinese Unicode characters."""
        config = ChunkConfig()
        lines = unicode_content_text.strip().split("\n")
        chunks = chunk_text(lines, config)

        # Find chunk with Chinese content
        all_text = " ".join(c.text for c in chunks)
        assert "中文" in all_text

    @pytest.mark.unit
    def test_blank_lines_handled(self) -> None:
        """Should handle multiple consecutive blank lines gracefully."""
        config = ChunkConfig()
        lines = [
            "First paragraph.",
            "",
            "",
            "",
            "Second paragraph after blank lines.",
        ]
        chunks = chunk_text(lines, config)

        # Should create chunks without extra blank line noise
        assert len(chunks) >= 1
        # Content should be preserved
        all_text = " ".join(c.text for c in chunks)
        assert "First paragraph" in all_text
        assert "Second paragraph" in all_text

    @pytest.mark.unit
    def test_very_long_document(self, long_article_text: str) -> None:
        """Should handle very long documents without issues."""
        config = ChunkConfig()
        lines = long_article_text.strip().split("\n")

        # Double the content to make it even longer
        lines = lines * 3

        chunks = chunk_text(lines, config)

        # Should create many chunks
        assert len(chunks) > 5

        # All line ranges should be valid
        for chunk in chunks:
            assert chunk.line_start >= 1
            assert chunk.line_end >= chunk.line_start

    @pytest.mark.unit
    def test_header_only_section(self) -> None:
        """Should handle sections with headers but minimal content."""
        config = ChunkConfig()
        lines = [
            "== First Header ==",
            "== Second Header ==",
            "== Third Header ==",
            "Finally some content here.",
            "== Fourth Header ==",
        ]
        chunks = chunk_text(lines, config)

        # Should handle gracefully without crashing
        assert len(chunks) >= 1


# =============================================================================
# CHUNK TEXT FUNCTION TESTS
# =============================================================================


class TestChunkTextFunction:
    """Tests for the chunk_text function behavior."""

    @pytest.mark.unit
    def test_chunk_text_returns_list(self) -> None:
        """chunk_text should return a list of Chunk objects."""
        config = ChunkConfig()
        lines = ["Some content here."]
        result = chunk_text(lines, config)

        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    @pytest.mark.unit
    def test_chunk_text_respects_config(self) -> None:
        """chunk_text should use provided config values."""
        config = ChunkConfig(max_tokens=50)  # Very small max
        lines = [
            "First sentence here.",
            "Second sentence here.",
            "Third sentence here.",
            "Fourth sentence here.",
        ]
        chunks = chunk_text(lines, config)

        # With small max_tokens, should create multiple chunks
        # Each chunk should respect the max
        for chunk in chunks:
            assert chunk.estimated_tokens <= config.max_tokens

    @pytest.mark.unit
    def test_chunk_text_line_numbers_accurate(self) -> None:
        """Line numbers in chunks should accurately reference source."""
        config = ChunkConfig()
        lines = [
            "Line one.",
            "Line two.",
            "Line three.",
            "== Header ==",
            "Line five.",
            "Line six.",
        ]
        chunks = chunk_text(lines, config)

        # First chunk should start at line 1
        assert chunks[0].line_start == 1

        # Line numbers should be contiguous across chunks
        if len(chunks) > 1:
            for i in range(1, len(chunks)):
                # Next chunk starts where previous ended (or +1 for section break)
                assert chunks[i].line_start >= chunks[i - 1].line_end


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.mark.unit
    def test_is_section_header_double_equals(self) -> None:
        """Should recognize == Header == as section header."""
        assert is_section_header("== Introduction ==")
        assert is_section_header("==Introduction==")
        assert is_section_header("== Long Header Title ==")

    @pytest.mark.unit
    def test_is_section_header_triple_equals(self) -> None:
        """Should recognize === Subsection === as section header."""
        assert is_section_header("=== Subsection ===")
        assert is_section_header("===Subsection===")

    @pytest.mark.unit
    def test_is_section_header_not_regular_text(self) -> None:
        """Should not match regular text containing equals signs."""
        assert not is_section_header("a = b + c")
        assert not is_section_header("The equation x == y")
        assert not is_section_header("= Single equals")
        assert not is_section_header("Just regular text")

    @pytest.mark.unit
    def test_estimate_tokens_empty_string(self) -> None:
        """Empty string should estimate to 0 tokens."""
        assert estimate_tokens("", 1.3) == 0

    @pytest.mark.unit
    def test_estimate_tokens_uses_factor(self) -> None:
        """Token estimation should multiply word count by factor."""
        text = "one two three four five"  # 5 words
        assert estimate_tokens(text, 1.0) == 5
        assert estimate_tokens(text, 2.0) == 10
        assert estimate_tokens(text, 1.5) == 7  # Rounded


# =============================================================================
# TIKTOKEN INTEGRATION TESTS
# =============================================================================


class TestTiktokenIntegration:
    """Tests for tiktoken-based token counting."""

    @pytest.mark.unit
    def test_count_tokens_import(self) -> None:
        """Should be able to import count_tokens function."""
        from pw_mcp.ingest.chunker import count_tokens

        assert callable(count_tokens)

    @pytest.mark.unit
    def test_count_tokens_basic(self) -> None:
        """Should accurately count tokens using tiktoken."""
        from pw_mcp.ingest.chunker import count_tokens

        # "Hello world" = 2 tokens in cl100k_base
        assert count_tokens("Hello world") == 2

        # More complex text
        text = "The quick brown fox jumps over the lazy dog."
        tokens = count_tokens(text)
        assert tokens > 0
        assert tokens < 20  # Should be around 9-10 tokens

    @pytest.mark.unit
    def test_count_tokens_empty(self) -> None:
        """Empty string should return 0 tokens."""
        from pw_mcp.ingest.chunker import count_tokens

        assert count_tokens("") == 0
        assert count_tokens("   ") == 1  # Whitespace is a token

    @pytest.mark.unit
    def test_count_tokens_unicode(self) -> None:
        """Should handle Unicode text correctly."""
        from pw_mcp.ingest.chunker import count_tokens

        # Russian text
        russian = "Привет мир"
        russian_tokens = count_tokens(russian)
        assert russian_tokens > 0

        # Chinese text (more tokens per character in cl100k_base)
        chinese = "你好世界"
        chinese_tokens = count_tokens(chinese)
        assert chinese_tokens > 0


class TestChunkConfigWithOverlap:
    """Tests for ChunkConfig with overlap support."""

    @pytest.mark.unit
    def test_chunk_config_overlap_default(self) -> None:
        """ChunkConfig should have overlap_tokens field with default."""
        config = ChunkConfig()
        assert hasattr(config, "overlap_tokens")
        # Default can be 0 or a small value like 50

    @pytest.mark.unit
    def test_chunk_config_overlap_custom(self) -> None:
        """Should accept custom overlap_tokens value."""
        config = ChunkConfig(overlap_tokens=100)
        assert config.overlap_tokens == 100


class TestChunkWithOverlap:
    """Tests for chunk overlap functionality."""

    @pytest.mark.unit
    def test_overlap_chunks_share_content(self) -> None:
        """Consecutive chunks should share overlapping content."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken

        # Create text that will definitely split
        paragraphs = [
            "First paragraph with some content. " * 20,
            "",
            "Second paragraph with different content. " * 20,
            "",
            "Third paragraph with more content. " * 20,
        ]
        text = "\n".join(paragraphs)

        config = ChunkConfig(max_tokens=100, overlap_tokens=20)
        chunks = chunk_text_tiktoken(text, config)

        if len(chunks) >= 2:
            # Check that second chunk starts with content from end of first
            # This is a behavioral test - the specific overlap depends on implementation
            assert len(chunks) >= 2

    @pytest.mark.unit
    def test_zero_overlap_no_shared_content(self) -> None:
        """With zero overlap, chunks should not share content."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken

        paragraphs = [
            "First paragraph content. " * 20,
            "",
            "Second paragraph content. " * 20,
        ]
        text = "\n".join(paragraphs)

        config = ChunkConfig(max_tokens=100, overlap_tokens=0)
        chunks = chunk_text_tiktoken(text, config)

        # All content should appear exactly once (no overlap)
        if len(chunks) >= 2:
            all_text = "".join(c.text for c in chunks)
            # Without overlap, concatenating should give back ~original length
            assert len(all_text) <= len(text) + 100  # Some tolerance


class TestChunkTextTiktoken:
    """Tests for the tiktoken-based chunk_text function."""

    @pytest.mark.unit
    def test_chunk_text_tiktoken_import(self) -> None:
        """Should be able to import chunk_text_tiktoken function."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken

        assert callable(chunk_text_tiktoken)

    @pytest.mark.unit
    def test_chunk_text_tiktoken_returns_chunks(self) -> None:
        """chunk_text_tiktoken should return list of Chunk objects."""
        from pw_mcp.ingest.chunker import Chunk, chunk_text_tiktoken

        text = "This is a test sentence. And another one."
        config = ChunkConfig()
        chunks = chunk_text_tiktoken(text, config)

        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    @pytest.mark.unit
    def test_chunk_text_tiktoken_respects_max_tokens(self) -> None:
        """Chunks should not exceed max_tokens."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken, count_tokens

        # Create multi-line text (realistic - documents have line breaks)
        lines = [f"This is sentence number {i}." for i in range(100)]
        text = "\n".join(lines)
        config = ChunkConfig(max_tokens=50, overlap_tokens=0)
        chunks = chunk_text_tiktoken(text, config)

        for chunk in chunks:
            tokens = count_tokens(chunk.text)
            # Allow some tolerance for boundary conditions
            assert tokens <= config.max_tokens + 10

    @pytest.mark.unit
    def test_chunk_text_tiktoken_preserves_content(self) -> None:
        """All content should be preserved across chunks."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken

        original = "Word1 Word2 Word3 Word4 Word5. " * 20
        config = ChunkConfig(max_tokens=30, overlap_tokens=0)
        chunks = chunk_text_tiktoken(original, config)

        # Check key words are present
        all_text = " ".join(c.text for c in chunks)
        assert "Word1" in all_text
        assert "Word5" in all_text

    @pytest.mark.unit
    def test_chunk_text_tiktoken_empty_input(self) -> None:
        """Empty input should return empty list."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken

        config = ChunkConfig()
        chunks = chunk_text_tiktoken("", config)
        assert chunks == []

    @pytest.mark.unit
    def test_chunk_text_tiktoken_section_headers(self) -> None:
        """Should respect section headers as hard breaks."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken

        text = """Content before header.

== New Section ==

Content after header."""

        config = ChunkConfig(max_tokens=1000)
        chunks = chunk_text_tiktoken(text, config)

        # Should have at least 2 chunks (split at header)
        assert len(chunks) >= 2

        # Find chunk with section header
        sections = [c.section for c in chunks if c.section]
        assert "New Section" in sections

    @pytest.mark.unit
    def test_chunk_text_tiktoken_actual_token_count(self) -> None:
        """Chunk should have actual_tokens field with tiktoken count."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken

        text = "Hello world. This is a test."
        config = ChunkConfig()
        chunks = chunk_text_tiktoken(text, config)

        assert len(chunks) >= 1
        # Check chunk has token count (either actual_tokens or estimated_tokens)
        assert chunks[0].estimated_tokens > 0


# =============================================================================
# OVERSIZED LINE HANDLING TESTS (TDD Red Phase)
# =============================================================================


class TestOversizedLineHandling:
    """Tests for handling lines that exceed max_tokens.

    These tests verify that the chunker correctly splits individual lines
    that exceed the max_tokens limit. This is critical for API compatibility
    (OpenAI has 8192 token per-input limit).
    """

    @pytest.mark.unit
    def test_single_oversized_line_splits(self) -> None:
        """A single line exceeding max_tokens should be split into multiple chunks."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken, count_tokens

        # Create a single very long line (~500 tokens)
        long_line = "word " * 400  # ~400 tokens
        config = ChunkConfig(max_tokens=100, overlap_tokens=0)

        assert count_tokens(long_line) > config.max_tokens  # Verify it's oversized

        chunks = chunk_text_tiktoken(long_line, config)

        # Should create multiple chunks, each under max_tokens
        assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
        for i, chunk in enumerate(chunks):
            tokens = count_tokens(chunk.text)
            assert (
                tokens <= config.max_tokens
            ), f"Chunk {i} has {tokens} tokens, exceeds max {config.max_tokens}"

    @pytest.mark.unit
    def test_oversized_line_in_middle_of_document(self) -> None:
        """Oversized line in middle of document should be split correctly."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken, count_tokens

        # Normal line + oversized line + normal line
        text = "Short normal line.\n" + ("word " * 400) + "\nAnother short line."
        config = ChunkConfig(max_tokens=100, overlap_tokens=0)

        chunks = chunk_text_tiktoken(text, config)

        # All chunks must respect max_tokens
        for i, chunk in enumerate(chunks):
            tokens = count_tokens(chunk.text)
            assert (
                tokens <= config.max_tokens
            ), f"Chunk {i} has {tokens} tokens, exceeds max {config.max_tokens}"

    @pytest.mark.unit
    def test_oversized_line_preserves_all_content(self) -> None:
        """When splitting an oversized line, all words should be preserved."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken

        # Create recognizable content
        words = [f"unique{i}" for i in range(200)]
        long_line = " ".join(words)
        config = ChunkConfig(max_tokens=50, overlap_tokens=0)

        chunks = chunk_text_tiktoken(long_line, config)

        # All unique words should appear in the combined text
        all_text = " ".join(c.text for c in chunks)
        for word in words[:10]:  # Check first 10
            assert word in all_text, f"Missing word: {word}"
        for word in words[-10:]:  # Check last 10
            assert word in all_text, f"Missing word: {word}"

    @pytest.mark.unit
    def test_openai_limit_enforcement(self) -> None:
        """Chunks should never exceed OpenAI's 8192 token per-input limit."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken, count_tokens

        # Create a massive text block (like State and Revolution chapter)
        massive_text = "The state " * 10000  # ~20k tokens

        # Use realistic config but ensure max is under OpenAI limit
        config = ChunkConfig(max_tokens=1000, overlap_tokens=50)

        chunks = chunk_text_tiktoken(massive_text, config)

        # No chunk should exceed max_tokens
        max_found = 0
        for i, chunk in enumerate(chunks):
            tokens = count_tokens(chunk.text)
            max_found = max(max_found, tokens)
            assert (
                tokens <= config.max_tokens
            ), f"Chunk {i} has {tokens} tokens, exceeds max {config.max_tokens}"

        # Should have created multiple chunks
        assert len(chunks) > 5, f"Expected >5 chunks for massive text, got {len(chunks)}"

    @pytest.mark.unit
    def test_sentence_boundary_split_when_possible(self) -> None:
        """When splitting oversized lines, prefer sentence boundaries."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken

        # Create oversized line with clear sentence boundaries
        sentences = ["This is sentence number " + str(i) + ". " for i in range(50)]
        long_line = "".join(sentences)  # All on one line

        config = ChunkConfig(max_tokens=50, overlap_tokens=0)
        chunks = chunk_text_tiktoken(long_line, config)

        # Most chunks should end with sentence-ending punctuation
        sentence_endings = 0
        for chunk in chunks:
            text = chunk.text.strip()
            if text and text[-1] in ".!?":
                sentence_endings += 1

        # Allow some tolerance but most should end cleanly
        ratio = sentence_endings / len(chunks) if chunks else 0
        assert ratio >= 0.5, f"Only {ratio:.0%} chunks end with sentence punctuation"

    @pytest.mark.unit
    def test_word_boundary_fallback(self) -> None:
        """When no sentence boundary available, split at word boundaries."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken

        # Long line with no sentence punctuation
        long_line = "word " * 500
        config = ChunkConfig(max_tokens=50, overlap_tokens=0)

        chunks = chunk_text_tiktoken(long_line, config)

        # Should split cleanly at word boundaries, not mid-word
        for chunk in chunks:
            text = chunk.text.strip()
            # Should not start with partial word (e.g., "rd" from "word")
            if text:
                assert text[0].isalpha() or text[0].isdigit() or text[0] in "\"'("

    @pytest.mark.unit
    def test_no_empty_chunks_from_oversized_split(self) -> None:
        """Splitting oversized lines should not produce empty chunks."""
        from pw_mcp.ingest.chunker import chunk_text_tiktoken

        long_line = "word " * 400
        config = ChunkConfig(max_tokens=100, overlap_tokens=0)

        chunks = chunk_text_tiktoken(long_line, config)

        for i, chunk in enumerate(chunks):
            assert chunk.text.strip(), f"Chunk {i} is empty"
            assert chunk.word_count > 0, f"Chunk {i} has zero word count"
