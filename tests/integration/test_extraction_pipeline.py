"""Integration tests for the extraction pipeline (TDD Green Phase).

These tests verify that the full extraction pipeline correctly:
1. Parses MediaWiki markup
2. Extracts infoboxes, citations, links, categories
3. Produces clean text with metadata

The pipeline module is now implemented - tests should PASS.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from pw_mcp.ingest.extraction import extract_article

if TYPE_CHECKING:
    from collections.abc import Callable


class TestExtractionPipelineIntegration:
    """Integration tests for full article extraction."""

    @pytest.mark.integration
    def test_extract_politician_article(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract all components from politician article."""
        # Load a complete article fixture
        article_text = load_fixture("infoboxes", "politician.txt")
        # Add some body content for a complete test
        article_text += (
            "\n\nAbraham Lincoln was a president.\n\n[[Category:Presidents of the United States]]"
        )
        result = extract_article(article_text)

        # Verify infobox extracted
        assert result.infobox is not None
        assert result.infobox.type == "politician"
        assert result.infobox.fields["name"] == "Abraham Lincoln"

        # Verify links extracted (from infobox)
        assert len(result.internal_links) > 0

        # Verify categories extracted
        assert len(result.categories) > 0
        assert "Presidents of the United States" in result.categories

        # Verify clean text produced
        assert "{{Infobox" not in result.clean_text
        assert "president" in result.clean_text.lower()

    @pytest.mark.integration
    def test_extract_article_without_infobox(self) -> None:
        """Should handle articles without infoboxes."""
        text = """'''Simple Article'''

This is a simple article about [[something]].

[[Category:Test]]
"""
        result = extract_article(text)
        assert result.infobox is None
        assert "simple article" in result.clean_text.lower()
        assert len(result.internal_links) >= 1

    @pytest.mark.integration
    def test_extract_library_article(self) -> None:
        """Should handle Library namespace articles (books)."""
        # Library articles often have {{Library work}} template
        # and are much longer (up to 46k lines)
        text = """== Chapter 1 ==

The history of all hitherto existing society is the history of class struggles.

== Chapter 2 ==

More content here.

[[Category:Marxist texts]]
"""
        result = extract_article(text, source_path="Library/Communist Manifesto.txt")
        assert result.namespace == "Library"
        assert "Chapter 1" in result.sections
        assert "Chapter 2" in result.sections


class TestCleanTextGeneration:
    """Tests for clean text output from extraction."""

    @pytest.mark.integration
    def test_clean_text_removes_infobox(self, politician_infobox: str) -> None:
        """Clean text should not contain infobox markup."""
        result = extract_article(politician_infobox + "\n\nArticle content.")
        assert "{{Infobox" not in result.clean_text
        assert "Article content" in result.clean_text

    @pytest.mark.integration
    def test_clean_text_removes_categories(self) -> None:
        """Clean text should not contain category links."""
        text = "Content.\n\n[[Category:Test Category]]"
        result = extract_article(text)
        assert "[[Category:" not in result.clean_text

    @pytest.mark.integration
    def test_clean_text_preserves_internal_links_as_text(self) -> None:
        """Clean text should convert [[links]] to plain text."""
        text = "See [[Article|the article]] for more."
        result = extract_article(text)
        assert "the article" in result.clean_text
        assert "[[" not in result.clean_text

    @pytest.mark.integration
    def test_clean_text_preserves_blockquotes(self) -> None:
        """Clean text should preserve blockquote content."""
        text = """Content before.

<blockquote>
This is an important quote.
</blockquote>

Content after."""
        result = extract_article(text)
        assert "important quote" in result.clean_text
        # Blockquotes should be preserved but perhaps reformatted


class TestMetadataExtraction:
    """Tests for metadata extraction from articles."""

    @pytest.mark.integration
    def test_extract_section_headers(self) -> None:
        """Should extract section headers (== Header ==)."""
        text = """== Introduction ==
Some text.

== History ==
More text.

=== Early Period ===
Details.
"""
        result = extract_article(text)
        sections = result.sections
        assert "Introduction" in sections
        assert "History" in sections
        assert "Early Period" in sections

    @pytest.mark.integration
    def test_extract_reference_sections(self) -> None:
        """Should identify reference sections for removal."""
        text = """Content.

== References ==
<references />

== See also ==
* [[Related Article]]
"""
        result = extract_article(text)
        # References section should be flagged/removed
        # See also links should be extracted
        assert "References" in result.sections
        assert "See also" in result.sections

    @pytest.mark.integration
    def test_count_references(self) -> None:
        """Should count number of references in article."""
        text = """
Text<ref>{{Citation|author=A}}</ref> and more<ref>{{Citation|author=B}}</ref>.
"""
        result = extract_article(text)
        assert result.reference_count == 2


class TestNamespaceHandling:
    """Tests for handling different ProleWiki namespaces."""

    @pytest.mark.integration
    def test_detect_main_namespace(self) -> None:
        """Should identify Main namespace articles."""
        # From file path: prolewiki-exports/Main/Article.txt
        text = "Simple article content."
        result = extract_article(text, source_path="Main/Article.txt")
        assert result.namespace == "Main"

    @pytest.mark.integration
    def test_detect_library_namespace(self) -> None:
        """Should identify Library namespace articles."""
        text = "Book content."
        result = extract_article(text, source_path="Library/Capital Vol1.txt")
        assert result.namespace == "Library"

    @pytest.mark.integration
    def test_detect_essays_namespace(self) -> None:
        """Should identify Essays namespace articles."""
        text = "Essay content."
        result = extract_article(text, source_path="Essays/On Imperialism.txt")
        assert result.namespace == "Essays"

    @pytest.mark.integration
    def test_detect_prolewiki_namespace(self) -> None:
        """Should identify ProleWiki (meta) namespace articles."""
        text = "Guidelines content."
        result = extract_article(text, source_path="ProleWiki/Guidelines.txt")
        assert result.namespace == "ProleWiki"


class TestArticleQualityFlags:
    """Tests for extracting article quality indicators."""

    @pytest.mark.integration
    def test_detect_stub_template(self) -> None:
        """Should detect {{Stub}} template."""
        text = "{{Stub}}Short article content."
        result = extract_article(text)
        assert result.is_stub is True

    @pytest.mark.integration
    def test_count_citation_needed(self) -> None:
        """Should count {{Citation needed}} templates."""
        text = """
Claim one.{{Citation needed}}
Claim two.{{Citation needed}}
Claim three is sourced.<ref>{{Citation|author=A}}</ref>
"""
        result = extract_article(text)
        assert result.citation_needed_count == 2

    @pytest.mark.integration
    def test_detect_article_with_blockquotes(self) -> None:
        """Should flag articles containing blockquotes."""
        text = """Content.
<blockquote>A quote.</blockquote>
More content."""
        result = extract_article(text)
        assert result.has_blockquote is True


class TestEdgeCases:
    """Tests for edge cases in extraction pipeline."""

    @pytest.mark.integration
    def test_empty_article(self) -> None:
        """Should handle empty article gracefully."""
        text = ""
        result = extract_article(text)
        assert result.clean_text == ""
        assert result.infobox is None

    @pytest.mark.integration
    def test_article_with_only_infobox(self, politician_infobox: str) -> None:
        """Should handle article with only infobox (no body text)."""
        result = extract_article(politician_infobox)
        assert result.infobox is not None
        assert result.clean_text.strip() == ""

    @pytest.mark.integration
    def test_malformed_templates(self) -> None:
        """Should handle malformed templates gracefully."""
        text = "{{Infobox unclosed|field=value"
        # Should not crash
        result = extract_article(text)
        assert result is not None

    @pytest.mark.integration
    def test_deeply_nested_templates(self) -> None:
        """Should handle deeply nested templates."""
        text = "{{A|{{B|{{C|value}}}}}}"
        result = extract_article(text)
        # Should not crash or hang
        assert result is not None


class TestFileBasedExtraction:
    """Tests for extracting from actual corpus files."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_extract_from_real_file(self, mediawiki_dir: Path) -> None:
        """Should extract from real prolewiki-exports file."""
        # This test uses actual corpus files if available
        corpus_path = Path("prolewiki-exports/Main/Abraham Lincoln.txt")
        if not corpus_path.exists():
            # Fall back to fixture
            fixture_path = mediawiki_dir / "infoboxes" / "politician.txt"
            if not fixture_path.exists():
                pytest.skip("Fixture not available")
            text = fixture_path.read_text()
            result = extract_article(text, source_path="Main/Abraham Lincoln.txt")
            assert result.infobox is not None
        else:
            text = corpus_path.read_text()
            result = extract_article(text)
            assert result.infobox is not None
