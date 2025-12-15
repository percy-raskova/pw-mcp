"""Unit tests for quote parser (TDD Red Phase).

These tests define the expected interface for the quote parser.
Tests will fail until the parser is implemented.

Quote templates in ProleWiki:
- {{Quote | "text" | attribution}} - Quoted text with source
- {{Quote | text}} - Quoted text without attribution
"""

from typing import TYPE_CHECKING

import pytest

from pw_mcp.ingest.parsers.quote import parse_quotes

if TYPE_CHECKING:
    from collections.abc import Callable


class TestQuoteDetection:
    """Tests for detecting {{Quote}} templates."""

    @pytest.mark.unit
    def test_detect_simple_quote(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should detect {{Quote}} template with attribution."""
        content = load_fixture("quotes", "simple_quote.txt")
        results = parse_quotes(content)
        assert len(results) == 1

    @pytest.mark.unit
    def test_detect_quote_without_attribution(
        self, load_fixture: "Callable[[str, str], str]"
    ) -> None:
        """Should detect {{Quote}} template without attribution."""
        content = load_fixture("quotes", "quote_no_attribution.txt")
        results = parse_quotes(content)
        assert len(results) == 1

    @pytest.mark.unit
    def test_no_quotes_returns_empty_list(self) -> None:
        """Should return empty list for text without quotes."""
        text = "This is plain text with no quote templates."
        results = parse_quotes(text)
        assert results == []


class TestQuoteTextExtraction:
    """Tests for extracting quote text from {{Quote}} templates."""

    @pytest.mark.unit
    def test_extract_quote_text(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract the quoted text."""
        content = load_fixture("quotes", "simple_quote.txt")
        results = parse_quotes(content)
        assert results[0].text is not None
        assert "revolutionary soul" in results[0].text

    @pytest.mark.unit
    def test_extract_quote_text_without_surrounding_quotes(
        self, load_fixture: "Callable[[str, str], str]"
    ) -> None:
        """Should strip surrounding quotation marks from text."""
        content = load_fixture("quotes", "simple_quote.txt")
        results = parse_quotes(content)
        # The text should not start/end with quotation marks
        assert not results[0].text.startswith('"')
        assert not results[0].text.endswith('"')

    @pytest.mark.unit
    def test_extract_multiline_quote_text(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract multiline quote text."""
        content = load_fixture("quotes", "multiline_quote.txt")
        results = parse_quotes(content)
        assert results[0].text is not None
        assert "systematic suppression" in results[0].text


class TestQuoteAttribution:
    """Tests for extracting attribution from {{Quote}} templates."""

    @pytest.mark.unit
    def test_extract_attribution(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract attribution from quote."""
        content = load_fixture("quotes", "simple_quote.txt")
        results = parse_quotes(content)
        assert results[0].attribution is not None
        assert "Lenin" in results[0].attribution
        assert "State and Revolution" in results[0].attribution

    @pytest.mark.unit
    def test_attribution_is_none_when_not_provided(
        self, load_fixture: "Callable[[str, str], str]"
    ) -> None:
        """Should set attribution to None when not provided."""
        content = load_fixture("quotes", "quote_no_attribution.txt")
        results = parse_quotes(content)
        assert results[0].attribution is None

    @pytest.mark.unit
    def test_extract_multiline_quote_attribution(
        self, load_fixture: "Callable[[str, str], str]"
    ) -> None:
        """Should extract attribution from multiline quote."""
        content = load_fixture("quotes", "multiline_quote.txt")
        results = parse_quotes(content)
        assert results[0].attribution is not None
        assert "Lenin" in results[0].attribution


class TestMultipleQuotes:
    """Tests for extracting multiple quotes from text."""

    @pytest.mark.unit
    def test_extract_multiple_quotes(self) -> None:
        """Should extract all quotes from text with multiple {{Quote}} templates."""
        text = """
        {{Quote | First quote text | Author One}}
        Some text in between.
        {{Quote | Second quote text | Author Two}}
        """
        results = parse_quotes(text)
        assert len(results) == 2
        assert "First quote" in results[0].text
        assert "Second quote" in results[1].text


class TestQuoteEdgeCases:
    """Tests for edge cases in quote parsing."""

    @pytest.mark.unit
    def test_quote_with_pipes_in_text(self) -> None:
        """Should handle quotes that might contain pipe characters in nested markup."""
        # This tests that the parser correctly identifies the boundary between
        # quote text and attribution even with complex content
        text = """{{Quote | The choice is simple | Karl Marx}}"""
        results = parse_quotes(text)
        assert len(results) == 1
        assert results[0].text == "The choice is simple"
        assert results[0].attribution == "Karl Marx"

    @pytest.mark.unit
    def test_quote_with_whitespace_trimmed(self) -> None:
        """Should trim whitespace from quote text and attribution."""
        text = """{{Quote |   Text with spaces   |   Author with spaces   }}"""
        results = parse_quotes(text)
        assert results[0].text == "Text with spaces"
        assert results[0].attribution == "Author with spaces"
