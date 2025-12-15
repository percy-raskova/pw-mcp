"""Unit tests for citation parser (TDD Green Phase).

These tests define the expected interface for the citation parser.
Tests now call the implemented parser and verify behavior.

Citation types from ProleWiki:
- {{Citation|...}} - Book citations with author, year, title, chapter, page, etc.
- {{Web citation|...}} - Web articles with url, newspaper, date, author
- {{News citation|...}} - News articles
- {{YouTube citation|...}} - YouTube videos with channel, url, quote
- {{Library citation|...}} - Links to Library namespace articles
"""

from typing import TYPE_CHECKING

import pytest

from pw_mcp.ingest.parsers.citation import parse_citations

if TYPE_CHECKING:
    from collections.abc import Callable


class TestCitationDetection:
    """Tests for detecting citation templates and their types."""

    @pytest.mark.unit
    def test_detect_book_citation(self, citation_template: str) -> None:
        """Should detect {{Citation}} template."""
        results = parse_citations(citation_template)
        assert len(results) == 1
        assert results[0].type == "book"

    @pytest.mark.unit
    def test_detect_web_citation(self, web_citation: str) -> None:
        """Should detect {{Web citation}} template."""
        results = parse_citations(web_citation)
        assert len(results) == 1
        assert results[0].type == "web"

    @pytest.mark.unit
    def test_detect_news_citation(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should detect {{News citation}} template."""
        content = load_fixture("citations", "news_citation.txt")
        results = parse_citations(content)
        assert len(results) == 1
        assert results[0].type == "news"

    @pytest.mark.unit
    def test_detect_youtube_citation(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should detect {{YouTube citation}} template."""
        content = load_fixture("citations", "youtube_citation.txt")
        results = parse_citations(content)
        assert len(results) == 1
        assert results[0].type == "youtube"

    @pytest.mark.unit
    def test_detect_library_citation(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should detect {{Library citation}} template."""
        content = load_fixture("citations", "library_citation.txt")
        results = parse_citations(content)
        assert len(results) == 1
        assert results[0].type == "library"

    @pytest.mark.unit
    def test_no_citations_returns_empty_list(self) -> None:
        """Should return empty list for text without citations."""
        text = "This is plain text with no citation templates."
        results = parse_citations(text)
        assert results == []


class TestBookCitationFields:
    """Tests for extracting fields from {{Citation}} templates."""

    @pytest.mark.unit
    def test_extract_author(self, citation_template: str) -> None:
        """Should extract author field, handling [[links]]."""
        results = parse_citations(citation_template)
        assert results[0].author == "Domenico Losurdo"

    @pytest.mark.unit
    def test_extract_year(self, citation_template: str) -> None:
        """Should extract year field."""
        results = parse_citations(citation_template)
        assert results[0].year == "2011"

    @pytest.mark.unit
    def test_extract_title(self, citation_template: str) -> None:
        """Should extract title field."""
        results = parse_citations(citation_template)
        assert results[0].title == "Liberalism: A Counter-History"

    @pytest.mark.unit
    def test_extract_chapter(self, citation_template: str) -> None:
        """Should extract chapter field."""
        results = parse_citations(citation_template)
        assert results[0].chapter is not None
        assert "Liberalism and Racial Slavery" in results[0].chapter

    @pytest.mark.unit
    def test_extract_page(self, citation_template: str) -> None:
        """Should extract page number."""
        results = parse_citations(citation_template)
        assert results[0].page == "55"

    @pytest.mark.unit
    def test_extract_publisher(self, citation_template: str) -> None:
        """Should extract publisher."""
        results = parse_citations(citation_template)
        assert results[0].publisher == "Verso"

    @pytest.mark.unit
    def test_extract_isbn(self, citation_template: str) -> None:
        """Should extract ISBN."""
        results = parse_citations(citation_template)
        assert results[0].isbn == "9781844676934"

    @pytest.mark.unit
    def test_extract_urls(self, citation_template: str) -> None:
        """Should extract lg and pdf URLs."""
        results = parse_citations(citation_template)
        assert results[0].lg_url is not None
        assert "libgen.rs" in results[0].lg_url
        assert results[0].pdf_url is not None
        assert "cloudflare-ipfs.com" in results[0].pdf_url


class TestWebCitationFields:
    """Tests for extracting fields from {{Web citation}} templates."""

    @pytest.mark.unit
    def test_extract_web_author(self, web_citation: str) -> None:
        """Should extract author from web citation."""
        results = parse_citations(web_citation)
        assert results[0].author == "Ed Rampell"

    @pytest.mark.unit
    def test_extract_newspaper(self, web_citation: str) -> None:
        """Should extract newspaper field, resolving [[links]]."""
        results = parse_citations(web_citation)
        assert results[0].newspaper == "CovertAction Magazine"

    @pytest.mark.unit
    def test_extract_web_title(self, web_citation: str) -> None:
        """Should extract article title."""
        results = parse_citations(web_citation)
        assert results[0].title is not None
        assert "Oliver Stone" in results[0].title

    @pytest.mark.unit
    def test_extract_web_date(self, web_citation: str) -> None:
        """Should extract date in ISO format."""
        results = parse_citations(web_citation)
        assert results[0].date == "2022-03-18"

    @pytest.mark.unit
    def test_extract_url(self, web_citation: str) -> None:
        """Should extract URL."""
        results = parse_citations(web_citation)
        assert results[0].url is not None
        assert "covertactionmagazine.com" in results[0].url


class TestNewsCitationFields:
    """Tests for extracting fields from {{News citation}} templates."""

    @pytest.mark.unit
    def test_extract_news_newspaper(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract newspaper from news citation."""
        content = load_fixture("citations", "news_citation.txt")
        results = parse_citations(content)
        assert results[0].newspaper == "TeleSur"

    @pytest.mark.unit
    def test_extract_retrieved_date(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract retrieved date."""
        content = load_fixture("citations", "news_citation.txt")
        results = parse_citations(content)
        assert results[0].retrieved == "2022-04-23"


class TestYouTubeCitationFields:
    """Tests for extracting fields from {{YouTube citation}} templates."""

    @pytest.mark.unit
    def test_extract_channel(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract YouTube channel name."""
        content = load_fixture("citations", "youtube_citation.txt")
        results = parse_citations(content)
        assert results[0].channel == "The Kavernacle"

    @pytest.mark.unit
    def test_extract_youtube_url(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract YouTube URL with timestamp."""
        content = load_fixture("citations", "youtube_citation.txt")
        results = parse_citations(content)
        assert results[0].url is not None
        assert "youtube.com" in results[0].url
        assert "t=1680s" in results[0].url

    @pytest.mark.unit
    def test_extract_quote(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract quote field."""
        content = load_fixture("citations", "youtube_citation.txt")
        results = parse_citations(content)
        assert results[0].quote is not None
        assert "I don't want people" in results[0].quote


class TestLibraryCitationFields:
    """Tests for extracting fields from {{Library citation}} templates."""

    @pytest.mark.unit
    def test_extract_library_link(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract link to Library namespace article."""
        content = load_fixture("citations", "library_citation.txt")
        results = parse_citations(content)
        assert results[0].link == "Ancient Civilisations of East and West"


class TestMultipleCitations:
    """Tests for extracting multiple citations from text."""

    @pytest.mark.unit
    def test_extract_multiple_citations_from_article(self) -> None:
        """Should extract all citations from article text."""
        text = """
        Some text.<ref>{{Citation|author=Author1|title=Title1}}</ref>
        More text.<ref>{{Web citation|url=http://example.com|title=Example}}</ref>
        Final text.<ref name=":0">{{Citation|author=Author2|title=Title2}}</ref>
        """
        results = parse_citations(text)
        assert len(results) == 3

    @pytest.mark.unit
    def test_handle_named_refs(self) -> None:
        """Should handle named refs and ref reuse."""
        text = """
        First use.<ref name=":0">{{Citation|author=Test|title=Test}}</ref>
        Second use.<ref name=":0" />
        """
        results = parse_citations(text)
        # Should return 1 unique citation, not 2
        assert len(results) == 1


class TestCitationEdgeCases:
    """Tests for edge cases in citation parsing."""

    @pytest.mark.unit
    def test_citation_with_linked_author(self, citation_template: str) -> None:
        """Should strip [[]] from author name."""
        # Author is [[Domenico Losurdo]] in template
        results = parse_citations(citation_template)
        assert results[0].author == "Domenico Losurdo"
        assert "[[" not in results[0].author

    @pytest.mark.unit
    def test_citation_with_linked_newspaper(self, web_citation: str) -> None:
        """Should strip [[]] from newspaper name."""
        # Newspaper is [[CovertAction Magazine]] in template
        results = parse_citations(web_citation)
        assert results[0].newspaper == "CovertAction Magazine"

    @pytest.mark.unit
    def test_citation_with_multiline_quote(self) -> None:
        """Should handle quotes spanning multiple lines."""
        text = """{{Web citation|title=Test|quote=This is a quote
that spans multiple lines
with line breaks.}}"""
        results = parse_citations(text)
        assert results[0].quote is not None
        assert "multiple lines" in results[0].quote

    @pytest.mark.unit
    def test_citation_with_special_characters_in_url(self) -> None:
        """Should preserve special characters in URLs."""
        text = """{{Web citation|url=https://example.com/path?param=value&other=123%20test}}"""
        results = parse_citations(text)
        assert results[0].url is not None
        assert "param=value" in results[0].url
        assert "123%20test" in results[0].url


class TestVideoCitationDetection:
    """Tests for detecting {{Video citation}} templates."""

    @pytest.mark.unit
    def test_detect_video_citation(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should detect {{Video citation}} template."""
        content = load_fixture("citations", "video_citation.txt")
        results = parse_citations(content)
        assert len(results) == 1
        assert results[0].type == "video"


class TestVideoCitationFields:
    """Tests for extracting fields from {{Video citation}} templates."""

    @pytest.mark.unit
    def test_extract_video_url(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract video URL."""
        content = load_fixture("citations", "video_citation.txt")
        results = parse_citations(content)
        assert results[0].url is not None
        assert "youtube.com" in results[0].url
        assert "9TYK9Mu_dzA" in results[0].url

    @pytest.mark.unit
    def test_extract_video_channel(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract channel name."""
        content = load_fixture("citations", "video_citation.txt")
        results = parse_citations(content)
        assert results[0].channel == "Second Thought"

    @pytest.mark.unit
    def test_extract_video_title(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract video title."""
        content = load_fixture("citations", "video_citation.txt")
        results = parse_citations(content)
        assert results[0].title is not None
        assert "Neither Left Nor Right" in results[0].title

    @pytest.mark.unit
    def test_extract_video_date(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract publication date."""
        content = load_fixture("citations", "video_citation.txt")
        results = parse_citations(content)
        assert results[0].date == "2022-03-18"


class TestTextciteDetection:
    """Tests for detecting {{Textcite}} templates."""

    @pytest.mark.unit
    def test_detect_textcite(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should detect {{Textcite}} template as book type."""
        content = load_fixture("citations", "textcite.txt")
        results = parse_citations(content)
        assert len(results) == 1
        assert results[0].type == "book"


class TestTextciteFields:
    """Tests for extracting fields from {{Textcite}} templates."""

    @pytest.mark.unit
    def test_extract_textcite_author(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract author from Textcite."""
        content = load_fixture("citations", "textcite.txt")
        results = parse_citations(content)
        assert results[0].author == "Fabio Giovannini"

    @pytest.mark.unit
    def test_extract_textcite_year(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract year from Textcite."""
        content = load_fixture("citations", "textcite.txt")
        results = parse_citations(content)
        assert results[0].year == "2004"

    @pytest.mark.unit
    def test_extract_textcite_title(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract title from Textcite."""
        content = load_fixture("citations", "textcite.txt")
        results = parse_citations(content)
        assert results[0].title == "Breve storia dell'anticomunismo"

    @pytest.mark.unit
    def test_extract_textcite_city_as_published_location(
        self, load_fixture: "Callable[[str, str], str]"
    ) -> None:
        """Should map city field to published_location."""
        content = load_fixture("citations", "textcite.txt")
        results = parse_citations(content)
        assert results[0].published_location == "Roma"

    @pytest.mark.unit
    def test_extract_textcite_publisher(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract publisher from Textcite."""
        content = load_fixture("citations", "textcite.txt")
        results = parse_citations(content)
        assert results[0].publisher == "Datanews Editrice"

    @pytest.mark.unit
    def test_extract_textcite_isbn(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract ISBN from Textcite."""
        content = load_fixture("citations", "textcite.txt")
        results = parse_citations(content)
        assert results[0].isbn == "9788879812511"

    @pytest.mark.unit
    def test_extract_textcite_lg_url(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract lg field as lg_url."""
        content = load_fixture("citations", "textcite.txt")
        results = parse_citations(content)
        assert results[0].lg_url is not None
        assert "libgen.rs" in results[0].lg_url

    @pytest.mark.unit
    def test_extract_textcite_translation_title(
        self, load_fixture: "Callable[[str, str], str]"
    ) -> None:
        """Should extract trans field as translation_title."""
        content = load_fixture("citations", "textcite.txt")
        results = parse_citations(content)
        assert results[0].translation_title == "Brief history of anti-communism"

    @pytest.mark.unit
    def test_extract_textcite_translation_language(
        self, load_fixture: "Callable[[str, str], str]"
    ) -> None:
        """Should extract translang field as translation_language."""
        content = load_fixture("citations", "textcite.txt")
        results = parse_citations(content)
        assert results[0].translation_language == "Italian"


class TestTextciteWebFields:
    """Tests for extracting fields from {{Textcite}} with web URL."""

    @pytest.mark.unit
    def test_extract_textcite_web_url(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should map web field to url."""
        content = load_fixture("citations", "textcite_web.txt")
        results = parse_citations(content)
        assert results[0].url is not None
        assert "pcb.org.br" in results[0].url

    @pytest.mark.unit
    def test_extract_textcite_web_author(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract author from Textcite with web URL."""
        content = load_fixture("citations", "textcite_web.txt")
        results = parse_citations(content)
        assert results[0].author == "Partido Comunista Brasileiro"
