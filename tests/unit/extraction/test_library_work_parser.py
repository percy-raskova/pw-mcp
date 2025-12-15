"""Unit tests for Library work parser (TDD Green Phase).

These tests define the expected interface for the Library work parser.
Tests are now enabled with the parser implementation.

Library work templates from ProleWiki Library namespace:
- {{Library work|title=...|author=...|type=Book|...}}
- Supports: Book, Article, Speech, Pamphlet, Song lyrics, Poem, etc.
"""

from typing import TYPE_CHECKING

import pytest

from pw_mcp.ingest.parsers.library_work import parse_library_work

if TYPE_CHECKING:
    from collections.abc import Callable


class TestLibraryWorkDetection:
    """Tests for detecting Library work templates and their types."""

    @pytest.mark.unit
    def test_detect_library_work_template(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should detect {{Library work}} template."""
        content = load_fixture("library_work", "book_full.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.title == "Fundamentals of Marxism Leninism"

    @pytest.mark.unit
    def test_detect_work_type_book(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should detect type=Book."""
        content = load_fixture("library_work", "book_full.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.work_type == "Book"

    @pytest.mark.unit
    def test_detect_work_type_article(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should detect type=Article."""
        content = load_fixture("library_work", "article.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.work_type == "Article"

    @pytest.mark.unit
    def test_detect_work_type_speech(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should detect type=Speech."""
        content = load_fixture("library_work", "speech.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.work_type == "Speech"

    @pytest.mark.unit
    def test_detect_work_type_song_lyrics(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should detect type=Song lyrics."""
        content = load_fixture("library_work", "song_lyrics.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.work_type == "Song lyrics"

    @pytest.mark.unit
    def test_no_library_work_returns_none(self) -> None:
        """Should return None for text without Library work template."""
        text = "This is plain text with no Library work template."
        result = parse_library_work(text)
        assert result is None


class TestLibraryWorkFieldExtraction:
    """Tests for extracting fields from Library work templates."""

    @pytest.mark.unit
    def test_extract_title(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract title field."""
        content = load_fixture("library_work", "book_full.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.title == "Fundamentals of Marxism Leninism"

    @pytest.mark.unit
    def test_extract_author(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract author field."""
        content = load_fixture("library_work", "book_full.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.author == "Otto Kuusinen"

    @pytest.mark.unit
    def test_extract_multiple_authors(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract multiple authors as list."""
        content = load_fixture("library_work", "multiple_authors.txt")
        result = parse_library_work(content)
        assert result is not None
        assert "John Bellamy Foster" in result.authors
        assert "Vijay Prashad" in result.authors
        assert len(result.authors) == 4

    @pytest.mark.unit
    def test_extract_publisher(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract publisher field."""
        content = load_fixture("library_work", "book_full.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.publisher == "Foreign Languages Publishing House"

    @pytest.mark.unit
    def test_extract_published_date(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract published_date field."""
        content = load_fixture("library_work", "book_full.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.published_date == "1960"

    @pytest.mark.unit
    def test_extract_published_location(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract published_location field."""
        content = load_fixture("library_work", "book_full.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.published_location == "Moscow"

    @pytest.mark.unit
    def test_extract_edition_date(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract edition_date field."""
        content = load_fixture("library_work", "book_full.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.edition_date == "1963"

    @pytest.mark.unit
    def test_extract_source_url(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract source URL."""
        content = load_fixture("library_work", "book_full.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.source_url is not None
        assert "redstarpublishers.org" in result.source_url

    @pytest.mark.unit
    def test_extract_pdf_url(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract pdf URL."""
        content = load_fixture("library_work", "speech.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.pdf_url is not None
        assert "archive.org" in result.pdf_url


class TestTranslatedWorks:
    """Tests for translated work metadata."""

    @pytest.mark.unit
    def test_extract_translator(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract translated by field."""
        content = load_fixture("library_work", "translated.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.translator == "Vic Schneierson"

    @pytest.mark.unit
    def test_extract_original_language(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract original language field."""
        content = load_fixture("library_work", "translated.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.original_language == "Russian"


class TestSpeechAndDateVariants:
    """Tests for speech/spoken_on and date variations."""

    @pytest.mark.unit
    def test_extract_spoken_on_date(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should extract spoken on date for speeches."""
        content = load_fixture("library_work", "speech.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.spoken_on == "1968"

    @pytest.mark.unit
    def test_extract_iso_date_format(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should handle ISO date format (YYYY-MM-DD)."""
        content = load_fixture("library_work", "article.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.published_date == "2014-02-18"


class TestLibraryWorkEdgeCases:
    """Tests for edge cases in Library work parsing."""

    @pytest.mark.unit
    def test_multiline_template(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should handle multiline template format."""
        content = load_fixture("library_work", "multiline.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.title == "Central Military Commission"

    @pytest.mark.unit
    def test_author_with_wiki_link(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should strip [[wiki links]] from author name."""
        content = load_fixture("library_work", "song_lyrics.txt")
        # Author is [[Immortal Technique]] in template
        result = parse_library_work(content)
        assert result is not None
        assert result.author == "Immortal Technique"
        assert "[[" not in result.author

    @pytest.mark.unit
    def test_missing_optional_fields(self) -> None:
        """Should handle missing optional fields gracefully."""
        text = "{{Library work|title=Simple Work|author=Simple Author}}"
        result = parse_library_work(text)
        assert result is not None
        assert result.title == "Simple Work"
        assert result.publisher is None
        assert result.source_url is None

    @pytest.mark.unit
    def test_source_with_wiki_link_syntax(self) -> None:
        """Should handle source with [url text] syntax."""
        text = "{{Library work|title=Test|source=[https://example.com Example]}}"
        result = parse_library_work(text)
        assert result is not None
        assert result.source_url is not None
        assert "example.com" in result.source_url


class TestLibraryWorkRemoval:
    """Tests for removing Library work template from text."""

    @pytest.mark.unit
    def test_remaining_text_after_removal(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should return remaining text with template removed."""
        content = load_fixture("library_work", "book_full.txt")
        full_text = content + "\n\n== Chapter 1 ==\nContent here."
        result = parse_library_work(full_text)
        assert result is not None
        assert "{{Library work" not in result.remaining_text
        assert "Chapter 1" in result.remaining_text

    @pytest.mark.unit
    def test_only_template_returns_empty_remaining(
        self, load_fixture: "Callable[[str, str], str]"
    ) -> None:
        """Should return empty remaining_text when only template present."""
        content = load_fixture("library_work", "book_full.txt")
        result = parse_library_work(content)
        assert result is not None
        assert result.remaining_text.strip() == ""
