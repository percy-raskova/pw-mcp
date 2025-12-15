"""Unit tests for link parser (TDD Red Phase).

These tests define the expected interface for the link parser.
The parser module does not exist yet - tests should SKIP initially.

Link types from MediaWiki:
- [[Target]] - Simple internal link
- [[Target|Display]] - Piped link (display text differs from target)
- [[Category:Name]] - Category link (special handling)
- [https://url text] - External link
"""

import pytest

from pw_mcp.ingest.parsers import (
    count_categories,
    count_internal_links,
    get_unique_targets,
    parse_links,
)


class TestSimpleLinkExtraction:
    """Tests for extracting simple [[Target]] links."""

    @pytest.mark.unit
    def test_extract_simple_link(self) -> None:
        """Should extract simple [[Target]] link."""
        text = "He opposed [[slavery]] and fought for freedom."
        results = parse_links(text)
        assert len(results) == 1
        assert results[0].target == "slavery"
        assert results[0].display == "slavery"
        assert results[0].link_type == "internal"

    @pytest.mark.unit
    def test_extract_multiple_simple_links(self, simple_links: str) -> None:
        """Should extract multiple simple links from text."""
        results = parse_links(simple_links)
        assert len(results) >= 3

    @pytest.mark.unit
    def test_simple_link_with_spaces(self) -> None:
        """Should handle links with spaces in target."""
        text = "See [[Communist Party of China]] for details."
        results = parse_links(text)
        assert results[0].target == "Communist Party of China"


class TestPipedLinkExtraction:
    """Tests for extracting [[Target|Display]] piped links."""

    @pytest.mark.unit
    def test_extract_piped_link(self) -> None:
        """Should extract [[Target|Display]] link with both parts."""
        text = "Born in [[United States of America|United States]]."
        results = parse_links(text)
        assert results[0].target == "United States of America"
        assert results[0].display == "United States"

    @pytest.mark.unit
    def test_extract_multiple_piped_links(self, piped_links: str) -> None:
        """Should extract all piped links from text."""
        results = parse_links(piped_links)
        targets = [r.target for r in results]
        assert "United States of America" in targets
        assert "Whig Party (United States)" in targets

    @pytest.mark.unit
    def test_piped_link_with_parentheses(self) -> None:
        """Should handle targets with parentheses (disambiguation)."""
        text = "Member of [[Whig Party (United States)|Whig Party]]."
        results = parse_links(text)
        assert results[0].target == "Whig Party (United States)"
        assert results[0].display == "Whig Party"


class TestCategoryLinkExtraction:
    """Tests for extracting [[Category:Name]] links."""

    @pytest.mark.unit
    def test_extract_category_link(self) -> None:
        """Should detect and extract category links."""
        text = "[[Category:Presidents of the United States]]"
        results = parse_links(text)
        assert results[0].link_type == "category"
        assert results[0].target == "Presidents of the United States"

    @pytest.mark.unit
    def test_extract_multiple_categories(self, category_links: str) -> None:
        """Should extract all category links."""
        results = parse_links(category_links, link_type="category")
        assert len(results) >= 4

    @pytest.mark.unit
    def test_category_extraction_returns_category_names(self, category_links: str) -> None:
        """Should return just the category name without 'Category:' prefix."""
        results = parse_links(category_links, link_type="category")
        categories = [r.target for r in results]
        assert "1809 births" in categories
        assert not any("Category:" in cat for cat in categories)


class TestExternalLinkExtraction:
    """Tests for extracting [url text] external links."""

    @pytest.mark.unit
    def test_extract_external_link_with_text(self) -> None:
        """Should extract [url text] external link."""
        text = "Visit [https://example.com Example Site] for more."
        results = parse_links(text)
        external = [r for r in results if r.link_type == "external"]
        assert len(external) == 1
        assert external[0].target == "https://example.com"
        assert external[0].display == "Example Site"

    @pytest.mark.unit
    def test_extract_external_link_without_text(self) -> None:
        """Should extract [url] external link without display text."""
        text = "See [https://example.com/path] for details."
        results = parse_links(text)
        external = [r for r in results if r.link_type == "external"]
        assert external[0].target == "https://example.com/path"
        assert external[0].display == "https://example.com/path"

    @pytest.mark.unit
    def test_external_link_with_special_chars(self) -> None:
        """Should preserve special characters in external URLs."""
        text = "[https://example.com/path?q=test&page=1 Link]"
        results = parse_links(text)
        assert "q=test&page=1" in results[0].target


class TestLinkFiltering:
    """Tests for filtering links by type."""

    @pytest.mark.unit
    def test_filter_internal_links_only(self) -> None:
        """Should be able to get only internal links."""
        text = """[[Internal Link]] and [https://external.com external]
        and [[Category:Test Category]]"""
        results = parse_links(text, link_type="internal")
        assert len(results) == 1
        assert results[0].target == "Internal Link"

    @pytest.mark.unit
    def test_filter_categories_only(self) -> None:
        """Should be able to get only category links."""
        text = """[[Internal Link]] and [[Category:Test]] and [[Category:Other]]"""
        results = parse_links(text, link_type="category")
        assert len(results) == 2

    @pytest.mark.unit
    def test_get_all_links_by_default(self) -> None:
        """Should return all link types when no filter specified."""
        text = """[[Internal]] [https://ext.com ext] [[Category:Cat]]"""
        results = parse_links(text)
        assert len(results) == 3


class TestLinkEdgeCases:
    """Tests for edge cases in link parsing."""

    @pytest.mark.unit
    def test_nested_brackets_in_display(self) -> None:
        """Should handle nested brackets in display text."""
        text = "[[Target|Display [with brackets]]]"
        # This is malformed, but should not crash
        results = parse_links(text)
        assert len(results) >= 0  # At least doesn't crash

    @pytest.mark.unit
    def test_empty_link_target(self) -> None:
        """Should handle empty link target gracefully."""
        text = "Some [[]] empty link."
        results = parse_links(text)
        # Should skip empty target links
        assert len(results) == 0

    @pytest.mark.unit
    def test_link_with_section_anchor(self) -> None:
        """Should handle links with #section anchors."""
        text = "See [[Article#Section]] for details."
        results = parse_links(text)
        assert results[0].target == "Article"
        assert results[0].section == "Section"

    @pytest.mark.unit
    def test_link_with_unicode(self) -> None:
        """Should handle Unicode in link targets."""
        text = "See [[Kommunistische Partei Deutschlands|KPD]]."
        results = parse_links(text)
        assert results[0].target == "Kommunistische Partei Deutschlands"

    @pytest.mark.unit
    def test_multiple_pipes_in_link(self) -> None:
        """Should handle multiple pipes (malformed but possible)."""
        text = "[[Target|Display|Extra]]"
        # Behavior: take first pipe as separator, rest is display text
        results = parse_links(text)
        assert results[0].target == "Target"
        assert results[0].display == "Display|Extra"

    @pytest.mark.unit
    def test_link_at_start_of_line(self) -> None:
        """Should extract links at the start of lines."""
        text = "[[Link]] at the beginning."
        results = parse_links(text)
        assert results[0].target == "Link"

    @pytest.mark.unit
    def test_link_at_end_of_line(self) -> None:
        """Should extract links at the end of lines."""
        text = "At the end [[Link]]"
        results = parse_links(text)
        assert results[0].target == "Link"

    @pytest.mark.unit
    def test_consecutive_links(self) -> None:
        """Should extract consecutive links without space."""
        text = "[[Link1]][[Link2]][[Link3]]"
        results = parse_links(text)
        assert len(results) == 3


class TestLinkPositions:
    """Tests for tracking link positions in text."""

    @pytest.mark.unit
    def test_track_link_position(self) -> None:
        """Should track start and end positions of links."""
        text = "Before [[Target]] after."
        results = parse_links(text, include_positions=True)
        assert results[0].start == 7  # Position of first [
        assert results[0].end == 17  # Position after last ]

    @pytest.mark.unit
    def test_link_positions_for_replacement(self) -> None:
        """Should provide positions suitable for text replacement."""
        text = "See [[Article|article]] for more."
        results = parse_links(text, include_positions=True)
        # Using positions, we should be able to replace with display text:
        new_text = text[: results[0].start] + results[0].display + text[results[0].end :]
        assert new_text == "See article for more."


class TestLinkCounts:
    """Tests for counting links (for metadata)."""

    @pytest.mark.unit
    def test_count_internal_links(self) -> None:
        """Should count internal links for link_count metadata."""
        text = """
        He opposed [[slavery]] and supported [[freedom]].
        See [[Category:Test]] for categorization.
        """
        # Internal links: slavery, freedom (not category)
        count = count_internal_links(text)
        assert count == 2

    @pytest.mark.unit
    def test_count_categories(self) -> None:
        """Should count category links."""
        text = """
        [[Category:A]]
        [[Category:B]]
        [[Category:C]]
        """
        count = count_categories(text)
        assert count == 3

    @pytest.mark.unit
    def test_get_unique_link_targets(self) -> None:
        """Should return unique set of link targets."""
        text = """
        [[Target]] appears twice: [[Target]] and [[Other]].
        """
        targets = get_unique_targets(text)
        assert len(targets) == 2
        assert "Target" in targets
        assert "Other" in targets
