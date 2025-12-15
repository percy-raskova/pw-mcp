"""Unit tests for infobox parser (TDD Green Phase).

These tests validate the infobox parser implementation.
"""

from typing import TYPE_CHECKING

import pytest

from pw_mcp.ingest.parsers.infobox import detect_infobox_type, parse_infobox

if TYPE_CHECKING:
    from collections.abc import Callable


class TestInfoboxDetection:
    """Tests for detecting infobox presence and type."""

    @pytest.mark.unit
    def test_detect_politician_infobox(self, politician_infobox: str) -> None:
        """Should detect {{Infobox politician}} and return type='politician'."""
        result = parse_infobox(politician_infobox)
        assert result is not None
        assert result.type == "politician"
        assert result.fields["name"] == "Abraham Lincoln"

    @pytest.mark.unit
    def test_detect_country_infobox(self, country_infobox: str) -> None:
        """Should detect {{Infobox country}} and return type='country'."""
        result = parse_infobox(country_infobox)
        assert result is not None
        assert result.type == "country"
        assert result.fields["name"] == "Kingdom of Norway"

    @pytest.mark.unit
    def test_detect_political_party_infobox(self, political_party_infobox: str) -> None:
        """Should detect {{Infobox political party}} and return type='political_party'."""
        result = parse_infobox(political_party_infobox)
        assert result is not None
        assert result.type == "political_party"
        assert result.fields["name"] == "Communist Party of Germany"

    @pytest.mark.unit
    def test_detect_person_infobox(self, person_infobox: str) -> None:
        """Should detect {{Infobox person}} and return type='person'."""
        result = parse_infobox(person_infobox)
        assert result is not None
        assert result.type == "person"

    @pytest.mark.unit
    def test_detect_philosopher_infobox(self, philosopher_infobox: str) -> None:
        """Should detect {{Infobox philosopher}} and return type='philosopher'."""
        result = parse_infobox(philosopher_infobox)
        assert result is not None
        assert result.type == "philosopher"
        assert result.fields["name"] == "Ludwig Feuerbach"

    @pytest.mark.unit
    def test_detect_organization_infobox(self, organization_infobox: str) -> None:
        """Should detect {{Infobox organization}} and return type='organization'."""
        result = parse_infobox(organization_infobox)
        assert result is not None
        assert result.type == "organization"
        assert result.fields["name"] == "Central Military Commission"

    @pytest.mark.unit
    def test_detect_youtuber_infobox(self, youtuber_infobox: str) -> None:
        """Should detect {{Infobox youtuber}} and return type='youtuber'."""
        result = parse_infobox(youtuber_infobox)
        assert result is not None
        assert result.type == "youtuber"
        assert result.fields["name"] == "The Kavernacle"

    @pytest.mark.unit
    def test_no_infobox_returns_none(self) -> None:
        """Should return None for text without infobox."""
        text = "This is plain text with no infobox template."
        result = parse_infobox(text)
        assert result is None

    @pytest.mark.unit
    def test_detect_infobox_type_lightweight(self, politician_infobox: str) -> None:
        """Should detect infobox type without full parsing."""
        result = detect_infobox_type(politician_infobox)
        assert result == "politician"

    @pytest.mark.unit
    def test_detect_infobox_type_returns_none(self) -> None:
        """Should return None when no infobox present."""
        text = "No infobox here."
        result = detect_infobox_type(text)
        assert result is None


class TestInfoboxFieldExtraction:
    """Tests for extracting fields from infoboxes."""

    @pytest.mark.unit
    def test_extract_name_field(self, politician_infobox: str) -> None:
        """Should extract |name= field as subject_name."""
        result = parse_infobox(politician_infobox)
        assert result is not None
        assert result.fields["name"] == "Abraham Lincoln"

    @pytest.mark.unit
    def test_extract_birth_date_string(self, politician_infobox: str) -> None:
        """Should extract |birth_date= field."""
        result = parse_infobox(politician_infobox)
        assert result is not None
        assert result.fields["birth_date"] == "February 12, 1809"

    @pytest.mark.unit
    def test_extract_political_orientation_with_links(self, politician_infobox: str) -> None:
        """Should extract [[links]] from political_orientation as list."""
        result = parse_infobox(politician_infobox)
        assert result is not None
        orientations = result.fields["political_orientation"]
        # Field has <br> tags so should be a list
        assert isinstance(orientations, list)
        assert "Liberalism" in orientations
        assert "White supremacy" in orientations

    @pytest.mark.unit
    def test_extract_nationality(self, country_infobox: str) -> None:
        """Should extract nationality/capital from country infobox."""
        result = parse_infobox(country_infobox)
        assert result is not None
        capital = result.fields["capital"]
        assert capital is not None
        assert "Oslo" in str(capital)

    @pytest.mark.unit
    def test_extract_numeric_fields(self, country_infobox: str) -> None:
        """Should extract numeric fields like area_km2, population_estimate."""
        result = parse_infobox(country_infobox)
        assert result is not None
        assert result.fields["area_km2"] == "385,207"
        assert result.fields["population_estimate"] == "5,425,270"

    @pytest.mark.unit
    def test_extract_founders_list(self, political_party_infobox: str) -> None:
        """Should extract founders as list of linked names."""
        result = parse_infobox(political_party_infobox)
        assert result is not None
        founders = result.fields["founders"]
        # Field has <br> tags so should be a list
        assert isinstance(founders, list)
        assert "Rosa Luxemburg" in founders
        assert "Karl Liebknecht" in founders

    @pytest.mark.unit
    def test_extract_empty_name_field(self, person_infobox: str) -> None:
        """Should handle empty |name= field (Jean-Francois Champollion)."""
        # Person infobox has |name=| (empty)
        result = parse_infobox(person_infobox)
        assert result is not None
        # Empty field should be None after processing
        assert result.fields.get("name") is None or result.fields.get("name") == ""


class TestInfoboxEdgeCases:
    """Tests for edge cases in infobox parsing."""

    @pytest.mark.unit
    def test_unicode_native_name(self, unicode_infobox: str) -> None:
        """Should handle Unicode in native_name field (Hittite cuneiform)."""
        # Suppiluliuma I has native_name with Hittite cuneiform script
        result = parse_infobox(unicode_infobox)
        assert result is not None
        native = result.fields.get("native_name", "")
        assert native is not None
        assert len(str(native)) > 0  # Should preserve Unicode

    @pytest.mark.unit
    def test_nested_templates_in_values(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should handle nested {{templates}} within field values."""
        content = load_fixture("edge_cases", "nested_templates.txt")
        result = parse_infobox(content)
        # The birth_date contains {{birth date|1809|2|12}}
        # Parser should either preserve it or extract the date
        assert result is not None
        # Should have parsed the infobox successfully
        assert result.type == "politician"

    @pytest.mark.unit
    def test_multiline_field_values(self, load_fixture: "Callable[[str, str], str]") -> None:
        """Should handle field values spanning multiple lines."""
        content = load_fixture("edge_cases", "multiline_params.txt")
        result = parse_infobox(content)
        assert result is not None
        assert result.fields["name"] == "Central Military Commission"

    @pytest.mark.unit
    def test_external_link_in_field(self, organization_infobox: str) -> None:
        """Should extract external links like [https://... text]."""
        result = parse_infobox(organization_infobox)
        assert result is not None
        website = result.fields.get("website", "")
        assert website is not None
        assert "eng.mod.gov.cn" in str(website)

    @pytest.mark.unit
    def test_br_tags_in_values(self, politician_infobox: str) -> None:
        """Should handle <br> tags separating multiple values."""
        # political_party has multiple values separated by <br>
        result = parse_infobox(politician_infobox)
        assert result is not None
        parties = result.fields.get("political_party", "")
        # Should split on <br> resulting in a list
        assert isinstance(parties, list)

    @pytest.mark.unit
    def test_date_with_parenthetical_age(self, politician_infobox: str) -> None:
        """Should handle dates with '(aged X)' suffix."""
        # death_date=April 15, 1865 (aged 56)
        result = parse_infobox(politician_infobox)
        assert result is not None
        death = result.fields.get("death_date", "")
        assert death is not None
        assert "1865" in str(death)


class TestInfoboxRemoval:
    """Tests for removing infobox from text (for clean_text output)."""

    @pytest.mark.unit
    def test_remove_infobox_from_text(self, politician_infobox: str) -> None:
        """Should be able to remove infobox, leaving empty string."""
        result = parse_infobox(politician_infobox)
        assert result is not None
        clean = result.remaining_text
        assert "{{Infobox" not in clean
        assert clean.strip() == ""  # Only infobox in fixture

    @pytest.mark.unit
    def test_preserve_text_after_infobox(self) -> None:
        """Should preserve article text that follows the infobox."""
        text_with_article = """{{Infobox politician|name=Test}}

'''Test Person''' was a politician."""
        result = parse_infobox(text_with_article)
        assert result is not None
        assert "Test Person" in result.remaining_text
        assert "politician" in result.remaining_text
