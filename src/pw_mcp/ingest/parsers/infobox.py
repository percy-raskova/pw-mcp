"""Infobox parser for MediaWiki templates.

This module extracts infobox data from ProleWiki articles.
Infoboxes are structured metadata templates that appear at the
top of articles, containing key facts about the subject.

Supported infobox types:
- politician, country, political_party, person, revolutionary
- essay, philosopher, company, settlement, military_person
- organization, guerilla_organization, youtuber, military_conflict
- book, religion, website, transcript, library_work
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, get_args

import mwparserfromhell

from pw_mcp.ingest.parsers.types import InfoboxData, InfoboxType

if TYPE_CHECKING:
    from mwparserfromhell.nodes import Template
    from mwparserfromhell.wikicode import Wikicode

# Pattern to detect {{Infobox TYPE|...}} templates (case-insensitive)
# Matches "Infobox politician", "Infobox country", etc.
INFOBOX_PATTERN: re.Pattern[str] = re.compile(
    r"\{\{\s*Infobox\s+(\w+(?:\s+\w+)?)",
    re.IGNORECASE,
)

# Pattern to extract internal wiki links [[Target]] or [[Target|Display]]
WIKI_LINK_PATTERN: re.Pattern[str] = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# Pattern to split on <br> tags (with optional / and whitespace)
BR_TAG_PATTERN: re.Pattern[str] = re.compile(
    r"<br\s*/?\s*>",
    re.IGNORECASE,
)

# Map template type strings to InfoboxType literals
# Handles underscore/space variations
INFOBOX_TYPE_MAP: dict[str, InfoboxType] = {
    "politician": "politician",
    "country": "country",
    "political party": "political_party",
    "political_party": "political_party",
    "person": "person",
    "revolutionary": "revolutionary",
    "essay": "essay",
    "philosopher": "philosopher",
    "company": "company",
    "settlement": "settlement",
    "military person": "military_person",
    "military_person": "military_person",
    "organization": "organization",
    "guerilla organization": "guerilla_organization",
    "guerilla_organization": "guerilla_organization",
    "youtuber": "youtuber",
    "military conflict": "military_conflict",
    "military_conflict": "military_conflict",
    "book": "book",
    "religion": "religion",
    "website": "website",
    "transcript": "transcript",
    "library work": "library_work",
    "library_work": "library_work",
}

# Valid InfoboxType values from the Literal type
VALID_INFOBOX_TYPES: frozenset[str] = frozenset(get_args(InfoboxType))


def _normalize_type_name(type_str: str) -> str:
    """Normalize infobox type string for lookup.

    Converts to lowercase and normalizes underscores/spaces.

    Args:
        type_str: Raw type string from template name.

    Returns:
        Normalized type string for lookup.

    Examples:
        >>> _normalize_type_name("Political Party")
        'political party'
        >>> _normalize_type_name("military_person")
        'military_person'
    """
    return type_str.strip().lower()


def _get_infobox_type(type_str: str) -> InfoboxType | None:
    """Convert raw type string to InfoboxType.

    Args:
        type_str: Type string extracted from template name.

    Returns:
        InfoboxType if recognized, None otherwise.

    Examples:
        >>> _get_infobox_type("politician")
        'politician'
        >>> _get_infobox_type("Political Party")
        'political_party'
    """
    normalized = _normalize_type_name(type_str)
    return INFOBOX_TYPE_MAP.get(normalized)


def _extract_wiki_links(text: str) -> list[str]:
    """Extract wiki link targets from text.

    Finds all [[Target]] or [[Target|Display]] links and returns
    the target article names.

    Args:
        text: Text that may contain wiki link markup.

    Returns:
        List of link target article names.

    Examples:
        >>> _extract_wiki_links("[[Oslo]] and [[Karl Marx|Marx]]")
        ['Oslo', 'Karl Marx']
    """
    return WIKI_LINK_PATTERN.findall(text)


def _strip_wiki_links(text: str) -> str:
    """Strip [[wiki links]] from text, keeping display text.

    For [[Target|Display]], keeps Display.
    For [[Target]], keeps Target.

    Args:
        text: Text that may contain wiki link markup.

    Returns:
        Text with wiki link markup removed.

    Examples:
        >>> _strip_wiki_links("[[Oslo]]")
        'Oslo'
        >>> _strip_wiki_links("[[Karl Marx|Marx]]")
        'Marx'
    """
    # First handle piped links [[Target|Display]]
    result = re.sub(r"\[\[[^|\]]*\|([^\]]+)\]\]", r"\1", text)
    # Then handle simple links [[Target]]
    result = re.sub(r"\[\[([^\]]+)\]\]", r"\1", result)
    return result


def _split_br_values(text: str) -> list[str]:
    """Split text on <br> tags into a list of values.

    Args:
        text: Text that may contain <br> tags.

    Returns:
        List of values split on <br> tags, stripped of whitespace.

    Examples:
        >>> _split_br_values("A<br>B<br />C")
        ['A', 'B', 'C']
    """
    parts = BR_TAG_PATTERN.split(text)
    return [p.strip() for p in parts if p.strip()]


def _has_br_tags(text: str) -> bool:
    """Check if text contains <br> tags.

    Args:
        text: Text to check.

    Returns:
        True if text contains <br> tags.
    """
    return BR_TAG_PATTERN.search(text) is not None


def _process_field_value(
    raw_value: str,
    extract_links: bool = True,
) -> tuple[str | list[str] | None, list[str]]:
    """Process a field value, extracting links and splitting on <br> tags.

    Args:
        raw_value: Raw parameter value from template.
        extract_links: Whether to extract wiki links.

    Returns:
        Tuple of (processed_value, extracted_links).
        processed_value may be:
        - None if empty
        - str if single value
        - list[str] if contains <br> tags or multiple links

    Examples:
        >>> _process_field_value("[[Oslo]]")
        ('Oslo', ['Oslo'])
        >>> _process_field_value("A<br>B")
        (['A', 'B'], [])
    """
    if not raw_value or not raw_value.strip():
        return None, []

    raw_value = raw_value.strip()
    links: list[str] = []

    if extract_links:
        links = _extract_wiki_links(raw_value)

    # Check for <br> tags indicating multiple values
    if _has_br_tags(raw_value):
        parts = _split_br_values(raw_value)
        # Strip wiki link markup from each part
        stripped_parts = [_strip_wiki_links(p) for p in parts]
        return stripped_parts, links

    # Single value - strip wiki link markup
    stripped_value = _strip_wiki_links(raw_value)
    return stripped_value, links


def _get_param_value(template: Template, param_name: str) -> str | None:
    """Get parameter value from template.

    Args:
        template: mwparserfromhell Template object.
        param_name: Parameter name to retrieve.

    Returns:
        Parameter value as string, or None if not present.
    """
    if template.has(param_name):
        value = str(template.get(param_name).value).strip()
        # Return empty string for explicitly empty params (|name=|)
        # This is different from missing params
        return value if value else ""
    return None


def _find_infobox_template(wikicode: Wikicode) -> Template | None:
    """Find the first Infobox template in parsed wikicode.

    Args:
        wikicode: Parsed MediaWiki content.

    Returns:
        The Infobox Template object, or None if not found.
    """
    templates = wikicode.filter_templates()
    for template in templates:
        name = str(template.name).strip().lower()
        if name.startswith("infobox"):
            return template
    return None


def _extract_type_from_template_name(template_name: str) -> str | None:
    """Extract the infobox type from a template name.

    Args:
        template_name: Full template name like "Infobox politician".

    Returns:
        The type portion (e.g., "politician"), or None if not an infobox.

    Examples:
        >>> _extract_type_from_template_name("Infobox politician")
        'politician'
        >>> _extract_type_from_template_name("Infobox political party")
        'political party'
    """
    name = template_name.strip()
    # Match case-insensitive "Infobox" prefix
    if name.lower().startswith("infobox"):
        # Extract everything after "Infobox " (with space)
        type_part = name[7:].strip()  # len("Infobox") = 7
        return type_part if type_part else None
    return None


def detect_infobox_type(text: str) -> InfoboxType | None:
    """Detect the type of infobox in the text without full parsing.

    This is a lightweight check to determine if an infobox exists
    and what type it is, without extracting all field values.

    Args:
        text: MediaWiki markup text to check.

    Returns:
        The InfoboxType if detected, None if no infobox found.

    Examples:
        >>> detect_infobox_type("{{Infobox politician|name=Test}}")
        'politician'

        >>> detect_infobox_type("No infobox here.")
        None
    """
    # Use regex for quick detection
    match = INFOBOX_PATTERN.search(text)
    if match is None:
        return None

    type_str = match.group(1)
    return _get_infobox_type(type_str)


def parse_infobox(text: str) -> InfoboxData | None:
    """Extract infobox data from MediaWiki text.

    Parses {{Infobox TYPE|...}} templates, extracting:
    - The infobox type (politician, country, etc.)
    - All field values (|field=value|...)
    - Internal links within field values
    - The remaining article text after infobox removal

    Args:
        text: MediaWiki markup text potentially containing an infobox.

    Returns:
        InfoboxData if an infobox was found, None otherwise.
        The remaining_text field contains the article with the
        infobox removed.

    Examples:
        >>> result = parse_infobox("{{Infobox politician|name=Test}}")
        >>> result.type
        'politician'
        >>> result.fields['name']
        'Test'

        >>> parse_infobox("No infobox here.")
        None
    """
    # Quick check - does text contain an infobox?
    if not INFOBOX_PATTERN.search(text):
        return None

    try:
        wikicode: Wikicode = mwparserfromhell.parse(text)
    except Exception:
        # If parsing fails, return None
        return None

    template = _find_infobox_template(wikicode)
    if template is None:
        return None

    # Extract type from template name
    template_name = str(template.name).strip()
    type_str = _extract_type_from_template_name(template_name)
    if type_str is None:
        return None

    infobox_type = _get_infobox_type(type_str)
    if infobox_type is None:
        return None

    # Extract all parameters
    fields: dict[str, str | list[str] | None] = {}
    raw_fields: dict[str, str] = {}
    all_links: list[str] = []

    for param in template.params:
        param_name = str(param.name).strip()
        raw_value = str(param.value).strip()

        # Store raw value
        raw_fields[param_name] = raw_value

        # Process the value
        processed_value, links = _process_field_value(raw_value)
        fields[param_name] = processed_value
        all_links.extend(links)

    # Remove the infobox template from text to get remaining_text
    try:
        wikicode.remove(template)
        remaining_text = str(wikicode).strip()
    except Exception:
        # Fallback: try to remove via string replacement
        remaining_text = text.replace(str(template), "").strip()

    return InfoboxData(
        type=infobox_type,
        fields=fields,
        remaining_text=remaining_text,
        extracted_links=all_links,
        raw_fields=raw_fields,
    )
