"""Library work parser for MediaWiki templates.

This module extracts Library work template data from ProleWiki Library namespace.
Library work templates provide metadata about books, articles, speeches, and
other works in the Marxist canon.

Template format:
    {{Library work|title=...|author=...|type=Book|...}}

Supported work types:
    Book, Article, Speech, Pamphlet, Song lyrics, Poem, etc.

Uses mwparserfromhell for reliable MediaWiki template parsing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import mwparserfromhell

from pw_mcp.ingest.parsers.types import LibraryWorkData

if TYPE_CHECKING:
    from mwparserfromhell.nodes import Template
    from mwparserfromhell.wikicode import Wikicode

# Pattern to detect {{Library work|...}} templates (case-insensitive)
LIBRARY_WORK_PATTERN: re.Pattern[str] = re.compile(
    r"\{\{\s*Library\s+work\s*[|\}]",
    re.IGNORECASE,
)

# Pattern to match [[Target]] or [[Target|Display]] wiki links
WIKI_LINK_PATTERN: re.Pattern[str] = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")

# Pattern to match [url text] external link syntax
EXTERNAL_LINK_PATTERN: re.Pattern[str] = re.compile(r"\[(\S+)\s+[^\]]+\]")

# Field name mappings from template params to LibraryWorkData attributes
# Template param (lowercase) -> attribute name
FIELD_MAPPINGS: dict[str, str] = {
    "title": "title",
    "author": "author",
    "type": "work_type",
    "publisher": "publisher",
    "published_date": "published_date",
    "published_location": "published_location",
    "edition_date": "edition_date",
    "source": "source_url",
    "pdf": "pdf_url",
    "translated by": "translator",
    "original language": "original_language",
    "spoken on": "spoken_on",
}


def _strip_wiki_links(text: str) -> str:
    """Strip [[wiki links]] from text, keeping display text.

    Handles both [[Target]] and [[Target|Display]] formats.

    Args:
        text: Text that may contain wiki link markup.

    Returns:
        Text with wiki links replaced by their display text.

    Examples:
        >>> _strip_wiki_links("[[Immortal Technique]]")
        'Immortal Technique'
        >>> _strip_wiki_links("[[Karl Marx|Marx]]")
        'Marx'
        >>> _strip_wiki_links("Plain text")
        'Plain text'
    """
    # First handle piped links [[Target|Display]]
    result = re.sub(r"\[\[[^|\]]*\|([^\]]+)\]\]", r"\1", text)
    # Then handle simple links [[Target]]
    result = re.sub(r"\[\[([^\]]+)\]\]", r"\1", result)
    return result


def _extract_url_from_source(source_value: str) -> str:
    """Extract URL from source field, handling [url text] syntax.

    Args:
        source_value: Raw source field value.

    Returns:
        The URL extracted from the source field.

    Examples:
        >>> _extract_url_from_source("https://example.com")
        'https://example.com'
        >>> _extract_url_from_source("[https://example.com Example]")
        'https://example.com'
    """
    # Check for [url text] syntax
    match = EXTERNAL_LINK_PATTERN.match(source_value.strip())
    if match:
        return match.group(1)
    # Return as-is (plain URL or already processed)
    return source_value.strip()


def _split_authors(author_string: str) -> list[str]:
    """Split comma-separated authors into a list.

    Args:
        author_string: Author field value, possibly comma-separated.

    Returns:
        List of individual author names, stripped and cleaned.

    Examples:
        >>> _split_authors("John Bellamy Foster, Vijay Prashad")
        ['John Bellamy Foster', 'Vijay Prashad']
        >>> _split_authors("Single Author")
        ['Single Author']
    """
    authors = [a.strip() for a in author_string.split(",")]
    return [a for a in authors if a]  # Filter empty strings


def _get_param_value(template: Template, param_name: str) -> str | None:
    """Get parameter value from template, stripping whitespace.

    Handles both exact param names and case-insensitive matching
    for params with spaces (e.g., "translated by", "spoken on").

    Args:
        template: mwparserfromhell Template object.
        param_name: Parameter name to retrieve.

    Returns:
        Parameter value as string, or None if not present.
    """
    # Try exact match first
    if template.has(param_name):
        value = str(template.get(param_name).value).strip()
        return value if value else None

    # Try case variations for params with spaces
    # MediaWiki is case-insensitive for param names
    for param in template.params:
        name = str(param.name).strip().lower()
        if name == param_name.lower():
            value = str(param.value).strip()
            return value if value else None

    return None


def _find_library_work_template(wikicode: Wikicode) -> Template | None:
    """Find the Library work template in parsed wikicode.

    Args:
        wikicode: Parsed MediaWiki content.

    Returns:
        The Library work Template object, or None if not found.
    """
    templates = wikicode.filter_templates()
    for template in templates:
        name = str(template.name).strip().lower()
        if name == "library work":
            return template
    return None


def parse_library_work(text: str) -> LibraryWorkData | None:
    """Parse Library work template from MediaWiki text.

    Extracts metadata from {{Library work|...}} templates including:
    - title, author, work_type (type=Book/Article/Speech/etc.)
    - publisher, published_date, published_location, edition_date
    - source_url (from source=), pdf_url (from pdf=)
    - translator (from translated by=), original_language
    - spoken_on (for speeches)

    Handles:
    - Wiki link syntax [[Author]] in author field
    - Multiple comma-separated authors
    - External link syntax [url text] in source field
    - Multiline template format

    Args:
        text: MediaWiki markup text potentially containing a Library work template.

    Returns:
        LibraryWorkData if a Library work template was found, None otherwise.
        The remaining_text field contains the text with the template removed.

    Examples:
        >>> result = parse_library_work("{{Library work|title=Test|author=Marx}}")
        >>> result.title
        'Test'
        >>> result.author
        'Marx'

        >>> parse_library_work("No template here.")
        None
    """
    # Quick check - does text contain a Library work template?
    if not LIBRARY_WORK_PATTERN.search(text):
        return None

    try:
        wikicode: Wikicode = mwparserfromhell.parse(text)
    except Exception:
        # If parsing fails, return None
        return None

    template = _find_library_work_template(wikicode)
    if template is None:
        return None

    # Extract title (required field)
    title = _get_param_value(template, "title")
    if title is None:
        return None

    # Build kwargs for LibraryWorkData
    kwargs: dict[str, str | list[str] | None] = {"title": title}

    # Extract all mapped fields
    for param_name, attr_name in FIELD_MAPPINGS.items():
        if attr_name == "title":
            continue  # Already handled

        value = _get_param_value(template, param_name)
        if value is not None:
            # Process based on field type
            if attr_name == "source_url":
                value = _extract_url_from_source(value)
            elif attr_name == "author":
                # Strip wiki links from author
                value = _strip_wiki_links(value)
            # Other fields use value as-is

            kwargs[attr_name] = value

    # Handle multiple authors
    author_value = kwargs.get("author")
    if author_value is not None and isinstance(author_value, str):
        if "," in author_value:
            # Multiple authors - populate authors list
            kwargs["authors"] = _split_authors(author_value)
        else:
            # Single author - still populate authors list with one entry
            kwargs["authors"] = [author_value]

    # Remove the template from text to get remaining_text
    try:
        wikicode.remove(template)
        remaining_text = str(wikicode).strip()
    except Exception:
        # Fallback: try to remove via string replacement
        remaining_text = text.replace(str(template), "").strip()

    kwargs["remaining_text"] = remaining_text

    return LibraryWorkData(**kwargs)  # type: ignore[arg-type]
