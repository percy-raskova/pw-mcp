"""Citation parser for MediaWiki templates.

This module extracts citations from ProleWiki templates:
- {{Citation|...}} - Book citations with author, year, title, chapter, page
- {{Web citation|...}} - Web articles with url, newspaper, date, author
- {{News citation|...}} - News articles
- {{YouTube citation|...}} - YouTube videos with channel, url, quote
- {{Library citation|...}} - Links to Library namespace articles

Uses mwparserfromhell for reliable MediaWiki template parsing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import mwparserfromhell

from pw_mcp.ingest.parsers.types import Citation, CitationType

if TYPE_CHECKING:
    from mwparserfromhell.nodes import Template
    from mwparserfromhell.wikicode import Wikicode

# Compile patterns once at module level for performance
# Pattern to match [[Target]] or [[Target|Display]] wiki links
WIKI_LINK_PATTERN: re.Pattern[str] = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")

# Pattern to match <ref>...</ref> tags with optional name attribute
# Handles: <ref>content</ref>, <ref name="foo">content</ref>, <ref name=foo>content</ref>
REF_TAG_PATTERN: re.Pattern[str] = re.compile(
    r'<ref(?:\s+name\s*=\s*["\']?([^"\'>\s]+)["\']?)?\s*>(.*?)</ref>',
    re.DOTALL | re.IGNORECASE,
)

# Pattern to match self-closing <ref name="..." /> tags (ref reuse)
REF_SELFCLOSE_PATTERN: re.Pattern[str] = re.compile(
    r'<ref\s+name\s*=\s*["\']?([^"\'>\s/]+)["\']?\s*/\s*>',
    re.IGNORECASE,
)

# Template name to CitationType mapping
TEMPLATE_TYPE_MAP: dict[str, CitationType] = {
    "citation": "book",
    "web citation": "web",
    "news citation": "news",
    "youtube citation": "youtube",
    "library citation": "library",
    "video citation": "video",
    "textcite": "book",  # Textcite is a text/book citation format
}

# Field name mappings from template params to Citation attributes
# Format: template_param -> citation_attr
FIELD_MAPPINGS: dict[str, str] = {
    "author": "author",
    "title": "title",
    "year": "year",
    "url": "url",
    "archive": "archive_url",
    "archive_url": "archive_url",
    "mia": "mia_url",
    "lg": "lg_url",
    "pdf": "pdf_url",
    "publisher": "publisher",
    "newspaper": "newspaper",
    "chapter": "chapter",
    "page": "page",
    "pages": "page",  # Map 'pages' to 'page' as well
    "isbn": "isbn",
    "date": "date",
    "retrieved": "retrieved",
    "channel": "channel",
    "quote": "quote",
    "link": "link",
    # Textcite-specific field mappings
    "city": "published_location",  # City of publication
    "web": "url",  # Alternative URL field in Textcite
    "trans": "translation_title",  # Translated title
    "translang": "translation_language",  # Original language
}


def _strip_wiki_links(text: str) -> str:
    """Strip [[wiki links]] from text, keeping display text.

    Handles both [[Target]] and [[Target|Display]] formats.

    Args:
        text: Text that may contain wiki link markup.

    Returns:
        Text with wiki links replaced by their display text.

    Examples:
        >>> _strip_wiki_links("[[Domenico Losurdo]]")
        'Domenico Losurdo'
        >>> _strip_wiki_links("[[CovertAction Magazine|CAM]]")
        'CAM'
        >>> _strip_wiki_links("Plain text")
        'Plain text'
    """

    # For [[Target|Display]], use Display; for [[Target]], use Target
    def replace_link(match: re.Match[str]) -> str:
        return match.group(1)

    # First handle piped links [[Target|Display]]
    result = re.sub(r"\[\[[^|\]]*\|([^\]]+)\]\]", replace_link, text)
    # Then handle simple links [[Target]]
    result = re.sub(r"\[\[([^\]]+)\]\]", replace_link, result)
    return result


def _get_citation_type(template_name: str) -> CitationType | None:
    """Get CitationType from template name.

    Args:
        template_name: MediaWiki template name (case-insensitive).

    Returns:
        CitationType if recognized, None otherwise.

    Examples:
        >>> _get_citation_type("Citation")
        'book'
        >>> _get_citation_type("Web citation")
        'web'
        >>> _get_citation_type("Unknown")
        None
    """
    normalized = template_name.strip().lower()
    return TEMPLATE_TYPE_MAP.get(normalized)


def _get_param_value(template: Template, param_name: str) -> str | None:
    """Get parameter value from template, stripping whitespace.

    Args:
        template: mwparserfromhell Template object.
        param_name: Parameter name to retrieve.

    Returns:
        Parameter value as string, or None if not present.
    """
    if template.has(param_name):
        value = str(template.get(param_name).value).strip()
        return value if value else None
    return None


def _parse_template(template: Template) -> Citation | None:
    """Parse a single citation template into a Citation object.

    Args:
        template: mwparserfromhell Template object.

    Returns:
        Citation object if template is a recognized citation type, None otherwise.
    """
    template_name = str(template.name).strip()
    citation_type = _get_citation_type(template_name)

    if citation_type is None:
        return None

    # Start building Citation with required type
    citation_kwargs: dict[str, str | None] = {"type": citation_type}

    # Extract all mapped fields
    for param_name, attr_name in FIELD_MAPPINGS.items():
        value = _get_param_value(template, param_name)
        if value is not None:
            # Strip wiki links from text fields (not URLs)
            if attr_name not in ("url", "archive_url", "mia_url", "lg_url", "pdf_url"):
                value = _strip_wiki_links(value)
            citation_kwargs[attr_name] = value

    return Citation(**citation_kwargs)  # type: ignore[arg-type]


def _parse_text_for_templates(text: str) -> list[Citation]:
    """Parse text for citation templates (without ref tag handling).

    Args:
        text: MediaWiki markup text.

    Returns:
        List of Citation objects found in templates.
    """
    citations: list[Citation] = []

    try:
        wikicode: Wikicode = mwparserfromhell.parse(text)
        templates = wikicode.filter_templates()

        for template in templates:
            citation = _parse_template(template)
            if citation is not None:
                citations.append(citation)

    except Exception:
        # If parsing fails, return empty list rather than crashing
        pass

    return citations


def parse_ref_tags(text: str) -> list[Citation]:
    """Extract citations from <ref>...</ref> tags.

    This is a lower-level function that focuses on the ref tag structure.
    It handles:
    - <ref>content</ref> - Anonymous refs
    - <ref name="foo">content</ref> - Named refs (first occurrence)
    - <ref name="foo" /> - Named refs (reuse, skipped)

    Args:
        text: MediaWiki markup text with ref tags.

    Returns:
        List of Citation objects from ref tags.
        Named refs appearing multiple times are deduplicated.

    Examples:
        >>> parse_ref_tags('<ref name=":0">{{Citation|title=Test}}</ref>')
        [Citation(type='book', title='Test', ref_name=':0')]
    """
    citations: list[Citation] = []
    seen_ref_names: set[str] = set()

    # Find all <ref>...</ref> tags
    for match in REF_TAG_PATTERN.finditer(text):
        ref_name = match.group(1)  # May be None for anonymous refs
        ref_content = match.group(2)

        # Skip if we've already seen this named ref
        if ref_name is not None:
            if ref_name in seen_ref_names:
                continue
            seen_ref_names.add(ref_name)

        # Parse the content inside the ref tag for templates
        inner_citations = _parse_text_for_templates(ref_content)

        # Attach ref_name to each citation found
        for citation in inner_citations:
            citation.ref_name = ref_name
            citations.append(citation)

    return citations


def parse_citations(text: str) -> list[Citation]:
    """Extract all citations from MediaWiki text.

    Parses citation templates including:
    - {{Citation|...}} - Book/general citations
    - {{Web citation|...}} - Web articles
    - {{News citation|...}} - News articles
    - {{YouTube citation|...}} - YouTube videos
    - {{Library citation|...}} - Library namespace links

    Handles named refs by deduplicating: if the same ref name appears
    multiple times, only one Citation is returned.

    The function first checks for <ref>...</ref> tags and parses citations
    within them. If no ref tags are found, it parses the entire text for
    citation templates directly.

    Args:
        text: MediaWiki markup text containing citation templates.

    Returns:
        List of Citation objects extracted from the text.
        Named refs that appear multiple times are deduplicated.

    Examples:
        >>> parse_citations("Text.<ref>{{Citation|author=Marx}}</ref>")
        [Citation(type='book', author='Marx', ...)]

        >>> parse_citations("No citations here.")
        []
    """
    # Check if text contains ref tags
    has_ref_tags = REF_TAG_PATTERN.search(text) is not None

    if has_ref_tags:
        # Parse citations from within ref tags (handles deduplication)
        return parse_ref_tags(text)

    # No ref tags - parse templates directly from text
    return _parse_text_for_templates(text)
