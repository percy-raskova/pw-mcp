"""Quote parser for MediaWiki templates.

This module extracts block quotes from ProleWiki {{Quote}} templates:
- {{Quote | text | attribution}} - Quote with source attribution
- {{Quote | text}} - Quote without attribution

Uses mwparserfromhell for reliable MediaWiki template parsing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mwparserfromhell

from pw_mcp.ingest.parsers.types import QuoteData

if TYPE_CHECKING:
    from mwparserfromhell.nodes import Template
    from mwparserfromhell.wikicode import Wikicode


def _strip_surrounding_quotes(text: str) -> str:
    """Strip surrounding quotation marks from text.

    Handles both standard double quotes and typographic quotes.

    Args:
        text: Text that may have surrounding quotation marks.

    Returns:
        Text with leading/trailing quotation marks removed.

    Examples:
        >>> _strip_surrounding_quotes('"Hello world"')
        'Hello world'
        >>> _strip_surrounding_quotes('No quotes here')
        'No quotes here'
    """
    # Strip standard double quotes
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    # Strip typographic quotes (left/right double)
    if text.startswith("\u201c") and text.endswith("\u201d"):
        return text[1:-1]
    return text


def _parse_quote_template(template: Template) -> QuoteData | None:
    """Parse a single {{Quote}} template into a QuoteData object.

    The Quote template uses positional parameters:
    - Parameter 1: The quote text
    - Parameter 2: The attribution (optional)

    Args:
        template: mwparserfromhell Template object.

    Returns:
        QuoteData object if template is a Quote template, None otherwise.
    """
    template_name = str(template.name).strip().lower()

    if template_name != "quote":
        return None

    # Extract positional parameter 1 (quote text)
    text: str | None = None
    attribution: str | None = None

    if template.has("1"):
        text = str(template.get("1").value).strip()
        text = _strip_surrounding_quotes(text)

    # Extract positional parameter 2 (attribution) if present
    if template.has("2"):
        attr_value = str(template.get("2").value).strip()
        if attr_value:
            attribution = attr_value

    # Cannot create QuoteData without text
    if text is None or text == "":
        return None

    return QuoteData(text=text, attribution=attribution)


def parse_quotes(text: str) -> list[QuoteData]:
    """Extract all quotes from MediaWiki text.

    Parses {{Quote}} templates and extracts:
    - The quoted text content
    - Optional attribution (author, work, date)

    Args:
        text: MediaWiki markup text containing Quote templates.

    Returns:
        List of QuoteData objects extracted from the text.

    Examples:
        >>> parse_quotes('{{Quote | "Test quote" | Karl Marx}}')
        [QuoteData(text='Test quote', attribution='Karl Marx')]

        >>> parse_quotes('{{Quote | Simple quote without attribution}}')
        [QuoteData(text='Simple quote without attribution', attribution=None)]

        >>> parse_quotes('No quotes here.')
        []
    """
    quotes: list[QuoteData] = []

    try:
        wikicode: Wikicode = mwparserfromhell.parse(text)
        templates = wikicode.filter_templates()

        for template in templates:
            quote_data = _parse_quote_template(template)
            if quote_data is not None:
                quotes.append(quote_data)

    except Exception:
        # If parsing fails, return empty list rather than crashing
        pass

    return quotes
