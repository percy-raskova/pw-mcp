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

from pw_mcp.ingest.parsers.types import InfoboxData, InfoboxType


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

    Raises:
        NotImplementedError: Parser not yet implemented.

    Examples:
        >>> parse_infobox("{{Infobox politician|name=Test}}")
        InfoboxData(type='politician', fields={'name': 'Test'}, ...)

        >>> parse_infobox("No infobox here.")
        None
    """
    raise NotImplementedError("Parser not implemented yet")


def detect_infobox_type(text: str) -> InfoboxType | None:
    """Detect the type of infobox in the text without full parsing.

    This is a lightweight check to determine if an infobox exists
    and what type it is, without extracting all field values.

    Args:
        text: MediaWiki markup text to check.

    Returns:
        The InfoboxType if detected, None if no infobox found.

    Raises:
        NotImplementedError: Parser not yet implemented.

    Examples:
        >>> detect_infobox_type("{{Infobox politician|name=Test}}")
        'politician'

        >>> detect_infobox_type("No infobox here.")
        None
    """
    raise NotImplementedError("Parser not implemented yet")
