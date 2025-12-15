"""Parser modules for MediaWiki extraction.

This package provides parsers for extracting structured data from
MediaWiki markup used in ProleWiki:

- **Links**: Internal [[links]], [[Category:]] links, and external [url] links
- **Citations**: {{Citation}}, {{Web citation}}, {{News citation}}, etc.
- **Infoboxes**: {{Infobox politician}}, {{Infobox country}}, etc.

All parsers are currently stubs that raise NotImplementedError.
This allows tests to transition from "skip" to "fail with NotImplementedError"
as part of the TDD Green Phase setup.

Example usage::

    from pw_mcp.ingest.parsers import parse_links, Link

    links = parse_links("See [[Article]] for more.")
    # Currently raises NotImplementedError

Public API:
    Types:
        - Link: Dataclass for parsed links
        - Citation: Dataclass for parsed citations
        - InfoboxData: Dataclass for parsed infobox data
        - LinkType: Literal type for link categories
        - CitationType: Literal type for citation categories
        - InfoboxType: Literal type for infobox categories

    Functions:
        - parse_links: Extract all links from text
        - count_internal_links: Count non-category internal links
        - count_categories: Count category links
        - get_unique_targets: Get unique set of link targets
        - parse_citations: Extract all citations from text
        - parse_ref_tags: Extract citations from <ref> tags
        - parse_infobox: Extract infobox data from text
        - detect_infobox_type: Detect infobox type without full parsing
"""

from pw_mcp.ingest.parsers.citation import (
    parse_citations,
    parse_ref_tags,
)
from pw_mcp.ingest.parsers.infobox import (
    detect_infobox_type,
    parse_infobox,
)
from pw_mcp.ingest.parsers.link import (
    count_categories,
    count_internal_links,
    get_unique_targets,
    parse_links,
)
from pw_mcp.ingest.parsers.types import (
    Citation,
    CitationType,
    InfoboxData,
    InfoboxType,
    Link,
    LinkType,
)

__all__ = [
    "Citation",
    "CitationType",
    "InfoboxData",
    "InfoboxType",
    "Link",
    "LinkType",
    "count_categories",
    "count_internal_links",
    "detect_infobox_type",
    "get_unique_targets",
    "parse_citations",
    "parse_infobox",
    "parse_links",
    "parse_ref_tags",
]
