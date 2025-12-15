"""Parser modules for MediaWiki extraction.

This package provides parsers for extracting structured data from
MediaWiki markup used in ProleWiki:

- **Links**: Internal [[links]], [[Category:]] links, and external [url] links
- **Citations**: {{Citation}}, {{Web citation}}, {{News citation}}, etc.
- **Infoboxes**: {{Infobox politician}}, {{Infobox country}}, etc.
- **Quotes**: {{Quote}} block quote templates

Example usage::

    from pw_mcp.ingest.parsers import parse_links, Link

    links = parse_links("See [[Article]] for more.")

Public API:
    Types:
        - Link: Dataclass for parsed links
        - Citation: Dataclass for parsed citations
        - InfoboxData: Dataclass for parsed infobox data
        - QuoteData: Dataclass for parsed quote templates
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
        - parse_quotes: Extract block quotes from text
"""

from pw_mcp.ingest.parsers.citation import (
    parse_citations,
    parse_ref_tags,
)
from pw_mcp.ingest.parsers.infobox import (
    detect_infobox_type,
    parse_infobox,
)
from pw_mcp.ingest.parsers.library_work import (
    parse_library_work,
)
from pw_mcp.ingest.parsers.link import (
    count_categories,
    count_internal_links,
    get_unique_targets,
    parse_links,
)
from pw_mcp.ingest.parsers.quote import (
    parse_quotes,
)
from pw_mcp.ingest.parsers.types import (
    ArticleData,
    Citation,
    CitationType,
    InfoboxData,
    InfoboxType,
    LibraryWorkData,
    Link,
    LinkType,
    QuoteData,
)

__all__ = [
    "ArticleData",
    "Citation",
    "CitationType",
    "InfoboxData",
    "InfoboxType",
    "LibraryWorkData",
    "Link",
    "LinkType",
    "QuoteData",
    "count_categories",
    "count_internal_links",
    "detect_infobox_type",
    "get_unique_targets",
    "parse_citations",
    "parse_infobox",
    "parse_library_work",
    "parse_links",
    "parse_quotes",
    "parse_ref_tags",
]
