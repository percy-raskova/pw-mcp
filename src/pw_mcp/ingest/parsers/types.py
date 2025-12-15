"""Shared type definitions for parser modules.

This module contains dataclasses and type aliases used across
the link, citation, and infobox parsers.
"""

from dataclasses import dataclass, field
from typing import Literal

# Type aliases for literal string types
LinkType = Literal["internal", "category", "external"]
"""Type of MediaWiki link: internal [[]], category [[Category:]], or external []."""

CitationType = Literal["book", "web", "news", "library", "youtube"]
"""Type of citation template used in ProleWiki.

Note: "book" corresponds to {{Citation|...}} template.
"""

InfoboxType = Literal[
    "politician",
    "country",
    "political_party",
    "person",
    "revolutionary",
    "essay",
    "philosopher",
    "company",
    "settlement",
    "military_person",
    "organization",
    "guerilla_organization",
    "youtuber",
    "military_conflict",
    "book",
    "religion",
    "website",
    "transcript",
    "library_work",
]
"""Type of infobox template detected in article."""


@dataclass
class Link:
    """Represents a MediaWiki link.

    Attributes:
        target: Article title (for internal/category) or URL (for external).
        display: Display text shown to reader.
        link_type: Type of link (internal, category, or external).
        section: Optional section anchor (from [[Article#Section]]).
        start: Optional start position in source text.
        end: Optional end position in source text.
    """

    target: str
    display: str
    link_type: LinkType
    section: str | None = None
    start: int | None = None
    end: int | None = None


@dataclass
class Citation:
    """Represents a citation/reference from MediaWiki.

    Attributes:
        type: Type of citation template (book, web, news, youtube, library).
        title: Title of the cited work.
        author: Primary author name (wiki links stripped).
        year: Publication year as string.
        url: Primary URL for the source.
        archive_url: Archive.org or similar backup URL.
        mia_url: Marxists Internet Archive URL.
        lg_url: Library Genesis URL.
        pdf_url: Direct PDF link.
        publisher: Publisher name.
        newspaper: Newspaper/publication name (for web/news citations).
        chapter: Chapter title (for book citations).
        page: Page number or range.
        isbn: ISBN number.
        date: Publication date (ISO format YYYY-MM-DD).
        retrieved: Date content was retrieved (ISO format).
        channel: YouTube channel name.
        quote: Quoted text from the source.
        ref_name: Named reference identifier (for <ref name="...">).
        link: Link to Library namespace article.
    """

    type: CitationType
    title: str | None = None
    author: str | None = None
    year: str | None = None
    url: str | None = None
    archive_url: str | None = None
    mia_url: str | None = None
    lg_url: str | None = None
    pdf_url: str | None = None
    publisher: str | None = None
    newspaper: str | None = None
    chapter: str | None = None
    page: str | None = None
    isbn: str | None = None
    date: str | None = None
    retrieved: str | None = None
    channel: str | None = None
    quote: str | None = None
    ref_name: str | None = None
    link: str | None = None


@dataclass
class InfoboxData:
    """Represents parsed infobox data.

    Attributes:
        type: Type of infobox (politician, country, etc.).
        fields: Parsed field values (some may be lists for multi-value fields).
        remaining_text: Article text with infobox removed.
        extracted_links: Internal links found within infobox fields.
        raw_fields: Original field values before processing.
    """

    type: InfoboxType
    fields: dict[str, str | list[str] | None]
    remaining_text: str
    extracted_links: list[str] = field(default_factory=list)
    raw_fields: dict[str, str] = field(default_factory=dict)
