"""Extraction pipeline for MediaWiki articles.

This module provides the main entry point for extracting structured data
from ProleWiki MediaWiki articles. It orchestrates the individual parsers
(link, citation, infobox) to produce a complete ArticleData object.

The extraction pipeline:
1. Parses infobox (if present) and removes from text
2. Extracts citations from <ref> tags
3. Extracts all links (internal, category, external)
4. Produces clean text with markup removed
5. Extracts metadata (sections, namespace, quality flags)
"""

from __future__ import annotations

import re
from typing import Final

from pw_mcp.ingest.parsers import (
    ArticleData,
    Citation,
    Link,
    parse_citations,
    parse_infobox,
    parse_links,
)
from pw_mcp.ingest.parsers.types import InfoboxData

# Compile patterns once at module level for performance

# Pattern to match section headers (== Header == or === Subheader ===)
SECTION_HEADER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(={2,6})\s*([^=]+?)\s*\1\s*$",
    re.MULTILINE,
)

# Pattern to match <ref>...</ref> tags (for counting and removal)
REF_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<ref(?:\s+[^>]*)?>.*?</ref>",
    re.DOTALL | re.IGNORECASE,
)

# Pattern to match self-closing <ref ... /> tags
REF_SELFCLOSE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<ref\s+[^>]*/\s*>",
    re.IGNORECASE,
)

# Pattern to match [[Category:Name]] links
CATEGORY_LINK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\[\[Category:([^\]|]+)(?:\|[^\]]*)?\]\]",
    re.IGNORECASE,
)

# Pattern to match {{Stub}} template (case-insensitive)
STUB_TEMPLATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\{\{\s*[Ss]tub\s*\}\}",
)

# Pattern to match {{Citation needed}} template (case-insensitive)
CITATION_NEEDED_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\{\{\s*[Cc]itation\s+needed\s*\}\}",
)

# Pattern to match <blockquote> tags
BLOCKQUOTE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<blockquote\b[^>]*>",
    re.IGNORECASE,
)

# Pattern to match internal links [[Target]] or [[Target|Display]]
# For clean text generation: converts to display text
INTERNAL_LINK_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]",
)

# Pattern to match wiki formatting (bold/italic)
WIKI_FORMATTING_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"'{2,5}",
)

# Pattern for <references /> or <references></references> tags
REFERENCES_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"<references\s*(?:/>|>.*?</references>)",
    re.DOTALL | re.IGNORECASE,
)

# Valid ProleWiki namespaces
VALID_NAMESPACES: Final[frozenset[str]] = frozenset(["Main", "Library", "Essays", "ProleWiki"])

# Default namespace when none can be determined
DEFAULT_NAMESPACE: Final[str] = "Main"


def _detect_namespace(source_path: str | None) -> str:
    """Detect namespace from source file path.

    Args:
        source_path: Relative path like "Main/Article.txt" or None.

    Returns:
        Namespace string (Main, Library, Essays, ProleWiki).
        Returns DEFAULT_NAMESPACE if path is None or namespace unknown.

    Examples:
        >>> _detect_namespace("Main/Article.txt")
        'Main'
        >>> _detect_namespace("Library/Capital Vol1.txt")
        'Library'
        >>> _detect_namespace(None)
        'Main'
    """
    if source_path is None:
        return DEFAULT_NAMESPACE

    # Handle paths with or without leading slash
    path = source_path.lstrip("/")

    # Split on first separator to get namespace
    if "/" in path:
        namespace_part = path.split("/", maxsplit=1)[0]
        if namespace_part in VALID_NAMESPACES:
            return namespace_part

    return DEFAULT_NAMESPACE


def _extract_sections(text: str) -> list[str]:
    """Extract section headers from MediaWiki text.

    Finds all == Header == style section markers and returns the
    header text (without the = signs).

    Args:
        text: MediaWiki markup text.

    Returns:
        List of section header names, in order of appearance.

    Examples:
        >>> _extract_sections("== History ==\\nText\\n=== Early ===")
        ['History', 'Early']
    """
    sections: list[str] = []
    for match in SECTION_HEADER_PATTERN.finditer(text):
        header_text = match.group(2).strip()
        if header_text:
            sections.append(header_text)
    return sections


def _count_references(text: str) -> int:
    """Count the number of <ref> tags in text.

    Counts both <ref>...</ref> and <ref name="..." /> forms.

    Args:
        text: MediaWiki markup text.

    Returns:
        Total count of reference tags.

    Examples:
        >>> _count_references("Text<ref>A</ref> and<ref name='x'>B</ref>")
        2
    """
    full_refs = len(REF_TAG_PATTERN.findall(text))
    self_close_refs = len(REF_SELFCLOSE_PATTERN.findall(text))
    return full_refs + self_close_refs


def _extract_categories(text: str) -> list[str]:
    """Extract category names from [[Category:X]] links.

    Args:
        text: MediaWiki markup text.

    Returns:
        List of category names (without "Category:" prefix).

    Examples:
        >>> _extract_categories("[[Category:Test]][[Category:Another]]")
        ['Test', 'Another']
    """
    categories: list[str] = []
    for match in CATEGORY_LINK_PATTERN.finditer(text):
        category_name = match.group(1).strip()
        if category_name:
            categories.append(category_name)
    return categories


def _detect_quality_flags(text: str) -> tuple[bool, int, bool]:
    """Detect article quality flags.

    Args:
        text: MediaWiki markup text.

    Returns:
        Tuple of (is_stub, citation_needed_count, has_blockquote).
    """
    is_stub = STUB_TEMPLATE_PATTERN.search(text) is not None
    citation_needed_count = len(CITATION_NEEDED_PATTERN.findall(text))
    has_blockquote = BLOCKQUOTE_PATTERN.search(text) is not None

    return is_stub, citation_needed_count, has_blockquote


def _convert_links_to_text(text: str) -> str:
    """Convert [[wiki links]] to plain text.

    For [[Target|Display]], keeps Display.
    For [[Target]], keeps Target.

    Args:
        text: Text containing wiki link markup.

    Returns:
        Text with links converted to plain text.

    Examples:
        >>> _convert_links_to_text("See [[Article|the article]] for more.")
        'See the article for more.'
    """

    def replace_link(match: re.Match[str]) -> str:
        target = match.group(1)
        display = match.group(2)
        # Use display text if provided, otherwise use target
        return display if display else target

    return INTERNAL_LINK_PATTERN.sub(replace_link, text)


def _generate_clean_text(
    text: str,
    infobox: InfoboxData | None,
) -> str:
    """Generate clean text from MediaWiki markup.

    Removes:
    - Infobox (uses remaining_text from parsed infobox)
    - [[Category:X]] links
    - <ref>...</ref> tags
    - <references /> tags
    - {{Stub}} and {{Citation needed}} templates
    - Wiki formatting (bold/italic)

    Converts:
    - [[Link|Text]] to "Text"
    - [[Link]] to "Link"

    Preserves:
    - <blockquote> content (tags may be stripped but content kept)
    - Section headers as plain text (= signs removed)

    Args:
        text: Original MediaWiki markup.
        infobox: Parsed infobox data (provides remaining_text).

    Returns:
        Clean text suitable for embedding.
    """
    # Start with infobox-removed text if available
    clean = infobox.remaining_text if infobox is not None else text

    # Remove <ref>...</ref> tags
    clean = REF_TAG_PATTERN.sub("", clean)

    # Remove self-closing ref tags
    clean = REF_SELFCLOSE_PATTERN.sub("", clean)

    # Remove <references /> tags
    clean = REFERENCES_TAG_PATTERN.sub("", clean)

    # Remove [[Category:X]] links
    clean = CATEGORY_LINK_PATTERN.sub("", clean)

    # Remove {{Stub}} templates
    clean = STUB_TEMPLATE_PATTERN.sub("", clean)

    # Remove {{Citation needed}} templates
    clean = CITATION_NEEDED_PATTERN.sub("", clean)

    # Convert [[links]] to plain text
    clean = _convert_links_to_text(clean)

    # Remove wiki formatting (bold/italic)
    clean = WIKI_FORMATTING_PATTERN.sub("", clean)

    # Convert section headers to plain text (remove = signs)
    def replace_header(match: re.Match[str]) -> str:
        return match.group(2).strip()

    clean = SECTION_HEADER_PATTERN.sub(replace_header, clean)

    # Clean up excessive whitespace while preserving paragraph breaks
    # Split into lines, strip each, rejoin
    lines = clean.split("\n")
    lines = [line.strip() for line in lines]

    # Collapse multiple blank lines into single blank line
    result_lines: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and prev_blank:
            continue
        result_lines.append(line)
        prev_blank = is_blank

    clean = "\n".join(result_lines).strip()

    return clean


def extract_article(
    text: str,
    source_path: str | None = None,
) -> ArticleData:
    """Extract structured data from MediaWiki article.

    This is the main entry point for the extraction pipeline. It orchestrates
    all parsers to produce a complete ArticleData object with:
    - Clean text (markup removed)
    - Parsed infobox (if present)
    - All citations
    - All internal links
    - Categories
    - Section headers
    - Namespace
    - Quality flags

    Args:
        text: MediaWiki markup text of the article.
        source_path: Optional relative path to source file (e.g., "Main/Article.txt").
            Used to detect namespace.

    Returns:
        ArticleData with all extracted information.

    Examples:
        >>> result = extract_article("{{Infobox politician|name=Test}}\\nContent.")
        >>> result.infobox.fields['name']
        'Test'
        >>> "Content" in result.clean_text
        True
    """
    # Parse infobox (may be None)
    infobox = parse_infobox(text)

    # Parse citations
    citations: list[Citation] = parse_citations(text)

    # Parse all links
    all_links = parse_links(text)

    # Separate internal links and categories
    internal_links: list[Link] = [link for link in all_links if link.link_type == "internal"]
    categories: list[str] = [link.target for link in all_links if link.link_type == "category"]

    # If no category links found via parse_links, try extracting directly
    # (some category links might be malformed)
    if not categories:
        categories = _extract_categories(text)

    # Extract section headers
    sections = _extract_sections(text)

    # Detect namespace from source path
    namespace = _detect_namespace(source_path)

    # Count references
    reference_count = _count_references(text)

    # Detect quality flags
    is_stub, citation_needed_count, has_blockquote = _detect_quality_flags(text)

    # Generate clean text
    clean_text = _generate_clean_text(text, infobox)

    return ArticleData(
        clean_text=clean_text,
        infobox=infobox,
        citations=citations,
        internal_links=internal_links,
        categories=categories,
        sections=sections,
        namespace=namespace,
        reference_count=reference_count,
        is_stub=is_stub,
        citation_needed_count=citation_needed_count,
        has_blockquote=has_blockquote,
    )
