"""Link parser for MediaWiki markup.

This module extracts links from MediaWiki text:
- [[Target]] - Simple internal link
- [[Target|Display]] - Piped link (display text differs from target)
- [[Category:Name]] - Category link (special handling)
- [https://url text] - External link
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pw_mcp.ingest.parsers.types import Link, LinkType

if TYPE_CHECKING:
    from re import Match

# Compile patterns once at module level for performance
# Internal link pattern: [[Target]] or [[Target|Display]]
# Uses non-greedy matching to handle consecutive links
INTERNAL_LINK_PATTERN: re.Pattern[str] = re.compile(r"\[\[([^\[\]|]+?)(?:\|([^\[\]]*))?\]\]")

# External link pattern: [https://url] or [https://url display text]
EXTERNAL_LINK_PATTERN: re.Pattern[str] = re.compile(r"\[(https?://[^\s\]]+)(?:\s+([^\]]+))?\]")

# Prefixes that should be ignored (file/media links)
IGNORED_PREFIXES: frozenset[str] = frozenset(["File:", "Image:", "Media:"])

# Interwiki prefixes to ignore
INTERWIKI_PREFIXES: frozenset[str] = frozenset(
    ["ru:", "de:", "es:", "fr:", "zh:", "pt:", "ar:", "ja:", "ko:"]
)

# Category prefix constant
CATEGORY_PREFIX: str = "Category:"


def _split_anchor(target: str) -> tuple[str, str | None]:
    """Split target into article title and section anchor.

    Args:
        target: Link target that may contain #section.

    Returns:
        Tuple of (article_title, section_or_none).

    Examples:
        >>> _split_anchor("Article#Section")
        ('Article', 'Section')
        >>> _split_anchor("Article")
        ('Article', None)
    """
    if "#" in target:
        parts = target.split("#", maxsplit=1)
        return parts[0], parts[1] if len(parts) > 1 and parts[1] else None
    return target, None


def _should_ignore_target(target: str) -> bool:
    """Check if a link target should be ignored.

    Args:
        target: The link target to check.

    Returns:
        True if the target should be ignored.
    """
    # Skip empty targets
    if not target or not target.strip():
        return True

    # Skip file/media links
    if any(target.startswith(prefix) for prefix in IGNORED_PREFIXES):
        return True

    # Skip interwiki links
    return any(target.startswith(prefix) for prefix in INTERWIKI_PREFIXES)


def _parse_internal_link(match: Match[str], include_positions: bool) -> Link | None:
    """Parse a single internal link match.

    Args:
        match: Regex match object for internal link.
        include_positions: Whether to include start/end positions.

    Returns:
        Link object or None if link should be ignored.
    """
    raw_target = match.group(1)

    # Skip links that should be ignored
    if _should_ignore_target(raw_target):
        return None

    # Check if this is a category link
    if raw_target.startswith(CATEGORY_PREFIX):
        category_name = raw_target[len(CATEGORY_PREFIX) :]
        return Link(
            target=category_name,
            display=category_name,
            link_type="category",
            section=None,
            start=match.start() if include_positions else None,
            end=match.end() if include_positions else None,
        )

    # Regular internal link
    target, section = _split_anchor(raw_target)

    # Handle display text - group(2) is everything after first pipe
    # If display_text is None (no pipe), use target
    # If display_text is empty string (pipe with nothing after), use target
    # Otherwise use display_text as-is (handles multiple pipes case)
    display_text = match.group(2)
    display = display_text if display_text else target

    return Link(
        target=target,
        display=display,
        link_type="internal",
        section=section,
        start=match.start() if include_positions else None,
        end=match.end() if include_positions else None,
    )


def _parse_external_link(match: Match[str], include_positions: bool) -> Link:
    """Parse a single external link match.

    Args:
        match: Regex match object for external link.
        include_positions: Whether to include start/end positions.

    Returns:
        Link object for external link.
    """
    url = match.group(1)
    display_text = match.group(2)

    return Link(
        target=url,
        display=display_text.strip() if display_text else url,
        link_type="external",
        section=None,
        start=match.start() if include_positions else None,
        end=match.end() if include_positions else None,
    )


def parse_links(
    text: str,
    link_type: LinkType | None = None,
    include_positions: bool = False,
) -> list[Link]:
    """Extract links from MediaWiki markup.

    Args:
        text: MediaWiki markup text to parse.
        link_type: Optional filter to return only links of a specific type.
            If None, returns all link types.
        include_positions: If True, populate start/end fields for each link.

    Returns:
        List of Link objects extracted from the text.

    Examples:
        >>> parse_links("He opposed [[slavery]] and fought.")
        [Link(target='slavery', display='slavery', link_type='internal')]

        >>> parse_links("[[Category:Test]]", link_type="category")
        [Link(target='Test', display='Test', link_type='category')]
    """
    results: list[Link] = []

    # Parse internal/category links (only if not filtering to external only)
    if link_type != "external":
        for match in INTERNAL_LINK_PATTERN.finditer(text):
            link = _parse_internal_link(match, include_positions)
            # Apply type filter if specified, skip None results
            if link is not None and (link_type is None or link.link_type == link_type):
                results.append(link)

    # Parse external links (only if not filtering to internal/category)
    if link_type is None or link_type == "external":
        for match in EXTERNAL_LINK_PATTERN.finditer(text):
            link = _parse_external_link(match, include_positions)
            results.append(link)

    # Sort by position if positions are tracked, to maintain text order
    if include_positions:
        results.sort(key=lambda lnk: lnk.start or 0)

    return results


def count_internal_links(text: str) -> int:
    """Count the number of internal links in text.

    Category links are not counted as internal links.

    Args:
        text: MediaWiki markup text to parse.

    Returns:
        Number of internal (non-category) links.

    Examples:
        >>> count_internal_links("[[Foo]] and [[Bar]] and [[Category:C]]")
        2
    """
    return len(parse_links(text, link_type="internal"))


def count_categories(text: str) -> int:
    """Count the number of category links in text.

    Args:
        text: MediaWiki markup text to parse.

    Returns:
        Number of category links.

    Examples:
        >>> count_categories("[[Category:A]][[Category:B]]")
        2
    """
    return len(parse_links(text, link_type="category"))


def get_unique_targets(text: str) -> set[str]:
    """Get unique set of internal link targets.

    Useful for building a link graph or determining article connections.
    Category links are not included in the result.

    Args:
        text: MediaWiki markup text to parse.

    Returns:
        Set of unique internal link target article titles.

    Examples:
        >>> get_unique_targets("[[A]] appears twice: [[A]] and [[B]].")
        {'A', 'B'}
    """
    links = parse_links(text, link_type="internal")
    return {link.target for link in links}
