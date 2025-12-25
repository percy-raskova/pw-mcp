"""MediaWiki section header detection.

This module provides functions to detect and parse MediaWiki-style
section headers (== Title ==, === Subsection ===, etc.).
"""

from __future__ import annotations

import re

# Section header pattern: matches == Title ==, === Subsection ===, etc.
# Captures the equals signs to ensure balanced opening/closing
SECTION_HEADER_PATTERN = re.compile(r"^(={2,6})\s*([^=]+?)\s*\1\s*$")


def is_section_header(line: str) -> bool:
    """Check if a line is a MediaWiki section header.

    Section headers have the form: == Title == or === Subsection === etc.
    The number of equals signs must match on both sides (2-6).

    Args:
        line: The line to check

    Returns:
        True if the line is a section header, False otherwise

    Examples:
        >>> is_section_header("== Introduction ==")
        True
        >>> is_section_header("=== Subsection ===")
        True
        >>> is_section_header("a = b + c")
        False
    """
    return SECTION_HEADER_PATTERN.match(line) is not None


def extract_section_title(line: str) -> str:
    """Extract the title from a section header line.

    Args:
        line: A section header line (must pass is_section_header check)

    Returns:
        The section title with whitespace stripped

    Example:
        >>> extract_section_title("== Introduction ==")
        'Introduction'
    """
    match = SECTION_HEADER_PATTERN.match(line)
    if match:
        return match.group(2).strip()
    return ""
