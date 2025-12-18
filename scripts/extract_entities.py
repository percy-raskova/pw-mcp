#!/usr/bin/env python3
"""
Extract verified entities from ProleWiki corpus.

Extracts:
- Article titles (from filenames)
- Internal wiki links [[Entity]]
- Categories [[Category:X]]
- Library references [[Library:Work]]
- Infobox person names

Outputs a JSON whitelist for use in entity verification reward function.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def extract_article_titles(corpus_dir: Path) -> set[str]:
    """Extract article titles from filenames."""
    titles = set()
    for txt_file in corpus_dir.rglob("*.txt"):
        # Remove .txt extension and get the title
        title = txt_file.stem
        # Also get namespace prefix
        namespace = txt_file.parent.name
        titles.add(title)
        # Store with namespace for disambiguation
        if namespace in ("Main", "Library", "Essays"):
            titles.add(f"{namespace}:{title}")
    return titles


def extract_wiki_links(content: str) -> set[str]:
    """Extract internal wiki links [[Entity]] or [[Entity|Display]]."""
    # Match [[Link]] or [[Link|Display text]]
    pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"
    matches = re.findall(pattern, content)

    links = set()
    for match in matches:
        # Skip special prefixes we'll handle separately
        if match.startswith(("Category:", "File:", "Image:")):
            continue
        # Clean up the link
        link = match.strip()
        if link:
            links.add(link)
    return links


def extract_categories(content: str) -> set[str]:
    """Extract categories [[Category:Name]]."""
    pattern = r"\[\[Category:([^\]|]+)(?:\|[^\]]+)?\]\]"
    matches = re.findall(pattern, content)
    return {m.strip() for m in matches if m.strip()}


def extract_library_refs(content: str) -> set[str]:
    """Extract library references [[Library:Work]]."""
    pattern = r"\[\[Library:([^\]|]+)(?:\|[^\]]+)?\]\]"
    matches = re.findall(pattern, content)
    return {m.strip() for m in matches if m.strip()}


def extract_infobox_names(content: str) -> set[str]:
    """Extract names from infobox person templates."""
    names = set()

    # Match | name = Value patterns in infoboxes
    name_pattern = r"\|\s*name\s*=\s*([^\n|]+)"
    matches = re.findall(name_pattern, content, re.IGNORECASE)
    for match in matches:
        name = match.strip()
        if name and not name.startswith(("{{", "[[", "File:", "<")):
            names.add(name)

    # Match | birth_name = Value
    birth_name_pattern = r"\|\s*birth_name\s*=\s*([^\n|]+)"
    matches = re.findall(birth_name_pattern, content, re.IGNORECASE)
    for match in matches:
        name = match.strip()
        if name and not name.startswith(("{{", "[[", "File:", "<")):
            names.add(name)

    return names


def extract_dates_and_years(content: str) -> set[str]:
    """Extract years and date references."""
    dates = set()

    # Four-digit years
    year_pattern = r"\b(1[0-9]{3}|20[0-2][0-9])\b"
    years = re.findall(year_pattern, content)
    dates.update(years)

    return dates


def categorize_entity(entity: str) -> str:
    """Attempt to categorize an entity based on patterns."""
    entity_lower = entity.lower()

    # Organizations/Parties
    if any(
        kw in entity_lower
        for kw in [
            "party",
            "communist",
            "socialist",
            "international",
            "league",
            "union",
            "front",
            "movement",
            "organization",
            "committee",
            "council",
            "congress",
            "federation",
            "association",
        ]
    ):
        return "organization"

    # Countries/Regions
    if any(
        kw in entity_lower
        for kw in [
            "republic",
            "kingdom",
            "empire",
            "soviet",
            "people's",
            "democratic republic",
            "socialist republic",
        ]
    ):
        return "location"

    # Events
    if any(
        kw in entity_lower
        for kw in [
            "revolution",
            "war",
            "strike",
            "uprising",
            "coup",
            "massacre",
            "incident",
            "crisis",
            "election",
        ]
    ):
        return "event"

    # Concepts/Theory
    if any(
        kw in entity_lower
        for kw in [
            "ism",
            "theory",
            "materialism",
            "dialectic",
            "class",
            "capitalism",
            "socialism",
            "communism",
            "imperialism",
        ]
    ):
        return "concept"

    # Library works often have specific patterns
    if entity.startswith("Library:"):
        return "work"

    # Default - likely a person or general entity
    return "general"


def main() -> None:
    corpus_dir = Path("/home/user/projects/pw-mcp/prolewiki-exports")
    output_dir = Path("/home/user/projects/pw-mcp/training_data")

    if not corpus_dir.exists():
        print(f"ERROR: Corpus directory not found: {corpus_dir}")
        sys.exit(1)

    print("=== ProleWiki Entity Extraction ===")
    print()

    # Collect all entities
    all_entities: dict[str, set[str]] = defaultdict(set)
    entity_counts: Counter[str] = Counter()
    file_count = 0

    # Process each namespace
    for namespace_dir in corpus_dir.iterdir():
        if not namespace_dir.is_dir():
            continue

        namespace = namespace_dir.name
        print(f"Processing {namespace}/...")

        for txt_file in namespace_dir.glob("*.txt"):
            file_count += 1

            # Article title from filename
            title = txt_file.stem
            all_entities["article_titles"].add(title)
            entity_counts[title] += 1

            # Read content
            try:
                content = txt_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                try:
                    content = txt_file.read_text(encoding="latin-1")
                except Exception as e:
                    print(f"  Warning: Could not read {txt_file.name}: {e}")
                    continue

            # Extract internal links
            links = extract_wiki_links(content)
            all_entities["wiki_links"].update(links)
            for link in links:
                entity_counts[link] += 1

            # Extract categories
            categories = extract_categories(content)
            all_entities["categories"].update(categories)

            # Extract library references
            library_refs = extract_library_refs(content)
            all_entities["library_works"].update(library_refs)

            # Extract infobox names
            names = extract_infobox_names(content)
            all_entities["person_names"].update(names)

            # Extract years (for date verification)
            years = extract_dates_and_years(content)
            all_entities["years"].update(years)

    print()
    print(f"Processed {file_count} files")
    print()

    # Build combined entity set
    combined_entities: set[str] = set()
    combined_entities.update(all_entities["article_titles"])
    combined_entities.update(all_entities["wiki_links"])
    combined_entities.update(all_entities["library_works"])
    combined_entities.update(all_entities["person_names"])

    # Categorize entities
    categorized: dict[str, list[str]] = defaultdict(list)
    for entity in sorted(combined_entities):
        category = categorize_entity(entity)
        categorized[category].append(entity)

    # Print statistics
    print("=== Extraction Statistics ===")
    print(f"Article titles: {len(all_entities['article_titles']):,}")
    print(f"Wiki links: {len(all_entities['wiki_links']):,}")
    print(f"Categories: {len(all_entities['categories']):,}")
    print(f"Library works: {len(all_entities['library_works']):,}")
    print(f"Person names: {len(all_entities['person_names']):,}")
    print(f"Years referenced: {len(all_entities['years']):,}")
    print()
    print(f"Combined unique entities: {len(combined_entities):,}")
    print()
    print("=== By Category ===")
    for cat in sorted(categorized.keys()):
        print(f"  {cat}: {len(categorized[cat]):,}")
    print()

    # Most referenced entities
    print("=== Top 30 Most Referenced Entities ===")
    for entity, count in entity_counts.most_common(30):
        print(f"  {count:4d}x  {entity}")
    print()

    # Prepare output
    output = {
        "metadata": {
            "source": "prolewiki-exports",
            "files_processed": file_count,
            "extraction_date": "2025-12-18",
            "total_entities": len(combined_entities),
        },
        "statistics": {
            "article_titles": len(all_entities["article_titles"]),
            "wiki_links": len(all_entities["wiki_links"]),
            "categories": len(all_entities["categories"]),
            "library_works": len(all_entities["library_works"]),
            "person_names": len(all_entities["person_names"]),
            "years": len(all_entities["years"]),
        },
        "by_category": {cat: sorted(entities) for cat, entities in categorized.items()},
        "categories": sorted(all_entities["categories"]),
        "years": sorted(all_entities["years"]),
        "all_entities": sorted(combined_entities),
    }

    # Save output
    output_file = output_dir / "entity_whitelist.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
