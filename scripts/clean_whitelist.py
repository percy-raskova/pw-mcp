#!/usr/bin/env python3
"""Clean up entity whitelist by removing wiki markup artifacts."""

import json
import re
from pathlib import Path


def clean_entity(entity: str) -> str:
    """Remove wiki markup and normalize entity."""
    # Remove wiki bold/italic markers
    entity = re.sub(r"'''?", "", entity)
    # Remove leading/trailing whitespace
    entity = entity.strip()
    # Remove anchor fragments
    if "#" in entity:
        entity = entity.split("#")[0]
    return entity


def is_valid_entity(entity: str) -> bool:
    """Check if entity is a valid name (not markup artifact)."""
    if not entity:
        return False
    # Skip pure punctuation or very short
    if len(entity) < 2:
        return False
    # Skip things that look like wiki formatting
    if entity.startswith(("{{", "[[", "<", "|", "=")):
        return False
    if entity.endswith(("}}", "]]", ">")):
        return False
    # Skip URLs
    if entity.startswith(("http://", "https://", "www.")):
        return False
    # Skip pure numbers (but keep years)
    return not (entity.isdigit() and len(entity) != 4)


def main() -> None:
    input_file = Path("/home/user/projects/pw-mcp/training_data/entity_whitelist.json")
    output_file = Path("/home/user/projects/pw-mcp/training_data/entity_whitelist_clean.json")

    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    # Clean all entities
    cleaned_entities: set[str] = set()
    for entity in data["all_entities"]:
        cleaned = clean_entity(entity)
        if is_valid_entity(cleaned):
            cleaned_entities.add(cleaned)

    # Also create lowercase versions for case-insensitive matching
    lowercase_entities = {e.lower() for e in cleaned_entities}

    # Clean categories
    cleaned_categories = {
        clean_entity(cat) for cat in data["categories"] if is_valid_entity(clean_entity(cat))
    }

    print(f"Original entities: {len(data['all_entities']):,}")
    print(f"Cleaned entities: {len(cleaned_entities):,}")
    print(f"Removed: {len(data['all_entities']) - len(cleaned_entities):,}")
    print()

    # Build output
    output = {
        "metadata": {
            **data["metadata"],
            "cleaned": True,
            "cleaning_date": "2025-12-18",
            "total_entities": len(cleaned_entities),
            "lowercase_count": len(lowercase_entities),
        },
        "entities": sorted(cleaned_entities),
        "entities_lowercase": sorted(lowercase_entities),
        "categories": sorted(cleaned_categories),
        "years": data["years"],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
