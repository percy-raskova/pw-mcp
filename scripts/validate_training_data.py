#!/usr/bin/env python3
"""
Validate all training data against schema and manifest.

Checks:
1. All JSONL files parse correctly
2. All records conform to qa_record.schema.json
3. Manifest record counts match actual files
4. No duplicate IDs
5. All required metadata fields present

Usage:
    python scripts/validate_training_data.py
    python scripts/validate_training_data.py --verbose
    python scripts/validate_training_data.py --skip-schema  # Skip schema validation
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    import jsonschema  # type: ignore[import-untyped]

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    jsonschema = None

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_schema(schema_path: Path) -> dict[str, object] | None:
    """Load JSON schema from file."""
    if not HAS_JSONSCHEMA:
        print("Warning: jsonschema not installed, skipping schema validation")
        return None

    if not schema_path.exists():
        print(f"Warning: Schema file not found: {schema_path}")
        return None

    with schema_path.open() as f:
        result: dict[str, object] = json.load(f)
        return result


def validate_jsonl_syntax(file_path: Path) -> list[str]:
    """Validate JSONL file syntax."""
    errors: list[str] = []

    try:
        with file_path.open() as f:
            for line_num, line in enumerate(f, 1):
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    errors.append(f"{file_path}:{line_num}: JSON parse error: {e}")
    except FileNotFoundError:
        errors.append(f"{file_path}: File not found")
    except OSError as e:
        errors.append(f"{file_path}: IO error: {e}")

    return errors


def validate_schema_compliance(
    file_path: Path,
    schema: dict[str, object] | None,
) -> list[str]:
    """Validate records against JSON schema."""
    if schema is None:
        return []

    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(schema)

    with file_path.open() as f:
        for line_num, line in enumerate(f, 1):
            record = json.loads(line)
            validation_errors = list(validator.iter_errors(record))
            for err in validation_errors:
                path = ".".join(str(p) for p in err.absolute_path)
                errors.append(f"{file_path}:{line_num}: {path}: {err.message}")

    return errors


def check_duplicate_ids(files: list[Path]) -> list[str]:
    """Check for duplicate qa_ids across all files."""
    errors: list[str] = []
    all_ids: Counter[str] = Counter()

    for file_path in files:
        with file_path.open() as f:
            for _line_num, line in enumerate(f, 1):
                record = json.loads(line)
                qa_id = record.get("qa_id")
                if qa_id:
                    all_ids[qa_id] += 1

    for qa_id, count in all_ids.items():
        if count > 1:
            errors.append(f"Duplicate qa_id '{qa_id}' appears {count} times")

    return errors


def validate_manifest_counts(
    manifest_path: Path,
    files: list[Path],
) -> list[str]:
    """Validate manifest record counts match actual files."""
    if not HAS_YAML:
        print("Warning: PyYAML not installed, skipping manifest validation")
        return []

    if not manifest_path.exists():
        return [f"Manifest not found: {manifest_path}"]

    errors: list[str] = []

    with manifest_path.open() as f:
        manifest = yaml.safe_load(f)

    # Build actual counts
    actual_counts: dict[str, int] = {}
    for file_path in files:
        rel_path = str(file_path.relative_to(file_path.parent.parent))
        with file_path.open() as f:
            actual_counts[rel_path] = sum(1 for _ in f)

    # Check manifest entries
    for entry in manifest.get("files", []):
        filename = entry.get("filename", "")
        expected = entry.get("record_count", 0)
        actual = actual_counts.get(filename)

        if actual is None:
            continue  # File not in our list, skip
        if expected != actual:
            errors.append(f"Manifest mismatch for {filename}: expected {expected}, actual {actual}")

    return errors


def get_all_training_files(base_path: Path) -> list[Path]:
    """Get all training data JSONL files."""
    files: list[Path] = []

    # Source files
    sources_dir = base_path / "sources"
    if sources_dir.exists():
        files.extend(sources_dir.glob("**/*.jsonl"))

    # Synthetic files
    files.extend(base_path.glob("synthetic_*.jsonl"))

    # Legacy files
    for legacy in ["curated_qa.jsonl", "grpo_dataset.jsonl"]:
        legacy_path = base_path / legacy
        if legacy_path.exists():
            files.append(legacy_path)

    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate training data files")
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("training_data"),
        help="Base directory for training data",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("training_data/schema/qa_record.schema.json"),
        help="Path to JSON schema",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip schema validation (useful for legacy files)",
    )
    args = parser.parse_args()

    print(f"Validating training data in {args.base}...\n")

    # Get all files
    files = get_all_training_files(args.base)
    print(f"Found {len(files)} training data files\n")

    all_errors: list[str] = []
    total_records = 0

    # Load schema
    schema = None if args.skip_schema else load_schema(args.schema)

    # Validate each file
    for file_path in files:
        if args.verbose:
            print(f"Checking {file_path.relative_to(args.base)}...")

        # Syntax check
        syntax_errors = validate_jsonl_syntax(file_path)
        all_errors.extend(syntax_errors)

        if syntax_errors:
            print(f"  SYNTAX ERROR: {len(syntax_errors)} issues")
            continue

        # Count records
        with file_path.open() as f:
            count = sum(1 for _ in f)
            total_records += count

        if args.verbose:
            print(f"  OK ({count} records)")

        # Schema check (skip for legacy files)
        if schema and "sources/" in str(file_path):
            schema_errors = validate_schema_compliance(file_path, schema)
            all_errors.extend(schema_errors[:10])  # Limit errors per file
            if schema_errors:
                print(f"  SCHEMA ERRORS: {len(schema_errors)} issues")

    # Check for duplicates
    print("\nChecking for duplicate IDs...")
    dup_errors = check_duplicate_ids(files)
    all_errors.extend(dup_errors)
    if dup_errors:
        print(f"  Found {len(dup_errors)} duplicate IDs")
    else:
        print("  No duplicates found")

    # Check manifest
    manifest_path = args.base / "MANIFEST.yaml"
    print(f"\nValidating manifest {manifest_path}...")
    manifest_errors = validate_manifest_counts(manifest_path, files)
    all_errors.extend(manifest_errors)
    if manifest_errors:
        for err in manifest_errors[:5]:
            print(f"  {err}")
    else:
        print("  Manifest counts match")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Files checked: {len(files)}")
    print(f"Total records: {total_records}")
    print(f"Errors found: {len(all_errors)}")

    if all_errors:
        print("\nErrors:")
        for error in all_errors[:20]:
            print(f"  - {error}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
        exit(1)
    else:
        print("\nAll validations passed!")
        exit(0)


if __name__ == "__main__":
    main()
