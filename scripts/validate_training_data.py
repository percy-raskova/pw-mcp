#!/usr/bin/env python3
"""Validate training data files against JSON Schema.

This script validates JSONL training data files against the formal
JSON Schema defined in training_data/schema/training_record.schema.json.

Usage:
    # Validate all JSONL files in training_data/
    uv run python scripts/validate_training_data.py

    # Validate specific file
    uv run python scripts/validate_training_data.py training_data/curated_qa.jsonl

    # Validate with verbose output
    uv run python scripts/validate_training_data.py --verbose

    # Generate statistics
    uv run python scripts/validate_training_data.py --stats

    # Check manifest integrity (SHA256 verification)
    uv run python scripts/validate_training_data.py --verify-checksums
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

# Try to import jsonschema, provide helpful error if missing
try:
    from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    Draft202012Validator = None


@dataclass
class ValidationResult:
    """Result of validating a single file."""

    filename: str
    total_records: int
    valid_records: int
    invalid_records: int
    errors: list[tuple[int, str]]  # (line_number, error_message)
    schema_version: str

    @property
    def is_valid(self) -> bool:
        """Return True if all records are valid."""
        return self.invalid_records == 0

    @property
    def validity_rate(self) -> float:
        """Return percentage of valid records."""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load JSON Schema from file."""
    result: dict[str, Any] = json.loads(schema_path.read_text())
    return result


def validate_record(record: dict[str, Any], validator: Draft202012Validator) -> list[str]:
    """Validate a single record against schema, return list of errors."""
    errors = []
    for error in validator.iter_errors(record):
        path = ".".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
        errors.append(f"{path}: {error.message}")
    return errors


def validate_file(
    filepath: Path,
    validator: Draft202012Validator,
    *,
    verbose: bool = False,
) -> ValidationResult:
    """Validate all records in a JSONL file."""
    errors: list[tuple[int, str]] = []
    valid_count = 0
    total_count = 0

    with filepath.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total_count += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((line_num, f"JSON parse error: {e}"))
                continue

            record_errors = validate_record(record, validator)
            if record_errors:
                for err in record_errors:
                    errors.append((line_num, err))
                    if verbose:
                        print(f"  Line {line_num}: {err}", file=sys.stderr)
            else:
                valid_count += 1

    return ValidationResult(
        filename=filepath.name,
        total_records=total_count,
        valid_records=valid_count,
        invalid_records=total_count - valid_count,
        errors=errors,
        schema_version="2020-12",
    )


def validate_legacy_format(filepath: Path, *, verbose: bool = False) -> ValidationResult:
    """Validate legacy format files (instruction/response only)."""
    errors: list[tuple[int, str]] = []
    valid_count = 0
    total_count = 0

    with filepath.open() as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total_count += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append((line_num, f"JSON parse error: {e}"))
                continue

            # Check for required fields in legacy format
            record_errors = []
            if "instruction" not in record and "prompt" not in record:
                record_errors.append("Missing 'instruction' or 'prompt' field")
            if "response" not in record and "answer" not in record:
                record_errors.append("Missing 'response' or 'answer' field")

            if record_errors:
                for err in record_errors:
                    errors.append((line_num, err))
                    if verbose:
                        print(f"  Line {line_num}: {err}", file=sys.stderr)
            else:
                valid_count += 1

    return ValidationResult(
        filename=filepath.name,
        total_records=total_count,
        valid_records=valid_count,
        invalid_records=total_count - valid_count,
        errors=errors,
        schema_version="legacy",
    )


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256_hash = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def verify_manifest_checksums(manifest_path: Path, data_dir: Path) -> list[str]:
    """Verify SHA256 checksums in manifest match actual files."""
    import yaml

    errors = []
    manifest = yaml.safe_load(manifest_path.read_text())

    for file_entry in manifest.get("files", []):
        filename = file_entry["filename"]
        expected_hash = file_entry["sha256"]
        filepath = data_dir / filename

        if not filepath.exists():
            errors.append(f"{filename}: File not found")
            continue

        actual_hash = compute_sha256(filepath)
        if actual_hash != expected_hash:
            errors.append(
                f"{filename}: Hash mismatch\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )

    return errors


def collect_statistics(data_dir: Path) -> dict[str, Any]:
    """Collect statistics across all JSONL files."""
    stats: dict[str, Any] = {
        "total_files": 0,
        "total_records": 0,
        "by_file": {},
        "categories": Counter(),
        "instruction_lengths": [],
        "response_lengths": [],
    }

    for filepath in data_dir.glob("*.jsonl"):
        stats["total_files"] += 1
        file_records = 0

        with filepath.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    file_records += 1
                    stats["total_records"] += 1

                    # Get instruction/response text
                    instruction = record.get("instruction") or record.get("prompt", [{}])
                    if isinstance(instruction, list):
                        instruction = instruction[-1].get("content", "") if instruction else ""
                    response = record.get("response") or record.get("answer", "")

                    stats["instruction_lengths"].append(len(str(instruction)))
                    stats["response_lengths"].append(len(str(response)))

                    # Extract categories if present
                    if "metadata" in record:
                        cats = record["metadata"].get("classification", {}).get("categories", [])
                        for cat in cats:
                            stats["categories"][cat] += 1

                except json.JSONDecodeError:
                    pass

        stats["by_file"][filepath.name] = file_records

    # Compute averages
    if stats["instruction_lengths"]:
        stats["avg_instruction_length"] = sum(stats["instruction_lengths"]) / len(
            stats["instruction_lengths"]
        )
    if stats["response_lengths"]:
        stats["avg_response_length"] = sum(stats["response_lengths"]) / len(
            stats["response_lengths"]
        )

    # Clean up raw lists
    del stats["instruction_lengths"]
    del stats["response_lengths"]
    stats["categories"] = dict(stats["categories"].most_common(20))

    return stats


def find_jsonl_files(data_dir: Path) -> Iterator[Path]:
    """Find all JSONL files in directory."""
    yield from data_dir.glob("*.jsonl")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate training data against JSON Schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Specific files to validate (default: all JSONL in training_data/)",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("training_data/schema/training_record.schema.json"),
        help="Path to JSON Schema file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("training_data"),
        help="Directory containing training data",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show individual validation errors"
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Validate against legacy format (instruction/response only)",
    )
    parser.add_argument("--stats", action="store_true", help="Generate and display statistics")
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="Verify SHA256 checksums from MANIFEST.yaml",
    )

    args = parser.parse_args()

    # Statistics mode
    if args.stats:
        print("Collecting statistics...")
        stats = collect_statistics(args.data_dir)
        print(f"\nTotal files: {stats['total_files']}")
        print(f"Total records: {stats['total_records']}")
        print("\nRecords by file:")
        for filename, count in stats["by_file"].items():
            print(f"  {filename}: {count}")
        if "avg_instruction_length" in stats:
            print(f"\nAverage instruction length: {stats['avg_instruction_length']:.0f} chars")
        if "avg_response_length" in stats:
            print(f"Average response length: {stats['avg_response_length']:.0f} chars")
        if stats["categories"]:
            print("\nTop categories:")
            for cat, count in stats["categories"].items():
                print(f"  {cat}: {count}")
        return 0

    # Checksum verification mode
    if args.verify_checksums:
        manifest_path = args.data_dir / "MANIFEST.yaml"
        if not manifest_path.exists():
            print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
            return 1

        try:
            import yaml  # noqa: F401
        except ImportError:
            print("ERROR: PyYAML required for manifest verification", file=sys.stderr)
            print("Install with: uv add pyyaml", file=sys.stderr)
            return 1

        print("Verifying checksums...")
        errors = verify_manifest_checksums(manifest_path, args.data_dir)
        if errors:
            print("\nChecksum verification FAILED:")
            for error in errors:
                print(f"  {error}")
            return 1
        print("All checksums verified successfully!")
        return 0

    # Schema validation mode
    if not args.legacy and not HAS_JSONSCHEMA:
        print("ERROR: jsonschema package required for schema validation", file=sys.stderr)
        print("Install with: uv add jsonschema", file=sys.stderr)
        print("Or use --legacy flag for basic format validation", file=sys.stderr)
        return 1

    # Load schema
    validator = None
    if not args.legacy:
        if not args.schema.exists():
            print(f"ERROR: Schema not found: {args.schema}", file=sys.stderr)
            return 1
        schema = load_schema(args.schema)
        validator = Draft202012Validator(schema)

    # Find files to validate
    files = args.files if args.files else list(find_jsonl_files(args.data_dir))

    if not files:
        print("No JSONL files found to validate", file=sys.stderr)
        return 1

    # Validate each file
    results: list[ValidationResult] = []
    for filepath in files:
        if not filepath.exists():
            print(f"WARNING: File not found: {filepath}", file=sys.stderr)
            continue

        print(f"Validating {filepath.name}...")

        if args.legacy or validator is None:
            result = validate_legacy_format(filepath, verbose=args.verbose)
        else:
            result = validate_file(filepath, validator, verbose=args.verbose)

        results.append(result)

        status = "PASS" if result.is_valid else "FAIL"
        print(
            f"  {status}: {result.valid_records}/{result.total_records} valid "
            f"({result.validity_rate:.1f}%)"
        )

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    total_records = sum(r.total_records for r in results)
    total_valid = sum(r.valid_records for r in results)
    total_invalid = sum(r.invalid_records for r in results)
    all_valid = all(r.is_valid for r in results)

    print(f"Files validated: {len(results)}")
    print(f"Total records: {total_records}")
    print(f"Valid records: {total_valid}")
    print(f"Invalid records: {total_invalid}")
    print(f"Overall validity: {(total_valid / total_records * 100) if total_records else 0:.1f}%")

    if all_valid:
        print("\n✓ All files passed validation!")
        return 0
    print("\n✗ Some files have validation errors")
    return 1


if __name__ == "__main__":
    sys.exit(main())
