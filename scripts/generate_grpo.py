#!/usr/bin/env python3
"""
Regenerate grpo_dataset.jsonl from source files.

This script replaces the static grpo_dataset.jsonl with a generated version
that pulls from all training_data/sources/**/*.jsonl files plus
training_data/synthetic_*.jsonl files.

Usage:
    python scripts/generate_grpo.py
    python scripts/generate_grpo.py --output training_data/grpo_dataset.jsonl
    python scripts/generate_grpo.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SYSTEM_PROMPT = """You are a Marxist-Leninist assistant trained on ProleWiki and critical theory.
Think through political theory questions using dialectical materialist analysis.
Show your reasoning in <think> tags, then provide a clear, well-sourced answer."""


def transform_to_grpo(record: dict[str, object]) -> dict[str, object]:
    """
    Transform instruction/response record to GRPO format.

    Input format (qa_record schema or simple instruction/response):
    {
        "qa_id": "...",
        "instruction": "...",
        "response": "...",
        "source": {...},
        ...
    }

    Output format (GRPO):
    {
        "prompt": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "answer": "..."
    }
    """
    instruction = record.get("instruction", "")
    response = record.get("response", "")

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        "answer": response,
    }


def collect_source_files(base_path: Path) -> list[Path]:
    """
    Collect all source files in consistent order.

    Order:
    1. Author-attributed sources (training_data/sources/**/*.jsonl)
    2. Synthetic correction data (training_data/synthetic_*.jsonl)
    """
    sources: list[Path] = []

    # 1. Author-attributed sources (sorted by path for consistency)
    sources_dir = base_path / "sources"
    if sources_dir.exists():
        sources.extend(sorted(sources_dir.glob("**/*.jsonl")))

    # 2. Synthetic correction data
    sources.extend(sorted(base_path.glob("synthetic_*.jsonl")))

    return sources


def generate_grpo_dataset(
    base_path: Path,
    output_path: Path,
    dry_run: bool = False,
) -> tuple[int, str]:
    """
    Generate GRPO dataset from all source files.

    Returns (record_count, sha256_hash).
    """
    source_files = collect_source_files(base_path)

    if dry_run:
        print(f"Would read from {len(source_files)} source files:")
        for sf in source_files:
            print(f"  - {sf.relative_to(base_path)}")
        return 0, ""

    records: list[dict[str, object]] = []

    for source_file in source_files:
        with source_file.open() as f:
            for line in f:
                record = json.loads(line)
                grpo_record = transform_to_grpo(record)
                records.append(grpo_record)

        print(f"  Processed {source_file.relative_to(base_path)}")

    # Write output
    with output_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    # Calculate SHA-256
    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()

    return len(records), sha256


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GRPO dataset from source files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_data/grpo_dataset.jsonl"),
        help="Output path for GRPO dataset",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("training_data"),
        help="Base directory for source files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview source files without generating output",
    )
    args = parser.parse_args()

    print(f"Generating GRPO dataset from {args.base}...")

    count, sha256 = generate_grpo_dataset(
        args.base,
        args.output,
        args.dry_run,
    )

    if not args.dry_run:
        print(f"\nGenerated {count} GRPO records")
        print(f"Output: {args.output}")
        print(f"SHA-256: {sha256}")
    else:
        print("\n[DRY-RUN] No output generated")


if __name__ == "__main__":
    main()
