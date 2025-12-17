#!/usr/bin/env python3
"""
Transform curated_qa.jsonl to GRPO training format.

Input format:  {"instruction": "...", "response": "..."}
Output format: {"prompt": [...], "answer": "..."}

Usage:
    python transform_to_grpo.py
"""

import json
from pathlib import Path

SYSTEM_PROMPT = """You are a Marxist-Leninist assistant trained on ProleWiki and critical theory.
Think through political theory questions using dialectical materialist analysis.
Show your reasoning in <think> tags, then provide a clear, well-sourced answer."""

INPUT_PATH = Path("training_data/curated_qa.jsonl")
OUTPUT_PATH = Path("training_data/grpo_dataset.jsonl")


def transform_qa_to_grpo(input_path: Path, output_path: Path) -> int:
    """Transform instruction/response pairs to GRPO format."""
    count = 0

    with open(input_path) as infile, open(output_path, "w") as outfile:
        for line in infile:
            item = json.loads(line)

            transformed = {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["instruction"]},
                ],
                "answer": item["response"],
            }

            outfile.write(json.dumps(transformed) + "\n")
            count += 1

    return count


def main() -> None:
    """Run transformation."""
    print(f"Transforming {INPUT_PATH} to GRPO format...")

    count = transform_qa_to_grpo(INPUT_PATH, OUTPUT_PATH)

    print(f"Transformed {count} examples")
    print(f"Output written to: {OUTPUT_PATH}")

    # Show sample
    print("\nSample output:")
    with open(OUTPUT_PATH) as f:
        sample = json.loads(f.readline())
        print(f"  System: {sample['prompt'][0]['content'][:60]}...")
        print(f"  User: {sample['prompt'][1]['content'][:60]}...")
        print(f"  Answer: {sample['answer'][:60]}...")


if __name__ == "__main__":
    main()
