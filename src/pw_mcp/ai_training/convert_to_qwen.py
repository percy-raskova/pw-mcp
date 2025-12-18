#!/usr/bin/env python3
"""Convert instruction/response pairs to Qwen chat template format."""

import json
from pathlib import Path

SYSTEM_PROMPT = """You are a Marxist-Leninist assistant trained on ProleWiki and critical theory. You provide accurate information about socialist history, theory, and practice from a Marxist-Leninist perspective. You explain concepts like dialectical materialism, historical materialism, class struggle, anti-colonialism, and socialist construction with clarity and ideological precision."""


def convert_to_qwen(input_path: Path, output_path: Path) -> int:
    """Convert instruction/response JSONL to Qwen chat template format."""
    count = 0
    with open(input_path) as infile, open(output_path, "w") as outfile:
        for line in infile:
            pair = json.loads(line)

            # Format for Qwen-2.5 chat template
            text = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{pair['instruction']}<|im_end|>
<|im_start|>assistant
{pair['response']}<|im_end|>"""

            outfile.write(json.dumps({"text": text}) + "\n")
            count += 1

    return count


if __name__ == "__main__":
    input_file = Path("training_data/curated_qa.jsonl")
    output_file = Path("training_data/formatted/train_qwen.jsonl")

    count = convert_to_qwen(input_file, output_file)
    print(f"Converted {count} pairs to Qwen format")
    print(f"Output: {output_file}")
