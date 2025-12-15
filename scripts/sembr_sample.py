#!/usr/bin/env python3
"""Run sembr on a sample of files to estimate processing time."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

# Configuration
MODEL = "admko/sembr2023-distilbert-base-multilingual-cased"
INPUT_LIST = Path("/tmp/sembr_sample_files.txt")
OUTPUT_DIR = Path("/home/user/projects/pw-mcp/prolewiki-sembr-sample")
SOURCE_DIR = Path("/home/user/projects/pw-mcp/prolewiki-exports")


def main() -> None:
    files = INPUT_LIST.read_text().strip().split("\n")
    total_files = len(files)
    total_bytes = 0
    processed = 0
    errors = 0

    print(f"Processing {total_files} files with model: {MODEL}")
    print("-" * 60)

    start_time = time.time()

    for i, filepath in enumerate(files, 1):
        src = Path(filepath)
        if not src.exists():
            print(f"[{i}/{total_files}] SKIP: {src.name} (not found)")
            errors += 1
            continue

        # Determine output path (preserve namespace subdirectory)
        relative = src.relative_to(SOURCE_DIR)
        dest = OUTPUT_DIR / relative
        dest.parent.mkdir(parents=True, exist_ok=True)

        file_size = src.stat().st_size
        total_bytes += file_size

        try:
            # Run sembr
            result = subprocess.run(
                ["uv", "run", "sembr", "-m", MODEL, "-i", str(src), "-o", str(dest)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                processed += 1
                status = "OK"
            else:
                errors += 1
                status = "ERR"
        except subprocess.TimeoutExpired:
            errors += 1
            status = "TIMEOUT"
        except Exception as e:
            errors += 1
            status = f"ERR: {e}"

        elapsed = time.time() - start_time

        print(
            f"[{i}/{total_files}] {status}: {src.name[:40]:<40} "
            f"({file_size:,} bytes, {elapsed:.1f}s elapsed)"
        )

    # Summary
    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Completed: {processed}/{total_files} files")
    print(f"Errors: {errors}")
    print(f"Total bytes: {total_bytes:,}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Avg per file: {elapsed/total_files:.2f}s")
    print(f"Throughput: {total_bytes/elapsed/1024:.1f} KB/s")

    # Estimate full corpus
    # ~204MB total, 5222 files
    full_corpus_bytes = 204 * 1024 * 1024
    full_corpus_files = 5222
    est_time_by_bytes = full_corpus_bytes / (total_bytes / elapsed) if total_bytes > 0 else 0
    est_time_by_files = (elapsed / total_files) * full_corpus_files

    print("-" * 60)
    print("ESTIMATED TIME FOR FULL CORPUS (5,222 files, 204MB):")
    print(f"  By file count: {est_time_by_files/60:.1f} minutes")
    print(f"  By byte count: {est_time_by_bytes/60:.1f} minutes")


if __name__ == "__main__":
    main()
