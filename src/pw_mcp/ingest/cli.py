"""CLI for corpus ingestion pipeline.

Commands:
    pw-ingest sembr     Apply semantic linebreaking to extracted text
    pw-ingest           Legacy ingestion (not yet implemented)

Examples:
    # Check if sembr server is running
    pw-ingest sembr --check-only

    # Process full corpus
    pw-ingest sembr -i extracted/ -o sembr/

    # Process 10 files for testing
    pw-ingest sembr --sample 10
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from pw_mcp.ingest.linebreaker import SembrConfig, SembrResult


def _create_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="pw-ingest",
        description="Ingest ProleWiki corpus into ChromaDB",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Legacy arguments (for backward compatibility when no subcommand given)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("prolewiki-exports"),
        help="Path to ProleWiki exports directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chroma_data"),
        help="Path to ChromaDB data directory",
    )
    parser.add_argument(
        "--semantic-linebreak",
        action="store_true",
        help="Apply semantic linebreaking to source files in-place",
    )

    # =========================================================================
    # SEMBR SUBCOMMAND
    # =========================================================================
    sembr_parser = subparsers.add_parser(
        "sembr",
        help="Apply semantic linebreaking using sembr server",
        description=(
            "Process text files through sembr server for semantic linebreaking. "
            "Requires sembr server to be running: mise run sembr-server"
        ),
    )

    sembr_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("extracted"),
        help="Input directory containing .txt files (default: extracted/)",
    )
    sembr_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sembr"),
        help="Output directory for processed files (default: sembr/)",
    )
    sembr_parser.add_argument(
        "-s",
        "--server",
        default="http://localhost:8384",
        help="Sembr server URL (default: http://localhost:8384)",
    )
    sembr_parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check server health, don't process files",
    )
    sembr_parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Process only N files (for testing)",
    )
    sembr_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent file processing (default: 10)",
    )
    sembr_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    sembr_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout in seconds (default: 60.0)",
    )

    return parser


def _run_sembr_check(server_url: str) -> bool:
    """Check sembr server health and print status.

    Args:
        server_url: URL of the sembr server

    Returns:
        True if server is healthy, False otherwise
    """
    # Import here to avoid circular imports and speed up --help
    from pw_mcp.ingest.linebreaker import SembrConfig, check_server_health

    config = SembrConfig(server_url=server_url)
    healthy = check_server_health(config)

    if healthy:
        print(f"✓ Sembr server is healthy at {server_url}")
        return True
    else:
        print(f"✗ Sembr server is not responding at {server_url}")
        print("  Start the server with: mise run sembr-server")
        return False


def _run_sembr_process(args: argparse.Namespace) -> int:
    """Run sembr processing on input directory.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Import here to avoid circular imports and speed up --help
    from pw_mcp.ingest.linebreaker import (
        SembrConfig,
        SembrServerError,
        SembrTimeoutError,
        check_server_health,
    )

    input_dir: Path = args.input
    output_dir: Path = args.output
    server_url: str = args.server
    sample_count: int | None = args.sample
    max_concurrent: int = args.max_concurrent
    show_progress: bool = not args.no_progress
    timeout: float = args.timeout

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # Create config
    config = SembrConfig(
        server_url=server_url,
        timeout_seconds=timeout,
    )

    # Print header
    print("Sembr Processing")
    print("=" * 40)
    print(f"Server:  {server_url}", end=" ")

    # Check server health
    if not check_server_health(config):
        print("✗")
        print("\nError: Sembr server is not responding")
        print("Start the server with: mise run sembr-server")
        return 1
    print("✓")

    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")

    # Count input files
    input_files = list(input_dir.rglob("*.txt"))
    total_files = len(input_files)

    if total_files == 0:
        print("\nNo .txt files found in input directory")
        return 1

    # Apply sample limit
    if sample_count is not None:
        total_files = min(sample_count, total_files)
        print(f"Sample:  {total_files} files (of {len(input_files)} total)")

    print(f"Files:   {total_files}")
    print()

    # Progress callback
    processed_count = 0
    start_time = time.perf_counter()

    def progress_callback(current: int, total: int, filename: str) -> None:
        nonlocal processed_count
        processed_count = current
        if show_progress:
            print(f"[{current}/{total}] {filename}")

    # Run async processing
    try:
        results = asyncio.run(
            _run_sembr_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                config=config,
                sample_count=sample_count,
                max_concurrent=max_concurrent,
                progress_callback=progress_callback if show_progress else None,
            )
        )

        elapsed = time.perf_counter() - start_time
        print()
        print(f"Complete: {len(results)} files processed in {elapsed:.1f}s")

        # Summary stats
        total_lines = sum(r.line_count for r in results)
        total_words = sum(r.input_word_count for r in results)
        print(f"Total:    {total_lines:,} lines, {total_words:,} words")

        return 0

    except SembrServerError as e:
        print(f"\nError: Server error - {e}")
        return 1
    except SembrTimeoutError as e:
        print(f"\nError: Request timed out - {e}")
        return 1
    except KeyboardInterrupt:
        print(f"\n\nInterrupted after processing {processed_count} files")
        return 130  # Standard exit code for SIGINT


async def _run_sembr_batch(
    input_dir: Path,
    output_dir: Path,
    config: SembrConfig,
    sample_count: int | None,
    max_concurrent: int,
    progress_callback: Callable[[int, int, str], None] | None,
) -> list[SembrResult]:
    """Run sembr batch processing with optional sampling.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        config: Sembr configuration
        sample_count: Optional limit on files to process
        max_concurrent: Maximum concurrent processing
        progress_callback: Optional progress callback

    Returns:
        List of processing results
    """
    from pw_mcp.ingest.linebreaker import process_batch

    # If sampling, we need to limit the files ourselves
    if sample_count is not None:
        # Create a temporary directory with limited files
        input_files = sorted(input_dir.rglob("*.txt"))[:sample_count]

        # Process files individually with progress
        results: list[SembrResult] = []
        total = len(input_files)

        # Import process_file for individual processing
        from pw_mcp.ingest.linebreaker import process_file

        for i, input_file in enumerate(input_files):
            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path

            if progress_callback:
                progress_callback(i + 1, total, str(relative_path))

            result = await process_file(input_file, output_file, config)
            results.append(result)

        return results
    else:
        # Full batch processing
        return await process_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            max_concurrent=max_concurrent,
        )


def _run_legacy_ingest(args: argparse.Namespace) -> int:
    """Run legacy ingestion pipeline (not yet implemented).

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Semantic linebreak: {args.semantic_linebreak}")
    print("Ingestion pipeline not yet implemented.")
    return 0


def main() -> None:
    """Run the ingestion pipeline."""
    parser = _create_parser()
    args = parser.parse_args()

    if args.command == "sembr":
        # Handle sembr subcommand
        if args.check_only:
            success = _run_sembr_check(args.server)
            sys.exit(0 if success else 1)
        else:
            exit_code = _run_sembr_process(args)
            sys.exit(exit_code)
    elif args.command is None:
        # No subcommand - legacy behavior or show help
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)
        else:
            exit_code = _run_legacy_ingest(args)
            sys.exit(exit_code)
    else:
        # Unknown subcommand (shouldn't happen with argparse)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
