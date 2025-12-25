"""CLI for corpus ingestion pipeline.

Commands:
    pw-ingest extract   Extract text from MediaWiki exports
    pw-ingest chunk     Chunk text into embedding-ready segments (tiktoken-based)
    pw-ingest embed     Generate vector embeddings for chunks
    pw-ingest load      Load chunks and embeddings into ChromaDB

Pipeline:
    extract → chunk → embed → load

Examples:
    # Chunk extracted text (uses tiktoken for accurate token counting)
    pw-ingest chunk -i extracted/ -o chunks/

    # Chunk with random sample
    pw-ingest chunk --sample 100

    # Generate embeddings
    pw-ingest embed -i chunks/ -o embeddings/

    # Embed with sample
    pw-ingest embed --sample 50

    # Load into ChromaDB
    pw-ingest load --chunks-dir ./chunks --embeddings-dir ./embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# =============================================================================
# LOGGING SETUP
# =============================================================================

# Module-level logger
logger = logging.getLogger("pw_ingest")


def _setup_logging(log_dir: Path | None = None, verbose: bool = False) -> Path:
    """Configure logging with file and console handlers.

    Args:
        log_dir: Directory for log files (default: ./logs/)
        verbose: If True, set console to DEBUG level

    Returns:
        Path to the log file
    """
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pw_ingest_{timestamp}.log"

    # Configure root logger
    logger.setLevel(logging.DEBUG)

    # File handler - captures everything with full detail
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler - less verbose unless --verbose flag
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    # Clear existing handlers and add new ones
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized - log file: {log_file}")
    return log_file


def _log_exception(msg: str, exc: Exception) -> None:
    """Log an exception with full traceback to file.

    Args:
        msg: Context message describing what failed
        exc: The exception that was raised
    """
    logger.error(f"{msg}: {type(exc).__name__}: {exc}")
    logger.debug(f"Traceback:\n{traceback.format_exc()}")


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
    # EXTRACT SUBCOMMAND
    # =========================================================================
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract clean text from MediaWiki source files",
        description=(
            "Parse MediaWiki files and extract clean text plus metadata. "
            "Removes templates, infoboxes, and wiki markup while preserving content."
        ),
    )

    extract_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("prolewiki-exports"),
        help="Input directory containing MediaWiki .txt files (default: prolewiki-exports/)",
    )
    extract_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("extracted"),
        help="Output directory for extracted text (default: extracted/)",
    )
    extract_parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Process only N randomly selected files (for testing)",
    )
    extract_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    extract_parser.add_argument(
        "--flat",
        action="store_true",
        help="Process flat directory (infer namespace from filename prefix: Essay_, Library_, etc.)",
    )

    # =========================================================================
    # CHUNK SUBCOMMAND
    # =========================================================================
    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Chunk extracted text into embedding-ready segments",
        description=(
            "Process extracted text files through tiktoken-based chunker to create "
            "embedding-ready JSONL files with accurate token counts and overlap."
        ),
    )

    chunk_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("extracted"),
        help="Input directory containing extracted .txt files (default: extracted/)",
    )
    chunk_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("chunks"),
        help="Output directory for .jsonl files (default: chunks/)",
    )
    chunk_parser.add_argument(
        "-e",
        "--extracted",
        type=Path,
        default=Path("extracted/articles"),
        help="Directory containing extracted metadata JSON (default: extracted/articles/)",
    )
    chunk_parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Process only N randomly selected files (for testing)",
    )
    chunk_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    chunk_parser.add_argument(
        "--target-tokens",
        type=int,
        default=600,
        help="Target token count per chunk (default: 600)",
    )
    chunk_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Maximum token count per chunk (default: 1000)",
    )
    chunk_parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=50,
        help="Token overlap between chunks for RAG context continuity (default: 50)",
    )
    chunk_parser.add_argument(
        "--min-words",
        type=int,
        default=10,
        help="Minimum word count per chunk; smaller chunks filtered out (default: 10)",
    )

    # =========================================================================
    # EMBED SUBCOMMAND
    # =========================================================================
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate vector embeddings for chunks",
        description=(
            "Process chunked JSONL files through Ollama or OpenAI to generate vector "
            "embeddings. For Ollama: requires server running with embeddinggemma. "
            "For OpenAI: requires OPENAI_API_KEY in .env or environment."
        ),
    )

    embed_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("chunks"),
        help="Input directory containing .jsonl files (default: chunks/)",
    )
    embed_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("embeddings"),
        help="Output directory for .npy files (default: embeddings/)",
    )
    embed_parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="Embedding provider: ollama (local) or openai (API) (default: ollama)",
    )
    embed_parser.add_argument(
        "--model",
        default="embeddinggemma",
        help="Embedding model (default: embeddinggemma for ollama, text-embedding-3-large for openai)",
    )
    embed_parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Embedding dimensions (default: 768 for ollama, 1536 for openai)",
    )
    embed_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding API calls (default: 32)",
    )
    embed_parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Process only N randomly selected files (for testing)",
    )
    embed_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    embed_parser.add_argument(
        "--host",
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)",
    )

    # =========================================================================
    # LOAD SUBCOMMAND
    # =========================================================================
    load_parser = subparsers.add_parser(
        "load",
        help="Load chunks and embeddings into ChromaDB",
        description=(
            "Load pre-computed chunks (JSONL) and embeddings (NPY) into ChromaDB. "
            "Chunks and embeddings must be in parallel directory structures."
        ),
    )

    load_parser.add_argument(
        "-c",
        "--chunks-dir",
        type=Path,
        default=Path("chunks"),
        help="Directory containing .jsonl chunk files (default: chunks/)",
    )
    load_parser.add_argument(
        "-e",
        "--embeddings-dir",
        type=Path,
        default=Path("embeddings"),
        help="Directory containing .npy embedding files (default: embeddings/)",
    )
    load_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("chroma_data"),
        help="ChromaDB persistence directory (default: chroma_data/)",
    )
    load_parser.add_argument(
        "--collection",
        default="prolewiki_chunks",
        help="ChromaDB collection name (default: prolewiki_chunks)",
    )
    load_parser.add_argument(
        "--dimensions",
        type=int,
        default=1536,
        help="Embedding dimensions (default: 1536 for OpenAI text-embedding-3-large)",
    )
    load_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    load_parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing collection before loading (fresh start)",
    )

    # =========================================================================
    # DIAGNOSE SUBCOMMAND
    # =========================================================================
    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Analyze pipeline state and identify issues",
        description=(
            "Scan pipeline directories to identify empty, missing, or orphaned files. "
            "Provides a summary of pipeline health and actionable recommendations."
        ),
    )

    diagnose_parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("extracted"),
        help="Directory containing extracted .txt files (default: extracted/)",
    )
    diagnose_parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("chunks"),
        help="Directory containing .jsonl chunk files (default: chunks/)",
    )
    diagnose_parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("embeddings"),
        help="Directory containing .npy embedding files (default: embeddings/)",
    )
    diagnose_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show per-file details for issues",
    )
    diagnose_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    diagnose_parser.add_argument(
        "--validate-content",
        action="store_true",
        help="Enable content-level validation (checks for duplicate chunks)",
    )
    diagnose_parser.add_argument(
        "--duplication-threshold",
        type=float,
        default=0.1,
        help="Flag files with duplication ratio above this threshold (default: 0.1 = 10%%)",
    )

    # =========================================================================
    # REPAIR SUBCOMMAND
    # =========================================================================
    repair_parser = subparsers.add_parser(
        "repair",
        help="Fix pipeline issues identified by diagnose",
        description=(
            "Repair pipeline issues by reprocessing empty files, deleting orphaned "
            "outputs, or forcing regeneration of specific stages."
        ),
    )

    repair_parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=Path("extracted"),
        help="Directory containing extracted .txt files (default: extracted/)",
    )
    repair_parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("chunks"),
        help="Directory containing .jsonl chunk files (default: chunks/)",
    )
    repair_parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("embeddings"),
        help="Directory containing .npy embedding files (default: embeddings/)",
    )
    repair_parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("prolewiki-exports"),
        help="Source MediaWiki export directory (default: prolewiki-exports/)",
    )
    repair_parser.add_argument(
        "--action",
        choices=["reprocess-empty", "delete-orphaned", "delete-empty", "delete-corrupt"],
        required=True,
        help=(
            "Repair action: reprocess-empty (re-extract empty files from source), "
            "delete-orphaned (remove outputs without sources), "
            "delete-empty (remove 0-byte files), "
            "delete-corrupt (remove files with excessive duplication)"
        ),
    )
    repair_parser.add_argument(
        "--stage",
        choices=["extracted", "chunks", "embeddings", "all"],
        default="all",
        help="Which stage to repair (default: all)",
    )
    repair_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    repair_parser.add_argument(
        "--duplication-threshold",
        type=float,
        default=0.1,
        help="For delete-corrupt: threshold for excessive duplication (default: 0.1 = 10%%)",
    )

    return parser


def _infer_namespace_from_filename(filename: str) -> str:
    """Infer namespace from filename prefix for flat directory mode.

    Args:
        filename: The filename to analyze

    Returns:
        Namespace string: "Essays", "Library", "ProleWiki", or "Main" (default)
    """
    if filename.startswith("Essay_"):
        return "Essays"
    if filename.startswith("Library_"):
        return "Library"
    if filename.startswith("ProleWiki_"):
        return "ProleWiki"
    return "Main"


def _run_extract_process(args: argparse.Namespace) -> int:
    """Run extraction process on MediaWiki source files.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error, 130 for interrupt)
    """
    import json

    from pw_mcp.ingest.extraction import extract_article

    input_dir: Path = args.input
    output_dir: Path = args.output
    sample_count: int | None = args.sample
    show_progress: bool = not args.no_progress
    flat_mode: bool = args.flat

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # Print header
    print("Extract Processing")
    print("=" * 40)
    print(f"Input:     {input_dir}")
    print(f"Output:    {output_dir}")
    if flat_mode:
        print("Mode:      flat (namespace from filename prefix)")

    # Discover input files
    input_files: list[Path] = []
    if flat_mode:
        # Flat mode: find all .txt files, infer namespace from filename
        input_files = list(input_dir.rglob("*.txt"))
    else:
        # Standard mode: organized by namespace subdirectories
        for subdir in ["Main", "Library", "Essays", "ProleWiki"]:
            namespace_dir = input_dir / subdir
            if namespace_dir.exists():
                input_files.extend(namespace_dir.rglob("*.txt"))

    total_available = len(input_files)

    if total_available == 0:
        print("\nNo .txt files found in input directory")
        return 1

    # Filter out files that already have corresponding output (resume support)
    files_to_process: list[Path] = []
    skipped_count = 0
    for input_file in input_files:
        # Determine namespace from filename (flat) or path (standard)
        if flat_mode:
            namespace = _infer_namespace_from_filename(input_file.name)
        else:
            relative_to_input = input_file.relative_to(input_dir)
            namespace = relative_to_input.parts[0] if relative_to_input.parts else "Main"

        # Output structure: extracted/{namespace}/{filename}.json
        output_file = output_dir / namespace / input_file.name

        if output_file.exists():
            skipped_count += 1
        else:
            files_to_process.append(input_file)

    # Apply sample limit with random selection
    if sample_count is not None:
        if sample_count < len(files_to_process):
            files_to_process = random.sample(files_to_process, sample_count)
        print(f"Sample:    {len(files_to_process)} files (randomly selected)")

    total_files = len(files_to_process)

    if skipped_count > 0:
        print(f"Skipped:   {skipped_count} files (already processed)")
    print(f"Files:     {total_files}")
    print()

    if total_files == 0:
        print("No files to process (all already extracted)")
        return 0

    # Create output directories
    for namespace in ["Main", "Library", "Essays", "ProleWiki"]:
        (output_dir / namespace).mkdir(parents=True, exist_ok=True)
    (output_dir / "articles").mkdir(parents=True, exist_ok=True)

    # Process files
    processed_count = 0
    error_count = 0
    start_time = time.perf_counter()

    try:
        for i, input_file in enumerate(files_to_process):
            # Determine namespace from filename (flat) or path (standard)
            if flat_mode:
                namespace = _infer_namespace_from_filename(input_file.name)
            else:
                relative_to_input = input_file.relative_to(input_dir)
                namespace = relative_to_input.parts[0] if relative_to_input.parts else "Main"

            # Progress reporting
            if show_progress:
                print(f"[{i + 1}/{total_files}] {namespace}/{input_file.name}")

            try:
                # Read source file
                source_text = input_file.read_text(encoding="utf-8")

                # Extract article data
                article_data = extract_article(source_text, str(input_file))

                # Write clean text
                text_output = output_dir / namespace / input_file.name
                text_output.write_text(article_data.clean_text, encoding="utf-8")

                # Write metadata as JSON
                meta_output = output_dir / "articles" / (input_file.stem + ".json")
                meta_dict = {
                    "title": input_file.stem,  # Title from filename
                    "namespace": article_data.namespace,
                    "categories": article_data.categories,
                    "internal_links": [
                        {"target": link.target, "display": link.display}
                        for link in article_data.internal_links
                    ],
                    "sections": article_data.sections,
                    "is_stub": article_data.is_stub,
                    "citation_needed_count": article_data.citation_needed_count,
                    "has_blockquote": article_data.has_blockquote,
                    "source_file": str(input_file),
                    # Phase B: Serialize structured metadata for enriched chunking
                    "infobox": asdict(article_data.infobox) if article_data.infobox else None,
                    "library_work": asdict(article_data.library_work)
                    if article_data.library_work
                    else None,
                }
                meta_output.write_text(
                    json.dumps(meta_dict, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                processed_count += 1

            except Exception as e:
                error_count += 1
                if show_progress:
                    print(f"  Error: {e}")

        elapsed = time.perf_counter() - start_time
        print()
        print(f"Complete:  {processed_count} files processed in {elapsed:.1f}s")
        if error_count > 0:
            print(f"Errors:    {error_count} files failed")

        return 0

    except KeyboardInterrupt:
        print(f"\n\nInterrupted after processing {processed_count} files")
        return 130  # Standard exit code for SIGINT


def _load_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load article metadata from JSON file.

    Args:
        metadata_path: Path to the metadata JSON file

    Returns:
        Dictionary with metadata, or empty dict with minimal defaults if not found
    """
    if metadata_path.exists():
        content = metadata_path.read_text(encoding="utf-8")
        return dict(json.loads(content))
    return {}


def _run_chunk_process(args: argparse.Namespace) -> int:
    """Run chunking process on extracted text files.

    Uses tiktoken for accurate token counting and supports chunk overlap
    for RAG context continuity.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error, 130 for interrupt)
    """
    # Import here to avoid circular imports and speed up --help
    from pw_mcp.ingest.chunker import ChunkConfig, chunk_article, write_chunks_jsonl

    input_dir: Path = args.input
    output_dir: Path = args.output
    extracted_dir: Path = args.extracted
    sample_count: int | None = args.sample
    show_progress: bool = not args.no_progress
    target_tokens: int = args.target_tokens
    max_tokens: int = args.max_tokens
    overlap_tokens: int = args.overlap_tokens
    min_words: int = args.min_words

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # Create chunking config with tiktoken
    config = ChunkConfig(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        min_words=min_words,
    )

    # Print header
    print("Chunk Processing (tiktoken)")
    print("=" * 40)
    print(f"Input:     {input_dir}")
    print(f"Output:    {output_dir}")
    print(f"Metadata:  {extracted_dir}")
    print(f"Target:    {target_tokens} tokens")
    print(f"Max:       {max_tokens} tokens")
    print(f"Overlap:   {overlap_tokens} tokens")
    print(f"Min words: {min_words}")

    # Discover input files
    input_files = list(input_dir.rglob("*.txt"))
    total_available = len(input_files)

    if total_available == 0:
        print("\nNo .txt files found in input directory")
        return 1

    # Filter out files that already have corresponding output (resume support)
    files_to_process: list[Path] = []
    skipped_count = 0
    for input_file in input_files:
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path.with_suffix(".jsonl")
        if output_file.exists():
            skipped_count += 1
        else:
            files_to_process.append(input_file)

    # Apply sample limit with random selection
    if sample_count is not None:
        if sample_count < len(files_to_process):
            files_to_process = random.sample(files_to_process, sample_count)
        print(f"Sample:    {len(files_to_process)} files (randomly selected)")

    total_files = len(files_to_process)

    if skipped_count > 0:
        print(f"Skipped:   {skipped_count} files (already processed)")
    print(f"Files:     {total_files}")
    print()

    if total_files == 0:
        print("No files to process (all already chunked)")
        return 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    processed_count = 0
    total_chunks = 0
    total_micro_filtered = 0
    total_duplicates_removed = 0
    missing_meta_count = 0
    start_time = time.perf_counter()

    try:
        for i, input_file in enumerate(files_to_process):
            # Calculate relative path for output structure preservation
            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_suffix(".jsonl")

            # Progress reporting
            if show_progress:
                print(f"[{i + 1}/{total_files}] {relative_path}")

            # Find corresponding metadata file
            # The metadata file should have same relative path but .json extension
            # Metadata is in extracted_dir, which may have different structure
            # Try multiple locations: same relative path, or just the stem.json
            meta_path_relative = extracted_dir / relative_path.with_suffix(".json")
            meta_path_flat = extracted_dir / (input_file.stem + ".json")

            metadata: dict[str, Any] = {}
            if meta_path_relative.exists():
                metadata = _load_metadata(meta_path_relative)
            elif meta_path_flat.exists():
                metadata = _load_metadata(meta_path_flat)
            else:
                # Log warning for missing metadata
                missing_meta_count += 1
                print(f"  Warning: Missing metadata for {relative_path}")
                # Provide minimal defaults
                metadata = {
                    "namespace": relative_path.parent.name if relative_path.parent.name else "Main",
                    "categories": [],
                    "internal_links": [],
                    "is_stub": False,
                    "citation_needed_count": 0,
                    "has_blockquote": False,
                }

            # Chunk the article
            chunked_article = chunk_article(input_file, metadata, config)

            # Write output with filtering
            stats = write_chunks_jsonl(chunked_article, output_file, config)

            processed_count += 1
            total_chunks += stats.chunks_written
            total_micro_filtered += stats.micro_chunks_filtered
            total_duplicates_removed += stats.consecutive_duplicates_removed

        elapsed = time.perf_counter() - start_time
        print()
        print(f"Complete:  {processed_count} files processed in {elapsed:.1f}s")
        print(f"Chunks:    {total_chunks:,} total")
        if total_micro_filtered > 0:
            print(f"Filtered:  {total_micro_filtered:,} micro-chunks (<{min_words} words)")
        if total_duplicates_removed > 0:
            print(f"Deduped:   {total_duplicates_removed:,} consecutive duplicates")
        if missing_meta_count > 0:
            print(f"Warnings:  {missing_meta_count} files with missing metadata")

        return 0

    except KeyboardInterrupt:
        print(f"\n\nInterrupted after processing {processed_count} files")
        return 130  # Standard exit code for SIGINT


def _run_embed_process(args: argparse.Namespace) -> int:
    """Run embedding process on chunked JSONL files.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error, 130 for interrupt)
    """
    # Import here to avoid circular imports and speed up --help
    from typing import Literal

    from pw_mcp.ingest.embedder import (
        EmbedConfig,
        OllamaConnectionError,
        OllamaModelError,
        OpenAIAuthError,
        OpenAIConnectionError,
        check_provider_ready,
        embed_article_chunks,
        write_embeddings_npy,
    )

    input_dir: Path = args.input
    output_dir: Path = args.output
    provider: Literal["ollama", "openai"] = args.provider
    batch_size: int = args.batch_size
    sample_count: int | None = args.sample
    show_progress: bool = not args.no_progress
    ollama_host: str = args.host

    # Auto-detect model and dimensions based on provider
    if args.dimensions is not None:
        dimensions = args.dimensions
    elif provider == "openai":
        dimensions = 1536
    else:
        dimensions = 768

    # Auto-detect model if using default and switching to openai
    if args.model == "embeddinggemma" and provider == "openai":
        model = "text-embedding-3-large"
    else:
        model = args.model

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # Create embedding config
    config = EmbedConfig(
        provider=provider,
        model=model,
        dimensions=dimensions,
        batch_size=batch_size,
        ollama_host=ollama_host,
    )

    # Print header
    print("Embed Processing")
    print("=" * 40)
    print(f"Provider:  {provider}", end=" ")

    # Check provider health
    if not check_provider_ready(config):
        print("X")
        if provider == "ollama":
            print("\nError: Ollama server is not responding or model not available")
            print("Start Ollama with: ollama serve")
            print(f"Ensure model is available: ollama pull {model}")
        else:
            print("\nError: OpenAI API is not responding or authentication failed")
            print("Ensure OPENAI_API_KEY is set in .env file or environment")
        return 1
    print("[OK]")

    print(f"Model:     {model}")
    print(f"Dims:      {dimensions}")
    print(f"Input:     {input_dir}")
    print(f"Output:    {output_dir}")
    print(f"Batch:     {batch_size}")

    # Discover input files
    input_files = list(input_dir.rglob("*.jsonl"))
    total_available = len(input_files)

    if total_available == 0:
        print("\nNo .jsonl files found in input directory")
        return 1

    # Filter out files that already have corresponding .npy output (resume support)
    files_to_process: list[Path] = []
    skipped_count = 0
    for input_file in input_files:
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path.with_suffix(".npy")
        if output_file.exists():
            skipped_count += 1
        else:
            files_to_process.append(input_file)

    # Apply sample limit with random selection
    if sample_count is not None:
        if sample_count < len(files_to_process):
            files_to_process = random.sample(files_to_process, sample_count)
        print(f"Sample:    {len(files_to_process)} files (randomly selected)")

    total_files = len(files_to_process)

    if skipped_count > 0:
        print(f"Skipped:   {skipped_count} files (already processed)")
    print(f"Files:     {total_files}")
    print()

    if total_files == 0:
        print("No files to process (all already have embeddings)")
        return 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    processed_count = 0
    total_chunks = 0
    start_time = time.perf_counter()

    try:
        for i, input_file in enumerate(files_to_process):
            # Calculate relative path for output structure preservation
            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_suffix(".npy")

            # Progress reporting
            if show_progress:
                print(f"[{i + 1}/{total_files}] {relative_path}")

            # Embed the article chunks
            embedded = embed_article_chunks(input_file, config)

            # Write embeddings to .npy
            write_embeddings_npy(embedded, output_file)

            processed_count += 1
            total_chunks += embedded.num_chunks

        elapsed = time.perf_counter() - start_time
        print()
        print(f"Complete:  {processed_count} files processed in {elapsed:.1f}s")
        print(f"Chunks:    {total_chunks:,} embedded")

        return 0

    except OllamaConnectionError as e:
        print(f"\nError: Ollama connection failed - {e}")
        return 1
    except OllamaModelError as e:
        print(f"\nError: Ollama model error - {e}")
        return 1
    except OpenAIConnectionError as e:
        print(f"\nError: OpenAI connection failed - {e}")
        return 1
    except OpenAIAuthError as e:
        print(f"\nError: OpenAI authentication failed - {e}")
        return 1
    except KeyboardInterrupt:
        print(f"\n\nInterrupted after processing {processed_count} files")
        return 130  # Standard exit code for SIGINT


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


def _run_load_process(args: argparse.Namespace) -> int:
    """Load chunks and embeddings into ChromaDB.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error, 130 for interrupt)
    """

    from pw_mcp.db import ChromaDBConfig, ProleWikiDB

    chunks_dir: Path = args.chunks_dir
    embeddings_dir: Path = args.embeddings_dir
    output_dir: Path = args.output_dir
    collection_name: str = args.collection
    dimensions: int = args.dimensions
    show_progress: bool = not args.no_progress
    reset_collection: bool = args.reset

    # Validate input directories
    if not chunks_dir.exists():
        print(f"Error: Chunks directory does not exist: {chunks_dir}")
        return 1

    if not embeddings_dir.exists():
        print(f"Error: Embeddings directory does not exist: {embeddings_dir}")
        return 1

    # Print header
    print("ChromaDB Load")
    print("=" * 40)
    print(f"Chunks:     {chunks_dir}")
    print(f"Embeddings: {embeddings_dir}")
    print(f"Output:     {output_dir}")
    print(f"Collection: {collection_name}")
    print(f"Dimensions: {dimensions}")

    # Discover chunk files
    chunk_files = sorted(chunks_dir.rglob("*.jsonl"))
    total_files = len(chunk_files)

    if total_files == 0:
        print("\nNo .jsonl files found in chunks directory")
        return 1

    print(f"Files:      {total_files}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ChromaDB
    config = ChromaDBConfig(
        persist_path=output_dir,
        collection_name=collection_name,
        embedding_dimensions=dimensions,
    )
    db = ProleWikiDB(config)

    # Reset collection if requested
    if reset_collection:
        print("Resetting collection...")
        db.delete_collection()
        print()

    # Track stats
    articles_loaded = 0
    chunks_loaded = 0
    articles_skipped = 0
    errors: list[str] = []
    start_time = time.perf_counter()

    try:
        for i, chunk_file in enumerate(chunk_files):
            # Calculate relative path to find matching embedding file
            relative_path = chunk_file.relative_to(chunks_dir)
            embedding_file = embeddings_dir / relative_path.with_suffix(".npy")

            # Progress reporting
            if show_progress:
                print(f"[{i + 1}/{total_files}] {relative_path}", end="")

            # Check if embedding file exists
            if not embedding_file.exists():
                articles_skipped += 1
                if show_progress:
                    print(" (skipped: no embeddings)")
                continue

            try:
                # Load the article into ChromaDB
                num_chunks = db.load_article(chunk_file, embedding_file)
                articles_loaded += 1
                chunks_loaded += num_chunks

                if show_progress:
                    print(f" ({num_chunks} chunks)")

            except ValueError as e:
                errors.append(f"{relative_path}: {e}")
                if show_progress:
                    print(f" (error: {e})")
            except Exception as e:
                errors.append(f"{relative_path}: {type(e).__name__}: {e}")
                if show_progress:
                    print(f" (error: {type(e).__name__})")

        elapsed = time.perf_counter() - start_time
        print()
        print(f"Complete:   {articles_loaded} articles in {elapsed:.1f}s")
        print(f"Chunks:     {chunks_loaded:,} loaded")
        print(f"Collection: {db.count():,} total chunks")

        if articles_skipped > 0:
            print(f"Skipped:    {articles_skipped} (no embeddings)")

        if errors:
            print(f"Errors:     {len(errors)}")
            for err in errors[:5]:  # Show first 5 errors
                print(f"  - {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")

        return 0 if not errors else 1

    except KeyboardInterrupt:
        print(f"\n\nInterrupted after loading {articles_loaded} articles")
        print(f"Collection has {db.count():,} total chunks")
        return 130  # Standard exit code for SIGINT


def _validate_chunk_content(
    chunks_dir: Path,
    threshold: float = 0.1,
) -> tuple[dict[str, tuple[int, int, int, float]], int, int]:
    """Validate chunk files for content issues like excessive duplication.

    Scans all JSONL chunk files and detects files where the ratio of
    duplicate chunks exceeds the given threshold.

    Args:
        chunks_dir: Directory containing .jsonl chunk files
        threshold: Flag files with duplication ratio above this (default: 0.1 = 10%)

    Returns:
        Tuple of:
        - corrupt_files: dict mapping file_key to (total, unique, dups, ratio)
        - total_chunks: sum of all chunks across all files
        - total_duplicates: sum of all duplicate chunks across all files
    """
    corrupt_files: dict[str, tuple[int, int, int, float]] = {}
    total_chunks = 0
    total_duplicates = 0

    if not chunks_dir.exists():
        return corrupt_files, total_chunks, total_duplicates

    for jsonl_file in chunks_dir.rglob("*.jsonl"):
        texts: list[str] = []
        try:
            with open(jsonl_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunk = json.loads(line)
                        texts.append(chunk.get("text", ""))
        except (json.JSONDecodeError, OSError):
            # Skip malformed files
            continue

        if not texts:
            continue

        file_total = len(texts)
        unique_count = len(set(texts))
        dup_count = file_total - unique_count
        dup_ratio = dup_count / file_total if file_total > 0 else 0.0

        total_chunks += file_total
        total_duplicates += dup_count

        if dup_ratio > threshold:
            # Normalize key to namespace/stem format
            relative = jsonl_file.relative_to(chunks_dir)
            file_key = str(relative.with_suffix(""))
            corrupt_files[file_key] = (file_total, unique_count, dup_count, dup_ratio)

    return corrupt_files, total_chunks, total_duplicates


def _run_diagnose_process(args: argparse.Namespace) -> int:
    """Analyze pipeline state and identify issues.

    Scans pipeline directories to find:
    - Empty files (0 bytes)
    - Missing files (exist in one stage but not another)
    - Orphaned files (output without corresponding input)

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for healthy, 1 for issues found)
    """
    extracted_dir: Path = args.extracted_dir
    chunks_dir: Path = args.chunks_dir
    embeddings_dir: Path = args.embeddings_dir
    verbose: bool = args.verbose
    output_format: str = args.format
    validate_content: bool = args.validate_content
    duplication_threshold: float = args.duplication_threshold

    # Collect file information for each stage
    def scan_directory(directory: Path, extension: str) -> dict[str, tuple[Path, int, str]]:
        """Scan directory and collect file metadata as (path, size, namespace) tuples."""
        files: dict[str, tuple[Path, int, str]] = {}
        if not directory.exists():
            return files

        for file_path in directory.rglob(f"*{extension}"):
            # Skip the articles/ subdirectory (metadata, not content)
            if "articles" in file_path.parts:
                continue
            relative = file_path.relative_to(directory)
            # Normalize key: namespace/stem (without extension)
            key = str(relative.with_suffix(""))
            namespace = relative.parts[0] if len(relative.parts) > 1 else "root"
            files[key] = (file_path, file_path.stat().st_size, namespace)
        return files

    # Scan all stages
    extracted = scan_directory(extracted_dir, ".txt")
    chunks = scan_directory(chunks_dir, ".jsonl")
    embeddings = scan_directory(embeddings_dir, ".npy")

    # Calculate statistics (tuple indices: 0=path, 1=size, 2=namespace)
    extracted_total = len(extracted)
    extracted_empty = sum(1 for _path, size, _ns in extracted.values() if size == 0)
    chunks_total = len(chunks)
    chunks_empty = sum(1 for _path, size, _ns in chunks.values() if size == 0)
    embeddings_total = len(embeddings)
    embeddings_empty = sum(1 for _path, size, _ns in embeddings.values() if size == 0)

    # Find missing files
    extracted_keys = set(extracted.keys())
    chunks_keys = set(chunks.keys())
    embeddings_keys = set(embeddings.keys())

    # Files extracted but not chunked
    missing_chunks = extracted_keys - chunks_keys
    # Files chunked but not embedded (excluding empty chunks which can't embed)
    non_empty_chunks = {k for k, (_path, size, _ns) in chunks.items() if size > 0}
    missing_embeddings = non_empty_chunks - embeddings_keys
    # Orphaned files (output exists without input)
    orphaned_chunks = chunks_keys - extracted_keys
    orphaned_embeddings = embeddings_keys - chunks_keys

    # Group empty files by namespace
    empty_by_namespace: dict[str, int] = {}
    for _key, (_path, size, namespace) in extracted.items():
        if size == 0:
            empty_by_namespace[namespace] = empty_by_namespace.get(namespace, 0) + 1

    # Content validation (optional)
    corrupt_files: dict[str, tuple[int, int, int, float]] = {}
    content_total_duplicates = 0
    if validate_content:
        corrupt_files, _, content_total_duplicates = _validate_chunk_content(
            chunks_dir, duplication_threshold
        )

    # Determine overall health
    has_issues = bool(
        missing_chunks
        or missing_embeddings
        or orphaned_chunks
        or orphaned_embeddings
        or extracted_empty
        or corrupt_files  # Add content validation issues
    )

    if output_format == "json":
        # JSON output for programmatic use
        result = {
            "stages": {
                "extracted": {
                    "total": extracted_total,
                    "non_empty": extracted_total - extracted_empty,
                    "empty": extracted_empty,
                },
                "chunks": {
                    "total": chunks_total,
                    "non_empty": chunks_total - chunks_empty,
                    "empty": chunks_empty,
                },
                "embeddings": {
                    "total": embeddings_total,
                    "non_empty": embeddings_total - embeddings_empty,
                    "empty": embeddings_empty,
                },
            },
            "issues": {
                "missing_chunks": len(missing_chunks),
                "missing_embeddings": len(missing_embeddings),
                "orphaned_chunks": len(orphaned_chunks),
                "orphaned_embeddings": len(orphaned_embeddings),
            },
            "empty_by_namespace": empty_by_namespace,
            "healthy": not has_issues,
        }
        # Add content validation results if requested
        if validate_content:
            result["content_issues"] = {
                "excessive_duplicates": len(corrupt_files),
                "total_duplicate_chunks": content_total_duplicates,
                "files": [
                    {
                        "path": f"{key}.jsonl",
                        "total": total,
                        "duplicates": dups,
                        "ratio": ratio,
                    }
                    for key, (total, _unique, dups, ratio) in sorted(
                        corrupt_files.items(),
                        key=lambda x: -x[1][3],  # Sort by ratio desc
                    )
                ],
            }
        if verbose:
            result["details"] = {
                "missing_chunks": sorted(missing_chunks)[:50],
                "missing_embeddings": sorted(missing_embeddings)[:50],
                "orphaned_chunks": sorted(orphaned_chunks)[:50],
                "orphaned_embeddings": sorted(orphaned_embeddings)[:50],
            }
        print(json.dumps(result, indent=2))
    else:
        # Text output for human readability
        print("Pipeline Diagnosis")
        print("=" * 50)
        print()
        print("Stage Statistics")
        print("-" * 50)
        print(f"{'Stage':<15} {'Total':>10} {'Non-Empty':>12} {'Empty':>10}")
        print("-" * 50)
        print(
            f"{'extracted':<15} {extracted_total:>10,} "
            f"{extracted_total - extracted_empty:>12,} {extracted_empty:>10,}"
        )
        print(
            f"{'chunks':<15} {chunks_total:>10,} "
            f"{chunks_total - chunks_empty:>12,} {chunks_empty:>10,}"
        )
        print(
            f"{'embeddings':<15} {embeddings_total:>10,} "
            f"{embeddings_total - embeddings_empty:>12,} {embeddings_empty:>10,}"
        )
        print()

        if empty_by_namespace:
            print("Empty Files by Namespace")
            print("-" * 50)
            for ns, count in sorted(empty_by_namespace.items(), key=lambda x: -x[1]):
                print(f"  {ns}: {count:,}")
            print()

        # Content validation results
        if validate_content and corrupt_files:
            print(f"Content Issues (duplication > {duplication_threshold:.0%})")
            print("-" * 50)
            print(f"  Corrupt files: {len(corrupt_files):,}")
            print(f"  Total duplicate chunks: {content_total_duplicates:,}")
            if verbose:
                print()
                # Show top 10 worst files
                sorted_corrupt = sorted(
                    corrupt_files.items(),
                    key=lambda x: -x[1][3],  # Sort by ratio desc
                )
                for key, (total, _unique, dups, ratio) in sorted_corrupt[:10]:
                    print(f"    {ratio:>6.1%} ({dups:>5}/{total:>5}) {key}.jsonl")
                if len(corrupt_files) > 10:
                    print(f"    ... and {len(corrupt_files) - 10} more")
            print()

        if missing_chunks or missing_embeddings or orphaned_chunks or orphaned_embeddings:
            print("Issues Found")
            print("-" * 50)
            if missing_chunks:
                print(f"  Missing chunks (extracted without chunks): {len(missing_chunks):,}")
                if verbose:
                    for key in sorted(missing_chunks)[:10]:
                        print(f"    - {key}")
                    if len(missing_chunks) > 10:
                        print(f"    ... and {len(missing_chunks) - 10} more")
            if missing_embeddings:
                print(
                    f"  Missing embeddings (non-empty chunks without embeddings): "
                    f"{len(missing_embeddings):,}"
                )
                if verbose:
                    for key in sorted(missing_embeddings)[:10]:
                        print(f"    - {key}")
                    if len(missing_embeddings) > 10:
                        print(f"    ... and {len(missing_embeddings) - 10} more")
            if orphaned_chunks:
                print(f"  Orphaned chunks (chunks without extracted): {len(orphaned_chunks):,}")
                if verbose:
                    for key in sorted(orphaned_chunks)[:10]:
                        print(f"    - {key}")
                    if len(orphaned_chunks) > 10:
                        print(f"    ... and {len(orphaned_chunks) - 10} more")
            if orphaned_embeddings:
                print(
                    f"  Orphaned embeddings (embeddings without chunks): "
                    f"{len(orphaned_embeddings):,}"
                )
                if verbose:
                    for key in sorted(orphaned_embeddings)[:10]:
                        print(f"    - {key}")
                    if len(orphaned_embeddings) > 10:
                        print(f"    ... and {len(orphaned_embeddings) - 10} more")
            print()

        if has_issues:
            print("Recommendations")
            print("-" * 50)
            if missing_chunks:
                print("  - Run: pw-ingest chunk  (to generate missing chunks)")
            if missing_embeddings:
                print("  - Run: pw-ingest embed  (to generate missing embeddings)")
            if orphaned_chunks or orphaned_embeddings:
                print(
                    "  - Run: pw-ingest repair --action delete-orphaned  "
                    "(to clean up orphaned files)"
                )
            if extracted_empty:
                print(
                    f"  - Note: {extracted_empty} empty extracted files are likely "
                    "intentional (empty library pages)"
                )
            if corrupt_files:
                print(
                    "  - Run: pw-ingest repair --action delete-corrupt  "
                    "(to remove corrupted chunk files)"
                )
                print("  - Then: pw-ingest chunk && pw-ingest embed  (to regenerate)")
        else:
            print("Status: HEALTHY - No issues found")

    return 1 if has_issues else 0


def _run_repair_process(args: argparse.Namespace) -> int:
    """Repair pipeline issues.

    Supports actions:
    - reprocess-empty: Re-extract files that produced 0-byte output
    - delete-orphaned: Remove outputs without corresponding inputs
    - delete-empty: Remove 0-byte files
    - delete-corrupt: Remove chunk files with excessive duplication

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    extracted_dir: Path = args.extracted_dir
    chunks_dir: Path = args.chunks_dir
    embeddings_dir: Path = args.embeddings_dir
    source_dir: Path = args.source_dir
    action: str = args.action
    stage: str = args.stage
    dry_run: bool = args.dry_run

    print("Pipeline Repair")
    print("=" * 50)
    print(f"Action:    {action}")
    print(f"Stage:     {stage}")
    print(f"Dry run:   {dry_run}")
    print()

    files_affected = 0
    errors: list[str] = []

    def collect_files(directory: Path, extension: str) -> list[Path]:
        """Collect files with given extension."""
        if not directory.exists():
            return []
        return list(directory.rglob(f"*{extension}"))

    if action == "delete-empty":
        # Delete 0-byte files from specified stage(s)
        stages_to_check: list[tuple[Path, str]] = []
        if stage in ("extracted", "all"):
            stages_to_check.append((extracted_dir, ".txt"))
        if stage in ("chunks", "all"):
            stages_to_check.append((chunks_dir, ".jsonl"))
        if stage in ("embeddings", "all"):
            stages_to_check.append((embeddings_dir, ".npy"))

        for directory, ext in stages_to_check:
            for file_path in collect_files(directory, ext):
                if file_path.stat().st_size == 0:
                    relative = file_path.relative_to(directory)
                    if dry_run:
                        print(f"  Would delete: {directory.name}/{relative}")
                    else:
                        try:
                            file_path.unlink()
                            print(f"  Deleted: {directory.name}/{relative}")
                        except OSError as e:
                            errors.append(f"{relative}: {e}")
                    files_affected += 1

    elif action == "delete-orphaned":
        # Delete output files that don't have corresponding input files
        extracted_keys = {
            str(p.relative_to(extracted_dir).with_suffix(""))
            for p in collect_files(extracted_dir, ".txt")
            if "articles" not in p.parts
        }
        chunks_keys = {
            str(p.relative_to(chunks_dir).with_suffix(""))
            for p in collect_files(chunks_dir, ".jsonl")
        }

        if stage in ("chunks", "all"):
            orphaned_chunks = chunks_keys - extracted_keys
            for key in orphaned_chunks:
                file_path = chunks_dir / f"{key}.jsonl"
                if file_path.exists():
                    if dry_run:
                        print(f"  Would delete orphaned chunk: {key}.jsonl")
                    else:
                        try:
                            file_path.unlink()
                            print(f"  Deleted orphaned chunk: {key}.jsonl")
                        except OSError as e:
                            errors.append(f"{key}: {e}")
                    files_affected += 1

        if stage in ("embeddings", "all"):
            orphaned_embeddings = {
                str(p.relative_to(embeddings_dir).with_suffix(""))
                for p in collect_files(embeddings_dir, ".npy")
            } - chunks_keys
            for key in orphaned_embeddings:
                file_path = embeddings_dir / f"{key}.npy"
                if file_path.exists():
                    if dry_run:
                        print(f"  Would delete orphaned embedding: {key}.npy")
                    else:
                        try:
                            file_path.unlink()
                            print(f"  Deleted orphaned embedding: {key}.npy")
                        except OSError as e:
                            errors.append(f"{key}: {e}")
                    files_affected += 1

    elif action == "reprocess-empty":
        # Re-extract files that have 0-byte extracted output
        from pw_mcp.ingest.extraction import extract_article

        if not source_dir.exists():
            print(f"Error: Source directory does not exist: {source_dir}")
            return 1

        # Find empty extracted files
        empty_extracted: list[Path] = []
        for file_path in collect_files(extracted_dir, ".txt"):
            if "articles" in file_path.parts:
                continue
            if file_path.stat().st_size == 0:
                empty_extracted.append(file_path)

        print(f"Found {len(empty_extracted)} empty extracted files")

        for extracted_path in empty_extracted:
            relative = extracted_path.relative_to(extracted_dir)
            # Find corresponding source file
            source_path = source_dir / relative
            if not source_path.exists():
                errors.append(f"Source not found: {relative}")
                continue

            if dry_run:
                print(f"  Would reprocess: {relative}")
            else:
                try:
                    source_text = source_path.read_text(encoding="utf-8")
                    article_data = extract_article(source_text, str(source_path))
                    extracted_path.write_text(article_data.clean_text, encoding="utf-8")
                    print(f"  Reprocessed: {relative} ({len(article_data.clean_text)} chars)")
                except Exception as e:
                    errors.append(f"{relative}: {e}")
            files_affected += 1

    elif action == "delete-corrupt":
        # Delete chunk files with excessive duplication and their embeddings
        duplication_threshold: float = args.duplication_threshold
        print(f"Threshold: {duplication_threshold:.0%}")
        print()

        # Find corrupt files using content validation
        corrupt_files, _, total_duplicates = _validate_chunk_content(
            chunks_dir, duplication_threshold
        )

        if not corrupt_files:
            print("No corrupt files found.")
        else:
            print(f"Found {len(corrupt_files):,} corrupt files")
            print(f"Total duplicate chunks: {total_duplicates:,}")
            print()

            deleted_chunks = 0
            deleted_embeddings = 0

            for file_key, (_total, _unique, _dups, ratio) in sorted(
                corrupt_files.items(),
                key=lambda x: -x[1][3],  # Sort by ratio desc
            ):
                chunk_path = chunks_dir / f"{file_key}.jsonl"
                embed_path = embeddings_dir / f"{file_key}.npy"

                if dry_run:
                    print(f"  Would delete: {file_key}.jsonl ({ratio:.1%} duplicates)")
                else:
                    try:
                        if chunk_path.exists():
                            chunk_path.unlink()
                            deleted_chunks += 1
                            print(f"  Deleted: {file_key}.jsonl ({ratio:.1%} duplicates)")
                        if embed_path.exists():
                            embed_path.unlink()
                            deleted_embeddings += 1
                    except OSError as e:
                        errors.append(f"{file_key}: {e}")
                files_affected += 1

            if not dry_run:
                print()
                print(f"Deleted {deleted_chunks} chunk files, {deleted_embeddings} embedding files")

    # Summary
    print()
    if dry_run:
        print(f"Dry run complete: {files_affected} files would be affected")
    else:
        print(f"Repair complete: {files_affected} files processed")

    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
        return 1

    return 0


def main() -> None:
    """Run the ingestion pipeline."""
    parser = _create_parser()
    args = parser.parse_args()

    if args.command == "chunk":
        # Handle chunk subcommand
        exit_code = _run_chunk_process(args)
        sys.exit(exit_code)
    elif args.command == "embed":
        # Handle embed subcommand
        exit_code = _run_embed_process(args)
        sys.exit(exit_code)
    elif args.command == "extract":
        # Handle extract subcommand
        exit_code = _run_extract_process(args)
        sys.exit(exit_code)
    elif args.command == "load":
        # Handle load subcommand
        exit_code = _run_load_process(args)
        sys.exit(exit_code)
    elif args.command == "diagnose":
        # Handle diagnose subcommand
        exit_code = _run_diagnose_process(args)
        sys.exit(exit_code)
    elif args.command == "repair":
        # Handle repair subcommand
        exit_code = _run_repair_process(args)
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
