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

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # Create chunking config with tiktoken
    config = ChunkConfig(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
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

    # Discover input files
    input_files = list(input_dir.rglob("*.txt"))
    total_files = len(input_files)

    if total_files == 0:
        print("\nNo .txt files found in input directory")
        return 1

    # Apply sample limit with random selection
    if sample_count is not None:
        if sample_count < total_files:
            input_files = random.sample(input_files, sample_count)
        total_files = len(input_files)
        print(f"Sample:    {total_files} files (randomly selected)")

    print(f"Files:     {total_files}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files
    processed_count = 0
    total_chunks = 0
    missing_meta_count = 0
    start_time = time.perf_counter()

    try:
        for i, input_file in enumerate(input_files):
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

            # Write output
            write_chunks_jsonl(chunked_article, output_file)

            processed_count += 1
            total_chunks += len(chunked_article.chunks)

        elapsed = time.perf_counter() - start_time
        print()
        print(f"Complete:  {processed_count} files processed in {elapsed:.1f}s")
        print(f"Chunks:    {total_chunks:,} total")
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
