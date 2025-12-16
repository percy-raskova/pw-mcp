"""CLI for corpus ingestion pipeline.

Commands:
    pw-ingest sembr     Apply semantic linebreaking to extracted text
    pw-ingest chunk     Chunk sembr'd text into embedding-ready segments
    pw-ingest embed     Generate vector embeddings for chunks
    pw-ingest load      Load chunks and embeddings into ChromaDB
    pw-ingest           Legacy ingestion (not yet implemented)

Examples:
    # Check if sembr server is running
    pw-ingest sembr --check-only

    # Process full corpus through sembr
    pw-ingest sembr -i extracted/ -o sembr/

    # Process 10 files for testing
    pw-ingest sembr --sample 10

    # Chunk sembr'd text
    pw-ingest chunk -i sembr/ -o chunks/

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
import asyncio
import json
import logging
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from pw_mcp.ingest.linebreaker import SembrConfig, SembrResult


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


# =============================================================================
# SERVER MANAGEMENT
# =============================================================================

# Default timeout for server startup (seconds)
_SERVER_STARTUP_TIMEOUT = 30
_SERVER_STARTUP_POLL_INTERVAL = 1.0


def _reset_cuda_device(gpu_id: int = 0) -> bool:
    """Attempt to reset CUDA device state.

    Note: This may require root privileges on some systems.

    Args:
        gpu_id: GPU device ID to reset (default: 0)

    Returns:
        True if reset succeeded or was unnecessary, False on failure
    """
    try:
        # Try PyTorch CUDA reset first (doesn't require root)
        result = subprocess.run(
            [
                "python3",
                "-c",
                f"import torch; torch.cuda.device({gpu_id}); torch.cuda.empty_cache(); "
                f"torch.cuda.reset_peak_memory_stats({gpu_id}); print('CUDA cache cleared')",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            print(f"  CUDA cache cleared for GPU {gpu_id}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # If that didn't work, inform user about nvidia-smi reset
    print(f"  Note: For persistent CUDA errors, try: sudo nvidia-smi --gpu-reset -i {gpu_id}")
    return True  # Continue anyway


def _restart_sembr_server(
    server_url: str = "http://localhost:8384",
    timeout: float = _SERVER_STARTUP_TIMEOUT,
    reset_cuda: bool = True,
    gpu_id: int | None = None,
) -> bool:
    """Restart the sembr server to ensure clean CUDA state.

    This uses server_manager to properly stop and start the sembr server,
    ensuring CUDA cleanup and GPU environment setup.

    Args:
        server_url: URL of the sembr server (used to extract port)
        timeout: Maximum seconds to wait for server startup
        reset_cuda: Whether to attempt CUDA cache reset before starting
        gpu_id: GPU device ID to use (None for auto-detect)

    Returns:
        True if server started successfully, False otherwise
    """
    from pw_mcp.ingest.server_manager import (
        SEMBR_CONFIG,
        ServerConfig,
        start_server,
        stop_server,
    )

    # Extract port from URL
    port = 8384
    if ":" in server_url:
        port_str = server_url.split(":")[-1].split("/")[0]
        try:
            port = int(port_str)
        except ValueError:
            port = 8384

    print("Restarting sembr server (clean CUDA state)...")
    logger.info(f"Restarting sembr server on port {port}, gpu_id={gpu_id}")

    # Stop existing server (graceful with CUDA cleanup)
    if stop_server("sembr", graceful=True, timeout=5.0):
        logger.info("Stopped existing sembr server")
    else:
        logger.debug("No existing sembr server to stop or stop timed out")

    # Additional CUDA reset if requested
    if reset_cuda:
        _reset_cuda_device(gpu_id if gpu_id is not None else 0)

    # Create config with custom port if different from default
    config = SEMBR_CONFIG
    if port != 8384:
        config = ServerConfig(
            server_type="sembr",
            port=port,
            health_endpoint="/check",
            start_command=["uv", "run", "sembr", "--listen", "-p", str(port)],
            startup_timeout=timeout,
        )

    # Start server with GPU selection
    try:
        server = start_server(config, gpu_id=gpu_id)
        if server is not None:
            print(f"  Server started (PID: {server.pid}, GPU: {server.gpu_id})")
            logger.info(f"Sembr server started: PID={server.pid}, GPU={server.gpu_id}")
            return True
        else:
            print("  Error: Server failed to start")
            logger.error("start_server returned None")
            return False
    except Exception as e:
        print(f"  Error starting server: {e}")
        logger.error(f"Failed to start sembr server: {type(e).__name__}: {e}")
        return False


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
    sembr_parser.add_argument(
        "--restart-server",
        action="store_true",
        default=True,
        help="Restart sembr server before processing (default: True, ensures clean CUDA state)",
    )
    sembr_parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Don't restart server (use existing server instance)",
    )
    sembr_parser.add_argument(
        "--gpu",
        type=int,
        metavar="ID",
        help="GPU device ID to use (default: auto-detect available GPU)",
    )
    sembr_parser.add_argument(
        "--no-gpu-failover",
        action="store_true",
        help="Disable automatic GPU failover on errors (default: enabled)",
    )
    sembr_parser.add_argument(
        "--health-interval",
        type=float,
        default=60.0,
        metavar="SECS",
        help="Health check interval during batch processing (default: 60.0)",
    )
    sembr_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output",
    )

    # =========================================================================
    # CHUNK SUBCOMMAND
    # =========================================================================
    chunk_parser = subparsers.add_parser(
        "chunk",
        help="Chunk sembr'd text into embedding-ready segments",
        description=(
            "Process sembr'd text files through the chunker to create "
            "embedding-ready JSONL files with metadata."
        ),
    )

    chunk_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("sembr"),
        help="Input directory containing sembr'd .txt files (default: sembr/)",
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
    from pw_mcp.ingest.gpu_manager import (
        detect_gpus,
        get_available_gpu,
    )
    from pw_mcp.ingest.linebreaker import (
        SembrConfig,
        SembrContentError,
        SembrError,
        SembrServerError,
        SembrTimeoutError,
        check_server_health,
    )
    from pw_mcp.ingest.server_manager import (
        register_cleanup,
        setup_signal_handlers,
    )

    input_dir: Path = args.input
    output_dir: Path = args.output
    server_url: str = args.server
    sample_count: int | None = args.sample
    max_concurrent: int = args.max_concurrent
    show_progress: bool = not args.no_progress
    timeout: float = args.timeout
    should_restart: bool = args.restart_server and not args.no_restart
    gpu_id: int | None = getattr(args, "gpu", None)
    enable_gpu_failover: bool = not getattr(args, "no_gpu_failover", False)
    health_interval: float = getattr(args, "health_interval", 60.0)
    verbose: bool = getattr(args, "verbose", False)

    # Initialize logging
    log_file = _setup_logging(verbose=verbose or show_progress)
    logger.info(f"Starting sembr processing: {input_dir} -> {output_dir}")

    # Set up signal handlers for graceful shutdown
    setup_signal_handlers()
    register_cleanup()

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    # Print header
    print("Sembr Processing")
    print("=" * 40)
    print(f"Log file: {log_file}")

    # GPU detection and selection
    gpus = detect_gpus()
    if gpus:
        logger.info(f"Detected {len(gpus)} GPU(s)")
        for g in gpus:
            logger.debug(f"  GPU {g.index}: {g.name} ({g.memory_used_mb}/{g.memory_total_mb} MB)")

        # Select GPU
        if gpu_id is not None:
            if any(g.index == gpu_id for g in gpus):
                print(f"GPU:     {gpu_id} (specified)")
                logger.info(f"Using specified GPU {gpu_id}")
            else:
                print(f"Warning: GPU {gpu_id} not found, using auto-detection")
                logger.warning(f"Specified GPU {gpu_id} not found")
                gpu_id = get_available_gpu()
        else:
            gpu_id = get_available_gpu()
            if gpu_id is not None:
                print(f"GPU:     {gpu_id} (auto-detected)")
                logger.info(f"Auto-selected GPU {gpu_id}")

        if gpu_id is None:
            print("Warning: No GPU available, server may use any GPU")
            logger.warning("No GPU selected, server will use default")
    else:
        logger.info("No GPUs detected (nvidia-smi not available)")
        print("GPU:     None detected")

    # Restart server if requested (default: True for clean CUDA state)
    if should_restart:
        logger.info("Restarting sembr server for clean CUDA state")
        if not _restart_sembr_server(server_url, gpu_id=gpu_id):
            print("\nError: Failed to restart sembr server")
            print("Try starting manually: mise run sembr-server")
            logger.error("Failed to restart sembr server")
            return 1
    else:
        # Just check health if not restarting
        print(f"Server:  {server_url}", end=" ")
        if not check_server_health():
            print("X")
            print("\nError: Sembr server is not responding")
            print("Start the server with: mise run sembr-server")
            print("Or use --restart-server to auto-restart")
            logger.error(f"Sembr server not responding at {server_url}")
            return 1
        print("[OK]")
        logger.info(f"Server health check passed: {server_url}")

    # Create config
    config = SembrConfig(
        server_url=server_url,
        timeout_seconds=timeout,
    )

    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")

    # Discover input files
    all_input_files = list(input_dir.rglob("*.txt"))
    total_available = len(all_input_files)

    if total_available == 0:
        print("\nNo .txt files found in input directory")
        logger.warning("No .txt files found in input directory")
        return 1

    # Filter out already-processed files (resume support)
    files_to_process: list[Path] = []
    skipped_count = 0
    for input_file in all_input_files:
        relative_path = input_file.relative_to(input_dir)
        output_file = output_dir / relative_path
        if output_file.exists():
            skipped_count += 1
        else:
            files_to_process.append(input_file)

    # Apply sample limit with random selection
    if sample_count is not None:
        if sample_count < len(files_to_process):
            files_to_process = random.sample(files_to_process, sample_count)
        print(f"Sample:  {len(files_to_process)} files (randomly selected)")

    total_files = len(files_to_process)

    if skipped_count > 0:
        print(f"Skipped: {skipped_count} files (already processed)")
        logger.info(f"Skipping {skipped_count} already-processed files")
    print(f"Files:   {total_files}")
    print()

    if total_files == 0:
        print("No files to process (all already have sembr output)")
        logger.info("No files to process - all already complete")
        return 0

    logger.info(f"Processing {total_files} files with max_concurrent={max_concurrent}")

    # Progress tracking
    processed_count = 0
    error_count = 0
    errors: list[tuple[str, str]] = []  # (filename, error_message)
    start_time = time.perf_counter()

    def progress_callback(current: int, total: int, filename: str) -> None:
        nonlocal processed_count
        processed_count = current
        if show_progress:
            elapsed = time.perf_counter() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            print(f"[{current}/{total}] {filename} ({rate:.1f} files/s)")
        logger.debug(f"Processing [{current}/{total}]: {filename}")

    def error_callback(filename: str, error: Exception) -> None:
        nonlocal error_count
        error_count += 1
        error_msg = f"{type(error).__name__}: {error}"
        errors.append((filename, error_msg))
        _log_exception(f"Failed to process {filename}", error)
        if show_progress:
            print(f"  ERROR: {error_msg}")

    # Run async processing
    try:
        results = asyncio.run(
            _run_sembr_batch_with_errors(
                input_dir=input_dir,
                output_dir=output_dir,
                config=config,
                files_to_process=files_to_process,
                max_concurrent=max_concurrent,
                progress_callback=progress_callback if show_progress else None,
                error_callback=error_callback,
                health_interval=health_interval,
                enable_failover=enable_gpu_failover,
            )
        )

        elapsed = time.perf_counter() - start_time
        print()
        print(f"Complete: {len(results)} files processed in {elapsed:.1f}s")
        logger.info(f"Completed {len(results)} files in {elapsed:.1f}s")

        # Summary stats
        if results:
            total_lines = sum(r.line_count for r in results)
            total_words = sum(r.input_word_count for r in results)
            print(f"Total:    {total_lines:,} lines, {total_words:,} words")
            logger.info(f"Total: {total_lines:,} lines, {total_words:,} words")

        if error_count > 0:
            print(f"Errors:   {error_count} files failed")
            logger.warning(f"{error_count} files failed - see log for details")
            for filename, err in errors[:5]:
                print(f"  - {filename}: {err}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more (see log file)")

        return 0 if error_count == 0 else 1

    except SembrServerError as e:
        print(f"\nError: Server error - {e}")
        _log_exception("Server error during batch processing", e)
        return 1
    except SembrTimeoutError as e:
        print(f"\nError: Request timed out - {e}")
        _log_exception("Timeout during batch processing", e)
        return 1
    except SembrContentError as e:
        print(f"\nError: Content validation failed - {e}")
        _log_exception("Content validation error", e)
        return 1
    except SembrError as e:
        print(f"\nError: Sembr error - {e}")
        _log_exception("Sembr error during batch processing", e)
        return 1
    except KeyboardInterrupt:
        print(f"\n\nInterrupted after processing {processed_count} files")
        logger.warning(f"Interrupted by user after {processed_count} files")
        print(f"Progress saved - run again to resume from {processed_count + skipped_count} files")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")
        _log_exception("Unexpected error during batch processing", e)
        return 1


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
        # Full batch processing - filter out failed files (None values)
        batch_results = await process_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            config=config,
            progress_callback=progress_callback,
            max_concurrent=max_concurrent,
        )
        return [r for r in batch_results if r is not None]


async def _run_sembr_batch_with_errors(
    input_dir: Path,
    output_dir: Path,
    config: SembrConfig,
    files_to_process: list[Path],
    max_concurrent: int,
    progress_callback: Callable[[int, int, str], None] | None,
    error_callback: Callable[[str, Exception], None] | None,
    health_interval: float = 60.0,
    enable_failover: bool = True,
) -> list[SembrResult]:
    """Run sembr batch processing with per-file error handling.

    This version processes files with controlled concurrency, catches errors
    for individual files, monitors server health, and supports graceful shutdown.

    Args:
        input_dir: Input directory (for relative path calculation)
        output_dir: Output directory
        config: Sembr configuration
        files_to_process: List of specific files to process
        max_concurrent: Maximum concurrent processing (used for semaphore)
        progress_callback: Optional callback for progress updates
        error_callback: Optional callback for error reporting
        health_interval: Seconds between health checks (default: 60.0)
        enable_failover: Enable GPU failover on server errors (default: True)

    Returns:
        List of successful processing results
    """
    from pw_mcp.ingest.linebreaker import process_file
    from pw_mcp.ingest.server_manager import (
        HealthMonitor,
        is_shutdown_requested,
        restart_server,
    )

    results: list[SembrResult] = []
    total = len(files_to_process)
    semaphore = asyncio.Semaphore(max_concurrent)
    processed_count = 0

    # Set up health monitor
    def recovery_callback() -> bool:
        """Attempt to recover the sembr server."""
        logger.info("Attempting sembr server recovery")
        try:
            server = restart_server("sembr")
            if server is not None:
                logger.info(f"Server recovered on GPU {server.gpu_id}")
                return True
        except Exception as e:
            logger.error(f"Server recovery failed: {e}")
        return False

    health_monitor = HealthMonitor(
        server_type="sembr",
        interval=health_interval,
        max_recovery_attempts=3 if enable_failover else 1,
        recovery_callback=recovery_callback if enable_failover else None,
    )
    health_monitor.start()

    async def process_single(index: int, input_file: Path) -> SembrResult | None:
        """Process a single file with error handling."""
        nonlocal processed_count

        # Check for shutdown request before processing
        if is_shutdown_requested():
            logger.info(f"Shutdown requested, skipping {input_file.name}")
            return None

        async with semaphore:
            # Check health periodically (outside semaphore would deadlock)
            if health_monitor.should_check() and not health_monitor.check_and_recover():
                logger.error("Server unrecoverable, aborting batch")
                raise RuntimeError("Sembr server unrecoverable after recovery attempts")

            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path

            if progress_callback:
                progress_callback(index + 1, total, str(relative_path))

            try:
                result = await process_file(input_file, output_file, config)
                processed_count += 1
                logger.debug(f"Successfully processed: {relative_path}")
                return result
            except Exception as e:
                if error_callback:
                    error_callback(str(relative_path), e)
                return None

    try:
        # Process files with controlled concurrency
        tasks = [process_single(i, f) for i, f in enumerate(files_to_process)]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results (failed/skipped files) and exceptions
        for result in batch_results:
            if isinstance(result, BaseException):
                # Re-raise RuntimeError from server recovery failure
                if isinstance(result, RuntimeError):
                    raise result
                logger.error(f"Task raised exception: {result}")
            elif result is not None:
                # Type narrowing: result is now SembrResult
                results.append(result)

    finally:
        health_monitor.stop()
        logger.info(f"Batch processing complete: {len(results)}/{total} files succeeded")

    return results


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
    """Run chunking process on sembr'd text files.

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

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1

    # Create chunking config
    config = ChunkConfig(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
    )

    # Print header
    print("Chunk Processing")
    print("=" * 40)
    print(f"Input:     {input_dir}")
    print(f"Output:    {output_dir}")
    print(f"Metadata:  {extracted_dir}")
    print(f"Target:    {target_tokens} tokens")
    print(f"Max:       {max_tokens} tokens")

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

    if args.command == "sembr":
        # Handle sembr subcommand
        if args.check_only:
            success = _run_sembr_check(args.server)
            sys.exit(0 if success else 1)
        else:
            exit_code = _run_sembr_process(args)
            sys.exit(exit_code)
    elif args.command == "chunk":
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
