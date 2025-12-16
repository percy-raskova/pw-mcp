# Session Lessons - December 16, 2025

## Session Overview

This session accomplished two major tasks:
1. **ChromaDB Schema Implementation** - Built the database layer for storing embedded chunks
2. **CLI Logging & Resume Infrastructure** - Added robust logging and resume support after a sembr batch job froze

## ChromaDB Implementation Lessons

### Schema Design
- **ChromaDB is schema-less** for metadata, but consistent field usage is essential
- **Lists must be JSON strings** - ChromaDB metadata only supports: str, int, float, bool
- **No None values in metadata** - Use empty string `""` as sentinel
- **Precomputed embeddings** - Skip `embedding_function`, pass `embeddings=` directly to `add()`

### Type Safety
```python
# ChromaDB types are strict - use cast() for complex Where clauses
where_filter = cast("Where", {"$and": [{"field": {"$eq": value}}]})

# Metadata type includes SparseVector - accept Mapping[str, Any]
def deserialize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
```

### Collection Configuration
```python
collection = client.get_or_create_collection(
    name="prolewiki_chunks",
    metadata={
        "hnsw:space": "cosine",        # Semantic similarity
        "hnsw:construction_ef": 200,   # Build quality
        "hnsw:search_ef": 100,         # Search accuracy
    }
)
```

## Batch Processing Architecture

### Resume Support is Non-Negotiable
```python
# Skip already-processed files
for input_file in all_input_files:
    output_file = output_dir / input_file.relative_to(input_dir)
    if output_file.exists():
        skipped_count += 1
    else:
        files_to_process.append(input_file)
```

### Per-File Error Handling
```python
async def process_single(input_file: Path) -> Result | None:
    try:
        return await process_file(input_file, output_file, config)
    except Exception as e:
        error_callback(str(input_file), e)  # Log, don't crash
        return None

# Filter None results after batch completes
results = [r for r in batch_results if r is not None]
```

### Progress Rate Calculation
```python
elapsed = time.perf_counter() - start_time
rate = current / elapsed if elapsed > 0 else 0
print(f"[{current}/{total}] {filename} ({rate:.1f} files/s)")
```

## Logging Infrastructure

### Two-Tier Logging Pattern
```python
# File handler - everything with timestamps
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console handler - only warnings unless --verbose
console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
```

### Exception Logging Helper
```python
def _log_exception(msg: str, exc: Exception) -> None:
    logger.error(f"{msg}: {type(exc).__name__}: {exc}")
    logger.debug(f"Traceback:\n{traceback.format_exc()}")
```

### Log File Naming
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"pw_ingest_{timestamp}.log"
# Print path at startup so user knows where to look
print(f"Log file: {log_file}")
```

## GPU/CUDA Troubleshooting

### Key Lesson: CUDA Errors Can Corrupt Driver-Level State
- Process restart may NOT clear the error
- GPU memory stays allocated even after process dies
- Solution: Switch to different GPU or reboot

### Multi-GPU Switching
```bash
# Check GPU status
nvidia-smi --query-gpu=index,name,memory.used --format=csv

# Start process on specific GPU
CUDA_VISIBLE_DEVICES=1 uv run sembr --listen -p 8384
```

### GPU Memory as Health Indicator
```
index, name, memory.used [MiB]
0, NVIDIA GeForce RTX 3060, 698 MiB    # Corrupted - old model stuck
1, NVIDIA GeForce GTX 1650, 207 MiB    # Clean - newly loaded model
```

## GPU Manager Patterns (Phase 5.5)

### Detection via nvidia-smi
```python
from pw_mcp.ingest.gpu_manager import detect_gpus, get_available_gpu

# Detect all GPUs
gpus = detect_gpus()
for gpu in gpus:
    print(f"GPU {gpu.index}: {gpu.name} ({gpu.memory_used_mb}/{gpu.memory_total_mb} MB)")

# Auto-select available GPU (skips GPUs with >90% memory usage)
gpu_id = get_available_gpu(exclude=[0])  # Exclude corrupted GPU 0
```

### CUDA Corruption Detection
```python
from pw_mcp.ingest.gpu_manager import is_cuda_corrupted, reset_cuda_cache

# Heuristic: high memory + no processes = likely corrupted
if is_cuda_corrupted(gpu_id):
    reset_cuda_cache(gpu_id)  # Try to clear via PyTorch subprocess
```

### Environment Setup for GPU Selection
```python
from pw_mcp.ingest.gpu_manager import get_cuda_env

# Get env dict with CUDA_VISIBLE_DEVICES set
env = get_cuda_env(gpu_id=1)
subprocess.Popen(command, env=env)  # Process sees only GPU 1
```

## Server Manager Patterns (Phase 5.5)

### Server Lifecycle
```python
from pw_mcp.ingest.server_manager import (
    SEMBR_CONFIG,
    start_server,
    stop_server,
    restart_server,
    check_server_health,
)

# Start with auto-detected GPU
server = start_server(SEMBR_CONFIG)
print(f"Server PID: {server.pid}, GPU: {server.gpu_id}")

# Check health via HTTP
if not check_server_health("sembr"):
    # Restart with GPU failover
    server = restart_server("sembr", new_gpu=1)

# Graceful stop (SIGTERM, CUDA cleanup, then SIGKILL if needed)
stop_server("sembr", graceful=True, timeout=5.0)
```

### Health Monitoring During Batch Processing
```python
from pw_mcp.ingest.server_manager import HealthMonitor, restart_server

def recovery_callback() -> bool:
    server = restart_server("sembr")
    return server is not None

monitor = HealthMonitor(
    server_type="sembr",
    interval=60.0,  # Check every minute
    max_recovery_attempts=3,
    recovery_callback=recovery_callback,
)
monitor.start()

for file in files_to_process:
    if monitor.should_check() and not monitor.check_and_recover():
        raise RuntimeError("Server unrecoverable")
    process_file(file)

monitor.stop()
```

### Signal Handlers for Graceful Shutdown
```python
from pw_mcp.ingest.server_manager import (
    setup_signal_handlers,
    register_cleanup,
    is_shutdown_requested,
)

# Set up at start of CLI
setup_signal_handlers()  # Catches SIGINT, SIGTERM
register_cleanup()       # Stops all servers on exit

# Check in batch loop
if is_shutdown_requested():
    print("Graceful shutdown - saving progress...")
    break
```

## Sembr Server Reference

### Correct CLI Syntax
```bash
# WRONG
uv run sembr serve --host 0.0.0.0 --port 8384

# CORRECT
uv run sembr --listen -p 8384
```

### Endpoints
- `/check` - Health check (GET)
- `/rewrap` - Process text (POST with `text=` form data)

### Model Size
- distilbert-base-multilingual-cased: ~135M params
- Fits in 4GB GPU (GTX 1650 works fine)
- Server mode loads once, processes many (10-50x faster than per-file)

## Files Created/Modified

| File | Description |
|------|-------------|
| `src/pw_mcp/db/chroma.py` | ChromaDB interface (~290 lines) |
| `src/pw_mcp/db/__init__.py` | Module exports |
| `src/pw_mcp/ingest/cli.py` | Added `load` command + logging + GPU integration |
| `src/pw_mcp/ingest/gpu_manager.py` | GPU detection and CUDA management (~520 lines) |
| `src/pw_mcp/ingest/server_manager.py` | Server lifecycle management (~650 lines) |
| `tests/unit/db/test_chroma.py` | 17 unit tests |
| `tests/unit/ingest/test_gpu_manager.py` | 42 unit tests |
| `tests/unit/ingest/test_server_manager.py` | 47 unit tests |
| `tests/slow/test_gpu_integration.py` | GPU integration tests (~300 lines) |
| `tests/fixtures/chromadb/sample.jsonl` | Test chunks (3 records) |
| `tests/fixtures/chromadb/sample.npy` | Test embeddings (3x1536) |

## Commands Reference

```bash
# ChromaDB operations
uv run pw-ingest load --chunks-dir ./chunks --embeddings-dir ./embeddings
uv run pw-ingest load --reset  # Clear and reload

# Sembr server on alternate GPU
CUDA_VISIBLE_DEVICES=1 uv run sembr --listen -p 8384

# Monitor batch processing
tail -f /tmp/sembr_batch.log
ls -lt logs/pw_ingest_*.log | head -1  # Latest detailed log

# Check progress
find sembr/ -name "*.txt" | wc -l  # Processed count

# GPU status
nvidia-smi --query-gpu=index,name,memory.used --format=csv
```

## Current State (End of Session)

### Running Processes
- **Sembr server**: GPU 1 (GTX 1650), port 8384
- **Sembr batch job**: Background, processing 4,821 files
- **Log file**: `/tmp/sembr_batch.log` and `logs/pw_ingest_*.log`

### Completed
- 308 files already sembr'd (skipped on resume)
- ChromaDB loaded with 21 test articles (313 chunks)

### Database Location
- `chroma_data/` - ChromaDB persistent storage
- Collection: `prolewiki_chunks`

## Next Steps

1. **Wait for sembr batch to complete** (~4,800 files remaining)
2. **Run chunker on all sembr'd files**: `uv run pw-ingest chunk`
3. **Generate embeddings**: `uv run pw-ingest embed --provider openai`
4. **Load full corpus into ChromaDB**: `uv run pw-ingest load --reset`
5. **Wire up MCP server tools** in `src/pw_mcp/server.py`

## Meta-Lessons

1. **Logging is insurance** - The frozen job had no logs; diagnosis was impossible
2. **Resume functionality saves hours** - Without skip-existing, crashes mean restart
3. **Hardware failures need creative workarounds** - Switching GPUs saved a reboot
4. **Graceful degradation over hard failures** - Log errors, continue processing
5. **Capture session state** - This document enables context continuity
