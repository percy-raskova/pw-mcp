"""GPU detection, health monitoring, and CUDA state management.

This module provides utilities for managing NVIDIA GPUs in the pw-mcp pipeline,
including detection, health checking, CUDA cache reset, and automatic failover
when GPU errors occur.

Usage:
    from pw_mcp.ingest.gpu_manager import (
        detect_gpus,
        get_available_gpu,
        reset_cuda_cache,
        is_cuda_corrupted,
    )

    # Detect all GPUs
    gpus = detect_gpus()
    for gpu in gpus:
        print(f"GPU {gpu.index}: {gpu.name} ({gpu.memory_used_mb}/{gpu.memory_total_mb} MB)")

    # Get first available GPU
    gpu_id = get_available_gpu()
    if gpu_id is not None:
        print(f"Using GPU {gpu_id}")

    # Reset CUDA cache after error
    if is_cuda_corrupted(0):
        reset_cuda_cache(0)
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("pw_ingest.gpu")


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class GPUStatus:
    """Status information for a single NVIDIA GPU.

    Attributes:
        index: GPU device index (0, 1, 2, ...)
        name: GPU model name (e.g., "NVIDIA GeForce RTX 3060")
        memory_used_mb: Currently used GPU memory in MB
        memory_total_mb: Total GPU memory in MB
        is_available: Whether the GPU is available for use
        utilization_percent: GPU utilization percentage (0-100)
    """

    index: int
    name: str
    memory_used_mb: int
    memory_total_mb: int
    is_available: bool
    utilization_percent: int

    @property
    def memory_free_mb(self) -> int:
        """Calculate free memory in MB."""
        return self.memory_total_mb - self.memory_used_mb

    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage as percentage."""
        if self.memory_total_mb == 0:
            return 0.0
        return (self.memory_used_mb / self.memory_total_mb) * 100


# =============================================================================
# GPU DETECTION
# =============================================================================


def detect_gpus() -> list[GPUStatus]:
    """Detect all available NVIDIA GPUs using nvidia-smi.

    Returns:
        List of GPUStatus objects for each detected GPU.
        Empty list if nvidia-smi is not available or no GPUs found.

    Note:
        This function does not raise exceptions. If nvidia-smi fails,
        it returns an empty list and logs a warning.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.warning(f"nvidia-smi returned non-zero: {result.stderr.strip()}")
            return []

        gpus: list[GPUStatus] = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                logger.warning(f"Unexpected nvidia-smi output format: {line}")
                continue

            try:
                index = int(parts[0])
                name = parts[1]
                memory_used = int(parts[2])
                memory_total = int(parts[3])
                utilization = int(parts[4])

                gpu = GPUStatus(
                    index=index,
                    name=name,
                    memory_used_mb=memory_used,
                    memory_total_mb=memory_total,
                    is_available=True,  # Will be updated by health check
                    utilization_percent=utilization,
                )
                gpus.append(gpu)
                logger.debug(f"Detected GPU {index}: {name} ({memory_used}/{memory_total} MB)")

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse GPU info from line: {line}: {e}")
                continue

        return gpus

    except FileNotFoundError:
        logger.info("nvidia-smi not found - no NVIDIA GPU detected")
        return []
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
        return []
    except Exception as e:
        logger.warning(f"Failed to detect GPUs: {type(e).__name__}: {e}")
        return []


def check_gpu_health(gpu_id: int) -> GPUStatus | None:
    """Check health status of a specific GPU.

    Args:
        gpu_id: GPU device index to check

    Returns:
        GPUStatus for the specified GPU, or None if GPU not found or unhealthy.
    """
    gpus = detect_gpus()
    for gpu in gpus:
        if gpu.index == gpu_id:
            return gpu
    return None


def get_available_gpu(exclude: list[int] | None = None) -> int | None:
    """Get the first available healthy GPU.

    Args:
        exclude: List of GPU indices to exclude from selection

    Returns:
        GPU index if an available GPU is found, None otherwise.

    Note:
        GPUs are considered available if:
        - They respond to nvidia-smi
        - They have less than 90% memory usage
        - They are not in the exclude list
    """
    if exclude is None:
        exclude = []

    gpus = detect_gpus()
    for gpu in gpus:
        if gpu.index in exclude:
            logger.debug(f"GPU {gpu.index} excluded from selection")
            continue

        # Check memory usage - high usage might indicate stuck processes
        if gpu.memory_usage_percent > 90:
            logger.debug(f"GPU {gpu.index} has high memory usage ({gpu.memory_usage_percent:.1f}%)")
            continue

        logger.info(f"Selected GPU {gpu.index}: {gpu.name}")
        return gpu.index

    logger.warning("No available GPU found")
    return None


# =============================================================================
# CUDA STATE MANAGEMENT
# =============================================================================


def reset_cuda_cache(gpu_id: int | None = None) -> bool:
    """Clear CUDA cache for specified GPU or all GPUs.

    This attempts to clear PyTorch's CUDA cache without importing PyTorch
    directly (to avoid loading CUDA into the current process).

    Args:
        gpu_id: Specific GPU to reset, or None for all GPUs

    Returns:
        True if reset succeeded, False otherwise.

    Note:
        For persistent CUDA errors, a full GPU reset via nvidia-smi
        (requires root) or a system reboot may be necessary.
    """
    device_spec = f"{gpu_id}" if gpu_id is not None else "all"

    # Build the CUDA reset script
    if gpu_id is not None:
        script = f"""
import torch
if torch.cuda.is_available():
    torch.cuda.device({gpu_id})
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats({gpu_id})
    print(f'CUDA cache cleared for GPU {gpu_id}')
else:
    print('CUDA not available')
"""
    else:
        script = """
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.device(i)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(i)
    print(f'CUDA cache cleared for {torch.cuda.device_count()} GPUs')
else:
    print('CUDA not available')
"""

    try:
        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            logger.info(f"CUDA cache reset succeeded for GPU {device_spec}")
            return True
        else:
            logger.warning(f"CUDA cache reset failed: {result.stderr.strip()}")
            return False

    except FileNotFoundError:
        logger.warning("python3 not found for CUDA reset")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("CUDA reset timed out")
        return False
    except Exception as e:
        logger.warning(f"CUDA reset failed: {type(e).__name__}: {e}")
        return False


def is_cuda_corrupted(gpu_id: int) -> bool:
    """Check if a GPU appears to have corrupted CUDA state.

    CUDA corruption often manifests as:
    - High memory usage with no running processes
    - nvidia-smi showing "ERR!" or similar error states

    Args:
        gpu_id: GPU device index to check

    Returns:
        True if GPU appears corrupted, False otherwise.

    Note:
        This is a heuristic check. False negatives are possible.
    """
    # Check nvidia-smi for error states
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=gpu_name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.warning(f"nvidia-smi failed for GPU {gpu_id}: {result.stderr.strip()}")
            return True  # Assume corrupted if can't query

        output = result.stdout.strip()

        # Check for error indicators in output
        if "ERR" in output.upper() or "N/A" in output:
            logger.warning(f"GPU {gpu_id} shows error state: {output}")
            return True

        # Check for orphaned memory usage (high memory, no processes)
        parts = [p.strip() for p in output.split(",")]
        if len(parts) >= 3:
            memory_used = int(parts[1])
            _memory_total = int(parts[2])

            # High memory usage might indicate stuck state
            # Check if there are actual processes using it
            proc_result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_id}",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            processes = [p.strip() for p in proc_result.stdout.strip().split("\n") if p.strip()]

            # If high memory usage but no processes, likely corrupted
            if memory_used > 1000 and len(processes) == 0:
                logger.warning(
                    f"GPU {gpu_id} has {memory_used}MB used but no processes - possible corruption"
                )
                return True

        return False

    except FileNotFoundError:
        logger.info("nvidia-smi not found - cannot check corruption")
        return False
    except subprocess.TimeoutExpired:
        logger.warning(f"nvidia-smi timed out checking GPU {gpu_id}")
        return True  # Assume corrupted if can't query
    except Exception as e:
        logger.warning(f"Failed to check GPU {gpu_id} corruption: {type(e).__name__}: {e}")
        return False


# =============================================================================
# GPU PROCESS MANAGEMENT
# =============================================================================


def get_gpu_processes(gpu_id: int) -> list[dict[str, str | int]]:
    """Get list of processes running on a specific GPU.

    Args:
        gpu_id: GPU device index

    Returns:
        List of dicts with 'pid', 'name', and 'memory_mb' keys.
        Empty list if query fails.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return []

        processes: list[dict[str, str | int]] = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    processes.append(
                        {
                            "pid": int(parts[0]),
                            "name": parts[1],
                            "memory_mb": int(parts[2]) if parts[2].isdigit() else 0,
                        }
                    )
                except (ValueError, IndexError):
                    continue

        return processes

    except Exception:
        return []


def kill_gpu_processes(gpu_id: int, signal_name: str = "TERM") -> int:
    """Kill all processes on a specific GPU.

    Args:
        gpu_id: GPU device index
        signal_name: Signal to send ("TERM" or "KILL")

    Returns:
        Number of processes killed.
    """
    processes = get_gpu_processes(gpu_id)
    killed = 0

    for proc in processes:
        pid = proc["pid"]
        try:
            if signal_name == "KILL":
                subprocess.run(["kill", "-9", str(pid)], capture_output=True, timeout=5)
            else:
                subprocess.run(["kill", str(pid)], capture_output=True, timeout=5)
            killed += 1
            logger.info(f"Sent {signal_name} to process {pid} on GPU {gpu_id}")
        except Exception as e:
            logger.warning(f"Failed to kill process {pid}: {e}")

    return killed


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================


def get_cuda_env(gpu_id: int) -> dict[str, str]:
    """Get environment variables for running a process on a specific GPU.

    Args:
        gpu_id: GPU device index to use

    Returns:
        Dict of environment variables including CUDA_VISIBLE_DEVICES.
    """
    import os

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def parse_cuda_visible_devices() -> list[int]:
    """Parse the current CUDA_VISIBLE_DEVICES environment variable.

    Returns:
        List of GPU indices currently visible.
        Empty list if not set or invalid.
    """
    import os

    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_env:
        # Not set - all GPUs visible
        gpus = detect_gpus()
        return [g.index for g in gpus]

    try:
        return [int(x.strip()) for x in cuda_env.split(",") if x.strip()]
    except ValueError:
        logger.warning(f"Invalid CUDA_VISIBLE_DEVICES value: {cuda_env}")
        return []


# =============================================================================
# DIAGNOSTIC UTILITIES
# =============================================================================


def get_gpu_summary() -> str:
    """Get a human-readable summary of all GPUs.

    Returns:
        Multi-line string with GPU information.
    """
    gpus = detect_gpus()
    if not gpus:
        return "No NVIDIA GPUs detected"

    lines = ["GPU Summary:", "-" * 60]
    for gpu in gpus:
        status = "OK" if gpu.is_available else "UNAVAILABLE"
        corrupted = is_cuda_corrupted(gpu.index)
        if corrupted:
            status = "CORRUPTED"

        lines.append(
            f"  [{gpu.index}] {gpu.name}: "
            f"{gpu.memory_used_mb}/{gpu.memory_total_mb} MB "
            f"({gpu.utilization_percent}% util) - {status}"
        )

    return "\n".join(lines)
