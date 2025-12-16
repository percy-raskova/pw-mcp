"""Server lifecycle management for GPU-bound services.

This module provides utilities for starting, stopping, monitoring, and
managing GPU-bound servers like sembr and Ollama in the pw-mcp pipeline.

Features:
- Graceful start/stop/restart with proper CUDA cleanup
- Health monitoring during batch processing
- Automatic GPU failover on errors
- atexit cleanup for process termination
- Signal handling for graceful shutdown

Usage:
    from pw_mcp.ingest.server_manager import (
        start_server,
        stop_server,
        restart_server,
        check_server_health,
        SEMBR_CONFIG,
        OLLAMA_CONFIG,
    )

    # Start sembr server on GPU 0
    server = start_server(SEMBR_CONFIG, gpu_id=0)
    print(f"Server started with PID {server.pid}")

    # Check health
    if check_server_health("sembr"):
        print("Server is healthy")

    # Stop gracefully
    stop_server("sembr")
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import httpx

from pw_mcp.ingest.gpu_manager import get_available_gpu, get_cuda_env, reset_cuda_cache

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("pw_ingest.server")


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for a GPU server.

    Attributes:
        server_type: Identifier for the server type ("sembr", "ollama")
        port: Port number the server listens on
        health_endpoint: HTTP endpoint for health checks
        start_command: Command line arguments to start the server
        startup_timeout: Maximum seconds to wait for server to become healthy
        health_timeout: Timeout in seconds for health check requests
        graceful_shutdown_timeout: Seconds to wait for graceful shutdown before SIGKILL
    """

    server_type: str
    port: int
    health_endpoint: str
    start_command: list[str]
    startup_timeout: float = 60.0
    health_timeout: float = 5.0
    graceful_shutdown_timeout: float = 5.0


# Pre-configured server configurations
SEMBR_CONFIG = ServerConfig(
    server_type="sembr",
    port=8384,
    health_endpoint="/check",
    start_command=["uv", "run", "sembr", "--listen", "-p", "8384"],
    startup_timeout=120.0,  # Model loading can be slow
)

OLLAMA_CONFIG = ServerConfig(
    server_type="ollama",
    port=11434,
    health_endpoint="/api/tags",
    start_command=["ollama", "serve"],
    startup_timeout=30.0,
)


# =============================================================================
# SERVER PROCESS TRACKING
# =============================================================================


@dataclass
class ServerProcess:
    """Information about a tracked server process.

    Attributes:
        config: Server configuration
        pid: Process ID
        start_time: When the server was started
        gpu_id: GPU device index the server is using
        process: The subprocess.Popen object (if available)
    """

    config: ServerConfig
    pid: int
    start_time: datetime
    gpu_id: int
    process: subprocess.Popen[bytes] | None = field(default=None, repr=False)


# Module-level server registry
_servers: dict[str, ServerProcess] = {}

# Shutdown flag for graceful termination
_shutdown_requested: bool = False


# =============================================================================
# HEALTH CHECKING
# =============================================================================


def check_server_health(server_type: str, timeout: float | None = None) -> bool:
    """Check if a server is healthy and responding.

    Args:
        server_type: Server type to check ("sembr" or "ollama")
        timeout: Request timeout in seconds (default: from config)

    Returns:
        True if server is healthy, False otherwise.
    """
    # Get config for server type
    config = _get_config_for_type(server_type)
    if config is None:
        logger.warning(f"Unknown server type: {server_type}")
        return False

    if timeout is None:
        timeout = config.health_timeout

    url = f"http://localhost:{config.port}{config.health_endpoint}"

    try:
        response = httpx.get(url, timeout=timeout)
        if response.status_code == 200:
            logger.debug(f"{server_type} health check passed")
            return True
        else:
            logger.warning(f"{server_type} health check failed: status {response.status_code}")
            return False
    except httpx.ConnectError:
        logger.debug(f"{server_type} not responding (connection refused)")
        return False
    except httpx.TimeoutException:
        logger.warning(f"{server_type} health check timed out")
        return False
    except Exception as e:
        logger.warning(f"{server_type} health check error: {type(e).__name__}: {e}")
        return False


def _get_config_for_type(server_type: str) -> ServerConfig | None:
    """Get the configuration for a server type."""
    configs: dict[str, ServerConfig] = {
        "sembr": SEMBR_CONFIG,
        "ollama": OLLAMA_CONFIG,
    }
    return configs.get(server_type)


# =============================================================================
# SERVER LIFECYCLE
# =============================================================================


def start_server(config: ServerConfig, gpu_id: int | None = None) -> ServerProcess:
    """Start a server on the specified GPU.

    Args:
        config: Server configuration
        gpu_id: GPU device index (auto-selected if None)

    Returns:
        ServerProcess object with process information

    Raises:
        RuntimeError: If server fails to start or become healthy
    """
    server_type = config.server_type

    # Check if server is already running
    if server_type in _servers:
        existing = _servers[server_type]
        if _is_process_running(existing.pid):
            logger.info(f"{server_type} already running (PID {existing.pid})")
            return existing
        else:
            # Process died, remove from registry
            logger.warning(f"{server_type} process {existing.pid} is no longer running")
            del _servers[server_type]

    # Auto-select GPU if not specified
    if gpu_id is None:
        gpu_id = get_available_gpu()
        if gpu_id is None:
            raise RuntimeError("No available GPU found")

    logger.info(f"Starting {server_type} on GPU {gpu_id}")

    # Prepare environment with CUDA_VISIBLE_DEVICES
    env = get_cuda_env(gpu_id)

    # Build command with correct port if needed
    command = list(config.start_command)
    # Replace port placeholder if present
    command = [
        str(config.port)
        if arg == str(SEMBR_CONFIG.port) and config.port != SEMBR_CONFIG.port
        else arg
        for arg in command
    ]

    try:
        # Start the process
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            start_new_session=True,  # Detach from parent
        )

        logger.info(f"{server_type} process started (PID {process.pid})")

        # Wait for server to become healthy
        start_time = time.time()
        while time.time() - start_time < config.startup_timeout:
            if check_server_health(server_type, timeout=2.0):
                server_process = ServerProcess(
                    config=config,
                    pid=process.pid,
                    start_time=datetime.now(),
                    gpu_id=gpu_id,
                    process=process,
                )
                _servers[server_type] = server_process
                logger.info(f"{server_type} server healthy after {time.time() - start_time:.1f}s")
                return server_process

            # Check if process died
            if process.poll() is not None:
                raise RuntimeError(f"{server_type} process exited with code {process.returncode}")

            time.sleep(1.0)

        # Timeout - kill the process
        process.kill()
        raise RuntimeError(
            f"{server_type} failed to become healthy within {config.startup_timeout}s"
        )

    except FileNotFoundError as err:
        raise RuntimeError(f"Command not found: {command[0]}") from err
    except PermissionError as err:
        raise RuntimeError(f"Permission denied running: {command[0]}") from err


def stop_server(server_type: str, graceful: bool = True, timeout: float | None = None) -> bool:
    """Stop a running server.

    Args:
        server_type: Server type to stop
        graceful: If True, send SIGTERM first and wait
        timeout: Graceful shutdown timeout (default: from config)

    Returns:
        True if server was stopped, False if it wasn't running.
    """
    if server_type not in _servers:
        logger.debug(f"{server_type} not in registry")
        # Try to find and kill by pattern anyway
        return _kill_server_by_pattern(server_type)

    server = _servers[server_type]
    pid = server.pid

    if timeout is None:
        timeout = server.config.graceful_shutdown_timeout

    logger.info(f"Stopping {server_type} (PID {pid})")

    try:
        if graceful:
            # Send SIGTERM
            os.kill(pid, signal.SIGTERM)
            logger.debug(f"Sent SIGTERM to {server_type} (PID {pid})")

            # Wait for graceful shutdown
            start_time = time.time()
            while time.time() - start_time < timeout:
                if not _is_process_running(pid):
                    logger.info(f"{server_type} stopped gracefully")
                    break
                time.sleep(0.5)
            else:
                # Still running, send SIGKILL
                logger.warning(f"{server_type} didn't stop gracefully, sending SIGKILL")
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
        else:
            # Immediate kill
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)

    except ProcessLookupError:
        logger.debug(f"{server_type} process {pid} already gone")
    except PermissionError:
        logger.warning(f"Permission denied stopping {server_type} (PID {pid})")
        return False

    # Remove from registry
    del _servers[server_type]

    # Clear CUDA cache after stopping
    reset_cuda_cache(server.gpu_id)

    return True


def _kill_server_by_pattern(server_type: str) -> bool:
    """Kill server processes by pattern matching.

    This is used when the server is not in our registry but might
    still be running (e.g., started in a previous session).

    Args:
        server_type: Server type to kill

    Returns:
        True if any processes were killed.
    """
    patterns: dict[str, str] = {
        "sembr": "sembr --listen",
        "ollama": "ollama serve",
    }

    pattern = patterns.get(server_type)
    if pattern is None:
        return False

    try:
        result = subprocess.run(
            ["pkill", "-f", pattern],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            logger.info(f"Killed {server_type} processes matching '{pattern}'")
            return True
        return False
    except FileNotFoundError:
        logger.warning("pkill not available")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("pkill timed out")
        return False


def restart_server(server_type: str, new_gpu: int | None = None) -> ServerProcess:
    """Restart a server, optionally on a different GPU.

    Args:
        server_type: Server type to restart
        new_gpu: GPU to use (None = use previous or auto-select)

    Returns:
        New ServerProcess object

    Raises:
        RuntimeError: If restart fails
    """
    config = _get_config_for_type(server_type)
    if config is None:
        raise RuntimeError(f"Unknown server type: {server_type}")

    # Get previous GPU if not specified
    if new_gpu is None and server_type in _servers:
        new_gpu = _servers[server_type].gpu_id

    # Stop existing server
    stop_server(server_type)

    # Brief pause to ensure cleanup
    time.sleep(1.0)

    # Start fresh
    return start_server(config, gpu_id=new_gpu)


def get_server_status(server_type: str) -> ServerProcess | None:
    """Get status of a tracked server.

    Args:
        server_type: Server type to query

    Returns:
        ServerProcess if running and tracked, None otherwise.
    """
    if server_type not in _servers:
        return None

    server = _servers[server_type]
    if not _is_process_running(server.pid):
        # Process died, clean up
        del _servers[server_type]
        return None

    return server


def is_server_running(server_type: str) -> bool:
    """Check if a server is running (process exists and responds to health check).

    Args:
        server_type: Server type to check

    Returns:
        True if server is running and healthy.
    """
    status = get_server_status(server_type)
    if status is None:
        return False
    return check_server_health(server_type)


# =============================================================================
# PROCESS UTILITIES
# =============================================================================


def _is_process_running(pid: int) -> bool:
    """Check if a process is still running.

    Args:
        pid: Process ID to check

    Returns:
        True if process exists, False otherwise.
    """
    try:
        os.kill(pid, 0)  # Signal 0 = check existence
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Process exists but we can't signal it


# =============================================================================
# SIGNAL HANDLING
# =============================================================================


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful shutdown.

    This should be called at the start of a batch processing operation.
    Signals SIGINT and SIGTERM will set the shutdown flag, allowing
    the main loop to complete the current file before stopping.
    """
    global _shutdown_requested
    _shutdown_requested = False

    def handler(signum: int, _frame: object) -> None:
        global _shutdown_requested
        _shutdown_requested = True
        signal_name = signal.Signals(signum).name
        logger.info(f"Shutdown requested (received {signal_name})")

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def is_shutdown_requested() -> bool:
    """Check if a shutdown has been requested via signal.

    Returns:
        True if SIGINT or SIGTERM was received.
    """
    return _shutdown_requested


def reset_shutdown_flag() -> None:
    """Reset the shutdown flag.

    Call this after handling a shutdown to allow continued processing
    or after cleanup is complete.
    """
    global _shutdown_requested
    _shutdown_requested = False


# =============================================================================
# CLEANUP
# =============================================================================


def cleanup_all_servers() -> None:
    """Stop all tracked servers.

    This is called automatically on process exit via atexit.
    """
    logger.info("Cleaning up all servers...")

    for server_type in list(_servers.keys()):
        try:
            stop_server(server_type, graceful=True, timeout=3.0)
        except Exception as e:
            logger.warning(f"Error stopping {server_type}: {e}")


def register_cleanup() -> None:
    """Register atexit handler for cleanup.

    Call this once at application startup to ensure servers are
    cleaned up when the process exits.
    """
    atexit.register(cleanup_all_servers)
    logger.debug("Registered atexit cleanup handler")


# =============================================================================
# HEALTH MONITORING
# =============================================================================


class HealthMonitor:
    """Monitor server health during batch processing.

    Usage:
        monitor = HealthMonitor("sembr", interval=60.0)
        monitor.start()

        # In your batch loop:
        for file in files:
            if monitor.should_check():
                if not monitor.check_and_recover():
                    raise RuntimeError("Server unrecoverable")
            process_file(file)

        monitor.stop()
    """

    def __init__(
        self,
        server_type: str,
        interval: float = 60.0,
        max_recovery_attempts: int = 3,
        recovery_callback: Callable[[], bool] | None = None,
    ):
        """Initialize health monitor.

        Args:
            server_type: Server type to monitor
            interval: Seconds between health checks
            max_recovery_attempts: Max attempts to recover server
            recovery_callback: Optional callback for recovery (return True on success)
        """
        self.server_type = server_type
        self.interval = interval
        self.max_recovery_attempts = max_recovery_attempts
        self.recovery_callback = recovery_callback

        self._last_check_time: float = 0.0
        self._recovery_attempts: int = 0
        self._is_healthy: bool = True
        self._is_running: bool = False

    def start(self) -> None:
        """Start monitoring."""
        self._is_running = True
        self._last_check_time = time.time()
        self._recovery_attempts = 0
        self._is_healthy = check_server_health(self.server_type)

    def stop(self) -> None:
        """Stop monitoring."""
        self._is_running = False

    def should_check(self) -> bool:
        """Check if it's time for a health check.

        Returns:
            True if interval has elapsed since last check.
        """
        if not self._is_running:
            return False
        return time.time() - self._last_check_time >= self.interval

    def check_and_recover(self) -> bool:
        """Perform health check and attempt recovery if needed.

        Returns:
            True if server is healthy (or was recovered), False if unrecoverable.
        """
        self._last_check_time = time.time()

        if check_server_health(self.server_type):
            self._is_healthy = True
            self._recovery_attempts = 0
            return True

        logger.warning(f"{self.server_type} health check failed, attempting recovery")
        self._is_healthy = False

        # Attempt recovery
        for _ in range(self.max_recovery_attempts):
            self._recovery_attempts += 1
            logger.info(f"Recovery attempt {self._recovery_attempts}/{self.max_recovery_attempts}")

            # Use custom callback if provided
            if self.recovery_callback is not None:
                if self.recovery_callback():
                    self._is_healthy = True
                    return True
            else:
                # Default: restart server
                try:
                    restart_server(self.server_type)
                    if check_server_health(self.server_type):
                        self._is_healthy = True
                        logger.info(
                            f"{self.server_type} recovered after {self._recovery_attempts} attempts"
                        )
                        return True
                except Exception as e:
                    logger.warning(f"Recovery failed: {e}")

            time.sleep(2.0)

        logger.error(f"{self.server_type} unrecoverable after {self._recovery_attempts} attempts")
        return False

    @property
    def is_healthy(self) -> bool:
        """Current health status."""
        return self._is_healthy

    @property
    def recovery_attempts(self) -> int:
        """Number of recovery attempts made."""
        return self._recovery_attempts


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def ensure_server_running(server_type: str, gpu_id: int | None = None) -> ServerProcess:
    """Ensure a server is running, starting it if needed.

    Args:
        server_type: Server type ("sembr" or "ollama")
        gpu_id: GPU to use (auto-select if None)

    Returns:
        ServerProcess for the running server

    Raises:
        RuntimeError: If server cannot be started
    """
    # Check if already running and healthy
    if is_server_running(server_type):
        server = _servers.get(server_type)
        if server is not None:
            return server

    # Get config
    config = _get_config_for_type(server_type)
    if config is None:
        raise RuntimeError(f"Unknown server type: {server_type}")

    # Start the server
    return start_server(config, gpu_id=gpu_id)


def get_running_servers() -> list[str]:
    """Get list of currently running server types.

    Returns:
        List of server type names that are running.
    """
    running = []
    for server_type in list(_servers.keys()):
        if is_server_running(server_type):
            running.append(server_type)
    return running


def get_server_summary() -> str:
    """Get a human-readable summary of server status.

    Returns:
        Multi-line string with server information.
    """
    lines = ["Server Summary:", "-" * 50]

    for server_type, _config in [("sembr", SEMBR_CONFIG), ("ollama", OLLAMA_CONFIG)]:
        status = get_server_status(server_type)
        if status is not None:
            uptime = datetime.now() - status.start_time
            healthy = "HEALTHY" if check_server_health(server_type) else "UNHEALTHY"
            lines.append(
                f"  {server_type}: PID {status.pid} on GPU {status.gpu_id} "
                f"(uptime: {uptime.seconds}s) - {healthy}"
            )
        else:
            running = check_server_health(server_type)
            if running:
                lines.append(f"  {server_type}: Running (untracked)")
            else:
                lines.append(f"  {server_type}: Not running")

    return "\n".join(lines)
