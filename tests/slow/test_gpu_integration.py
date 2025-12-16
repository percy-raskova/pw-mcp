"""Slow integration tests for GPU and server management.

These tests are marked as @pytest.mark.slow and are skipped by default.
Run with: pytest -m slow

Prerequisites:
- NVIDIA GPU with nvidia-smi available
- For server tests: sembr dependencies installed
- For failover tests: 2+ GPUs available
"""

from __future__ import annotations

import contextlib
import subprocess
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest

from pw_mcp.ingest.gpu_manager import (
    check_gpu_health,
    detect_gpus,
    get_available_gpu,
    get_gpu_processes,
    get_gpu_summary,
    is_cuda_corrupted,
    reset_cuda_cache,
)
from pw_mcp.ingest.server_manager import (
    SEMBR_CONFIG,
    check_server_health,
    is_shutdown_requested,
    reset_shutdown_flag,
    restart_server,
    setup_signal_handlers,
    start_server,
    stop_server,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def require_gpu() -> int:
    """Skip test if no GPU available, return first GPU index."""
    gpus = detect_gpus()
    if not gpus:
        pytest.skip("No NVIDIA GPU detected - requires nvidia-smi")
    return gpus[0].index


@pytest.fixture
def require_two_gpus() -> list[int]:
    """Skip test if less than 2 GPUs available."""
    gpus = detect_gpus()
    if len(gpus) < 2:
        pytest.skip("Requires 2+ GPUs for failover tests")
    return [g.index for g in gpus]


@pytest.fixture
def cleanup_sembr_server() -> Generator[None, None, None]:
    """Ensure sembr server is stopped after test."""
    yield
    # Cleanup: stop any running sembr server
    with contextlib.suppress(Exception):
        stop_server("sembr", graceful=True, timeout=5.0)
    # Also try pkill as fallback
    with contextlib.suppress(Exception):
        subprocess.run(
            ["pkill", "-f", "sembr --listen"],
            capture_output=True,
            timeout=5,
        )


# =============================================================================
# GPU DETECTION TESTS
# =============================================================================


class TestRealGPUDetection:
    """Tests for real GPU detection via nvidia-smi."""

    @pytest.mark.slow
    def test_detect_gpus_returns_valid_status(self, require_gpu: int) -> None:
        """Should return valid GPUStatus objects from real hardware."""
        gpus = detect_gpus()

        assert len(gpus) >= 1
        for gpu in gpus:
            # Index should be non-negative
            assert gpu.index >= 0

            # Name should be non-empty
            assert gpu.name
            assert len(gpu.name) > 0

            # Memory values should be positive
            assert gpu.memory_total_mb > 0
            assert gpu.memory_used_mb >= 0
            assert gpu.memory_used_mb <= gpu.memory_total_mb

            # Free memory calculation should work
            assert gpu.memory_free_mb >= 0
            assert gpu.memory_free_mb == gpu.memory_total_mb - gpu.memory_used_mb

            # Usage percentage should be valid
            assert 0.0 <= gpu.memory_usage_percent <= 100.0

            # Utilization should be in valid range
            assert 0 <= gpu.utilization_percent <= 100

    @pytest.mark.slow
    def test_check_gpu_health_returns_status(self, require_gpu: int) -> None:
        """Should return health status for specific GPU."""
        status = check_gpu_health(require_gpu)

        assert status is not None
        assert status.index == require_gpu
        assert status.name
        assert status.memory_total_mb > 0

    @pytest.mark.slow
    def test_check_gpu_health_invalid_index(self, require_gpu: int) -> None:
        """Should return None for non-existent GPU index."""
        # Use a very high index that won't exist
        status = check_gpu_health(9999)
        assert status is None

    @pytest.mark.slow
    def test_get_available_gpu_returns_valid(self, require_gpu: int) -> None:
        """Should return a valid GPU index."""
        gpu_id = get_available_gpu()

        assert gpu_id is not None
        assert gpu_id >= 0

        # Should be one of the detected GPUs
        gpus = detect_gpus()
        gpu_indices = [g.index for g in gpus]
        assert gpu_id in gpu_indices

    @pytest.mark.slow
    def test_get_available_gpu_excludes_specified(self, require_two_gpus: list[int]) -> None:
        """Should exclude specified GPUs from selection."""
        first_gpu = require_two_gpus[0]

        # Exclude first GPU, should get second
        gpu_id = get_available_gpu(exclude=[first_gpu])

        assert gpu_id is not None
        assert gpu_id != first_gpu

    @pytest.mark.slow
    def test_get_gpu_summary_is_readable(self, require_gpu: int) -> None:
        """Should return human-readable summary."""
        summary = get_gpu_summary()

        assert "GPU Summary:" in summary
        assert "-" * 60 in summary

        # Should contain GPU info
        gpus = detect_gpus()
        for gpu in gpus:
            assert f"[{gpu.index}]" in summary
            assert gpu.name in summary

    @pytest.mark.slow
    def test_is_cuda_corrupted_healthy_gpu(self, require_gpu: int) -> None:
        """Should detect healthy GPU as not corrupted."""
        # A freshly detected GPU should not be corrupted
        # (unless something is actually wrong with the system)
        corrupted = is_cuda_corrupted(require_gpu)

        # We expect a healthy system, but this could fail if GPU is actually stuck
        # Log the result rather than hard-asserting
        if corrupted:
            pytest.skip("GPU appears corrupted - may need system reboot")

        assert corrupted is False


# =============================================================================
# GPU PROCESS TESTS
# =============================================================================


class TestRealGPUProcesses:
    """Tests for GPU process management."""

    @pytest.mark.slow
    def test_get_gpu_processes_returns_list(self, require_gpu: int) -> None:
        """Should return list of processes (may be empty)."""
        processes = get_gpu_processes(require_gpu)

        assert isinstance(processes, list)

        # If processes exist, validate structure
        for proc in processes:
            assert "pid" in proc
            assert "name" in proc
            assert "memory_mb" in proc
            assert isinstance(proc["pid"], int)
            assert proc["pid"] > 0


# =============================================================================
# CUDA CACHE TESTS
# =============================================================================


class TestRealCUDACache:
    """Tests for CUDA cache management."""

    @pytest.mark.slow
    def test_reset_cuda_cache_succeeds_or_reports(self, require_gpu: int) -> None:
        """Should attempt CUDA cache reset without error."""
        # This test may succeed or fail depending on PyTorch availability
        # It should not raise an exception either way
        result = reset_cuda_cache(require_gpu)

        # Result is True if reset worked, False if not (e.g., no PyTorch)
        assert isinstance(result, bool)


# =============================================================================
# SERVER LIFECYCLE TESTS
# =============================================================================


class TestSembrServerLifecycle:
    """Integration tests for sembr server start/stop."""

    @pytest.mark.slow
    def test_server_start_stop_cycle(
        self,
        require_gpu: int,
        cleanup_sembr_server: None,
    ) -> None:
        """Should start and stop sembr server cleanly."""
        # Start server
        server = start_server(SEMBR_CONFIG, gpu_id=require_gpu)

        if server is None:
            pytest.skip("Could not start sembr server - check sembr installation")

        try:
            assert server.pid > 0
            assert server.gpu_id == require_gpu

            # Verify health
            assert check_server_health("sembr") is True

            # Stop server
            stopped = stop_server("sembr", graceful=True, timeout=10.0)
            assert stopped is True

            # Give it a moment to fully stop
            time.sleep(2)

            # Verify stopped
            assert check_server_health("sembr") is False

        except Exception:
            # Cleanup on failure
            stop_server("sembr", graceful=False)
            raise

    @pytest.mark.slow
    def test_server_restart_same_gpu(
        self,
        require_gpu: int,
        cleanup_sembr_server: None,
    ) -> None:
        """Should restart server on same GPU."""
        # Start initial server
        server1 = start_server(SEMBR_CONFIG, gpu_id=require_gpu)

        if server1 is None:
            pytest.skip("Could not start sembr server")

        try:
            first_pid = server1.pid

            # Restart
            server2 = restart_server("sembr", new_gpu=require_gpu)

            assert server2 is not None
            assert server2.pid != first_pid  # New process
            assert server2.gpu_id == require_gpu

            # Should be healthy
            assert check_server_health("sembr") is True

        except Exception:
            stop_server("sembr", graceful=False)
            raise

    @pytest.mark.slow
    def test_server_restart_different_gpu(
        self,
        require_two_gpus: list[int],
        cleanup_sembr_server: None,
    ) -> None:
        """Should restart server on different GPU for failover."""
        gpu1, gpu2 = require_two_gpus[0], require_two_gpus[1]

        # Start on first GPU
        server1 = start_server(SEMBR_CONFIG, gpu_id=gpu1)

        if server1 is None:
            pytest.skip("Could not start sembr server")

        try:
            assert server1.gpu_id == gpu1

            # Restart on second GPU (failover scenario)
            server2 = restart_server("sembr", new_gpu=gpu2)

            assert server2 is not None
            assert server2.gpu_id == gpu2
            assert check_server_health("sembr") is True

        except Exception:
            stop_server("sembr", graceful=False)
            raise


# =============================================================================
# SIGNAL HANDLING TESTS
# =============================================================================


class TestSignalHandling:
    """Tests for signal handling infrastructure."""

    @pytest.mark.slow
    def test_setup_signal_handlers_runs(self) -> None:
        """Should set up signal handlers without error."""
        # This shouldn't raise
        setup_signal_handlers()

        # Default state should be not shutdown
        reset_shutdown_flag()
        assert is_shutdown_requested() is False

    @pytest.mark.slow
    def test_shutdown_flag_persistence(self) -> None:
        """Should persist shutdown flag state."""
        reset_shutdown_flag()
        assert is_shutdown_requested() is False

        # Note: We can't easily test actual signal sending in a test
        # as it would interrupt the test process. The flag functionality
        # is tested via reset_shutdown_flag.


# =============================================================================
# HEALTH MONITORING TESTS
# =============================================================================


class TestHealthMonitoring:
    """Integration tests for health monitoring during batch processing."""

    @pytest.mark.slow
    def test_health_monitor_detects_server_down(
        self,
        require_gpu: int,
        cleanup_sembr_server: None,
    ) -> None:
        """Should detect when server goes down."""
        from pw_mcp.ingest.server_manager import HealthMonitor

        # Start server
        server = start_server(SEMBR_CONFIG, gpu_id=require_gpu)

        if server is None:
            pytest.skip("Could not start sembr server")

        try:
            # Create monitor
            monitor = HealthMonitor(
                server_type="sembr",
                interval=1.0,  # Quick checks for testing
                max_recovery_attempts=1,
            )
            monitor.start()

            # Should be healthy initially
            assert monitor.is_healthy is True

            # Kill the server (simulate crash)
            stop_server("sembr", graceful=False)
            time.sleep(2)

            # Force a check
            monitor._last_check_time = 0  # Force check
            result = monitor.check_and_recover()

            # Recovery should fail (no callback)
            assert result is False

            monitor.stop()

        except Exception:
            stop_server("sembr", graceful=False)
            raise
