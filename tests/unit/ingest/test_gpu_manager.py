"""Unit tests for GPU manager module.

These tests mock subprocess calls to test GPU detection, health checking,
and CUDA state management without requiring actual GPU hardware.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from pw_mcp.ingest.gpu_manager import (
    GPUStatus,
    check_gpu_health,
    detect_gpus,
    get_available_gpu,
    get_cuda_env,
    get_gpu_processes,
    get_gpu_summary,
    is_cuda_corrupted,
    kill_gpu_processes,
    parse_cuda_visible_devices,
    reset_cuda_cache,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_nvidia_smi_output() -> str:
    """Sample nvidia-smi output with two GPUs."""
    return """0, NVIDIA GeForce RTX 3060, 512, 12288, 15
1, NVIDIA GeForce GTX 1650, 256, 4096, 5"""


@pytest.fixture
def mock_single_gpu_output() -> str:
    """Sample nvidia-smi output with single GPU."""
    return "0, NVIDIA GeForce RTX 3060, 512, 12288, 15"


@pytest.fixture
def mock_gpu_processes_output() -> str:
    """Sample nvidia-smi compute apps output."""
    return """12345, python, 1024
67890, ollama, 2048"""


# =============================================================================
# GPU STATUS DATACLASS TESTS
# =============================================================================


class TestGPUStatus:
    """Tests for GPUStatus dataclass."""

    def test_gpu_status_creation(self) -> None:
        """Test creating GPUStatus instance."""
        status = GPUStatus(
            index=0,
            name="NVIDIA GeForce RTX 3060",
            memory_used_mb=512,
            memory_total_mb=12288,
            is_available=True,
            utilization_percent=15,
        )

        assert status.index == 0
        assert status.name == "NVIDIA GeForce RTX 3060"
        assert status.memory_used_mb == 512
        assert status.memory_total_mb == 12288
        assert status.is_available is True
        assert status.utilization_percent == 15

    def test_memory_free_mb(self) -> None:
        """Test memory_free_mb property."""
        status = GPUStatus(
            index=0,
            name="Test GPU",
            memory_used_mb=1000,
            memory_total_mb=12288,
            is_available=True,
            utilization_percent=10,
        )

        assert status.memory_free_mb == 11288

    def test_memory_usage_percent(self) -> None:
        """Test memory_usage_percent property."""
        status = GPUStatus(
            index=0,
            name="Test GPU",
            memory_used_mb=6144,
            memory_total_mb=12288,
            is_available=True,
            utilization_percent=50,
        )

        assert status.memory_usage_percent == 50.0

    def test_memory_usage_percent_zero_total(self) -> None:
        """Test memory_usage_percent with zero total memory."""
        status = GPUStatus(
            index=0,
            name="Test GPU",
            memory_used_mb=0,
            memory_total_mb=0,
            is_available=True,
            utilization_percent=0,
        )

        assert status.memory_usage_percent == 0.0

    def test_gpu_status_is_frozen(self) -> None:
        """Test that GPUStatus is immutable."""
        status = GPUStatus(
            index=0,
            name="Test GPU",
            memory_used_mb=512,
            memory_total_mb=12288,
            is_available=True,
            utilization_percent=15,
        )

        with pytest.raises(AttributeError):
            status.index = 1  # type: ignore[misc]


# =============================================================================
# GPU DETECTION TESTS
# =============================================================================


class TestDetectGPUs:
    """Tests for detect_gpus function."""

    def test_detect_two_gpus(self, mock_nvidia_smi_output: str) -> None:
        """Test detecting multiple GPUs."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_nvidia_smi_output

        with patch("subprocess.run", return_value=mock_result):
            gpus = detect_gpus()

        assert len(gpus) == 2
        assert gpus[0].index == 0
        assert gpus[0].name == "NVIDIA GeForce RTX 3060"
        assert gpus[0].memory_used_mb == 512
        assert gpus[0].memory_total_mb == 12288
        assert gpus[1].index == 1
        assert gpus[1].name == "NVIDIA GeForce GTX 1650"

    def test_detect_single_gpu(self, mock_single_gpu_output: str) -> None:
        """Test detecting single GPU."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_single_gpu_output

        with patch("subprocess.run", return_value=mock_result):
            gpus = detect_gpus()

        assert len(gpus) == 1
        assert gpus[0].index == 0

    def test_detect_no_gpus_empty_output(self) -> None:
        """Test empty output from nvidia-smi."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            gpus = detect_gpus()

        assert len(gpus) == 0

    def test_detect_nvidia_smi_not_found(self) -> None:
        """Test handling when nvidia-smi is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            gpus = detect_gpus()

        assert len(gpus) == 0

    def test_detect_nvidia_smi_timeout(self) -> None:
        """Test handling nvidia-smi timeout."""
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)
        ):
            gpus = detect_gpus()

        assert len(gpus) == 0

    def test_detect_nvidia_smi_error(self) -> None:
        """Test handling nvidia-smi non-zero return code."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "NVIDIA driver error"

        with patch("subprocess.run", return_value=mock_result):
            gpus = detect_gpus()

        assert len(gpus) == 0

    def test_detect_malformed_output(self) -> None:
        """Test handling malformed nvidia-smi output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "invalid,output,format"

        with patch("subprocess.run", return_value=mock_result):
            gpus = detect_gpus()

        assert len(gpus) == 0


# =============================================================================
# GPU HEALTH CHECK TESTS
# =============================================================================


class TestCheckGPUHealth:
    """Tests for check_gpu_health function."""

    def test_check_existing_gpu(self, mock_nvidia_smi_output: str) -> None:
        """Test checking health of existing GPU."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_nvidia_smi_output

        with patch("subprocess.run", return_value=mock_result):
            status = check_gpu_health(0)

        assert status is not None
        assert status.index == 0

    def test_check_nonexistent_gpu(self, mock_single_gpu_output: str) -> None:
        """Test checking health of non-existent GPU."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_single_gpu_output

        with patch("subprocess.run", return_value=mock_result):
            status = check_gpu_health(5)

        assert status is None


# =============================================================================
# GET AVAILABLE GPU TESTS
# =============================================================================


class TestGetAvailableGPU:
    """Tests for get_available_gpu function."""

    def test_get_first_available(self, mock_nvidia_smi_output: str) -> None:
        """Test getting first available GPU."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_nvidia_smi_output

        with patch("subprocess.run", return_value=mock_result):
            gpu_id = get_available_gpu()

        assert gpu_id == 0

    def test_get_available_with_exclusion(self, mock_nvidia_smi_output: str) -> None:
        """Test getting available GPU with exclusion list."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_nvidia_smi_output

        with patch("subprocess.run", return_value=mock_result):
            gpu_id = get_available_gpu(exclude=[0])

        assert gpu_id == 1

    def test_get_available_all_excluded(self, mock_nvidia_smi_output: str) -> None:
        """Test when all GPUs are excluded."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_nvidia_smi_output

        with patch("subprocess.run", return_value=mock_result):
            gpu_id = get_available_gpu(exclude=[0, 1])

        assert gpu_id is None

    def test_get_available_no_gpus(self) -> None:
        """Test when no GPUs available."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            gpu_id = get_available_gpu()

        assert gpu_id is None

    def test_get_available_high_memory_usage(self) -> None:
        """Test skipping GPUs with high memory usage."""
        # GPU with 95% memory usage
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0, Test GPU, 11700, 12288, 95"

        with patch("subprocess.run", return_value=mock_result):
            gpu_id = get_available_gpu()

        assert gpu_id is None


# =============================================================================
# CUDA RESET TESTS
# =============================================================================


class TestResetCUDACache:
    """Tests for reset_cuda_cache function."""

    def test_reset_specific_gpu_success(self) -> None:
        """Test successful CUDA reset for specific GPU."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "CUDA cache cleared for GPU 0"

        with patch("subprocess.run", return_value=mock_result):
            result = reset_cuda_cache(0)

        assert result is True

    def test_reset_all_gpus_success(self) -> None:
        """Test successful CUDA reset for all GPUs."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "CUDA cache cleared for 2 GPUs"

        with patch("subprocess.run", return_value=mock_result):
            result = reset_cuda_cache()

        assert result is True

    def test_reset_failure(self) -> None:
        """Test CUDA reset failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "CUDA error"

        with patch("subprocess.run", return_value=mock_result):
            result = reset_cuda_cache(0)

        assert result is False

    def test_reset_python_not_found(self) -> None:
        """Test handling when python3 is not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = reset_cuda_cache(0)

        assert result is False

    def test_reset_timeout(self) -> None:
        """Test handling CUDA reset timeout."""
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="python3", timeout=30)
        ):
            result = reset_cuda_cache(0)

        assert result is False


# =============================================================================
# CUDA CORRUPTION CHECK TESTS
# =============================================================================


class TestIsCUDACorrupted:
    """Tests for is_cuda_corrupted function."""

    def test_healthy_gpu(self) -> None:
        """Test detecting healthy GPU."""
        # First call for GPU info
        gpu_info = MagicMock()
        gpu_info.returncode = 0
        gpu_info.stdout = "NVIDIA GeForce RTX 3060, 512, 12288"

        # Second call for processes
        proc_info = MagicMock()
        proc_info.returncode = 0
        proc_info.stdout = "12345"

        with patch("subprocess.run", side_effect=[gpu_info, proc_info]):
            result = is_cuda_corrupted(0)

        assert result is False

    def test_error_in_output(self) -> None:
        """Test detecting GPU with error state."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 3060, ERR!, 12288"

        with patch("subprocess.run", return_value=mock_result):
            result = is_cuda_corrupted(0)

        assert result is True

    def test_high_memory_no_processes(self) -> None:
        """Test detecting orphaned GPU memory."""
        # First call for GPU info - high memory
        gpu_info = MagicMock()
        gpu_info.returncode = 0
        gpu_info.stdout = "NVIDIA GeForce RTX 3060, 5000, 12288"

        # Second call for processes - empty
        proc_info = MagicMock()
        proc_info.returncode = 0
        proc_info.stdout = ""

        with patch("subprocess.run", side_effect=[gpu_info, proc_info]):
            result = is_cuda_corrupted(0)

        assert result is True

    def test_nvidia_smi_failure(self) -> None:
        """Test handling nvidia-smi failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error"

        with patch("subprocess.run", return_value=mock_result):
            result = is_cuda_corrupted(0)

        assert result is True  # Assume corrupted if can't query

    def test_nvidia_smi_not_found(self) -> None:
        """Test handling missing nvidia-smi."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = is_cuda_corrupted(0)

        assert result is False  # Can't determine, assume OK


# =============================================================================
# GPU PROCESS TESTS
# =============================================================================


class TestGetGPUProcesses:
    """Tests for get_gpu_processes function."""

    def test_get_processes_success(self, mock_gpu_processes_output: str) -> None:
        """Test getting GPU processes."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_gpu_processes_output

        with patch("subprocess.run", return_value=mock_result):
            processes = get_gpu_processes(0)

        assert len(processes) == 2
        assert processes[0]["pid"] == 12345
        assert processes[0]["name"] == "python"
        assert processes[0]["memory_mb"] == 1024

    def test_get_processes_empty(self) -> None:
        """Test getting processes when none running."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            processes = get_gpu_processes(0)

        assert len(processes) == 0

    def test_get_processes_failure(self) -> None:
        """Test handling nvidia-smi failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            processes = get_gpu_processes(0)

        assert len(processes) == 0


class TestKillGPUProcesses:
    """Tests for kill_gpu_processes function."""

    def test_kill_processes_term(self) -> None:
        """Test killing processes with SIGTERM."""
        # Mock get_gpu_processes
        mock_procs = [{"pid": 12345, "name": "python", "memory_mb": 1024}]

        with (
            patch("pw_mcp.ingest.gpu_manager.get_gpu_processes", return_value=mock_procs),
            patch("subprocess.run") as mock_run,
        ):
            killed = kill_gpu_processes(0, "TERM")

        assert killed == 1
        mock_run.assert_called_once()

    def test_kill_processes_kill(self) -> None:
        """Test killing processes with SIGKILL."""
        mock_procs = [{"pid": 12345, "name": "python", "memory_mb": 1024}]

        with (
            patch("pw_mcp.ingest.gpu_manager.get_gpu_processes", return_value=mock_procs),
            patch("subprocess.run") as mock_run,
        ):
            killed = kill_gpu_processes(0, "KILL")

        assert killed == 1
        # Check -9 was used
        args = mock_run.call_args[0][0]
        assert "-9" in args


# =============================================================================
# ENVIRONMENT TESTS
# =============================================================================


class TestGetCUDAEnv:
    """Tests for get_cuda_env function."""

    def test_get_env_sets_visible_devices(self) -> None:
        """Test that CUDA_VISIBLE_DEVICES is set."""
        env = get_cuda_env(1)
        assert env["CUDA_VISIBLE_DEVICES"] == "1"

    def test_get_env_preserves_other_vars(self) -> None:
        """Test that other environment variables are preserved."""
        with patch.dict("os.environ", {"PATH": "/usr/bin", "HOME": "/home/user"}):
            env = get_cuda_env(0)
            assert env["PATH"] == "/usr/bin"
            assert env["HOME"] == "/home/user"


class TestParseCUDAVisibleDevices:
    """Tests for parse_cuda_visible_devices function."""

    def test_parse_single_device(self) -> None:
        """Test parsing single device."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
            devices = parse_cuda_visible_devices()
        assert devices == [0]

    def test_parse_multiple_devices(self) -> None:
        """Test parsing multiple devices."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1,2"}):
            devices = parse_cuda_visible_devices()
        assert devices == [0, 1, 2]

    def test_parse_empty_returns_all(self, mock_nvidia_smi_output: str) -> None:
        """Test that empty env returns all detected GPUs."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_nvidia_smi_output

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("subprocess.run", return_value=mock_result),
        ):
            # Remove CUDA_VISIBLE_DEVICES if it exists
            import os

            env_backup = os.environ.get("CUDA_VISIBLE_DEVICES")
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            try:
                devices = parse_cuda_visible_devices()
            finally:
                if env_backup is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = env_backup

        assert devices == [0, 1]

    def test_parse_invalid_value(self) -> None:
        """Test parsing invalid CUDA_VISIBLE_DEVICES."""
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "invalid"}):
            devices = parse_cuda_visible_devices()
        assert devices == []


# =============================================================================
# DIAGNOSTIC TESTS
# =============================================================================


class TestGetGPUSummary:
    """Tests for get_gpu_summary function."""

    def test_summary_with_gpus(self, mock_nvidia_smi_output: str) -> None:
        """Test GPU summary output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = mock_nvidia_smi_output

        # Mock corruption check
        corruption_result = MagicMock()
        corruption_result.returncode = 0
        corruption_result.stdout = "NVIDIA GeForce RTX 3060, 512, 12288"

        proc_result = MagicMock()
        proc_result.returncode = 0
        proc_result.stdout = "12345"

        with patch(
            "subprocess.run",
            side_effect=[
                mock_result,
                corruption_result,
                proc_result,
                corruption_result,
                proc_result,
            ],
        ):
            summary = get_gpu_summary()

        assert "GPU Summary" in summary
        assert "RTX 3060" in summary
        assert "GTX 1650" in summary

    def test_summary_no_gpus(self) -> None:
        """Test GPU summary when no GPUs detected."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            summary = get_gpu_summary()

        assert "No NVIDIA GPUs detected" in summary
