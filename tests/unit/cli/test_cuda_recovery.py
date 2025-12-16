"""Unit tests for CUDA crash detection and recovery in CLI.

These tests verify that CUDA crashes are detected and that the server
can be automatically restarted to recover from device-side assert errors.

Test strategy:
- Test is_cuda_crash() detection patterns
- Test batch processing handles SembrSkipError correctly
- Test CUDA crash triggers server restart
"""

from __future__ import annotations

import pytest

# Import the functions we're testing
from pw_mcp.ingest.cli import is_cuda_crash

# =============================================================================
# CUDA CRASH DETECTION TESTS
# =============================================================================


class TestIsCudaCrash:
    """Tests for is_cuda_crash() function."""

    @pytest.mark.unit
    def test_cuda_error_lowercase(self) -> None:
        """Should detect 'cuda error' in exception message."""
        error = Exception("cuda error: device-side assert triggered")
        assert is_cuda_crash(error) is True

    @pytest.mark.unit
    def test_cuda_error_uppercase(self) -> None:
        """Should detect 'CUDA error' (case insensitive)."""
        error = Exception("CUDA error: out of memory")
        assert is_cuda_crash(error) is True

    @pytest.mark.unit
    def test_device_side_assert(self) -> None:
        """Should detect 'device-side assert' pattern."""
        error = Exception("device-side assert triggered at some location")
        assert is_cuda_crash(error) is True

    @pytest.mark.unit
    def test_cuda_error_assert(self) -> None:
        """Should detect 'cudaerrorassert' pattern."""
        error = Exception("cudaerrorassert: Kernel launch failed")
        assert is_cuda_crash(error) is True

    @pytest.mark.unit
    def test_unrelated_error_not_cuda(self) -> None:
        """Should return False for unrelated errors."""
        error = Exception("Connection refused")
        assert is_cuda_crash(error) is False

    @pytest.mark.unit
    def test_timeout_error_not_cuda(self) -> None:
        """Should return False for timeout errors."""
        error = Exception("Request timed out after 60s")
        assert is_cuda_crash(error) is False

    @pytest.mark.unit
    def test_empty_error_not_cuda(self) -> None:
        """Should return False for empty error message."""
        error = Exception("")
        assert is_cuda_crash(error) is False

    @pytest.mark.unit
    def test_server_error_not_cuda(self) -> None:
        """Should return False for generic server errors."""
        error = Exception("Server returned 500: Internal error")
        assert is_cuda_crash(error) is False

    @pytest.mark.unit
    def test_json_error_not_cuda(self) -> None:
        """Should return False for JSON parsing errors."""
        error = Exception("JSONDecodeError: Expecting value")
        assert is_cuda_crash(error) is False

    @pytest.mark.unit
    def test_mixed_case_detection(self) -> None:
        """Should handle mixed case in error messages."""
        error = Exception("CuDa ErRoR: device problem")
        assert is_cuda_crash(error) is True

    @pytest.mark.unit
    def test_sembr_server_error_with_cuda(self) -> None:
        """Should detect CUDA error wrapped in SembrServerError."""
        from pw_mcp.ingest.linebreaker import SembrServerError

        error = SembrServerError("Server error: cuda error: device-side assert")
        assert is_cuda_crash(error) is True

    @pytest.mark.unit
    def test_sembr_server_error_without_cuda(self) -> None:
        """Should return False for SembrServerError without CUDA message."""
        from pw_mcp.ingest.linebreaker import SembrServerError

        error = SembrServerError("Server error: Model not loaded")
        assert is_cuda_crash(error) is False
