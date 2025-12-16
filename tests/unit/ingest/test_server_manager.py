"""Unit tests for server manager module.

These tests mock subprocess and httpx calls to test server lifecycle
management without requiring actual server processes.
"""

from __future__ import annotations

import signal
import time
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from pw_mcp.ingest.server_manager import (
    OLLAMA_CONFIG,
    SEMBR_CONFIG,
    HealthMonitor,
    ServerConfig,
    ServerProcess,
    _is_process_running,
    _kill_server_by_pattern,
    _servers,
    check_server_health,
    cleanup_all_servers,
    ensure_server_running,
    get_running_servers,
    get_server_status,
    get_server_summary,
    is_server_running,
    is_shutdown_requested,
    register_cleanup,
    reset_shutdown_flag,
    restart_server,
    setup_signal_handlers,
    start_server,
    stop_server,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clear_server_registry() -> None:
    """Clear server registry before and after each test."""
    _servers.clear()
    reset_shutdown_flag()
    yield
    _servers.clear()


@pytest.fixture
def mock_healthy_response() -> MagicMock:
    """Mock a healthy HTTP response."""
    response = MagicMock()
    response.status_code = 200
    return response


@pytest.fixture
def mock_process() -> MagicMock:
    """Mock a subprocess.Popen object."""
    process = MagicMock()
    process.pid = 12345
    process.returncode = None
    process.poll.return_value = None
    return process


# =============================================================================
# SERVER CONFIG TESTS
# =============================================================================


class TestServerConfig:
    """Tests for ServerConfig dataclass."""

    def test_sembr_config_defaults(self) -> None:
        """Test SEMBR_CONFIG has correct values."""
        assert SEMBR_CONFIG.server_type == "sembr"
        assert SEMBR_CONFIG.port == 8384
        assert SEMBR_CONFIG.health_endpoint == "/check"
        assert "--listen" in SEMBR_CONFIG.start_command
        assert "-p" in SEMBR_CONFIG.start_command

    def test_ollama_config_defaults(self) -> None:
        """Test OLLAMA_CONFIG has correct values."""
        assert OLLAMA_CONFIG.server_type == "ollama"
        assert OLLAMA_CONFIG.port == 11434
        assert OLLAMA_CONFIG.health_endpoint == "/api/tags"
        assert "ollama" in OLLAMA_CONFIG.start_command

    def test_custom_config(self) -> None:
        """Test creating custom server config."""
        config = ServerConfig(
            server_type="test",
            port=9999,
            health_endpoint="/health",
            start_command=["test-server", "--port", "9999"],
            startup_timeout=30.0,
        )
        assert config.server_type == "test"
        assert config.port == 9999
        assert config.startup_timeout == 30.0

    def test_config_is_frozen(self) -> None:
        """Test that ServerConfig is immutable."""
        with pytest.raises(AttributeError):
            SEMBR_CONFIG.port = 9999  # type: ignore[misc]


class TestServerProcess:
    """Tests for ServerProcess dataclass."""

    def test_server_process_creation(self) -> None:
        """Test creating ServerProcess instance."""
        process = ServerProcess(
            config=SEMBR_CONFIG,
            pid=12345,
            start_time=datetime.now(),
            gpu_id=0,
        )
        assert process.pid == 12345
        assert process.gpu_id == 0
        assert process.process is None


# =============================================================================
# HEALTH CHECK TESTS
# =============================================================================


class TestCheckServerHealth:
    """Tests for check_server_health function."""

    def test_healthy_server(self, mock_healthy_response: MagicMock) -> None:
        """Test detecting healthy server."""
        with patch("httpx.get", return_value=mock_healthy_response):
            result = check_server_health("sembr")
        assert result is True

    def test_unhealthy_server_bad_status(self) -> None:
        """Test detecting unhealthy server by status code."""
        response = MagicMock()
        response.status_code = 500

        with patch("httpx.get", return_value=response):
            result = check_server_health("sembr")
        assert result is False

    def test_connection_refused(self) -> None:
        """Test handling connection refused."""
        import httpx

        with patch("httpx.get", side_effect=httpx.ConnectError("")):
            result = check_server_health("sembr")
        assert result is False

    def test_timeout(self) -> None:
        """Test handling request timeout."""
        import httpx

        with patch("httpx.get", side_effect=httpx.TimeoutException("")):
            result = check_server_health("sembr")
        assert result is False

    def test_unknown_server_type(self) -> None:
        """Test handling unknown server type."""
        result = check_server_health("unknown")
        assert result is False


# =============================================================================
# START SERVER TESTS
# =============================================================================


class TestStartServer:
    """Tests for start_server function."""

    def test_start_server_success(
        self,
        mock_process: MagicMock,
        mock_healthy_response: MagicMock,
    ) -> None:
        """Test successful server start."""
        with (
            patch("subprocess.Popen", return_value=mock_process),
            patch("httpx.get", return_value=mock_healthy_response),
            patch(
                "pw_mcp.ingest.server_manager.get_available_gpu",
                return_value=0,
            ),
            patch(
                "pw_mcp.ingest.server_manager.get_cuda_env",
                return_value={"CUDA_VISIBLE_DEVICES": "0"},
            ),
        ):
            server = start_server(SEMBR_CONFIG, gpu_id=0)

        assert server.pid == 12345
        assert server.gpu_id == 0
        assert "sembr" in _servers

    def test_start_server_auto_select_gpu(
        self,
        mock_process: MagicMock,
        mock_healthy_response: MagicMock,
    ) -> None:
        """Test server start with auto GPU selection."""
        with (
            patch("subprocess.Popen", return_value=mock_process),
            patch("httpx.get", return_value=mock_healthy_response),
            patch(
                "pw_mcp.ingest.server_manager.get_available_gpu",
                return_value=1,
            ),
            patch("pw_mcp.ingest.server_manager.get_cuda_env", return_value={}),
        ):
            server = start_server(SEMBR_CONFIG)

        assert server.gpu_id == 1

    def test_start_server_no_gpu_available(self) -> None:
        """Test error when no GPU available."""
        with patch(
            "pw_mcp.ingest.server_manager.get_available_gpu",
            return_value=None,
        ):
            with pytest.raises(RuntimeError, match="No available GPU"):
                start_server(SEMBR_CONFIG)

    def test_start_server_already_running(
        self,
        mock_healthy_response: MagicMock,
    ) -> None:
        """Test handling when server already running."""
        # Add existing server to registry
        existing = ServerProcess(
            config=SEMBR_CONFIG,
            pid=99999,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["sembr"] = existing

        with (
            patch(
                "pw_mcp.ingest.server_manager._is_process_running",
                return_value=True,
            ),
            patch("httpx.get", return_value=mock_healthy_response),
        ):
            server = start_server(SEMBR_CONFIG, gpu_id=0)

        assert server.pid == 99999  # Returns existing

    def test_start_server_command_not_found(self) -> None:
        """Test handling when command not found."""
        with (
            patch("subprocess.Popen", side_effect=FileNotFoundError),
            patch(
                "pw_mcp.ingest.server_manager.get_available_gpu",
                return_value=0,
            ),
            patch("pw_mcp.ingest.server_manager.get_cuda_env", return_value={}),
        ):
            with pytest.raises(RuntimeError, match="Command not found"):
                start_server(SEMBR_CONFIG, gpu_id=0)

    def test_start_server_health_timeout(
        self,
        mock_process: MagicMock,
    ) -> None:
        """Test handling when server doesn't become healthy."""
        import httpx

        # Configure short timeout for test
        short_config = ServerConfig(
            server_type="test",
            port=9999,
            health_endpoint="/check",
            start_command=["test"],
            startup_timeout=0.1,  # Very short timeout
        )

        with (
            patch("subprocess.Popen", return_value=mock_process),
            patch("httpx.get", side_effect=httpx.ConnectError("")),
            patch(
                "pw_mcp.ingest.server_manager.get_available_gpu",
                return_value=0,
            ),
            patch("pw_mcp.ingest.server_manager.get_cuda_env", return_value={}),
        ):
            with pytest.raises(RuntimeError, match="failed to become healthy"):
                start_server(short_config, gpu_id=0)

        mock_process.kill.assert_called_once()


# =============================================================================
# STOP SERVER TESTS
# =============================================================================


class TestStopServer:
    """Tests for stop_server function."""

    def test_stop_server_graceful(self) -> None:
        """Test graceful server stop."""
        # Add server to registry
        server = ServerProcess(
            config=SEMBR_CONFIG,
            pid=12345,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["sembr"] = server

        with (
            patch("os.kill") as mock_kill,
            patch(
                "pw_mcp.ingest.server_manager._is_process_running",
                side_effect=[True, False],  # Running, then stopped
            ),
            patch("pw_mcp.ingest.server_manager.reset_cuda_cache") as mock_reset,
        ):
            result = stop_server("sembr")

        assert result is True
        assert "sembr" not in _servers
        mock_kill.assert_called_once_with(12345, signal.SIGTERM)
        mock_reset.assert_called_once_with(0)

    def test_stop_server_forceful(self) -> None:
        """Test forceful server stop."""
        server = ServerProcess(
            config=SEMBR_CONFIG,
            pid=12345,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["sembr"] = server

        with (
            patch("os.kill") as mock_kill,
            patch("pw_mcp.ingest.server_manager.reset_cuda_cache"),
        ):
            result = stop_server("sembr", graceful=False)

        assert result is True
        mock_kill.assert_called_once_with(12345, signal.SIGKILL)

    def test_stop_server_not_in_registry(self) -> None:
        """Test stopping server not in registry."""
        with patch(
            "pw_mcp.ingest.server_manager._kill_server_by_pattern",
            return_value=False,
        ):
            result = stop_server("sembr")
        assert result is False

    def test_stop_server_requires_sigkill(self) -> None:
        """Test that SIGKILL is sent if graceful shutdown fails."""
        server = ServerProcess(
            config=ServerConfig(
                server_type="test",
                port=9999,
                health_endpoint="/check",
                start_command=["test"],
                graceful_shutdown_timeout=0.1,  # Very short
            ),
            pid=12345,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["test"] = server

        with (
            patch("os.kill") as mock_kill,
            patch(
                "pw_mcp.ingest.server_manager._is_process_running",
                return_value=True,  # Always running
            ),
            patch("pw_mcp.ingest.server_manager.reset_cuda_cache"),
        ):
            stop_server("test", graceful=True, timeout=0.1)

        # Should have called both SIGTERM and SIGKILL
        calls = mock_kill.call_args_list
        assert call(12345, signal.SIGTERM) in calls
        assert call(12345, signal.SIGKILL) in calls


# =============================================================================
# RESTART SERVER TESTS
# =============================================================================


class TestRestartServer:
    """Tests for restart_server function."""

    def test_restart_server(
        self,
        mock_process: MagicMock,
        mock_healthy_response: MagicMock,
    ) -> None:
        """Test restarting a server."""
        # Add existing server
        server = ServerProcess(
            config=SEMBR_CONFIG,
            pid=11111,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["sembr"] = server

        with (
            patch("os.kill"),
            patch(
                "pw_mcp.ingest.server_manager._is_process_running",
                side_effect=[True, False],
            ),
            patch("pw_mcp.ingest.server_manager.reset_cuda_cache"),
            patch("subprocess.Popen", return_value=mock_process),
            patch("httpx.get", return_value=mock_healthy_response),
            patch(
                "pw_mcp.ingest.server_manager.get_cuda_env",
                return_value={},
            ),
        ):
            new_server = restart_server("sembr")

        assert new_server.pid == mock_process.pid

    def test_restart_server_new_gpu(
        self,
        mock_process: MagicMock,
        mock_healthy_response: MagicMock,
    ) -> None:
        """Test restarting server on different GPU."""
        server = ServerProcess(
            config=SEMBR_CONFIG,
            pid=11111,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["sembr"] = server

        with (
            patch("os.kill"),
            patch(
                "pw_mcp.ingest.server_manager._is_process_running",
                side_effect=[True, False],
            ),
            patch("pw_mcp.ingest.server_manager.reset_cuda_cache"),
            patch("subprocess.Popen", return_value=mock_process),
            patch("httpx.get", return_value=mock_healthy_response),
            patch(
                "pw_mcp.ingest.server_manager.get_cuda_env",
                return_value={"CUDA_VISIBLE_DEVICES": "1"},
            ),
        ):
            new_server = restart_server("sembr", new_gpu=1)

        assert new_server.gpu_id == 1

    def test_restart_unknown_server(self) -> None:
        """Test restarting unknown server type."""
        with pytest.raises(RuntimeError, match="Unknown server type"):
            restart_server("unknown")


# =============================================================================
# SERVER STATUS TESTS
# =============================================================================


class TestGetServerStatus:
    """Tests for get_server_status function."""

    def test_get_status_running(self) -> None:
        """Test getting status of running server."""
        server = ServerProcess(
            config=SEMBR_CONFIG,
            pid=12345,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["sembr"] = server

        with patch(
            "pw_mcp.ingest.server_manager._is_process_running",
            return_value=True,
        ):
            status = get_server_status("sembr")

        assert status is not None
        assert status.pid == 12345

    def test_get_status_not_tracked(self) -> None:
        """Test getting status of untracked server."""
        status = get_server_status("sembr")
        assert status is None

    def test_get_status_process_died(self) -> None:
        """Test that dead process is cleaned up."""
        server = ServerProcess(
            config=SEMBR_CONFIG,
            pid=12345,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["sembr"] = server

        with patch(
            "pw_mcp.ingest.server_manager._is_process_running",
            return_value=False,
        ):
            status = get_server_status("sembr")

        assert status is None
        assert "sembr" not in _servers


class TestIsServerRunning:
    """Tests for is_server_running function."""

    def test_server_running_and_healthy(
        self,
        mock_healthy_response: MagicMock,
    ) -> None:
        """Test detecting running and healthy server."""
        server = ServerProcess(
            config=SEMBR_CONFIG,
            pid=12345,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["sembr"] = server

        with (
            patch(
                "pw_mcp.ingest.server_manager._is_process_running",
                return_value=True,
            ),
            patch("httpx.get", return_value=mock_healthy_response),
        ):
            result = is_server_running("sembr")

        assert result is True

    def test_server_not_tracked(self) -> None:
        """Test untracked server."""
        result = is_server_running("sembr")
        assert result is False


# =============================================================================
# UTILITY TESTS
# =============================================================================


class TestIsProcessRunning:
    """Tests for _is_process_running function."""

    def test_process_running(self) -> None:
        """Test detecting running process."""
        with patch("os.kill"):  # No exception = running
            result = _is_process_running(12345)
        assert result is True

    def test_process_not_running(self) -> None:
        """Test detecting dead process."""
        with patch("os.kill", side_effect=ProcessLookupError):
            result = _is_process_running(12345)
        assert result is False

    def test_process_permission_denied(self) -> None:
        """Test handling permission denied (process exists)."""
        with patch("os.kill", side_effect=PermissionError):
            result = _is_process_running(12345)
        assert result is True


class TestKillServerByPattern:
    """Tests for _kill_server_by_pattern function."""

    def test_kill_sembr_pattern(self) -> None:
        """Test killing sembr by pattern."""
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = _kill_server_by_pattern("sembr")

        assert result is True
        # Verify correct pattern used
        args = mock_run.call_args[0][0]
        assert "pkill" in args
        assert "-f" in args
        assert "sembr --listen" in args

    def test_kill_unknown_pattern(self) -> None:
        """Test unknown server type."""
        result = _kill_server_by_pattern("unknown")
        assert result is False


# =============================================================================
# SIGNAL HANDLING TESTS
# =============================================================================


class TestSignalHandling:
    """Tests for signal handling functions."""

    def test_setup_signal_handlers(self) -> None:
        """Test setting up signal handlers."""
        with patch("signal.signal") as mock_signal:
            setup_signal_handlers()

        # Should register SIGINT and SIGTERM handlers
        calls = mock_signal.call_args_list
        signals_registered = [c[0][0] for c in calls]
        assert signal.SIGINT in signals_registered
        assert signal.SIGTERM in signals_registered

    def test_is_shutdown_requested_default(self) -> None:
        """Test default shutdown flag is False."""
        reset_shutdown_flag()
        assert is_shutdown_requested() is False

    def test_reset_shutdown_flag(self) -> None:
        """Test resetting shutdown flag."""
        # Simulate shutdown requested
        from pw_mcp.ingest import server_manager

        server_manager._shutdown_requested = True
        assert is_shutdown_requested() is True

        reset_shutdown_flag()
        assert is_shutdown_requested() is False


# =============================================================================
# CLEANUP TESTS
# =============================================================================


class TestCleanup:
    """Tests for cleanup functions."""

    def test_cleanup_all_servers(self) -> None:
        """Test cleanup_all_servers stops all servers."""
        # Add servers to registry
        _servers["sembr"] = ServerProcess(
            config=SEMBR_CONFIG,
            pid=11111,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["ollama"] = ServerProcess(
            config=OLLAMA_CONFIG,
            pid=22222,
            start_time=datetime.now(),
            gpu_id=1,
        )

        with (
            patch("os.kill"),
            patch(
                "pw_mcp.ingest.server_manager._is_process_running",
                return_value=False,
            ),
            patch("pw_mcp.ingest.server_manager.reset_cuda_cache"),
        ):
            cleanup_all_servers()

        assert len(_servers) == 0

    def test_register_cleanup(self) -> None:
        """Test registering atexit handler."""
        with patch("atexit.register") as mock_register:
            register_cleanup()

        mock_register.assert_called_once_with(cleanup_all_servers)


# =============================================================================
# HEALTH MONITOR TESTS
# =============================================================================


class TestHealthMonitor:
    """Tests for HealthMonitor class."""

    def test_monitor_initialization(self) -> None:
        """Test monitor initialization."""
        monitor = HealthMonitor("sembr", interval=30.0)
        assert monitor.server_type == "sembr"
        assert monitor.interval == 30.0
        assert not monitor._is_running

    def test_monitor_start_stop(self, mock_healthy_response: MagicMock) -> None:
        """Test starting and stopping monitor."""
        with patch("httpx.get", return_value=mock_healthy_response):
            monitor = HealthMonitor("sembr")
            monitor.start()
            assert monitor._is_running
            assert monitor.is_healthy

            monitor.stop()
            assert not monitor._is_running

    def test_should_check_respects_interval(self) -> None:
        """Test that should_check respects interval."""
        monitor = HealthMonitor("sembr", interval=60.0)
        monitor.start()
        monitor._last_check_time = time.time()

        # Just started, should not need check yet
        assert not monitor.should_check()

        # Simulate time passing
        monitor._last_check_time = time.time() - 70  # 70 seconds ago
        assert monitor.should_check()

    def test_check_and_recover_healthy(
        self,
        mock_healthy_response: MagicMock,
    ) -> None:
        """Test check_and_recover with healthy server."""
        monitor = HealthMonitor("sembr")
        monitor.start()

        with patch("httpx.get", return_value=mock_healthy_response):
            result = monitor.check_and_recover()

        assert result is True
        assert monitor.is_healthy

    def test_check_and_recover_unhealthy(self) -> None:
        """Test check_and_recover attempts recovery."""
        import httpx

        monitor = HealthMonitor("sembr", max_recovery_attempts=1)
        monitor.start()

        # Custom callback that always fails
        callback_calls = []

        def failing_callback() -> bool:
            callback_calls.append(1)
            return False

        monitor.recovery_callback = failing_callback

        with patch("httpx.get", side_effect=httpx.ConnectError("")):
            result = monitor.check_and_recover()

        assert result is False
        assert not monitor.is_healthy
        assert len(callback_calls) == 1


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_ensure_server_running_already_running(
        self,
        mock_healthy_response: MagicMock,
    ) -> None:
        """Test ensure_server_running returns existing server."""
        server = ServerProcess(
            config=SEMBR_CONFIG,
            pid=12345,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["sembr"] = server

        with (
            patch(
                "pw_mcp.ingest.server_manager._is_process_running",
                return_value=True,
            ),
            patch("httpx.get", return_value=mock_healthy_response),
        ):
            result = ensure_server_running("sembr")

        assert result.pid == 12345

    def test_ensure_server_running_starts_new(
        self,
        mock_process: MagicMock,
        mock_healthy_response: MagicMock,
    ) -> None:
        """Test ensure_server_running starts server if not running."""
        with (
            patch("subprocess.Popen", return_value=mock_process),
            patch("httpx.get", return_value=mock_healthy_response),
            patch(
                "pw_mcp.ingest.server_manager.get_available_gpu",
                return_value=0,
            ),
            patch("pw_mcp.ingest.server_manager.get_cuda_env", return_value={}),
        ):
            result = ensure_server_running("sembr")

        assert result.pid == mock_process.pid

    def test_get_running_servers(self, mock_healthy_response: MagicMock) -> None:
        """Test get_running_servers returns list of running servers."""
        server = ServerProcess(
            config=SEMBR_CONFIG,
            pid=12345,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["sembr"] = server

        with (
            patch(
                "pw_mcp.ingest.server_manager._is_process_running",
                return_value=True,
            ),
            patch("httpx.get", return_value=mock_healthy_response),
        ):
            running = get_running_servers()

        assert "sembr" in running

    def test_get_server_summary(self) -> None:
        """Test get_server_summary returns formatted string."""
        import httpx

        server = ServerProcess(
            config=SEMBR_CONFIG,
            pid=12345,
            start_time=datetime.now(),
            gpu_id=0,
        )
        _servers["sembr"] = server

        with (
            patch(
                "pw_mcp.ingest.server_manager._is_process_running",
                return_value=True,
            ),
            patch("httpx.get", side_effect=httpx.ConnectError("")),
        ):
            summary = get_server_summary()

        assert "Server Summary" in summary
        assert "sembr" in summary
        assert "12345" in summary
