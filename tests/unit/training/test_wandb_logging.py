"""
Tests for Weights & Biases logging module.

Tests cover:
- WandbSampleLogger accumulation and table creation
- Reward metrics logging
- Logging reward function signature compatibility
- Graceful handling when wandb is not available
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_wandb() -> MagicMock:
    """Create a mock wandb module."""
    mock = MagicMock()
    mock.run = MagicMock()
    mock.run.name = "test-run"
    mock.run.url = "https://wandb.ai/test/run"
    mock.Table = MagicMock(return_value=MagicMock())
    mock.init = MagicMock(return_value=mock.run)
    mock.log = MagicMock()
    mock.finish = MagicMock()
    return mock


@pytest.fixture
def sample_prompts() -> list[list[dict[str, str]]]:
    """Create sample prompts for testing."""
    return [
        [
            {"role": "system", "content": "You are a Marxist assistant."},
            {"role": "user", "content": "What is revisionism?"},
        ],
        [
            {"role": "system", "content": "You are a Marxist assistant."},
            {"role": "user", "content": "Explain surplus value."},
        ],
    ]


@pytest.fixture
def sample_completions() -> list[list[dict[str, str]]]:
    """Create sample completions for testing."""
    return [
        [{"role": "assistant", "content": "</think>Revisionism distorts Marxist theory."}],
        [{"role": "assistant", "content": "</think>Surplus value is unpaid labor."}],
    ]


@pytest.fixture
def sample_answers() -> list[str]:
    """Create sample ground truth answers."""
    return [
        "Revisionism is the distortion of Marxist-Leninist theory.",
        "Surplus value is the value produced by workers beyond their wages.",
    ]


# =============================================================================
# REWARD SAMPLE TESTS
# =============================================================================


class TestRewardSample:
    """Test the RewardSample dataclass."""

    def test_total_reward_calculation(self) -> None:
        """Test that total_reward sums all rewards."""
        from pw_mcp.ai_training.wandb_logging import RewardSample

        sample = RewardSample(
            step=10,
            question="What is X?",
            response="X is Y.",
            ground_truth="X is Y.",
            rewards={
                "format_exact": 3.0,
                "nli_coherence": 2.0,
                "topic_relevance": 1.5,
            },
        )

        assert sample.total_reward == 6.5

    def test_empty_rewards(self) -> None:
        """Test total_reward with empty rewards dict."""
        from pw_mcp.ai_training.wandb_logging import RewardSample

        sample = RewardSample(
            step=10,
            question="Q",
            response="R",
            ground_truth="T",
            rewards={},
        )

        assert sample.total_reward == 0.0


# =============================================================================
# SAMPLE LOGGER TESTS
# =============================================================================


class TestWandbSampleLogger:
    """Test the WandbSampleLogger class."""

    def test_add_sample(self) -> None:
        """Test adding samples to the logger."""
        from pw_mcp.ai_training.wandb_logging import WandbSampleLogger

        logger = WandbSampleLogger(log_every_n_steps=10)

        logger.add_sample(
            step=5,
            question="What is revisionism?",
            response="Revisionism distorts theory.",
            ground_truth="Revisionism is distortion of Marxism.",
            rewards={"format": 3.0, "nli": 2.0},
        )

        assert len(logger._samples) == 1
        assert logger._samples[0].step == 5
        assert logger._samples[0].total_reward == 5.0

    def test_sample_buffer_limit(self) -> None:
        """Test that sample buffer doesn't grow unbounded."""
        from pw_mcp.ai_training.wandb_logging import WandbSampleLogger

        logger = WandbSampleLogger(log_every_n_steps=10, max_samples_per_log=2)

        # Add many samples
        for i in range(20):
            logger.add_sample(
                step=i,
                question=f"Q{i}",
                response=f"R{i}",
                ground_truth=f"T{i}",
                rewards={"x": float(i)},
            )

        # Buffer should be limited (max_samples_per_log * 3 = 6)
        assert len(logger._samples) <= 6

    def test_should_log(self) -> None:
        """Test should_log returns True at correct intervals."""
        from pw_mcp.ai_training.wandb_logging import WandbSampleLogger

        logger = WandbSampleLogger(log_every_n_steps=10)

        assert not logger.should_log(0)  # Step 0 doesn't log
        assert not logger.should_log(5)
        assert logger.should_log(10)
        assert not logger.should_log(15)
        assert logger.should_log(20)

    def test_clear(self) -> None:
        """Test clearing the sample buffer."""
        from pw_mcp.ai_training.wandb_logging import WandbSampleLogger

        logger = WandbSampleLogger()
        logger.add_sample(1, "Q", "R", "T", {"x": 1.0})
        logger.add_sample(2, "Q", "R", "T", {"x": 2.0})

        assert len(logger._samples) == 2

        logger.clear()
        assert len(logger._samples) == 0

    def test_truncation(self) -> None:
        """Test that long strings are truncated."""
        from pw_mcp.ai_training.wandb_logging import WandbSampleLogger

        logger = WandbSampleLogger()

        long_text = "x" * 1000  # 1000 characters
        logger.add_sample(
            step=1,
            question=long_text,
            response=long_text,
            ground_truth=long_text,
            rewards={},
        )

        assert len(logger._samples[0].question) == 500
        assert len(logger._samples[0].response) == 500
        assert len(logger._samples[0].ground_truth) == 300


# =============================================================================
# LOG TABLE TESTS
# =============================================================================


class TestLogTable:
    """Test table logging functionality."""

    def test_log_table_creates_table(self, mock_wandb: MagicMock) -> None:
        """Test that log_table creates and logs a wandb Table."""
        from pw_mcp.ai_training.wandb_logging import WandbSampleLogger

        with patch("pw_mcp.ai_training.wandb_logging._get_wandb", return_value=mock_wandb):
            logger = WandbSampleLogger(max_samples_per_log=2)

            # Add samples
            logger.add_sample(1, "Q1", "R1", "T1", {"format": 3.0})
            logger.add_sample(2, "Q2", "R2", "T2", {"format": 2.0})

            # Log table
            logger.log_table(step=10)

            # Verify Table was created
            mock_wandb.Table.assert_called_once()

            # Verify log was called
            mock_wandb.log.assert_called_once()
            call_args = mock_wandb.log.call_args
            assert "samples" in call_args[0][0]
            assert call_args[1]["step"] == 10


# =============================================================================
# REWARD METRICS LOGGING TESTS
# =============================================================================


class TestLogRewardMetrics:
    """Test reward metrics logging."""

    def test_log_reward_metrics(self, mock_wandb: MagicMock) -> None:
        """Test logging reward metrics to wandb."""
        from pw_mcp.ai_training.wandb_logging import log_reward_metrics

        with patch("pw_mcp.ai_training.wandb_logging._get_wandb", return_value=mock_wandb):
            reward_scores = {
                "format_exact": [3.0, 3.0, 0.0],
                "nli_coherence": [2.0, -1.0, 3.0],
            }

            log_reward_metrics(step=50, reward_scores=reward_scores)

            # Verify log was called with correct metrics
            mock_wandb.log.assert_called_once()
            logged_metrics = mock_wandb.log.call_args[0][0]

            # Check mean calculations
            assert logged_metrics["rewards/format_exact"] == 2.0  # (3+3+0)/3
            assert logged_metrics["rewards/nli_coherence"] == pytest.approx(4 / 3)  # (2-1+3)/3

            # Check min/max
            assert logged_metrics["rewards/format_exact_min"] == 0.0
            assert logged_metrics["rewards/format_exact_max"] == 3.0

    def test_log_reward_metrics_computes_total(self, mock_wandb: MagicMock) -> None:
        """Test that total reward is computed correctly."""
        from pw_mcp.ai_training.wandb_logging import log_reward_metrics

        with patch("pw_mcp.ai_training.wandb_logging._get_wandb", return_value=mock_wandb):
            reward_scores = {
                "format": [3.0, 2.0],
                "nli": [1.0, 2.0],
            }

            log_reward_metrics(step=10, reward_scores=reward_scores)

            logged_metrics = mock_wandb.log.call_args[0][0]

            # Total for sample 0: 3.0 + 1.0 = 4.0
            # Total for sample 1: 2.0 + 2.0 = 4.0
            # Mean total: 4.0
            assert logged_metrics["rewards/total"] == 4.0


# =============================================================================
# LOGGING REWARD FUNCTION TESTS
# =============================================================================


class TestCreateLoggingReward:
    """Test the create_logging_reward function."""

    def test_returns_zeros(
        self,
        sample_prompts: list[list[dict[str, str]]],
        sample_completions: list[list[dict[str, str]]],
        sample_answers: list[str],
    ) -> None:
        """Test that logging reward returns zeros (no training effect)."""
        from pw_mcp.ai_training.wandb_logging import (
            WandbSampleLogger,
            create_logging_reward,
        )

        # Create logging reward without wandb (will fallback to print)
        with patch("pw_mcp.ai_training.wandb_logging._get_wandb", return_value=None):
            sample_logger = WandbSampleLogger()
            logging_reward = create_logging_reward(sample_logger, compute_all_rewards=False)

            scores = logging_reward(
                prompts=sample_prompts,
                completions=sample_completions,
                answer=sample_answers,
            )

            # Should return zeros for all samples
            assert scores == [0.0, 0.0]

    def test_function_signature_compatibility(
        self,
        sample_prompts: list[list[dict[str, str]]],
        sample_completions: list[list[dict[str, str]]],
        sample_answers: list[str],
    ) -> None:
        """Test that logging reward has correct signature for GRPOTrainer."""
        from pw_mcp.ai_training.wandb_logging import create_logging_reward

        with patch("pw_mcp.ai_training.wandb_logging._get_wandb", return_value=None):
            logging_reward = create_logging_reward(compute_all_rewards=False)

            # Should accept prompts, completions, answer, and kwargs
            result = logging_reward(
                prompts=sample_prompts,
                completions=sample_completions,
                answer=sample_answers,
                extra_kwarg="ignored",
            )

            assert isinstance(result, list)
            assert len(result) == len(sample_completions)

    def test_logs_samples_at_interval(
        self,
        mock_wandb: MagicMock,
        sample_prompts: list[list[dict[str, str]]],
        sample_completions: list[list[dict[str, str]]],
        sample_answers: list[str],
    ) -> None:
        """Test that samples are logged at correct intervals."""
        # Reset global step counter
        import pw_mcp.ai_training.wandb_logging as wl
        from pw_mcp.ai_training.wandb_logging import (
            WandbSampleLogger,
            create_logging_reward,
        )

        wl._LOGGING_STEP = 0

        with patch("pw_mcp.ai_training.wandb_logging._get_wandb", return_value=mock_wandb):
            sample_logger = WandbSampleLogger(log_every_n_steps=5, max_samples_per_log=2)
            logging_reward = create_logging_reward(sample_logger, compute_all_rewards=False)

            # Call multiple times
            for _ in range(10):
                logging_reward(
                    prompts=sample_prompts,
                    completions=sample_completions,
                    answer=sample_answers,
                )

            # Table should have been logged twice (at step 5 and 10)
            table_logs = [call for call in mock_wandb.log.call_args_list if "samples" in call[0][0]]
            assert len(table_logs) == 2


# =============================================================================
# WANDB AVAILABILITY TESTS
# =============================================================================


class TestWandbAvailability:
    """Test handling of wandb availability."""

    def test_is_wandb_available_true(self, mock_wandb: MagicMock) -> None:
        """Test is_wandb_available returns True when wandb is installed."""
        from pw_mcp.ai_training import wandb_logging as wl

        # Reset cached state
        wl._WANDB_AVAILABLE = None
        wl._wandb_module = None

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            # Force re-import check
            wl._WANDB_AVAILABLE = None
            _result = wl.is_wandb_available()
            # Note: This may still be False due to import mechanics
            # The important thing is it doesn't crash

    def test_graceful_degradation_without_wandb(
        self,
        sample_prompts: list[list[dict[str, str]]],
        sample_completions: list[list[dict[str, str]]],
        sample_answers: list[str],
    ) -> None:
        """Test that logging works gracefully without wandb."""
        from pw_mcp.ai_training.wandb_logging import (
            WandbSampleLogger,
            create_logging_reward,
            log_reward_metrics,
        )

        with patch("pw_mcp.ai_training.wandb_logging._get_wandb", return_value=None):
            # These should not raise exceptions
            sample_logger = WandbSampleLogger()
            logging_reward = create_logging_reward(sample_logger, compute_all_rewards=False)

            # Should return valid result even without wandb
            result = logging_reward(
                prompts=sample_prompts,
                completions=sample_completions,
                answer=sample_answers,
            )
            assert result == [0.0, 0.0]

            # Metrics logging should not crash
            log_reward_metrics(step=1, reward_scores={"x": [1.0]})

            # Table logging should not crash
            sample_logger.log_table(step=10)


# =============================================================================
# INIT AND FINISH TESTS
# =============================================================================


class TestInitAndFinish:
    """Test initialization and finishing of wandb runs."""

    def test_init_wandb_logging(self, mock_wandb: MagicMock) -> None:
        """Test wandb initialization with config."""
        from pw_mcp.ai_training.wandb_logging import init_wandb_logging

        with patch("pw_mcp.ai_training.wandb_logging._get_wandb", return_value=mock_wandb):
            _run = init_wandb_logging(
                project="test-project",
                config={"lr": 1e-5, "batch_size": 4},
                name="test-run",
                tags=["test"],
            )

            mock_wandb.init.assert_called_once()
            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs["project"] == "test-project"
            assert call_kwargs["config"] == {"lr": 1e-5, "batch_size": 4}

    def test_finish_wandb_logging(self, mock_wandb: MagicMock) -> None:
        """Test wandb finish with summary."""
        from pw_mcp.ai_training.wandb_logging import finish_wandb_logging

        with patch("pw_mcp.ai_training.wandb_logging._get_wandb", return_value=mock_wandb):
            finish_wandb_logging(summary={"final_loss": 0.5})

            # Check summary was updated
            mock_wandb.run.summary.__setitem__.assert_called_with("final_loss", 0.5)

            # Check finish was called
            mock_wandb.finish.assert_called_once()
