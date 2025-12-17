#!/usr/bin/env python3
"""
Weights & Biases Logging for GRPO Training.

Provides comprehensive logging for debugging and monitoring GRPO fine-tuning:
- Per-step reward metrics (each reward function's mean)
- Sample tables showing question → response → reward breakdown
- Run configuration and hyperparameters
- Summary statistics at training end

Usage:
    from pw_mcp.ai_training.wandb_logging import (
        init_wandb_logging,
        WandbSampleLogger,
        create_logging_reward,
    )

    # Initialize
    run = init_wandb_logging(project="marxist-grpo", config={...})

    # Create logger and reward function
    sample_logger = WandbSampleLogger(log_every_n_steps=10)
    logging_reward = create_logging_reward(sample_logger)

    # Use in GRPOTrainer
    trainer = GRPOTrainer(
        reward_funcs=[..., logging_reward],
        ...
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# Global flag to track if wandb is available
_WANDB_AVAILABLE: bool | None = None
_wandb_module: Any = None


def _get_wandb() -> Any:
    """Lazily import and return wandb module."""
    global _WANDB_AVAILABLE, _wandb_module

    if _WANDB_AVAILABLE is None:
        try:
            import wandb

            _wandb_module = wandb
            _WANDB_AVAILABLE = True
        except ImportError:
            _WANDB_AVAILABLE = False
            _wandb_module = None

    return _wandb_module


def is_wandb_available() -> bool:
    """Check if wandb is installed and available."""
    _get_wandb()
    return _WANDB_AVAILABLE is True


# =============================================================================
# INITIALIZATION
# =============================================================================


def init_wandb_logging(
    project: str,
    config: dict[str, Any],
    name: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    mode: str = "online",
) -> Any:
    """
    Initialize Weights & Biases logging for GRPO training.

    Args:
        project: W&B project name (e.g., "marxist-grpo")
        config: Dictionary of hyperparameters and settings
        name: Optional run name (auto-generated if None)
        tags: Optional list of tags for filtering runs
        notes: Optional notes about this run
        mode: "online", "offline", or "disabled"

    Returns:
        wandb.Run object (or None if wandb unavailable)

    Example:
        run = init_wandb_logging(
            project="marxist-grpo",
            config={
                "model": "DeepSeek-R1-0528-Qwen3-8B",
                "learning_rate": 5e-6,
                "batch_size": 2,
                "max_steps": 250,
            },
            tags=["grpo", "marxist", "v1"],
        )
    """
    wandb = _get_wandb()
    if wandb is None:
        print("[WandbLogging] wandb not installed. Install with: pip install wandb")
        return None

    # Initialize run
    run = wandb.init(
        project=project,
        config=config,
        name=name,
        tags=tags or ["grpo", "marxist-leninist"],
        notes=notes,
        mode=mode,
    )

    # Define metrics with proper summaries
    _define_reward_metrics(run)

    print(f"[WandbLogging] Initialized run: {run.name}")
    print(f"[WandbLogging] View at: {run.url}")

    return run


def _define_reward_metrics(run: Any) -> None:
    """Define reward metrics with min/max/mean summaries."""
    reward_metrics = [
        "rewards/format_exact",
        "rewards/format_approx",
        "rewards/semantic_similarity",
        "rewards/terminology",
        "rewards/nli_coherence",
        "rewards/self_consistency",
        "rewards/structural_coherence",
        "rewards/topic_relevance",
        "rewards/interconnection_depth",
        "rewards/completeness",
        "rewards/total",
    ]

    for metric in reward_metrics:
        # Track min, max, and mean for each reward
        run.define_metric(metric, summary="mean")
        run.define_metric(f"{metric}_min", summary="min")
        run.define_metric(f"{metric}_max", summary="max")


# =============================================================================
# SAMPLE LOGGER
# =============================================================================


@dataclass
class RewardSample:
    """A single sample with its reward breakdown."""

    step: int
    question: str
    response: str
    ground_truth: str
    rewards: dict[str, float]

    @property
    def total_reward(self) -> float:
        """Sum of all rewards."""
        return sum(self.rewards.values())


@dataclass
class WandbSampleLogger:
    """
    Logs sample tables to W&B for debugging reward functions.

    Accumulates samples during training and logs them as a wandb.Table
    every N steps. This lets you inspect actual model outputs and
    understand why specific rewards were assigned.

    Example table:
        | step | question | response | ground_truth | format | nli | topic | depth | total |
        |------|----------|----------|--------------|--------|-----|-------|-------|-------|
        | 50   | What is..| The bour.| Revisionism..| 3.0    | 2.5 | 1.5   | 1.0   | 8.0   |
    """

    log_every_n_steps: int = 10
    max_samples_per_log: int = 4
    _samples: list[RewardSample] = field(default_factory=list)
    _step_counter: int = field(default=0)
    _table_columns: list[str] = field(
        default_factory=lambda: [
            "step",
            "question",
            "response",
            "ground_truth",
            "format_exact",
            "format_approx",
            "nli_coherence",
            "topic_relevance",
            "depth",
            "completeness",
            "total",
        ]
    )

    def add_sample(
        self,
        step: int,
        question: str,
        response: str,
        ground_truth: str,
        rewards: dict[str, float],
    ) -> None:
        """Add a sample to the buffer."""
        sample = RewardSample(
            step=step,
            question=question[:500],  # Truncate for table display
            response=response[:500],
            ground_truth=ground_truth[:300],
            rewards=rewards,
        )
        self._samples.append(sample)

        # Keep only recent samples
        max_buffer = self.max_samples_per_log * 3
        if len(self._samples) > max_buffer:
            self._samples = self._samples[-max_buffer:]

    def should_log(self, step: int) -> bool:
        """Check if we should log at this step."""
        return step > 0 and step % self.log_every_n_steps == 0

    def log_table(self, step: int) -> None:
        """Log accumulated samples as a wandb.Table."""
        wandb = _get_wandb()
        if wandb is None or not self._samples:
            return

        # Get recent samples
        samples_to_log = self._samples[-self.max_samples_per_log :]

        # Create table
        table = wandb.Table(columns=self._table_columns)

        for sample in samples_to_log:
            row = [
                sample.step,
                sample.question,
                sample.response,
                sample.ground_truth,
                sample.rewards.get("format_exact", 0.0),
                sample.rewards.get("format_approx", 0.0),
                sample.rewards.get("nli_coherence", 0.0),
                sample.rewards.get("topic_relevance", 0.0),
                sample.rewards.get("interconnection_depth", 0.0),
                sample.rewards.get("completeness", 0.0),
                sample.total_reward,
            ]
            table.add_data(*row)

        # Log the table
        wandb.log({"samples": table}, step=step)
        print(f"[WandbLogging] Logged {len(samples_to_log)} samples at step {step}")

    def clear(self) -> None:
        """Clear the sample buffer."""
        self._samples.clear()


# =============================================================================
# REWARD METRICS LOGGING
# =============================================================================


def log_reward_metrics(
    step: int,
    reward_scores: dict[str, list[float]],
) -> None:
    """
    Log reward metrics to wandb.

    Args:
        step: Training step number
        reward_scores: Dict mapping reward name to list of scores
                      e.g., {"format_exact": [3.0, 3.0, 0.0, 3.0]}
    """
    wandb = _get_wandb()
    if wandb is None:
        return

    metrics: dict[str, float] = {}

    for name, scores in reward_scores.items():
        if not scores:
            continue

        mean_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        metrics[f"rewards/{name}"] = mean_score
        metrics[f"rewards/{name}_min"] = min_score
        metrics[f"rewards/{name}_max"] = max_score

    # Compute total
    if reward_scores:
        all_totals = []
        num_samples = len(next(iter(reward_scores.values())))
        for i in range(num_samples):
            total = sum(scores[i] for scores in reward_scores.values() if i < len(scores))
            all_totals.append(total)

        if all_totals:
            metrics["rewards/total"] = sum(all_totals) / len(all_totals)
            metrics["rewards/total_min"] = min(all_totals)
            metrics["rewards/total_max"] = max(all_totals)

    wandb.log(metrics, step=step)


# =============================================================================
# LOGGING REWARD FUNCTION
# =============================================================================

# Global step counter for the logging reward
_LOGGING_STEP = 0


def create_logging_reward(
    sample_logger: WandbSampleLogger | None = None,
    compute_all_rewards: bool = True,
) -> Callable[..., list[float]]:
    """
    Create a reward function that logs metrics and samples to wandb.

    This replaces debug_print_reward with comprehensive wandb logging.
    The returned function computes ALL individual rewards internally,
    logs them to wandb, and returns [0.0] * len(completions) (no training effect).

    Args:
        sample_logger: WandbSampleLogger instance for sample table logging
        compute_all_rewards: If True, compute and log all reward functions

    Returns:
        A reward function compatible with GRPOTrainer

    Example:
        sample_logger = WandbSampleLogger(log_every_n_steps=10)
        logging_reward = create_logging_reward(sample_logger)

        trainer = GRPOTrainer(
            reward_funcs=[..., logging_reward],
            ...
        )
    """
    global _LOGGING_STEP

    def logging_reward(
        prompts: Sequence[Sequence[dict[str, str]]],
        completions: Sequence[Sequence[dict[str, str]]],
        answer: Sequence[str],
        **kwargs: object,
    ) -> list[float]:
        """Log rewards and samples to wandb. Returns 0.0 (no training effect)."""
        global _LOGGING_STEP
        _LOGGING_STEP += 1
        step = _LOGGING_STEP

        wandb = _get_wandb()
        if wandb is None or wandb.run is None:
            # Fallback to print if wandb not initialized
            if step % 10 == 0:
                print(f"[Step {step}] Q: {prompts[0][-1]['content'][:80]}...")
            return [0.0] * len(completions)

        # Compute all reward scores if requested
        if compute_all_rewards:
            reward_scores = _compute_all_reward_scores(prompts, completions, answer, **kwargs)
            log_reward_metrics(step, reward_scores)
        else:
            reward_scores = {}

        # Log samples periodically
        if sample_logger and sample_logger.should_log(step):
            # Add current batch to sample logger
            for i in range(min(sample_logger.max_samples_per_log, len(prompts))):
                question = prompts[i][-1]["content"]
                response = completions[i][0]["content"]
                truth = answer[i] if i < len(answer) else ""

                # Get individual rewards for this sample
                sample_rewards = {
                    name: scores[i] if i < len(scores) else 0.0
                    for name, scores in reward_scores.items()
                }

                sample_logger.add_sample(
                    step=step,
                    question=question,
                    response=response,
                    ground_truth=truth,
                    rewards=sample_rewards,
                )

            sample_logger.log_table(step)

        return [0.0] * len(completions)

    return logging_reward


def _compute_all_reward_scores(
    prompts: Sequence[Sequence[dict[str, str]]],
    completions: Sequence[Sequence[dict[str, str]]],
    answer: Sequence[str],
    **kwargs: object,
) -> dict[str, list[float]]:
    """
    Compute all reward function scores for logging.

    Returns dict mapping reward name to list of scores.
    """
    # Import reward functions here to avoid circular imports
    from pw_mcp.ai_training.grpo_rewards import (
        completeness_reward,
        interconnection_depth_reward,
        match_format_approximately,
        match_format_exactly,
        nli_coherence_reward,
        topic_relevance_reward,
    )

    reward_scores: dict[str, list[float]] = {}

    # Format rewards (don't need answer)
    try:
        reward_scores["format_exact"] = match_format_exactly(completions, **kwargs)
    except Exception as e:
        print(f"[WandbLogging] Error in format_exact: {e}")
        reward_scores["format_exact"] = [0.0] * len(completions)

    try:
        reward_scores["format_approx"] = match_format_approximately(completions, **kwargs)
    except Exception as e:
        print(f"[WandbLogging] Error in format_approx: {e}")
        reward_scores["format_approx"] = [0.0] * len(completions)

    # NLI coherence (needs answer)
    try:
        reward_scores["nli_coherence"] = nli_coherence_reward(completions, answer, **kwargs)
    except Exception as e:
        print(f"[WandbLogging] Error in nli_coherence: {e}")
        reward_scores["nli_coherence"] = [0.0] * len(completions)

    # Topic relevance (needs prompts)
    try:
        reward_scores["topic_relevance"] = topic_relevance_reward(prompts, completions, **kwargs)
    except Exception as e:
        print(f"[WandbLogging] Error in topic_relevance: {e}")
        reward_scores["topic_relevance"] = [0.0] * len(completions)

    # Interconnection depth
    try:
        reward_scores["interconnection_depth"] = interconnection_depth_reward(completions, **kwargs)
    except Exception as e:
        print(f"[WandbLogging] Error in interconnection_depth: {e}")
        reward_scores["interconnection_depth"] = [0.0] * len(completions)

    # Completeness (needs answer)
    try:
        reward_scores["completeness"] = completeness_reward(completions, answer, **kwargs)
    except Exception as e:
        print(f"[WandbLogging] Error in completeness: {e}")
        reward_scores["completeness"] = [0.0] * len(completions)

    return reward_scores


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def finish_wandb_logging(summary: dict[str, Any] | None = None) -> None:
    """
    Finish the wandb run with optional summary statistics.

    Args:
        summary: Optional dict of final summary metrics
    """
    wandb = _get_wandb()
    if wandb is None or wandb.run is None:
        return

    if summary:
        for key, value in summary.items():
            wandb.run.summary[key] = value

    wandb.finish()
    print("[WandbLogging] Run finished.")


def log_model_checkpoint(
    checkpoint_path: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Log a model checkpoint as a wandb artifact.

    Args:
        checkpoint_path: Path to the checkpoint directory
        metadata: Optional metadata about the checkpoint
    """
    wandb = _get_wandb()
    if wandb is None or wandb.run is None:
        return

    artifact = wandb.Artifact(
        name=f"checkpoint-{wandb.run.name}",
        type="model",
        metadata=metadata or {},
    )
    artifact.add_dir(checkpoint_path)
    wandb.log_artifact(artifact)
    print(f"[WandbLogging] Logged checkpoint: {checkpoint_path}")
