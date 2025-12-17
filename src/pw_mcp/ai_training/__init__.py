"""
AI Training Module for Marxist-Leninist LLM Fine-tuning.

This module contains reward functions and training utilities for GRPO
(Group Relative Policy Optimization) fine-tuning on ProleWiki corpus.

Components:
- grpo_rewards: Reward functions for GRPO training
- wandb_logging: Weights & Biases logging for training observability
- train_grpo_marxist: Main GRPO training script
- transform_to_grpo: Dataset transformation utilities
- convert_to_qwen: Qwen format conversion
"""

from pw_mcp.ai_training.grpo_rewards import (
    CONCEPT_EQUIVALENCES,
    DEPTH_MARKERS,
    DISCOURSE_CONNECTIVES,
    EXPLANATORY_PHRASES,
    HOLLOW_BUZZWORDS,
    MARXIST_TERMS,
    QUESTION_WORDS,
    completeness_reward,
    debug_print_reward,
    full_coherence_reward,
    interconnection_depth_reward,
    match_format_approximately,
    match_format_exactly,
    nli_coherence_reward,
    robust_coherence_reward,
    self_consistency_reward,
    semantic_similarity_reward,
    structural_coherence_reward,
    terminology_reward,
    topic_relevance_reward,
)
from pw_mcp.ai_training.wandb_logging import (
    RewardSample,
    WandbSampleLogger,
    create_logging_reward,
    finish_wandb_logging,
    init_wandb_logging,
    is_wandb_available,
    log_model_checkpoint,
    log_reward_metrics,
)

__all__ = [
    "CONCEPT_EQUIVALENCES",
    "DEPTH_MARKERS",
    "DISCOURSE_CONNECTIVES",
    "EXPLANATORY_PHRASES",
    "HOLLOW_BUZZWORDS",
    "MARXIST_TERMS",
    "QUESTION_WORDS",
    "RewardSample",
    "WandbSampleLogger",
    "completeness_reward",
    "create_logging_reward",
    "debug_print_reward",
    "finish_wandb_logging",
    "full_coherence_reward",
    "init_wandb_logging",
    "interconnection_depth_reward",
    "is_wandb_available",
    "log_model_checkpoint",
    "log_reward_metrics",
    "match_format_approximately",
    "match_format_exactly",
    "nli_coherence_reward",
    "robust_coherence_reward",
    "self_consistency_reward",
    "semantic_similarity_reward",
    "structural_coherence_reward",
    "terminology_reward",
    "topic_relevance_reward",
]
