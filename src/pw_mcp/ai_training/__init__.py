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
    CAPITULATION_PATTERNS,
    CLASS_ANALYSIS_MARKERS,
    CONCEPT_EQUIVALENCES,
    CONFIDENT_CLAIM_PATTERNS,
    DEPTH_MARKERS,
    DISCOURSE_CONNECTIVES,
    EXPLANATORY_PHRASES,
    FIRMNESS_PATTERNS,
    HOLLOW_BUZZWORDS,
    IDEOLOGICAL_CHALLENGE_PATTERNS,
    MARXIST_TERMS,
    QUESTION_WORDS,
    QUOTE_TO_REFUTE_PATTERNS,
    SELF_CRITICISM_MARKERS,
    UNCERTAINTY_PATTERNS,
    completeness_reward,
    debug_print_reward,
    entity_verification_reward,
    epistemic_calibration_reward,
    full_coherence_reward,
    ideological_firmness_reward,
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
    "CAPITULATION_PATTERNS",
    "CLASS_ANALYSIS_MARKERS",
    "CONCEPT_EQUIVALENCES",
    "CONFIDENT_CLAIM_PATTERNS",
    "DEPTH_MARKERS",
    "DISCOURSE_CONNECTIVES",
    "EXPLANATORY_PHRASES",
    "FIRMNESS_PATTERNS",
    "HOLLOW_BUZZWORDS",
    "IDEOLOGICAL_CHALLENGE_PATTERNS",
    "MARXIST_TERMS",
    "QUESTION_WORDS",
    "QUOTE_TO_REFUTE_PATTERNS",
    "SELF_CRITICISM_MARKERS",
    "UNCERTAINTY_PATTERNS",
    "RewardSample",
    "WandbSampleLogger",
    "completeness_reward",
    "create_logging_reward",
    "debug_print_reward",
    "entity_verification_reward",
    "epistemic_calibration_reward",
    "finish_wandb_logging",
    "full_coherence_reward",
    "ideological_firmness_reward",
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
