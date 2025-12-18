#!/usr/bin/env python3
"""
GRPO Fine-tuning for Marxist-Leninist Reasoning Model.

Trains DeepSeek-R1-0528-Qwen3-8B on ProleWiki corpus using GRPO
(Group Relative Policy Optimization) with custom reward functions.

Usage:
    # First transform data
    python transform_to_grpo.py

    # Then run training
    python train_grpo_marxist.py

Hardware: A40 (48GB) optimized
Expected time: ~1-2 hours for 250 steps
"""

from __future__ import annotations

import os
from pathlib import Path

# Set vLLM standby mode for better memory utilization
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from vllm import SamplingParams

from pw_mcp.ai_training.grpo_rewards import (
    completeness_reward,
    debug_print_reward,
    match_format_approximately,
    match_format_exactly,
    semantic_similarity_reward,
    terminology_reward,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model
MODEL_NAME = "unsloth/DeepSeek-R1-0528-Qwen3-8B"
MAX_SEQ_LENGTH = 2048  # Longer for detailed political theory responses
LORA_RANK = 32  # Same as original notebook

# Paths
DATA_PATH = Path("training_data/grpo_dataset.jsonl")
OUTPUT_DIR = Path("outputs/marxist-grpo")
LORA_OUTPUT = Path("outputs/marxist-grpo-lora")

# Training
MAX_STEPS = 250  # Cover most of 1058 samples
SAVE_STEPS = 50
LEARNING_RATE = 5e-6
WARMUP_RATIO = 0.1

# A40 optimized settings
GPU_MEMORY_UTILIZATION = 0.85
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 2
NUM_GENERATIONS = 4

# Completion limits
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 1500


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================


def main() -> None:
    """Run GRPO training."""
    print("=" * 60)
    print("Marxist-Leninist GRPO Training")
    print("=" * 60)

    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"VRAM: {gpu_mem:.1f} GB")
    else:
        raise RuntimeError("CUDA not available!")

    # =========================================================================
    # Load Model
    # =========================================================================
    print(f"\nLoading model: {MODEL_NAME}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,  # Enable vLLM
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    print(f"Model type: {model.config.model_type}")

    # =========================================================================
    # Apply LoRA
    # =========================================================================
    print("\nApplying LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,  # *2 speeds up training
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # =========================================================================
    # Load Dataset
    # =========================================================================
    print(f"\nLoading dataset: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n" "Run 'python transform_to_grpo.py' first!"
        )

    dataset = Dataset.from_json(str(DATA_PATH))
    print(f"Loaded {len(dataset)} examples")

    # Show sample
    sample = dataset[0]
    print(f"Sample prompt: {sample['prompt'][1]['content'][:60]}...")

    # =========================================================================
    # Configure vLLM Sampling
    # =========================================================================
    vllm_sampling_params = SamplingParams(
        min_p=0.1,
        top_p=1.0,  # No nucleus sampling (matches original template)
        top_k=-1,
        # NOTE: temperature is set in GRPOConfig, not here
        max_tokens=MAX_COMPLETION_LENGTH,
        stop=[tokenizer.eos_token],
        include_stop_str_in_output=True,
        seed=3407,
    )

    # =========================================================================
    # Configure Training
    # =========================================================================
    print("\nConfiguring GRPO trainer...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        # vLLM
        vllm_sampling_params=vllm_sampling_params,
        temperature=1.0,  # For GRPO training dynamics
        # Optimization
        learning_rate=LEARNING_RATE,
        weight_decay=0.001,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        # Batch settings
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_generations=NUM_GENERATIONS,
        # Sequence lengths
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_completion_length=MAX_COMPLETION_LENGTH,
        # Training duration
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        # Logging
        logging_steps=1,
        report_to="none",
        # Output
        output_dir=str(OUTPUT_DIR),
    )

    # =========================================================================
    # Create Trainer
    # =========================================================================
    print("\nInitializing trainer with reward functions:")
    print("  - match_format_exactly (+3.0 for </think>)")
    print("  - match_format_approximately (±0.5 for tags)")
    print("  - semantic_similarity_reward (+5.0 to -3.0)")
    print("  - terminology_reward (+0 to +2.0)")
    print("  - completeness_reward (±2.0)")
    print("  - debug_print_reward (monitoring)")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            match_format_exactly,
            match_format_approximately,
            semantic_similarity_reward,
            terminology_reward,
            completeness_reward,
            debug_print_reward,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    # =========================================================================
    # Train!
    # =========================================================================
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Steps: {MAX_STEPS}")
    print(f"Batch: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} x {NUM_GENERATIONS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    trainer.train()

    # =========================================================================
    # Save
    # =========================================================================
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    LORA_OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_lora(str(LORA_OUTPUT))
    print(f"LoRA saved to: {LORA_OUTPUT}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Test the model with and without LoRA")
    print("2. Export to GGUF if satisfied")
    print("3. Create Ollama Modelfile")


# =============================================================================
# TEST FUNCTION
# =============================================================================


def test_model() -> None:
    """Test the trained model."""
    print("Loading model for testing...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )

    test_questions = [
        "What is revisionism in the Marxist sense?",
        "Explain the concept of surplus value.",
        "What is the dictatorship of the proletariat?",
        "How does dialectical materialism differ from idealism?",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        max_tokens=1024,
    )

    system_prompt = """You are a Marxist-Leninist assistant trained on ProleWiki.
Think through political theory questions using dialectical materialist analysis.
Show your reasoning in <think> tags, then provide a clear answer."""

    print("\n" + "=" * 60)
    print("TESTING WITHOUT LORA")
    print("=" * 60)

    for question in test_questions[:2]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        output = (
            model.fast_generate(text, sampling_params=sampling_params, lora_request=None)[0]
            .outputs[0]
            .text
        )
        print(f"\nQ: {question}")
        print(f"A: {output[:500]}...")

    print("\n" + "=" * 60)
    print("TESTING WITH LORA")
    print("=" * 60)

    for question in test_questions[:2]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        output = (
            model.fast_generate(
                text,
                sampling_params=sampling_params,
                lora_request=model.load_lora(str(LORA_OUTPUT)),
            )[0]
            .outputs[0]
            .text
        )
        print(f"\nQ: {question}")
        print(f"A: {output[:500]}...")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_model()
    else:
        main()
