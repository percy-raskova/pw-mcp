#!/usr/bin/env python3
"""
Fine-tune DeepSeek-R1-Distill-Qwen-7B-abliterated on ProleWiki Marxist-Leninist corpus.

Usage on RunPod:
  1. Upload this script and train_qwen.jsonl to /workspace/data/
  2. pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  3. pip install trl>=0.7.0 datasets accelerate bitsandbytes peft
  4. python train_marxist.py

Expected: ~30 min training, ~$0.30-0.50 on RTX 4090
"""

from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "huihui-ai/DeepSeek-R1-Distill-Qwen-7B-abliterated"
MAX_SEQ_LENGTH = 2048  # Qwen can go higher but 2048 is sufficient
LOAD_IN_4BIT = True

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training config
EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4  # Effective batch = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.1

# Paths - adjust for your environment
DATA_PATH = Path("/workspace/data/train_qwen.jsonl")  # RunPod
OUTPUT_DIR = Path("/workspace/outputs/marxist-deepseek")
CHECKPOINT_DIR = Path("/workspace/checkpoints")

# Fallback for local testing
if not DATA_PATH.exists():
    DATA_PATH = Path("training_data/formatted/train_qwen.jsonl")
    OUTPUT_DIR = Path("outputs/marxist-deepseek")
    CHECKPOINT_DIR = Path("checkpoints")


def load_dataset(path: Path) -> Dataset:
    """Load pre-formatted Qwen template dataset."""
    import json

    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)


def main() -> None:
    """Run fine-tuning."""
    print("=" * 60)
    print("Marxist-Leninist LLM Fine-Tuning")
    print("=" * 60)

    # Check CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        raise RuntimeError("CUDA not available - need GPU for training!")

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=None,  # Auto-detect (bf16 if available)
    )
    print(f"Model type: {model.config.model_type}")

    # Apply LoRA
    print("\nApplying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load dataset
    print(f"\nLoading dataset: {DATA_PATH}")
    dataset = load_dataset(DATA_PATH)

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Configure trainer
    print("\nConfiguring trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",  # Pre-formatted Qwen template
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            # Batch settings
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            # Learning rate
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=WARMUP_RATIO,
            # Training duration
            num_train_epochs=EPOCHS,
            # Memory optimization
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            # Logging
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            # Output
            output_dir=str(OUTPUT_DIR),
            seed=3407,
            report_to="none",  # or "wandb" if configured
        ),
    )

    # Train!
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Epochs: {EPOCHS}")
    print(
        f"Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}"
    )
    print(f"Learning rate: {LEARNING_RATE}")
    print()

    trainer.train()

    # Save final model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    lora_path = CHECKPOINT_DIR / "marxist-lora-adapter"
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print(f"LoRA adapter saved to: {lora_path}")

    # Export to GGUF
    print("\nExporting to GGUF (q4_k_m)...")
    gguf_path = CHECKPOINT_DIR / "marxist-deepseek-q4_k_m"
    model.save_pretrained_gguf(
        str(gguf_path),
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"GGUF exported to: {gguf_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print(f"1. Download: {gguf_path}/*.gguf")
    print("2. Create Ollama Modelfile (see ai-docs/finetune.yaml)")
    print("3. ollama create marxist-deepseek -f Modelfile")
    print("4. ollama run marxist-deepseek")
    print("\nDON'T FORGET TO STOP YOUR RUNPOD!")


if __name__ == "__main__":
    main()
