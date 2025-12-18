=========================================
AI Training Infrastructure Reference
=========================================

Annotated guide to the ``ai-docs/`` and ``training_data/`` directories
for Marxist-Leninist GRPO fine-tuning.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

This document provides a comprehensive annotated reference to the AI training
infrastructure in the pw-mcp project. The system uses GRPO (Group Relative
Policy Optimization) to fine-tune language models on Marxist-Leninist political
theory from the ProleWiki corpus.

Directory Structure
-------------------

.. code-block:: text

   pw-mcp/
   ├── ai-docs/                    # Token-efficient YAML references for AI assistants
   │   ├── index.yaml              # Master index of all reference files
   │   ├── finetune.yaml           # GRPO methodology and training configuration
   │   ├── reward-modeling.yaml    # Multi-layer reward function design
   │   ├── chatbot-ideology.yaml   # Training set design and question generation
   │   ├── runpod.yaml             # Cloud GPU setup instructions
   │   ├── project-status.yaml     # Implementation progress tracking
   │   └── ...                     # Additional reference files
   │
   ├── training_data/              # Datasets and training artifacts
   │   ├── curated_qa.jsonl        # 1,058 curated Q&A pairs (source)
   │   ├── grpo_dataset.jsonl      # GRPO-formatted training data
   │   ├── MODEL_CARD.yaml         # Dataset documentation and provenance
   │   ├── Marxist_GRPO_Training.ipynb  # Authoritative training notebook
   │   ├── LICENSE                 # AGPL-3.0-only
   │   └── formatted/              # Alternative format exports
   │       └── train_qwen.jsonl    # Qwen chat template format
   │
   └── src/pw_mcp/ai_training/     # Python module for GRPO rewards
       ├── grpo_rewards.py         # 13+ reward functions
       ├── wandb_logging.py        # Weights & Biases integration
       └── __init__.py             # Public API exports


ai-docs/ Directory
==================

Token-efficient YAML reference files designed for AI assistant consumption.
Each file provides structured lookup for specific aspects of the training system.

index.yaml
----------

**Purpose**: Master index of all reference files with descriptions and use cases.

**Key Sections**:

- ``files``: Maps each YAML file to its purpose, topics, and use cases
- ``ai_training_module``: Documents the Python module at ``src/pw_mcp/ai_training/``

**When to Consult**: Starting a new session, finding which file covers a topic.

finetune.yaml
-------------

**Purpose**: GRPO methodology and training configuration.

**Status**: ``IN_PROGRESS`` - Implementation complete, execution pending.

**Key Information**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Value
   * - Method
     - GRPO (Group Relative Policy Optimization)
   * - Base Model
     - ``unsloth/DeepSeek-R1-0528-Qwen3-8B``
   * - Dataset Size
     - 1,058 curated Q&A pairs
   * - Hardware
     - A40 48GB (RunPod)
   * - Training Steps
     - 250

**GRPO vs SFT Rationale**:

The project pivoted from SFT to GRPO because:

1. Political theory has no single "correct" answer (unlike math)
2. Open-ended prose requires semantic similarity, not exact string matching
3. Reward functions can encode domain expertise
4. Multi-layer rewards defeat adversarial "word soup" attacks

**LoRA Configuration**:

.. code-block:: yaml

   lora_config:
     rank: 64
     lora_alpha: 64
     target_modules:
       - q_proj   # Query projection (what to look for)
       - k_proj   # Key projection (how to be found)
       - v_proj   # Value projection (what info to carry)
       - o_proj   # Output projection (how to synthesize)
       - gate_proj  # Gate (what to amplify/suppress)
       - up_proj    # Up projection (what knowledge to access)
       - down_proj  # Down projection (what to preserve)
     use_gradient_checkpointing: unsloth

reward-modeling.yaml
--------------------

**Purpose**: Multi-layer reward function design for defeating reward hacking.

**Status**: ``IMPLEMENTED`` - 13+ functions with W&B logging.

**The Problem**: Naive rewards (keyword counting) are vulnerable to "word soup" -
models outputting random Marxist terminology without coherent meaning.

**The Solution**: Multi-layer rewards combining:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Layer
     - Function
   * - NLI Coherence
     - Check if response ENTAILS ground truth (BART-large-MNLI)
   * - Self-Consistency
     - Check for internal contradictions (no external ideology)
   * - Structural Coherence
     - Verify terms in syntactic roles via spaCy
   * - Topic Relevance
     - Ensure answer addresses what question asked
   * - Interconnection Depth
     - Reward deep analysis, penalize buzzword salad

**Ideological Bias Consideration**:

The document addresses concerns about NLI models trained on liberal media:

   "Testing shows BART-large-MNLI performs LOGICAL inference, not ideological
   judgment. We compare Marxist response against Marxist ground truth (from
   ProleWiki). The model isn't judging if Marxism is 'true' - it's checking if
   the response logically follows from the expected answer."

**Anti-Hacking Measures**:

.. code-block:: yaml

   HOLLOW_BUZZWORDS:
     # Penalized activist jargon without substance
     - "interconnected, interrelated, intersects with"
     - "centered, centering, uplift, do the work"
     - "problematic, harmful, toxic"

   DEPTH_MARKERS:
     # Bonuses for historical specificity
     - "in 1871", "Marx argued", "as Lenin wrote"
     - "for example", "specifically", "because"

   EXPLANATORY_PHRASES:
     # Indicators of actual explanation
     - "because the", "this is due to", "results from"
     - "is defined as", "means that", "therefore"

chatbot-ideology.yaml
---------------------

**Purpose**: Training set design for general Marxist-Leninist chatbot.

**Key Design Principles**:

1. **Data Source**: ProleWiki Library namespace chunks
2. **Question Generation**: Transform chunk metadata into natural questions
3. **System Prompt**: Defines chatbot persona (Marxist-Leninist assistant)

**Question Templates**:

.. code-block:: text

   Priority 1 (with section):
     "What does {author} say about {section}?"

   Priority 2 (with internal_links):
     "Explain {concept} from a Marxist perspective."

   Priority 3 (with categories):
     "Discuss {category} in Marxist theory."

   Fallback:
     "What does {author} teach us in this passage?"

**Quality Criteria**:

- Responses grounded in Marxist theory
- Materialist analysis (not idealist)
- Accurate to source texts
- Coherent and well-structured

runpod.yaml
-----------

**Purpose**: Cloud GPU setup instructions for training execution.

**Recommended Configuration**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Setting
     - Value
   * - GPU
     - A40 (48GB VRAM)
   * - Pricing
     - ~$0.79/hour spot, ~$1.14/hour on-demand
   * - Template
     - PyTorch 2.4 with CUDA 12.4
   * - Training Time
     - ~2-4 hours for 250 steps

**VRAM Breakdown**:

.. code-block:: text

   Model (4-bit):           ~4GB
   LoRA params:             ~1GB
   Optimizer state:         ~2GB
   Activations:             ~6GB (with gradient checkpointing)
   VLLM generation:         ~8GB (4 generations)
   Reward models:           ~2.5GB (NLI + embeddings + spaCy)
   ────────────────────────────────
   Total:                   ~24GB (safe on 48GB A40)


training_data/ Directory
========================

Contains the actual datasets, training notebooks, and supporting artifacts.

curated_qa.jsonl
----------------

**Purpose**: Primary source dataset of curated Q&A pairs.

**Statistics**:

- **Total pairs**: 1,058
- **Format**: JSONL with ``instruction`` and ``response`` fields
- **License**: AGPL-3.0-only

**Schema**:

.. code-block:: json

   {
     "instruction": "What is revisionism in the Marxist sense?",
     "response": "Revisionism refers to attempts to revise..."
   }

**Source Distribution**:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Category
     - Sources
   * - Primary Theory
     - Marx (Capital), Lenin (Collected Works), Fanon (Wretched of the Earth)
   * - Critical Historiography
     - Losurdo (Liberalism), Sousa (Soviet History), Pappe (Israel Myths)
   * - Anti-Colonial Thought
     - Sankara, Nkrumah, Dunbar-Ortiz, Rodney, George Jackson
   * - Feminist Marxism
     - Feinberg (Transgender Liberation), Kansas Socialist Book Club
   * - Historical Interviews
     - Stalin-Wells, Che-Howard, Assata Shakur, Mao (1939)

**Topic Keyword Frequency** (top 10):

.. code-block:: text

   Marx: 375      colonial: 276    gender: 160
   fascism: 126   Israel: 115      liberalism: 94
   women: 90      Lenin: 90        slavery: 88
   Soviet: 75

grpo_dataset.jsonl
------------------

**Purpose**: GRPO-formatted version of curated_qa.jsonl for training.

**Statistics**: 1,058 pairs (same content, different format)

**Schema**:

.. code-block:: json

   {
     "prompt": [
       {"role": "system", "content": "You are a Marxist-Leninist assistant..."},
       {"role": "user", "content": "What is revisionism?"}
     ],
     "answer": "Revisionism refers to attempts..."
   }

MODEL_CARD.yaml
---------------

**Purpose**: Dataset documentation following ML model card conventions.

**Key Sections**:

1. **Metadata**: Name, version, license (AGPL-3.0-only)
2. **Description**: Summary, motivation, intended use, out-of-scope uses
3. **Composition**: Source distribution, topic frequencies
4. **Ideological Position**: Explicit Marxist-Leninist commitment
5. **Limitations**: Coverage gaps, quality considerations
6. **Ethical Considerations**: Political nature, contested claims, responsible use
7. **Attribution**: Original contributions credited

**Ideological Position Statement**:

   "This dataset is explicitly committed to Marxist-Leninist political analysis.
   It does not claim ideological neutrality. The responses reflect historical
   materialist methodology and are sympathetic to socialist, anti-colonial,
   and anti-imperialist movements."

**Original Contributions**:

The dataset includes original essays with explicit permission:

- **AV Dremel**: ~80 Q&A pairs on fascism, COVID biology/politics, queer liberation
- **Persephone Raskova** (Kansas Socialist Book Club): ~60 Q&A pairs on political economy

Marxist_GRPO_Training.ipynb
---------------------------

**Purpose**: Self-contained Jupyter notebook for RunPod execution.

**Status**: AUTHORITATIVE REFERENCE for current implementation.

**Contents**:

1. **Installation**: Unsloth, TRL, vLLM dependencies
2. **Model Loading**: FastLanguageModel.from_pretrained with 4-bit quantization
3. **LoRA Configuration**: get_peft_model with all 7 target modules
4. **Dataset Loading**: From grpo_dataset.jsonl
5. **Reward Functions**: All 13+ functions defined inline (no external imports)
6. **W&B Integration**: WandbSampleLogger, create_logging_reward
7. **Training**: GRPOConfig + GRPOTrainer with A40-optimized settings
8. **Export**: LoRA saving, GGUF conversion for Ollama

**Why Inline Reward Functions?**

The notebook contains all reward functions inline rather than importing from
``src/pw_mcp/ai_training/`` because:

1. RunPod execution requires self-contained notebooks
2. No dependency on pw-mcp package installation
3. Easier debugging and modification during training

formatted/ Directory
--------------------

**Purpose**: Alternative format exports for different training frameworks.

**Contents**:

- ``train_qwen.jsonl`` (1.7MB): Qwen chat template format

**Qwen Template Format**:

.. code-block:: text

   <|im_start|>system
   You are a Marxist-Leninist assistant...<|im_end|>
   <|im_start|>user
   What is revisionism?<|im_end|>
   <|im_start|>assistant
   Revisionism refers to...<|im_end|>


Python Module Reference
=======================

Location: ``src/pw_mcp/ai_training/``

grpo_rewards.py
---------------

**Purpose**: 13+ reward functions for GRPO training.

**Function Categories**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Purpose
   * - ``match_format_exactly``
     - +3.0 for exact ``</think>`` pattern
   * - ``match_format_approximately``
     - +0.5/-1.0 for partial tag compliance
   * - ``semantic_similarity_reward``
     - Sentence-transformer embedding similarity
   * - ``terminology_reward``
     - Bonus for Marxist lexicon (DEPRECATED - gameable)
   * - ``nli_coherence_reward``
     - BART-large-MNLI entailment checking
   * - ``self_consistency_reward``
     - No internal contradictions
   * - ``structural_coherence_reward``
     - Terms in syntactic roles via spaCy
   * - ``topic_relevance_reward``
     - Question→answer concept coverage
   * - ``interconnection_depth_reward``
     - Anti-buzzword-salad detection
   * - ``completeness_reward``
     - Key concept coverage
   * - ``full_coherence_reward``
     - Combined 5-layer (RECOMMENDED)
   * - ``robust_coherence_reward``
     - Combined 3-layer (NLI + consistency + structure)
   * - ``debug_print_reward``
     - Development logging

**Constants Exported**:

- ``MARXIST_TERMS``: 100+ terms by category
- ``DISCOURSE_CONNECTIVES``: Analytical reasoning markers
- ``EXPLANATORY_PHRASES``: Causal reasoning indicators
- ``HOLLOW_BUZZWORDS``: Activist jargon penalty list
- ``DEPTH_MARKERS``: Historical specificity bonus list
- ``CONCEPT_EQUIVALENCES``: Synonym mapping (bourgeoisie ↔ capitalist class)

wandb_logging.py
----------------

**Purpose**: Weights & Biases integration for training observability.

**Key Components**:

.. code-block:: python

   # Initialize logging
   run = init_wandb_logging(
       project="marxist-grpo",
       config={"model": "DeepSeek-R1", "lr": 5e-6},
   )

   # Create sample logger
   sample_logger = WandbSampleLogger(
       log_every_n_steps=10,
       max_samples_per_log=4,
   )

   # Create logging reward (zero training effect)
   logging_reward = create_logging_reward(sample_logger)

   # Add to GRPOTrainer reward_funcs
   trainer = GRPOTrainer(
       reward_funcs=[..., logging_reward],
       ...
   )

**Graceful Degradation**:

All functions work without wandb installed - falls back to print statements.


Workflow: From Dataset to Deployment
====================================

.. code-block:: text

   1. DATASET PREPARATION
      └── curated_qa.jsonl (1,058 pairs)
          ↓ transform_to_grpo.py
      └── grpo_dataset.jsonl (GRPO format)

   2. RUNPOD SETUP
      └── Create A40 pod (ai-docs/runpod.yaml)
      └── Upload notebook + dataset
      └── Install Unsloth

   3. TRAINING EXECUTION
      └── Marxist_GRPO_Training.ipynb
          ├── Load model (DeepSeek-R1-Qwen3-8B)
          ├── Apply LoRA (7 target modules)
          ├── Configure rewards (full_coherence_reward)
          ├── Train (250 steps, ~2-4 hours)
          └── Monitor via W&B

   4. EXPORT
      └── Save LoRA adapter
      └── Merge to base model
      └── Convert to GGUF (q4_k_m)

   5. DEPLOYMENT
      └── Create Ollama Modelfile
      └── ollama create marxist-assistant
      └── Test with evaluation questions


Related Documentation
=====================

- ``docs/ideology-and-transformer-modules.rst``: How LoRA modules affect ideological output
- ``ai-docs/chromadb.yaml``: Vector database schema for RAG
- ``ai-docs/pipeline.yaml``: Ingestion pipeline architecture
- ``CLAUDE.md``: Project overview and development instructions


Appendix: Quick Reference Commands
==================================

.. code-block:: bash

   # Run tests for training module
   uv run pytest tests/unit/training/ -v

   # Check reward function implementation
   uv run python -c "from pw_mcp.ai_training import full_coherence_reward; print('OK')"

   # Validate dataset
   uv run python -c "
   import json
   with open('training_data/grpo_dataset.jsonl') as f:
       data = [json.loads(line) for line in f]
   print(f'Loaded {len(data)} examples')
   "

   # Start training (on RunPod)
   # Open Marxist_GRPO_Training.ipynb in JupyterLab
   # Run all cells
