# pw-mcp

ProleWiki MCP Server & ML Training Infrastructure - Semantic vector search and GRPO fine-tuning for Marxist-Leninist AI systems.

## Overview

This project provides two complementary capabilities:

1. **MCP Server**: Semantic search over the ProleWiki corpus (~5,000 articles) for AI assistants
2. **ML Training**: GRPO fine-tuning infrastructure for creating Marxist-Leninist language models

**Key Features:**
- Semantic search over encyclopedia articles, library books, and essays
- Multi-provider embeddings (OpenAI `text-embedding-3-large` or local Ollama)
- Semantic line breaking via [sembr](https://github.com/admko/sembr) for intelligent chunking
- ChromaDB vector storage with rich metadata
- **GRPO training system** with 1,165 curated Q&A pairs
- **JSON Schema validation** for training data quality assurance
- Composable pipeline with granular mise tasks

## Quick Start

```bash
# Install dependencies
uv sync

# Run the MCP server
uv run pw-mcp
```

## Ingestion Pipeline

The pipeline transforms MediaWiki exports into searchable vector embeddings:

```
prolewiki-exports/ → extracted/ → sembr/ → chunks/ → embeddings/ → ChromaDB
     (source)        (clean text)  (line-   (para-    (vectors)    (indexed)
                                   broken)   graphs)
```

### Pipeline Commands (via mise)

```bash
# Check pipeline status
mise run status

# Run full pipeline
mise run pipeline

# Run individual stages
mise run extract          # MediaWiki → clean text
mise run sembr            # Add semantic line breaks
mise run chunk            # Split into paragraphs
mise run embed            # Generate embeddings
mise run load             # Load into ChromaDB

# Resume interrupted pipeline (skips existing files)
mise run pipeline-continue

# Test with sample files
mise run pipeline-sample
```

### Embedding Providers

```bash
mise run embed-openai     # OpenAI API (requires OPENAI_API_KEY)
mise run embed-ollama     # Local Ollama (requires running Ollama server)
```

### Sembr Server

Semantic line breaking requires a running sembr server:

```bash
mise run sembr-server     # Start on GPU 0
mise run sembr-server-gpu1  # Start on GPU 1
mise run sembr-stop       # Stop server
mise run sembr-check      # Check server health
```

## AI Training Infrastructure

The `training_data/` directory contains a rigorous ML training system:

### Dataset

| File | Records | Purpose |
|------|---------|---------|
| `curated_qa.jsonl` | 1,058 | Human-curated Q&A from ProleWiki corpus |
| `synthetic_antisemitism_correction.jsonl` | 61 | Anti-Zionism/antisemitism distinction |
| `synthetic_cpc_ml_distinction.jsonl` | 34 | CPC vs ML theoretical clarity |
| `synthetic_prolewiki_facts.jsonl` | 12 | ProleWiki organizational facts |

### Schema & Validation

- **JSON Schema (2020-12)** for training records: `training_data/schema/`
- **MANIFEST.yaml** with SHA256 checksums for reproducibility
- **Validation script**: `uv run python scripts/validate_training_data.py`

### Training Notebooks

Self-contained Jupyter notebooks for RunPod/cloud GPU training:
- `Marxist_GRPO_Training.ipynb` - Main GRPO training notebook
- `TRL_Direct_GRPO_Training.ipynb` - TRL-native GRPO implementation

### Documentation

- `MODEL_CARD.yaml` - Dataset provenance and source distribution
- `TRAINING_DIARY.md` - Training iterations and discovered issues
- `ai-docs/training-schema.yaml` - Human-readable schema reference

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
mise run test             # All tests
mise run test-unit        # Unit tests only
mise run test-cov         # With coverage

# Code quality
mise run check            # Lint + typecheck
mise run pre-commit       # All pre-commit hooks
```

## Project Structure

```
pw-mcp/
├── src/pw_mcp/
│   ├── server.py              # MCP server entry point
│   ├── config.py              # Pydantic configuration
│   ├── ingest/                # Ingestion pipeline
│   │   ├── cli.py             # pw-ingest CLI
│   │   ├── mediawiki.py       # MediaWiki parser
│   │   ├── linebreaker.py     # Sembr integration
│   │   ├── chunker.py         # Semantic chunking
│   │   └── embedder.py        # Embedding generation
│   ├── db/
│   │   └── chroma.py          # ChromaDB interface
│   └── ai_training/           # GRPO training module
│       ├── grpo_rewards.py    # Reward functions
│       └── wandb_logging.py   # W&B integration
├── training_data/             # ML training datasets
│   ├── *.jsonl                # Training Q&A pairs
│   ├── schema/                # JSON Schema definitions
│   ├── MANIFEST.yaml          # Dataset inventory
│   └── MODEL_CARD.yaml        # Dataset documentation
├── ai-docs/                   # AI reference documentation (YAML)
├── scripts/                   # Utility scripts
├── tests/
├── prolewiki-exports/         # Source corpus (gitignored)
└── chroma_data/               # ChromaDB persistence (gitignored)
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
OPENAI_API_KEY=sk-...         # For OpenAI embeddings
OLLAMA_HOST=http://localhost:11434  # For local Ollama
```

## Data

The ProleWiki corpus (`prolewiki-exports/`) is **not included** in this repository at the request of ProleWiki staff. Pipeline output directories (`extracted/`, `sembr/`, `chunks/`, `embeddings/`, `chroma_data/`) are also gitignored but their structure is preserved via `.gitkeep` files.

To obtain the corpus, contact ProleWiki directly or use their public export tools.

## Acknowledgments

- **[ProleWiki](https://en.prolewiki.org/)** - The Marxist-Leninist encyclopedia that provides the source corpus. Special thanks to their staff for guidance on data usage.
- **[ChromaDB](https://www.trychroma.com/)** - The AI-native open-source vector database powering semantic search.
- **[claude-mem](https://github.com/thedotmack/claude-mem)** - Cross-session memory plugin for Claude Code that helped maintain context during development.
- **[sembr](https://github.com/admko/sembr)** - Semantic line breaking model for intelligent text chunking.

## License

AGPL-3.0 - See [LICENSE](LICENSE)
