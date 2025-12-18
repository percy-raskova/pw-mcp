# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# pw-mcp - ProleWiki MCP Server

## Project Overview

An MCP (Model Context Protocol) server providing AI assistants with semantic vector search over the ProleWiki corpus - a Marxist-Leninist encyclopedia. The releasable artifact is a ChromaDB database that can be distributed independently of the source corpus.

## Tech Stack

- **Language**: Python 3.12 (chosen for AI coding proficiency + ecosystem support)
- **Package Manager**: uv
- **Vector DB**: ChromaDB (persistent, local-first)
- **Embeddings**: OpenAI `text-embedding-3-large` (1536-dim, via API) or Ollama (local fallback)
- **MediaWiki Parsing**: mwparserfromhell
- **Token Counting**: tiktoken (cl100k_base encoding for OpenAI compatibility)
- **Server Protocol**: MCP via FastMCP
- **Configuration**: Pydantic models loading from `[tool.pw-mcp]` in pyproject.toml

## AI Reference Documentation

Token-efficient YAML references for AI assistants in `ai-docs/`:

| File | Purpose |
|------|---------|
| `index.yaml` | Index of all reference files with descriptions |
| `chromadb.yaml` | ChromaDB schema, operations, metadata patterns |
| `finetune.yaml` | GRPO methodology and training configuration |
| `reward-modeling.yaml` | Multi-layer reward function design |
| `chatbot-ideology.yaml` | Training set design, question generation |
| `runpod.yaml` | Cloud GPU setup instructions |
| `project-status.yaml` | Implementation progress tracking |

**When to consult ai-docs/:**
- Designing database schema → `chromadb.yaml#pw_schema`
- Writing query/filter logic → `chromadb.yaml#operators`
- Implementing batch ingestion → `chromadb.yaml#batching`
- Understanding GRPO training → `finetune.yaml`, `reward-modeling.yaml`
- Setting up RunPod GPU → `runpod.yaml`

## Corpus Structure

Source: `prolewiki-exports/` (gitignored, ~204MB, 5,222 files)

```
prolewiki-exports/
├── Main/           # Encyclopedia articles (~4,000+ files)
├── Library/        # Full books/documents (Marx, Lenin, etc.) - up to 46k lines
├── Essays/         # User-contributed analytical essays
└── ProleWiki/      # Meta/admin pages (governance, guidelines)
```

Format: MediaWiki markup with `[[links]]`, `{{templates}}`, `[[Category:tags]]`, refs

## Architecture Decisions

### Why Paragraph-Level Chunking (not 1 sentence = 1 vector)

Individual sentences often lack context:
```
Stalin implemented the Five-Year Plans.
These transformed the Soviet Union from an agrarian society into an industrial power.
```

"These transformed..." loses its referent if embedded alone.

**Solution**:
1. Chunk at paragraph/section level (~350-500 tokens) for embedding
2. Use 50-token overlap between chunks for context continuity
3. Store line offsets in metadata for precise citations

### Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. MediaWiki Parser (extract)                              │
│     prolewiki-exports/*.txt → clean text + metadata         │
│     Extract: categories, internal links, sections, refs     │
│                                                             │
│  2. Tiktoken Chunker (chunk)                                │
│     extracted text → chunks (~350-500 tokens)               │
│     - Respects section boundaries (== headers ==)           │
│     - Uses tiktoken for accurate token counting             │
│     - 50-token overlap for RAG context continuity           │
│     - Stores line offsets for precise citation              │
│                                                             │
│  3. Embedder (embed)                                        │
│     chunks → vectors via OpenAI text-embedding-3-large      │
│     (1536-dim, or Ollama 768-dim for local development)     │
│                                                             │
│  4. ChromaDB Loader (load)                                  │
│     vectors + metadata → persistent database                │
└─────────────────────────────────────────────────────────────┘
```

### Chunk Metadata Schema (MVP - 13 fields)

```python
{
    "chunk_id": str,           # "Main/Five-Year_Plans#0"
    "text": str,               # Chunk content (embedded)
    "article_title": str,      # "Five-Year Plans"
    "namespace": str,          # "Main", "Library", "Essays", "ProleWiki"
    "section": str | None,     # "Implementation" (from == headers ==)
    "chunk_index": int,        # Position within article
    "line_range": str,         # "45-52" for citation
    "word_count": int,         # 287
    "categories": list[str],   # ["Soviet economy", "Stalin era"]
    "internal_links": list[str],  # Referenced articles
    "is_stub": bool,           # Article marked as incomplete
    "citation_needed_count": int,  # Number of {{Citation needed}} markers
    "has_blockquote": bool,    # Contains <blockquote> content
}
```

## MCP Server Tools

```python
@mcp.tool()
async def search(query: str, limit: int = 5) -> str:
    """Semantic search over ProleWiki corpus."""

@mcp.tool()
async def get_article(title: str) -> str:
    """Retrieve full article by title."""

@mcp.tool()
async def list_categories() -> str:
    """List all categories in the corpus."""
```

## Project Structure

```
pw-mcp/
├── src/pw_mcp/
│   ├── __init__.py
│   ├── config.py              # Pydantic config from pyproject.toml
│   ├── server.py              # MCP server entry point
│   ├── ingest/
│   │   ├── cli.py             # pw-ingest CLI
│   │   ├── extraction.py      # Article extraction pipeline
│   │   ├── parsers/           # MediaWiki parsers (link, citation, infobox, etc.)
│   │   ├── chunker.py         # Tiktoken-based chunking
│   │   └── embedder.py        # OpenAI/Ollama embeddings
│   ├── db/
│   │   └── chroma.py          # ChromaDB interface
│   └── ai_training/           # GRPO fine-tuning module
│       ├── grpo_rewards.py    # 13+ reward functions
│       ├── wandb_logging.py   # Weights & Biases integration
│       └── transform_to_grpo.py  # Dataset conversion
├── ai-docs/                   # AI reference documentation (YAML)
├── docs/                      # RST documentation
│   ├── ideology-and-transformer-modules.rst
│   └── ai-training-reference.rst
├── training_data/             # ML training datasets
│   ├── curated_qa.jsonl       # 1,058 Q&A pairs (source)
│   ├── grpo_dataset.jsonl     # GRPO-formatted data
│   ├── MODEL_CARD.yaml        # Dataset documentation
│   └── Marxist_GRPO_Training.ipynb  # RunPod training notebook
├── tests/
│   └── unit/training/         # GRPO reward function tests
├── prolewiki-exports/         # Source corpus (gitignored)
├── chroma_data/               # ChromaDB persistence (gitignored)
├── pyproject.toml             # Includes [tool.pw-mcp] config section
└── uv.lock
```

## Development Environment

### Code Quality Tools

| Tool | Purpose | Config Location |
|------|---------|-----------------|
| **ruff** | Linting + formatting + import sorting | `pyproject.toml` |
| **mypy** | Static type checking (strict mode) | `pyproject.toml` |
| **pytest** | Testing with markers (unit/integration/slow) | `pyproject.toml` |
| **pre-commit** | Git hooks for automated checks | `.pre-commit-config.yaml` |
| **yamllint** | YAML file validation | `.yamllint.yaml` |

### Pre-commit Hooks

Runs automatically on `git commit`:
1. **ruff lint** - fix linting issues
2. **ruff format** - enforce code style
3. **mypy** - type checking
4. **trailing-whitespace** - clean up whitespace
5. **end-of-file-fixer** - ensure newlines
6. **check-yaml/json/toml** - validate config files
7. **yamllint** - advanced YAML validation

### Type Checking

Strict mypy configuration:
- `strict = true`
- `disallow_untyped_defs = true`
- `no_implicit_optional = true`
- Third-party libs with missing stubs are allowlisted in `[[tool.mypy.overrides]]`

## Commands

### Using mise (recommended)

```bash
mise run install          # Install all deps including dev
mise run hooks            # Install pre-commit hooks

mise run check            # Run lint + typecheck
mise run test             # Run all tests
mise run pre-commit       # Run all pre-commit hooks

mise run serve            # Start MCP server

# Pipeline stages
mise run extract          # Extract from MediaWiki
mise run chunk            # Chunk extracted text (tiktoken)
mise run embed            # Generate embeddings
mise run load             # Load into ChromaDB

mise run corpus-pipeline  # Run full pipeline
mise run sample-pipeline  # Run on sample files
```

### Using uv directly

```bash
# Development setup
uv sync --group dev
uv run pre-commit install

# Code quality
uv run ruff check src/          # Lint
uv run ruff format src/         # Format
uv run mypy src/pw_mcp/         # Type check
uv run pre-commit run --all-files  # All checks

# Testing
uv run pytest                   # All tests
uv run pytest -m unit           # Fast unit tests only
uv run pytest --cov             # With coverage

# Servers
uv run pw-mcp                   # Start MCP server
uv run pw-ingest                # Run ingestion
```

## AI Training Infrastructure

GRPO (Group Relative Policy Optimization) fine-tuning system for Marxist-Leninist language models.

### Training Module (`src/pw_mcp/ai_training/`)

| File | Purpose |
|------|---------|
| `grpo_rewards.py` | 13+ reward functions for GRPO training |
| `wandb_logging.py` | Weights & Biases integration |
| `transform_to_grpo.py` | Dataset format conversion |

### Key Reward Functions

| Function | Purpose |
|----------|---------|
| `full_coherence_reward` | Combined 5-layer check (RECOMMENDED) |
| `nli_coherence_reward` | BART-large-MNLI entailment |
| `self_consistency_reward` | Internal contradiction detection |
| `structural_coherence_reward` | spaCy syntactic analysis |
| `topic_relevance_reward` | Question-answer alignment |
| `interconnection_depth_reward` | Anti-buzzword-salad |

### Training Data (`training_data/`)

| File | Purpose |
|------|---------|
| `curated_qa.jsonl` | 1,058 curated Q&A pairs (source) |
| `grpo_dataset.jsonl` | GRPO-formatted training data |
| `MODEL_CARD.yaml` | Dataset provenance and documentation |
| `Marxist_GRPO_Training.ipynb` | Self-contained RunPod notebook |

### AI Training Commands

```bash
# Run training module tests
uv run pytest tests/unit/training/ -v

# Run single test file
uv run pytest tests/unit/training/test_grpo_rewards.py -v

# Validate reward function imports
uv run python -c "from pw_mcp.ai_training import full_coherence_reward"
```

## Future Considerations

- **Metadata normalization**: ProleWiki categories need cleaning/standardization
- **Incremental updates**: Currently full reindex; could add delta ingestion
- **Artifact distribution**: Package ChromaDB as downloadable release
