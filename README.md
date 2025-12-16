# pw-mcp

ProleWiki MCP Server - Semantic vector search over the ProleWiki Marxist-Leninist encyclopedia for AI assistants.

## Overview

An MCP (Model Context Protocol) server providing AI assistants with semantic search capabilities over the ProleWiki corpus (~5,000 articles). Features a multi-stage ingestion pipeline with semantic line breaking for optimal chunking.

**Key Features:**
- Semantic search over encyclopedia articles, library books, and essays
- Multi-provider embeddings (OpenAI `text-embedding-3-large` or local Ollama)
- Semantic line breaking via [sembr](https://github.com/admko/sembr) for intelligent chunking
- ChromaDB vector storage with rich metadata
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
│   ├── ingest/
│   │   ├── cli.py             # pw-ingest CLI
│   │   ├── mediawiki.py       # MediaWiki parser
│   │   ├── linebreaker.py     # Sembr integration
│   │   ├── chunker.py         # Semantic chunking
│   │   └── embedder.py        # Embedding generation
│   └── db/
│       └── chroma.py          # ChromaDB interface
├── ai-docs/                   # AI reference documentation (YAML)
├── tests/
│   ├── unit/
│   └── integration/
├── prolewiki-exports/         # Source corpus (gitignored)
├── extracted/                 # Pipeline stage output (gitignored)
├── sembr/                     # Pipeline stage output (gitignored)
├── chunks/                    # Pipeline stage output (gitignored)
├── embeddings/                # Pipeline stage output (gitignored)
└── chroma_data/               # ChromaDB persistence (gitignored)
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
OPENAI_API_KEY=sk-...         # For OpenAI embeddings
OLLAMA_HOST=http://localhost:11434  # For local Ollama
```

## License

MIT
