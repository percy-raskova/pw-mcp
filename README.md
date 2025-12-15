# pw-mcp

ProleWiki MCP Server - Vector-powered Marxist-Leninist encyclopedia search for AI.

## Overview

An MCP (Model Context Protocol) server that provides AI assistants with semantic search capabilities over the ProleWiki corpus. Uses ChromaDB for vector storage and local embeddings.

## Installation

```bash
uv sync
```

## Usage

### Run the MCP server

```bash
uv run pw-mcp
```

### Ingest the corpus

```bash
uv run pw-ingest --source prolewiki-exports --output chroma_data
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
```

## Project Structure

```
pw-mcp/
├── src/pw_mcp/
│   ├── server.py          # MCP server entry point
│   ├── ingest/            # Corpus ingestion pipeline
│   ├── embeddings/        # Embedding interface
│   └── db/                # ChromaDB interface
├── tests/
├── prolewiki-exports/     # Source corpus (gitignored)
└── chroma_data/           # ChromaDB persistence (gitignored)
```

## License

MIT
