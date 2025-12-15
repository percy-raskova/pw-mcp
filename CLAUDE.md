# pw-mcp - ProleWiki MCP Server

## Project Overview

An MCP (Model Context Protocol) server providing AI assistants with semantic vector search over the ProleWiki corpus - a Marxist-Leninist encyclopedia. The releasable artifact is a ChromaDB database that can be distributed independently of the source corpus.

## Tech Stack

- **Language**: Python 3.12 (chosen for AI coding proficiency + ecosystem support)
- **Package Manager**: uv
- **Vector DB**: ChromaDB (persistent, local-first)
- **Embeddings**: Ollama `embeddinggemma` (768-dim, local, Gemma3-based 307M params)
- **MediaWiki Parsing**: mwparserfromhell
- **Semantic Linebreaking**: sembr with `distilbert-base-multilingual-cased` (supports Chinese, Russian, etc.)
- **Server Protocol**: MCP via FastMCP
- **Configuration**: Pydantic models loading from `[tool.pw-mcp]` in pyproject.toml

## AI Reference Documentation

Token-efficient YAML references for AI assistants in `ai-docs/`:

| File | Purpose |
|------|---------|
| `index.yaml` | Index of all reference files with descriptions |
| `chromadb.yaml` | ChromaDB schema, operations, metadata patterns, interlinking |

**When to consult ai-docs/:**
- Designing database schema → `chromadb.yaml#pw_schema`
- Writing query/filter logic → `chromadb.yaml#operators`
- Implementing batch ingestion → `chromadb.yaml#batching`
- Wiki-style interlinking → `chromadb.yaml#interlinking`

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

### Why Semantic Linebreaking + Chunking (not 1 line = 1 vector)

Individual sentences often lack context:
```
Stalin implemented the Five-Year Plans.
These transformed the Soviet Union from an agrarian society into an industrial power.
```

"These transformed..." loses its referent if embedded alone.

**Solution**:
1. Sembr source files for clean organization
2. Chunk at paragraph/section level (~200-500 tokens) for embedding
3. Store line offsets in metadata for precise citations

### Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. MediaWiki Parser                                        │
│     prolewiki-exports/*.txt → clean text + metadata         │
│     Extract: categories, internal links, sections, refs     │
│                                                             │
│  2. Semantic Linebreaker (sembr)                            │
│     clean text → sembr'd text                               │
│     Creates semantic boundaries for chunking                │
│                                                             │
│  3. Chunker                                                 │
│     sembr'd text → chunks (~200-500 tokens)                 │
│     - Respects section boundaries (== headers ==)           │
│     - Groups semantic lines into coherent paragraphs        │
│     - Stores line offsets for precise citation              │
│                                                             │
│  4. Embedder                                                │
│     chunks → vectors via Ollama embeddinggemma (768-dim)    │
│                                                             │
│  5. ChromaDB Loader                                         │
│     vectors + metadata → persistent database                │
└─────────────────────────────────────────────────────────────┘
```

### Sembr Performance Note

**IMPORTANT**: Running sembr per-file is slow (~9s/file due to model loading).

Use **server mode** for batch processing:
```bash
# Terminal 1: Start sembr server (loads 135M param model once)
mise run sembr-server

# Terminal 2: Send files via HTTP to localhost:8384
# This avoids reloading the model for each file
```

Benchmark results (100 files sample):
- Per-file mode: ~9 seconds/file → **13-16 hours** for full corpus
- Server mode: Expected **10-50x faster** (model loaded once)

### Chunk Metadata Schema

```python
{
    "article_title": str,      # "Five-Year Plans"
    "namespace": str,          # "Main", "Library", "Essays", "ProleWiki"
    "section": str | None,     # "Implementation" (from == headers ==)
    "categories": list[str],   # ["Soviet economy", "Stalin era"]
    "internal_links": list[str],  # Referenced articles
    "source_file": str,        # "Main/Five-Year Plans.txt"
    "line_range": tuple[int, int],  # (45, 52) for citation
    "chunk_index": int,        # Position within article
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
│   │   ├── __init__.py
│   │   ├── cli.py             # pw-ingest CLI
│   │   ├── mediawiki.py       # MediaWiki parser
│   │   ├── linebreaker.py     # sembr wrapper
│   │   └── chunker.py         # Chunk creation
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embed.py           # sentence-transformers interface
│   └── db/
│       ├── __init__.py
│       └── chroma.py          # ChromaDB interface
├── ai-docs/                   # AI reference documentation (YAML)
│   ├── index.yaml             # Index of all reference files
│   └── chromadb.yaml          # ChromaDB operations & schema
├── tests/
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

mise run sembr-server     # Start sembr server for batch processing
mise run serve            # Start MCP server
mise run ingest           # Run corpus ingestion
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

## Future Considerations

- **Metadata normalization**: ProleWiki categories need cleaning/standardization
- **Incremental updates**: Currently full reindex; could add delta ingestion
- **Artifact distribution**: Package ChromaDB as downloadable release
- **Sembr server integration**: Build HTTP client to use sembr server mode for faster batch processing
