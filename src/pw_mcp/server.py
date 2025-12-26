"""ProleWiki MCP Server - exposes vector search tools to AI assistants."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from mcp.server.fastmcp import FastMCP

from pw_mcp.db.chroma import ChromaDBConfig, ProleWikiDB, SearchResult
from pw_mcp.ingest.embedder import EmbedConfig, embed_texts

mcp: FastMCP = FastMCP("prolewiki")

# Constants
MIN_LIMIT: int = 1
MAX_LIMIT: int = 20
DEFAULT_LIMIT: int = 5
CHROMA_DATA_PATH: Path = Path(__file__).parent.parent.parent.parent / "chroma_data"

# Embedding configuration - reused across queries
_EMBED_CONFIG: EmbedConfig = EmbedConfig(
    provider="openai",
    model="text-embedding-3-large",
    dimensions=1536,
)


def embed_query(query: str) -> list[float]:
    """Embed a single query string using OpenAI text-embedding-3-large.

    Args:
        query: The search query to embed.

    Returns:
        A 1536-dimensional embedding vector.
    """
    embeddings = embed_texts([query], _EMBED_CONFIG)
    return cast(list[float], embeddings[0].tolist())


def get_db() -> ProleWikiDB:
    """Get a ChromaDB instance for the ProleWiki corpus.

    Returns:
        ProleWikiDB instance connected to the persistent database.
    """
    config = ChromaDBConfig(persist_path=CHROMA_DATA_PATH)
    return ProleWikiDB(config)


def format_results(results: list[SearchResult]) -> str:
    """Format search results as markdown.

    Args:
        results: List of SearchResult objects from ChromaDB.

    Returns:
        Markdown-formatted string with each result on its own block.

    Format:
        **{article_title}** ({namespace}, S{section}, lines {line_range}) [score: {score:.2f}]:
        > {chunk_text}
    """
    if not results:
        return "No results found. Try a different query or broaden your search terms."

    formatted_parts: list[str] = []

    for result in results:
        metadata = result.metadata
        article_title = metadata.get("article_title", "Unknown")
        namespace = metadata.get("namespace", "Unknown")
        section = metadata.get("section")
        line_range = metadata.get("line_range", "?")

        # Convert cosine distance to similarity score
        # ChromaDB cosine distance: 0 = identical, higher = less similar
        # For normalized vectors, cosine distance is in [0, 2] range
        # Similarity = 1 - distance gives us [1, -1] for [0, 2] distance
        score = 1.0 - result.distance

        # Format section - use "(none)" if section is None or empty
        section_str = f"{section}" if section else "(none)"

        # Build the formatted result
        header = (
            f"**{article_title}** ({namespace}, "
            f"\u00a7{section_str}, lines {line_range}) "
            f"[score: {score:.2f}]:"
        )
        quote = f"> {result.text}"
        formatted_parts.append(f"{header}\n{quote}")

    return "\n\n".join(formatted_parts)


@mcp.tool()
async def search(query: str, limit: int = DEFAULT_LIMIT) -> str:
    """Search the ProleWiki encyclopedia using semantic vector search.

    Args:
        query: Natural language search query
        limit: Maximum number of results to return (default: 5, range: 1-20)

    Returns:
        Markdown-formatted search results with attribution.

    Raises:
        ValueError: If limit is outside the valid range (1-20).
    """
    # Validate limit parameter
    if limit < MIN_LIMIT or limit > MAX_LIMIT:
        raise ValueError(f"limit must be between {MIN_LIMIT} and {MAX_LIMIT}, got {limit}")

    # Embed the query
    query_embedding = embed_query(query)

    # Get database and perform search
    db = get_db()
    results = db.search(query_embedding, limit=limit)

    # Format and return results
    return format_results(results)


@mcp.tool()
async def get_article(title: str) -> str:
    """Retrieve a specific article by title.

    Args:
        title: The exact title of the article to retrieve
    """
    # TODO: Implement article retrieval
    return f"Article retrieval not yet implemented. Title: {title}"


@mcp.tool()
async def list_categories() -> str:
    """List all available categories in the ProleWiki corpus."""
    # TODO: Implement category listing from metadata
    return "Category listing not yet implemented."


def main() -> None:
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
