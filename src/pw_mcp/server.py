"""ProleWiki MCP Server - exposes vector search tools to AI assistants."""

from mcp.server.fastmcp import FastMCP

mcp: FastMCP = FastMCP("prolewiki")


@mcp.tool()
async def search(query: str, limit: int = 5) -> str:
    """Search the ProleWiki encyclopedia using semantic vector search.

    Args:
        query: Natural language search query
        limit: Maximum number of results to return (default: 5)
    """
    # TODO: Implement ChromaDB search
    return f"Search not yet implemented. Query: {query}, Limit: {limit}"


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
