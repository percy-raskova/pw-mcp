"""Type stubs for mcp.server.fastmcp module."""

from collections.abc import Callable, Coroutine
from typing import Any, Literal, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

class FastMCP:
    """FastMCP server for Model Context Protocol."""

    def __init__(
        self,
        name: str | None = None,
        instructions: str | None = None,
        website_url: str | None = None,
        *,
        debug: bool = False,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None: ...
    def tool(
        self,
        name: str | None = None,
        description: str | None = None,
    ) -> Callable[
        [Callable[P, Coroutine[Any, Any, R]]],
        Callable[P, Coroutine[Any, Any, R]],
    ]:
        """Decorator to register a function as an MCP tool."""
        ...
    def run(self, transport: str = "stdio") -> None:
        """Run the MCP server."""
        ...
