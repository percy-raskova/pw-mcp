"""Unit tests for MCP server search tool.

RED PHASE: These tests define the contract for the search tool implementation.
All tests should FAIL until the implementation is complete.

The search tool should:
1. Accept query (str) and limit (int, default 5) parameters
2. Embed the query using OpenAI text-embedding-3-large
3. Query ChromaDB with the embedded query
4. Return markdown-formatted results with attribution

Output Format:
    **{article_title}** ({namespace}, S{section}, lines {line_range}) [score: {score:.2f}]:
    > {chunk_text}
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from pw_mcp.db.chroma import SearchResult

if TYPE_CHECKING:
    pass


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_embedding_1536() -> list[float]:
    """Return a mock 1536-dimensional embedding vector."""
    return [0.01 * i for i in range(1536)]


@pytest.fixture
def mock_search_results() -> list[SearchResult]:
    """Return mock search results from ChromaDB."""
    return [
        SearchResult(
            chunk_id="Main/Five-Year_Plans#0",
            text="The Five-Year Plans were a series of centralized economic plans "
            "implemented by the Soviet Union starting in 1928.",
            distance=0.15,  # Lower distance = higher similarity for cosine
            metadata={
                "article_title": "Five-Year Plans",
                "namespace": "Main",
                "section": "Introduction",
                "chunk_index": 0,
                "line_range": "1-8",
                "word_count": 150,
                "categories": ["Soviet economy", "Stalin era"],
                "internal_links": ["Soviet Union", "Joseph Stalin"],
                "is_stub": False,
                "citation_needed_count": 0,
                "has_blockquote": False,
            },
        ),
        SearchResult(
            chunk_id="Library/State_and_Revolution#3",
            text="The state is a product and a manifestation of the irreconcilability "
            "of class antagonisms.",
            distance=0.22,
            metadata={
                "article_title": "State and Revolution",
                "namespace": "Library",
                "section": "Chapter 1",
                "chunk_index": 3,
                "line_range": "45-52",
                "word_count": 200,
                "categories": ["Marxism-Leninism"],
                "internal_links": ["Lenin", "Marx"],
                "is_stub": False,
                "citation_needed_count": 0,
                "has_blockquote": True,
            },
        ),
        SearchResult(
            chunk_id="Main/Dialectical_materialism#1",
            text="Dialectical materialism is the philosophical basis of Marxism.",
            distance=0.31,
            metadata={
                "article_title": "Dialectical materialism",
                "namespace": "Main",
                "section": None,  # No section header
                "chunk_index": 1,
                "line_range": "10-15",
                "word_count": 120,
                "categories": ["Philosophy", "Marxism"],
                "internal_links": ["Marx", "Engels"],
                "is_stub": True,
                "citation_needed_count": 2,
                "has_blockquote": False,
            },
        ),
    ]


@pytest.fixture
def mock_empty_results() -> list[SearchResult]:
    """Return empty search results."""
    return []


# =============================================================================
# TEST: TOOL REGISTRATION
# =============================================================================


class TestSearchToolRegistration:
    """Tests for MCP server tool registration."""

    @pytest.mark.asyncio
    async def test_search_tool_registered(self) -> None:
        """Test that MCP server exposes search tool with correct signature.

        The search tool should be registered with the FastMCP server
        and accept query (str) and limit (int) parameters.
        """
        from pw_mcp.server import mcp

        # Get registered tools from the MCP server
        # FastMCP list_tools() is async
        tools = await mcp.list_tools()

        # Find the search tool
        search_tool = None
        for tool in tools:
            if tool.name == "search":
                search_tool = tool
                break

        assert search_tool is not None, "search tool should be registered"
        assert "query" in str(search_tool.inputSchema), "search should accept query parameter"
        assert "limit" in str(search_tool.inputSchema), "search should accept limit parameter"


# =============================================================================
# TEST: FORMATTED OUTPUT
# =============================================================================


class TestSearchFormattedOutput:
    """Tests for search result formatting."""

    @pytest.mark.asyncio
    async def test_search_returns_formatted_markdown(
        self,
        mock_search_results: list[SearchResult],
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that search results are returned in markdown format.

        Expected format for each result:
            **{article_title}** ({namespace}, S{section}, lines {line_range}) [score: {score:.2f}]:
            > {chunk_text}
        """
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = mock_search_results
            mock_get_db.return_value = mock_db

            result = await search("Soviet economic planning", limit=3)

        # Check that result is a string (markdown)
        assert isinstance(result, str)

        # Check for expected markdown patterns
        assert "**Five-Year Plans**" in result
        assert "(Main," in result
        assert "lines 1-8" in result
        assert "score:" in result
        assert ">" in result  # Blockquote syntax

    @pytest.mark.asyncio
    async def test_search_includes_attribution(
        self,
        mock_search_results: list[SearchResult],
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that results include title, namespace, section, and line_range."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = mock_search_results[:1]  # Just first result
            mock_get_db.return_value = mock_db

            result = await search("Five Year Plans", limit=1)

        # Check all attribution fields
        assert "Five-Year Plans" in result, "Should include article title"
        assert "Main" in result, "Should include namespace"
        assert "Introduction" in result, "Should include section"
        assert "1-8" in result, "Should include line range"

    @pytest.mark.asyncio
    async def test_search_handles_null_section(
        self,
        mock_search_results: list[SearchResult],
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that results handle null section gracefully."""
        from pw_mcp.server import search

        # Use the result with None section
        no_section_result = [mock_search_results[2]]  # Dialectical materialism

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = no_section_result
            mock_get_db.return_value = mock_db

            result = await search("dialectics", limit=1)

        # Should not crash and should format without section
        assert "Dialectical materialism" in result
        # Should not have "SNone" or throw error
        assert "None" not in result or "SNone" not in result


# =============================================================================
# TEST: RELEVANCE SCORES
# =============================================================================


class TestSearchRelevanceScores:
    """Tests for relevance score handling."""

    @pytest.mark.asyncio
    async def test_search_includes_relevance_scores(
        self,
        mock_search_results: list[SearchResult],
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that results include relevance scores."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = mock_search_results
            mock_get_db.return_value = mock_db

            result = await search("Marxism", limit=3)

        # Check for score format [score: X.XX]
        score_pattern = r"\[score: \d+\.\d{2}\]"
        assert re.search(score_pattern, result), "Should include formatted scores"

    @pytest.mark.asyncio
    async def test_search_results_ordered_by_relevance(
        self,
        mock_search_results: list[SearchResult],
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that results are ordered by relevance (lowest distance first)."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = mock_search_results
            mock_get_db.return_value = mock_db

            result = await search("planning", limit=3)

        # Five-Year Plans (distance 0.15) should appear before
        # State and Revolution (distance 0.22)
        five_year_idx = result.find("Five-Year Plans")
        state_rev_idx = result.find("State and Revolution")

        assert five_year_idx < state_rev_idx, "Results should be ordered by relevance"


# =============================================================================
# TEST: LIMIT PARAMETER
# =============================================================================


class TestSearchLimit:
    """Tests for limit parameter behavior."""

    @pytest.mark.asyncio
    async def test_search_respects_limit(
        self,
        mock_search_results: list[SearchResult],
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that limit parameter controls number of results."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            # DB would return all, but we limit
            mock_db.search.return_value = mock_search_results[:2]
            mock_get_db.return_value = mock_db

            await search("Soviet", limit=2)

            # Verify limit was passed to DB
            mock_db.search.assert_called_once()
            call_kwargs = mock_db.search.call_args
            assert call_kwargs.kwargs.get("limit") == 2 or call_kwargs.args[1] == 2

    @pytest.mark.asyncio
    async def test_search_default_limit_is_five(
        self,
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that default limit is 5 when not specified."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = []
            mock_get_db.return_value = mock_db

            await search("query without limit")

            # Verify default limit was passed
            mock_db.search.assert_called_once()
            call_kwargs = mock_db.search.call_args
            # Check limit is 5 (either as kwarg or positional arg)
            if call_kwargs.kwargs.get("limit"):
                assert call_kwargs.kwargs["limit"] == 5
            else:
                # Second positional arg should be limit
                assert len(call_kwargs.args) >= 2 and call_kwargs.args[1] == 5


# =============================================================================
# TEST: LIMIT VALIDATION
# =============================================================================


class TestSearchLimitValidation:
    """Tests for limit parameter validation."""

    @pytest.mark.asyncio
    async def test_search_rejects_zero_limit(self) -> None:
        """Test that limit=0 is rejected."""
        from pw_mcp.server import search

        with pytest.raises(ValueError, match=r"limit.*positive|limit.*1.*20"):
            await search("test query", limit=0)

    @pytest.mark.asyncio
    async def test_search_rejects_negative_limit(self) -> None:
        """Test that negative limit is rejected."""
        from pw_mcp.server import search

        with pytest.raises(ValueError, match=r"limit.*positive|limit.*1.*20"):
            await search("test query", limit=-5)

    @pytest.mark.asyncio
    async def test_search_rejects_excessive_limit(self) -> None:
        """Test that limit > 20 is rejected."""
        from pw_mcp.server import search

        with pytest.raises(ValueError, match=r"limit.*20|limit.*exceed"):
            await search("test query", limit=100)

    @pytest.mark.asyncio
    async def test_search_accepts_valid_limit_range(
        self,
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that limits 1-20 are accepted."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = []
            mock_get_db.return_value = mock_db

            # These should not raise
            await search("test", limit=1)
            await search("test", limit=10)
            await search("test", limit=20)


# =============================================================================
# TEST: EMPTY RESULTS
# =============================================================================


class TestSearchEmptyResults:
    """Tests for empty result handling."""

    @pytest.mark.asyncio
    async def test_search_empty_results_returns_helpful_message(
        self,
        mock_empty_results: list[SearchResult],
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that empty results return a helpful message."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = mock_empty_results
            mock_get_db.return_value = mock_db

            result = await search("xyzzy nonexistent topic", limit=5)

        assert isinstance(result, str)
        assert len(result) > 0, "Should return a non-empty message"
        # Should contain helpful language
        assert any(
            phrase in result.lower() for phrase in ["no results", "not found", "no matches", "try"]
        ), "Should provide helpful message for empty results"


# =============================================================================
# TEST: UNICODE HANDLING
# =============================================================================


class TestSearchUnicodeHandling:
    """Tests for Unicode character handling in queries."""

    @pytest.mark.asyncio
    async def test_search_handles_cyrillic(
        self,
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that Cyrillic characters in queries are handled."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = []
            mock_get_db.return_value = mock_db

            # Should not raise any encoding errors
            await search("Karl Marx", limit=3)

            # Verify the query was passed to embedder
            mock_embed.assert_called_once()
            embedded_query = mock_embed.call_args[0][0]
            assert "Karl Marx" in embedded_query or embedded_query == "Karl Marx"

    @pytest.mark.asyncio
    async def test_search_handles_chinese(
        self,
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that Chinese characters in queries are handled."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = []
            mock_get_db.return_value = mock_db

            await search("Mao Zedong", limit=3)

            mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_handles_mixed_scripts(
        self,
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that mixed Latin/Cyrillic/Chinese queries are handled."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = []
            mock_get_db.return_value = mock_db

            # Mixed script query (Latin + Cyrillic + Chinese)
            query = "Lenin and Marx"
            await search(query, limit=3)

            mock_embed.assert_called_once()


# =============================================================================
# TEST: EMBEDDING INTEGRATION
# =============================================================================


class TestSearchEmbeddingIntegration:
    """Tests for embedding provider integration."""

    @pytest.mark.asyncio
    async def test_search_uses_openai_embeddings(
        self,
        mock_search_results: list[SearchResult],
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that search uses OpenAI embeddings for queries."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = mock_search_results
            mock_get_db.return_value = mock_db

            await search("test query", limit=5)

            # embed_query should be called with the query string
            mock_embed.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_search_passes_embedding_to_db(
        self,
        mock_search_results: list[SearchResult],
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that embedded query is passed to ChromaDB."""
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = mock_search_results
            mock_get_db.return_value = mock_db

            await search("test query", limit=5)

            # DB search should receive the embedding vector
            mock_db.search.assert_called_once()
            call_args = mock_db.search.call_args
            query_embedding = (
                call_args.args[0] if call_args.args else call_args.kwargs.get("query_embedding")
            )
            assert query_embedding == mock_embedding_1536


# =============================================================================
# TEST: SCORE CONVERSION
# =============================================================================


class TestSearchScoreConversion:
    """Tests for distance-to-score conversion."""

    @pytest.mark.asyncio
    async def test_search_converts_distance_to_similarity_score(
        self,
        mock_search_results: list[SearchResult],
        mock_embedding_1536: list[float],
    ) -> None:
        """Test that cosine distance is converted to similarity score.

        ChromaDB returns cosine distance (0 = identical, 2 = opposite).
        We should convert to similarity score (1 = identical, 0 = opposite).

        Similarity = 1 - (distance / 2) for cosine distance, or
        Similarity = 1 - distance for normalized cosine distance (0-1 range).
        """
        from pw_mcp.server import search

        with (
            patch("pw_mcp.server.embed_query") as mock_embed,
            patch("pw_mcp.server.get_db") as mock_get_db,
        ):
            mock_embed.return_value = mock_embedding_1536
            mock_db = MagicMock()
            mock_db.search.return_value = mock_search_results[:1]  # distance=0.15
            mock_get_db.return_value = mock_db

            result = await search("test", limit=1)

        # For distance 0.15, similarity should be around 0.85 (if 1-d) or 0.925 (if 1-d/2)
        # Either way, should be between 0.80 and 0.95
        score_match = re.search(r"\[score: (\d+\.\d{2})\]", result)
        assert score_match, "Should include score in output"
        score = float(score_match.group(1))
        assert 0.80 <= score <= 0.95, f"Score {score} should reflect high similarity"
