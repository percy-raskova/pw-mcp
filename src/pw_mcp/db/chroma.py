"""ChromaDB interface for ProleWiki vector storage and retrieval.

This module provides the database layer for storing and querying
embedded chunks from the ProleWiki corpus.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import chromadb
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection
    from chromadb.api.types import Where, WhereDocument


@dataclass
class ChromaDBConfig:
    """Configuration for ChromaDB persistent client."""

    persist_path: Path
    collection_name: str = "prolewiki_chunks"
    embedding_dimensions: int = 1536  # OpenAI text-embedding-3-large
    batch_size: int = 5000  # Below ChromaDB max batch size


@dataclass
class LoadStats:
    """Statistics from a batch load operation."""

    articles_loaded: int = 0
    chunks_loaded: int = 0
    articles_skipped: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """A single search result from semantic query."""

    chunk_id: str
    text: str
    distance: float
    metadata: dict[str, Any]


def serialize_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    """Convert chunk dict to ChromaDB-compatible metadata.

    ChromaDB metadata constraints:
    - No None values (use empty string for strings, -1 for optional ints)
    - Lists must be JSON strings
    - Supported types: str, int, float, bool
    """
    return {
        # Core identifier fields
        "article_title": chunk["article_title"],
        "namespace": chunk["namespace"],
        "section": chunk.get("section") or "",  # None → ""
        "chunk_index": chunk["chunk_index"],
        "line_range": chunk["line_range"],
        "word_count": chunk["word_count"],
        # Array fields stored as JSON strings
        "categories": json.dumps(chunk.get("categories", [])),
        "internal_links": json.dumps(chunk.get("internal_links", [])),
        # Quality flags
        "is_stub": chunk.get("is_stub", False),
        "citation_needed_count": chunk.get("citation_needed_count", 0),
        "has_blockquote": chunk.get("has_blockquote", False),
        # Phase B: Enriched metadata for MCP filtering
        "library_work_author": chunk.get("library_work_author") or "",
        "library_work_type": chunk.get("library_work_type") or "",
        "library_work_published_year": chunk.get("library_work_published_year") or -1,
        "infobox_type": chunk.get("infobox_type") or "",
        "political_orientation": chunk.get("political_orientation") or "",
        "primary_category": chunk.get("primary_category") or "",
        "category_count": chunk.get("category_count", 0),
    }


def deserialize_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert ChromaDB metadata back to original format.

    Reverses the transformations from serialize_metadata.

    Args:
        metadata: ChromaDB metadata mapping (may contain str, int, float, bool, None).

    Returns:
        Dict with deserialized categories, internal_links, and Phase B fields.
    """
    result: dict[str, Any] = dict(metadata)

    # Restore None for empty section
    if result.get("section") == "":
        result["section"] = None

    # Parse JSON arrays
    if "categories" in result and isinstance(result["categories"], str):
        result["categories"] = json.loads(result["categories"])
    if "internal_links" in result and isinstance(result["internal_links"], str):
        result["internal_links"] = json.loads(result["internal_links"])

    # Restore None for empty Phase B string fields
    for field_name in [
        "library_work_author",
        "library_work_type",
        "infobox_type",
        "political_orientation",
        "primary_category",
    ]:
        if result.get(field_name) == "":
            result[field_name] = None

    # Restore None for -1 in published_year
    if result.get("library_work_published_year") == -1:
        result["library_work_published_year"] = None

    return result


class ProleWikiDB:
    """ChromaDB interface for ProleWiki corpus.

    Manages a persistent ChromaDB collection containing embedded chunks
    from ProleWiki articles. Supports loading from pre-computed embeddings
    and semantic search with metadata filtering.

    Example:
        config = ChromaDBConfig(persist_path=Path("./chroma_data"))
        db = ProleWikiDB(config)
        db.load_article(chunks_path, embeddings_path)
        results = db.search(query_embedding, limit=5)
    """

    def __init__(self, config: ChromaDBConfig) -> None:
        """Initialize ChromaDB client and get/create collection.

        Args:
            config: Database configuration including persist path.
        """
        self.config = config
        self.client = chromadb.PersistentClient(path=str(config.persist_path))
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        """Create or retrieve the ProleWiki collection.

        Configures HNSW index with cosine distance for semantic similarity.
        No embedding function - we use precomputed embeddings.
        """
        return self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100,
            },
        )

    def load_article(
        self,
        chunks_path: Path,
        embeddings_path: Path,
    ) -> int:
        """Load a single article's chunks and embeddings into the database.

        Args:
            chunks_path: Path to JSONL file with chunk data.
            embeddings_path: Path to NPY file with embedding vectors.

        Returns:
            Number of chunks loaded.

        Raises:
            ValueError: If chunk count doesn't match embedding count.
            FileNotFoundError: If either file doesn't exist.
        """
        # Read chunks from JSONL
        chunks: list[dict[str, Any]] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

        if not chunks:
            return 0

        # Read embeddings from NPY
        embeddings: NDArray[np.float32] = np.load(embeddings_path)

        # Validate counts match
        if len(chunks) != embeddings.shape[0]:
            msg = (
                f"Chunk count ({len(chunks)}) doesn't match "
                f"embedding count ({embeddings.shape[0]}) "
                f"for {chunks_path.name}"
            )
            raise ValueError(msg)

        # Prepare data for ChromaDB
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]
        metadatas = [serialize_metadata(chunk) for chunk in chunks]
        embedding_list = embeddings.tolist()

        # Add to collection (upsert semantics via IDs)
        # Note: metadatas type is compatible but mypy has strict typing
        self.collection.add(
            ids=ids,
            embeddings=embedding_list,
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )

        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        where: Where | None = None,
        where_document: WhereDocument | None = None,
    ) -> list[SearchResult]:
        """Perform semantic search over the corpus.

        Args:
            query_embedding: Query vector (same dimensions as stored embeddings).
            limit: Maximum number of results to return.
            where: Optional metadata filter (ChromaDB where clause).
            where_document: Optional document content filter.

        Returns:
            List of SearchResult objects ordered by similarity.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=limit,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )

        search_results: list[SearchResult] = []

        # ChromaDB returns nested lists (one per query)
        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for i, chunk_id in enumerate(ids):
                search_results.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        text=documents[i] if i < len(documents) else "",
                        distance=distances[i] if i < len(distances) else 0.0,
                        metadata=deserialize_metadata(metadatas[i]) if i < len(metadatas) else {},
                    )
                )

        return search_results

    def get_article_chunks(
        self,
        article_title: str,
        namespace: str = "Main",
    ) -> list[dict[str, Any]]:
        """Retrieve all chunks for a specific article.

        Args:
            article_title: Title of the article.
            namespace: Wiki namespace (Main, Library, Essays, ProleWiki).

        Returns:
            List of chunks with metadata, ordered by chunk_index.
        """
        # Build where clause with proper ChromaDB filter syntax
        # Note: Using cast to satisfy ChromaDB's complex Where type
        where_filter = cast(
            "Where",
            {
                "$and": [
                    {"article_title": {"$eq": article_title}},
                    {"namespace": {"$eq": namespace}},
                ]
            },
        )
        results = self.collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )

        chunks: list[dict[str, Any]] = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                chunk: dict[str, Any] = {"chunk_id": chunk_id}
                if results["documents"] and i < len(results["documents"]):
                    chunk["text"] = results["documents"][i]
                if results["metadatas"] and i < len(results["metadatas"]):
                    chunk.update(deserialize_metadata(results["metadatas"][i]))
                chunks.append(chunk)

        # Sort by chunk_index
        chunks.sort(key=lambda c: c.get("chunk_index", 0))
        return chunks

    def count(self) -> int:
        """Return total number of chunks in the collection."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete the entire collection. Use with caution."""
        self.client.delete_collection(self.config.collection_name)
        self.collection = self._get_or_create_collection()
