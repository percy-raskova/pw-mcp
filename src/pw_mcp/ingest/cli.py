"""CLI for corpus ingestion pipeline."""

import argparse
from pathlib import Path


def main() -> None:
    """Run the ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest ProleWiki corpus into ChromaDB")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("prolewiki-exports"),
        help="Path to ProleWiki exports directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("chroma_data"),
        help="Path to ChromaDB data directory",
    )
    parser.add_argument(
        "--semantic-linebreak",
        action="store_true",
        help="Apply semantic linebreaking to source files in-place",
    )

    args = parser.parse_args()

    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Semantic linebreak: {args.semantic_linebreak}")
    print("Ingestion pipeline not yet implemented.")


if __name__ == "__main__":
    main()
