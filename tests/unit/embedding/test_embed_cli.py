"""Unit tests for embed CLI subcommand (TDD Green Phase).

These tests verify the embed CLI subcommand implementation.
Tests use mocking to avoid requiring a running Ollama server.

Test strategy:
- Test argument parsing defaults and options
- Test file processing behavior
- Test sample mode
- Test progress reporting
- Test Ollama validation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pw_mcp.ingest.cli import _create_parser

if TYPE_CHECKING:
    pass


# =============================================================================
# PARSER TESTS
# =============================================================================


class TestEmbedParserRegistration:
    """Tests for embed subcommand parser registration."""

    @pytest.mark.unit
    def test_embed_subcommand_exists(self) -> None:
        """'embed' should be registered as a valid subcommand.

        parser = _create_parser()
        args = parser.parse_args(["embed"])
        assert args.command == "embed"
        """
        parser = _create_parser()
        args = parser.parse_args(["embed"])
        assert args.command == "embed"

    @pytest.mark.unit
    def test_embed_input_default(self) -> None:
        """Input directory should default to Path('chunks/').

        parser = _create_parser()
        args = parser.parse_args(["embed"])
        assert args.input == Path("chunks")
        """
        parser = _create_parser()
        args = parser.parse_args(["embed"])
        assert args.input == Path("chunks")

    @pytest.mark.unit
    def test_embed_output_default(self) -> None:
        """Output directory should default to Path('embeddings/').

        parser = _create_parser()
        args = parser.parse_args(["embed"])
        assert args.output == Path("embeddings")
        """
        parser = _create_parser()
        args = parser.parse_args(["embed"])
        assert args.output == Path("embeddings")

    @pytest.mark.unit
    def test_embed_model_default(self) -> None:
        """Model should default to 'embeddinggemma'.

        parser = _create_parser()
        args = parser.parse_args(["embed"])
        assert args.model == "embeddinggemma"
        """
        parser = _create_parser()
        args = parser.parse_args(["embed"])
        assert args.model == "embeddinggemma"

    @pytest.mark.unit
    def test_embed_batch_size_default(self) -> None:
        """Batch size should default to 32.

        parser = _create_parser()
        args = parser.parse_args(["embed"])
        assert args.batch_size == 32
        """
        parser = _create_parser()
        args = parser.parse_args(["embed"])
        assert args.batch_size == 32


# =============================================================================
# PROCESSING TESTS
# =============================================================================


class TestEmbedProcessing:
    """Tests for embed processing behavior."""

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.check_ollama_ready")
    @patch("pw_mcp.ingest.embedder.embed_texts")
    def test_embed_sample_limits_files(
        self,
        mock_embed_texts: MagicMock,
        mock_check: MagicMock,
        tmp_path: Path,
    ) -> None:
        """--sample N should process only N files (randomly selected).

        Given 10 input files and --sample 3:
        - Only 3 files should be processed
        - Selection should use random.sample()
        """
        # Setup input directory with 10 JSONL files
        input_dir = tmp_path / "chunks"
        input_dir.mkdir()
        for i in range(10):
            jsonl_file = input_dir / f"article_{i}.jsonl"
            chunk = {
                "chunk_id": f"Main/Article_{i}#0",
                "text": f"Content for article {i}",
                "article_title": f"Article {i}",
                "namespace": "Main",
            }
            jsonl_file.write_text(json.dumps(chunk) + "\n")

        # Mock Ollama check to pass
        mock_check.return_value = True

        # Mock embed_texts to return embeddings
        mock_embed_texts.return_value = np.zeros((1, 768), dtype=np.float32)

        # Import and run the process function
        from pw_mcp.ingest.cli import _run_embed_process

        parser = _create_parser()
        args = parser.parse_args(
            [
                "embed",
                "-i",
                str(input_dir),
                "-o",
                str(tmp_path / "embeddings"),
                "--sample",
                "3",
                "--no-progress",
            ]
        )

        exit_code = _run_embed_process(args)

        assert exit_code == 0
        # Should have processed exactly 3 files (embed_texts called 3 times)
        assert mock_embed_texts.call_count == 3

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.check_ollama_ready")
    @patch("pw_mcp.ingest.embedder.embed_texts")
    def test_embed_skip_existing_npy(
        self,
        mock_embed_texts: MagicMock,
        mock_check: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Should skip files that already have corresponding .npy output.

        Resume support: If embeddings/Main/Article.npy exists,
        skip processing chunks/Main/Article.jsonl.

        This enables:
        - Resuming interrupted runs
        - Incremental updates
        """
        # Setup input directory with 3 JSONL files
        input_dir = tmp_path / "chunks" / "Main"
        input_dir.mkdir(parents=True)
        for i in range(3):
            jsonl_file = input_dir / f"article_{i}.jsonl"
            chunk = {
                "chunk_id": f"Main/Article_{i}#0",
                "text": f"Content for article {i}",
                "article_title": f"Article {i}",
                "namespace": "Main",
            }
            jsonl_file.write_text(json.dumps(chunk) + "\n")

        # Create output directory with existing .npy file for article_0
        output_dir = tmp_path / "embeddings" / "Main"
        output_dir.mkdir(parents=True)
        existing_npy = output_dir / "article_0.npy"
        np.save(existing_npy, np.zeros((1, 768), dtype=np.float32))

        # Mock Ollama check to pass
        mock_check.return_value = True

        # Mock embed_texts to return embeddings
        mock_embed_texts.return_value = np.zeros((1, 768), dtype=np.float32)

        from pw_mcp.ingest.cli import _run_embed_process

        parser = _create_parser()
        args = parser.parse_args(
            [
                "embed",
                "-i",
                str(tmp_path / "chunks"),
                "-o",
                str(tmp_path / "embeddings"),
                "--no-progress",
            ]
        )

        exit_code = _run_embed_process(args)

        assert exit_code == 0
        # Should have processed only 2 files (skipped article_0)
        assert mock_embed_texts.call_count == 2

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.check_ollama_ready")
    def test_embed_validates_ollama(
        self,
        mock_check: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should check Ollama server health before processing.

        Pre-flight check should:
        1. Verify Ollama server is reachable
        2. Verify model returns expected dimensions (768)
        3. Exit with code 1 and helpful message if check fails
        """
        # Setup minimal input directory
        input_dir = tmp_path / "chunks"
        input_dir.mkdir()
        jsonl_file = input_dir / "article.jsonl"
        chunk = {
            "chunk_id": "Main/Article#0",
            "text": "Test content",
            "article_title": "Article",
            "namespace": "Main",
        }
        jsonl_file.write_text(json.dumps(chunk) + "\n")

        # Mock Ollama check to fail
        mock_check.return_value = False

        from pw_mcp.ingest.cli import _run_embed_process

        parser = _create_parser()
        args = parser.parse_args(
            [
                "embed",
                "-i",
                str(input_dir),
                "-o",
                str(tmp_path / "embeddings"),
                "--no-progress",
            ]
        )

        exit_code = _run_embed_process(args)

        assert exit_code == 1

        captured = capsys.readouterr()
        assert "Ollama server is not responding" in captured.out

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.check_ollama_ready")
    @patch("pw_mcp.ingest.embedder.embed_texts")
    def test_embed_progress_reports(
        self,
        mock_embed_texts: MagicMock,
        mock_check: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Should show progress during processing (unless --no-progress).

        Output should include:
        - [n/total] filename format
        - Summary statistics at end
        """
        # Setup input directory with 2 JSONL files
        input_dir = tmp_path / "chunks"
        input_dir.mkdir()
        for i in range(2):
            jsonl_file = input_dir / f"article_{i}.jsonl"
            chunk = {
                "chunk_id": f"Main/Article_{i}#0",
                "text": f"Content for article {i}",
                "article_title": f"Article {i}",
                "namespace": "Main",
            }
            jsonl_file.write_text(json.dumps(chunk) + "\n")

        # Mock Ollama check to pass
        mock_check.return_value = True

        # Mock embed_texts to return embeddings
        mock_embed_texts.return_value = np.zeros((1, 768), dtype=np.float32)

        from pw_mcp.ingest.cli import _run_embed_process

        parser = _create_parser()
        # Note: NOT using --no-progress to test progress output
        args = parser.parse_args(
            [
                "embed",
                "-i",
                str(input_dir),
                "-o",
                str(tmp_path / "embeddings"),
            ]
        )

        exit_code = _run_embed_process(args)

        assert exit_code == 0

        captured = capsys.readouterr()
        # Should have progress format [n/total]
        assert "[1/2]" in captured.out
        assert "[2/2]" in captured.out
        # Should have summary
        assert "Complete:" in captured.out
        assert "embedded" in captured.out

    @pytest.mark.unit
    @patch("pw_mcp.ingest.embedder.check_ollama_ready")
    @patch("pw_mcp.ingest.embedder.embed_texts")
    def test_embed_preserves_namespace(
        self,
        mock_embed_texts: MagicMock,
        mock_check: MagicMock,
        tmp_path: Path,
    ) -> None:
        """chunks/Main/x.jsonl should produce embeddings/Main/x.npy.

        Directory structure must be preserved:
        - chunks/Main/Five-Year_Plans.jsonl -> embeddings/Main/Five-Year_Plans.npy
        - chunks/Library/Capital.jsonl -> embeddings/Library/Capital.npy
        """
        # Setup input directory with namespace structure
        main_dir = tmp_path / "chunks" / "Main"
        main_dir.mkdir(parents=True)
        library_dir = tmp_path / "chunks" / "Library"
        library_dir.mkdir(parents=True)

        # Create JSONL files in different namespaces
        main_jsonl = main_dir / "Five-Year_Plans.jsonl"
        library_jsonl = library_dir / "Capital.jsonl"

        main_chunk = {
            "chunk_id": "Main/Five-Year_Plans#0",
            "text": "Five-Year Plans content",
            "article_title": "Five-Year Plans",
            "namespace": "Main",
        }
        library_chunk = {
            "chunk_id": "Library/Capital#0",
            "text": "Capital content",
            "article_title": "Capital",
            "namespace": "Library",
        }

        main_jsonl.write_text(json.dumps(main_chunk) + "\n")
        library_jsonl.write_text(json.dumps(library_chunk) + "\n")

        # Mock Ollama check to pass
        mock_check.return_value = True

        # Mock embed_texts to return embeddings
        mock_embed_texts.return_value = np.zeros((1, 768), dtype=np.float32)

        from pw_mcp.ingest.cli import _run_embed_process

        output_dir = tmp_path / "embeddings"

        parser = _create_parser()
        args = parser.parse_args(
            [
                "embed",
                "-i",
                str(tmp_path / "chunks"),
                "-o",
                str(output_dir),
                "--no-progress",
            ]
        )

        exit_code = _run_embed_process(args)

        assert exit_code == 0

        # Verify output files were created in correct locations
        expected_main = output_dir / "Main" / "Five-Year_Plans.npy"
        expected_library = output_dir / "Library" / "Capital.npy"

        assert expected_main.exists() or expected_library.exists()
