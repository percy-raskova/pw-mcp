"""Unit tests for chunk CLI subcommand (TDD Red Phase).

These tests define the expected interface for the chunk CLI subcommand.
Tests should fail until the CLI implementation is complete.

Test strategy:
- Test argument parsing defaults and options
- Test file processing behavior
- Test sample mode
- Test progress reporting
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pw_mcp.ingest.cli import _create_parser, main

# =============================================================================
# PARSER TESTS
# =============================================================================


class TestChunkParserRegistration:
    """Tests for chunk subcommand parser registration."""

    @pytest.mark.unit
    def test_chunk_subcommand_exists(self) -> None:
        """'chunk' should be registered as a valid subcommand."""
        parser = _create_parser()
        # Parse with 'chunk' subcommand - should not raise
        args = parser.parse_args(["chunk"])
        assert args.command == "chunk"

    @pytest.mark.unit
    def test_chunk_input_default(self) -> None:
        """Input directory should default to Path('sembr')."""
        parser = _create_parser()
        args = parser.parse_args(["chunk"])
        assert args.input == Path("sembr")

    @pytest.mark.unit
    def test_chunk_output_default(self) -> None:
        """Output directory should default to Path('chunks')."""
        parser = _create_parser()
        args = parser.parse_args(["chunk"])
        assert args.output == Path("chunks")

    @pytest.mark.unit
    def test_chunk_extracted_default(self) -> None:
        """Extracted metadata directory should default to Path('extracted/articles')."""
        parser = _create_parser()
        args = parser.parse_args(["chunk"])
        assert args.extracted == Path("extracted/articles")

    @pytest.mark.unit
    def test_chunk_sample_optional(self) -> None:
        """--sample should accept int and default to None."""
        parser = _create_parser()

        # Default is None
        args_default = parser.parse_args(["chunk"])
        assert args_default.sample is None

        # Can provide int
        args_with_sample = parser.parse_args(["chunk", "--sample", "50"])
        assert args_with_sample.sample == 50

    @pytest.mark.unit
    def test_chunk_no_progress_flag(self) -> None:
        """--no-progress should be a boolean flag."""
        parser = _create_parser()

        # Default is False (show progress)
        args_default = parser.parse_args(["chunk"])
        assert args_default.no_progress is False

        # Flag sets to True
        args_no_progress = parser.parse_args(["chunk", "--no-progress"])
        assert args_no_progress.no_progress is True

    @pytest.mark.unit
    def test_chunk_target_tokens_default(self) -> None:
        """--target-tokens should default to 600."""
        parser = _create_parser()
        args = parser.parse_args(["chunk"])
        assert args.target_tokens == 600

    @pytest.mark.unit
    def test_chunk_max_tokens_default(self) -> None:
        """--max-tokens should default to 1000."""
        parser = _create_parser()
        args = parser.parse_args(["chunk"])
        assert args.max_tokens == 1000


# =============================================================================
# PROCESSING TESTS
# =============================================================================


class TestChunkProcessing:
    """Tests for chunk processing behavior."""

    @pytest.mark.unit
    def test_chunk_process_validates_input_dir(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should exit 1 if input directory does not exist."""
        nonexistent_input = tmp_path / "nonexistent_sembr"

        with patch("sys.argv", ["pw-ingest", "chunk", "-i", str(nonexistent_input)]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "does not exist" in captured.out.lower() or "error" in captured.out.lower()

    @pytest.mark.unit
    def test_chunk_process_discovers_txt_files(self, tmp_path: Path) -> None:
        """Should discover all .txt files using rglob('*.txt')."""
        input_dir = tmp_path / "sembr"
        input_dir.mkdir()

        # Create nested .txt files
        (input_dir / "Main").mkdir()
        (input_dir / "Main" / "article1.txt").write_text("== Header ==\nContent.")
        (input_dir / "Main" / "article2.txt").write_text("== Header ==\nMore content.")
        (input_dir / "Library").mkdir()
        (input_dir / "Library" / "book.txt").write_text("== Chapter ==\nBook content.")

        # Create metadata directory
        extracted_dir = tmp_path / "extracted" / "articles"
        extracted_dir.mkdir(parents=True)
        (extracted_dir / "Main").mkdir()
        (extracted_dir / "Library").mkdir()

        # Create minimal metadata files
        minimal_meta: dict[str, Any] = {
            "namespace": "Main",
            "categories": [],
            "internal_links": [],
            "is_stub": False,
            "citation_needed_count": 0,
            "has_blockquote": False,
        }
        (extracted_dir / "Main" / "article1.json").write_text(json.dumps(minimal_meta))
        (extracted_dir / "Main" / "article2.json").write_text(json.dumps(minimal_meta))
        (extracted_dir / "Library" / "book.json").write_text(
            json.dumps({**minimal_meta, "namespace": "Library"})
        )

        output_dir = tmp_path / "chunks"

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "chunk",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-e",
                str(extracted_dir),
                "--no-progress",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # Should succeed
            assert exc_info.value.code == 0

        # Should have created 3 output files
        output_files = list(output_dir.rglob("*.jsonl"))
        assert len(output_files) == 3

    @pytest.mark.unit
    def test_chunk_process_creates_output_dir(self, tmp_path: Path) -> None:
        """Should create output directory if it does not exist (mkdir -p)."""
        input_dir = tmp_path / "sembr"
        input_dir.mkdir()
        (input_dir / "test.txt").write_text("== Section ==\nSome content here.")

        extracted_dir = tmp_path / "extracted" / "articles"
        extracted_dir.mkdir(parents=True)
        (extracted_dir / "test.json").write_text(
            json.dumps({"namespace": "Main", "categories": [], "internal_links": []})
        )

        # Output dir does not exist
        output_dir = tmp_path / "deeply" / "nested" / "chunks"
        assert not output_dir.exists()

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "chunk",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-e",
                str(extracted_dir),
                "--no-progress",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        # Output directory should now exist
        assert output_dir.exists()

    @pytest.mark.unit
    def test_chunk_process_produces_jsonl(self, tmp_path: Path) -> None:
        """Should produce .jsonl files from .txt input files."""
        input_dir = tmp_path / "sembr"
        input_dir.mkdir()
        (input_dir / "article.txt").write_text("== Introduction ==\nThis is the content.")

        extracted_dir = tmp_path / "extracted" / "articles"
        extracted_dir.mkdir(parents=True)
        (extracted_dir / "article.json").write_text(
            json.dumps({"namespace": "Main", "categories": [], "internal_links": []})
        )

        output_dir = tmp_path / "chunks"

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "chunk",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-e",
                str(extracted_dir),
                "--no-progress",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        # Check that .jsonl file was created
        output_file = output_dir / "article.jsonl"
        assert output_file.exists()

        # Validate JSONL format (one JSON object per line)
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) >= 1
        for line in lines:
            parsed = json.loads(line)  # Should not raise
            assert "chunk_id" in parsed
            assert "text" in parsed

    @pytest.mark.unit
    def test_chunk_process_preserves_namespace(self, tmp_path: Path) -> None:
        """sembr/Main/x.txt should produce chunks/Main/x.jsonl."""
        input_dir = tmp_path / "sembr"
        (input_dir / "Main").mkdir(parents=True)
        (input_dir / "Main" / "article.txt").write_text("== Section ==\nContent here.")

        extracted_dir = tmp_path / "extracted" / "articles"
        (extracted_dir / "Main").mkdir(parents=True)
        (extracted_dir / "Main" / "article.json").write_text(
            json.dumps({"namespace": "Main", "categories": [], "internal_links": []})
        )

        output_dir = tmp_path / "chunks"

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "chunk",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-e",
                str(extracted_dir),
                "--no-progress",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        # Output should preserve namespace directory structure
        expected_output = output_dir / "Main" / "article.jsonl"
        assert expected_output.exists()

    @pytest.mark.unit
    def test_chunk_process_handles_missing_meta(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should warn and continue when metadata JSON is missing."""
        input_dir = tmp_path / "sembr"
        input_dir.mkdir()
        (input_dir / "no_meta.txt").write_text("== Section ==\nContent without metadata.")

        extracted_dir = tmp_path / "extracted" / "articles"
        extracted_dir.mkdir(parents=True)
        # Deliberately not creating the metadata file

        output_dir = tmp_path / "chunks"

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "chunk",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-e",
                str(extracted_dir),
                "--no-progress",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # Should still succeed (graceful handling)
            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        # Should have logged a warning about missing metadata
        assert "warning" in captured.out.lower() or "missing" in captured.out.lower()

        # Output file should still be created with minimal metadata
        output_file = output_dir / "no_meta.jsonl"
        assert output_file.exists()


# =============================================================================
# SAMPLE MODE TESTS
# =============================================================================


class TestChunkSampleMode:
    """Tests for --sample mode behavior."""

    @pytest.mark.unit
    def test_chunk_sample_limits_files(self, tmp_path: Path) -> None:
        """--sample N should process only N files."""
        input_dir = tmp_path / "sembr"
        input_dir.mkdir()

        # Create 10 input files
        for i in range(10):
            (input_dir / f"article_{i}.txt").write_text(f"== Section {i} ==\nContent {i}.")

        extracted_dir = tmp_path / "extracted" / "articles"
        extracted_dir.mkdir(parents=True)
        for i in range(10):
            (extracted_dir / f"article_{i}.json").write_text(
                json.dumps({"namespace": "Main", "categories": [], "internal_links": []})
            )

        output_dir = tmp_path / "chunks"

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "chunk",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-e",
                str(extracted_dir),
                "--sample",
                "3",
                "--no-progress",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        # Should have created exactly 3 output files
        output_files = list(output_dir.rglob("*.jsonl"))
        assert len(output_files) == 3

    @pytest.mark.unit
    def test_chunk_sample_random_selection(self, tmp_path: Path) -> None:
        """--sample should use random.sample for file selection."""
        input_dir = tmp_path / "sembr"
        input_dir.mkdir()

        # Create 10 input files
        for i in range(10):
            (input_dir / f"article_{i}.txt").write_text(f"== Section {i} ==\nContent {i}.")

        extracted_dir = tmp_path / "extracted" / "articles"
        extracted_dir.mkdir(parents=True)
        for i in range(10):
            (extracted_dir / f"article_{i}.json").write_text(
                json.dumps({"namespace": "Main", "categories": [], "internal_links": []})
            )

        output_dir = tmp_path / "chunks"

        # Mock random.sample to verify it's called
        with patch("random.sample") as mock_sample:
            # Make random.sample return the first 3 files
            def sample_side_effect(population: list[Path], k: int) -> list[Path]:
                return list(population)[:k]

            mock_sample.side_effect = sample_side_effect

            with patch(
                "sys.argv",
                [
                    "pw-ingest",
                    "chunk",
                    "-i",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                    "-e",
                    str(extracted_dir),
                    "--sample",
                    "3",
                    "--no-progress",
                ],
            ):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                assert exc_info.value.code == 0

            # random.sample should have been called
            mock_sample.assert_called_once()
            call_args = mock_sample.call_args
            assert call_args[0][1] == 3  # k=3


# =============================================================================
# PROGRESS TESTS
# =============================================================================


class TestChunkProgress:
    """Tests for progress reporting behavior."""

    @pytest.mark.unit
    def test_chunk_progress_reports_status(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should show [n/total] filename during processing."""
        input_dir = tmp_path / "sembr"
        input_dir.mkdir()
        (input_dir / "article1.txt").write_text("== Section ==\nContent 1.")
        (input_dir / "article2.txt").write_text("== Section ==\nContent 2.")

        extracted_dir = tmp_path / "extracted" / "articles"
        extracted_dir.mkdir(parents=True)
        for name in ["article1", "article2"]:
            (extracted_dir / f"{name}.json").write_text(
                json.dumps({"namespace": "Main", "categories": [], "internal_links": []})
            )

        output_dir = tmp_path / "chunks"

        # Do NOT use --no-progress so progress is shown
        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "chunk",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-e",
                str(extracted_dir),
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        # Should show progress like "[1/2]" or "[2/2]"
        assert "[1/" in captured.out or "[2/" in captured.out

    @pytest.mark.unit
    def test_chunk_no_progress_suppresses(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--no-progress should suppress progress output."""
        input_dir = tmp_path / "sembr"
        input_dir.mkdir()
        (input_dir / "article1.txt").write_text("== Section ==\nContent 1.")
        (input_dir / "article2.txt").write_text("== Section ==\nContent 2.")

        extracted_dir = tmp_path / "extracted" / "articles"
        extracted_dir.mkdir(parents=True)
        for name in ["article1", "article2"]:
            (extracted_dir / f"{name}.json").write_text(
                json.dumps({"namespace": "Main", "categories": [], "internal_links": []})
            )

        output_dir = tmp_path / "chunks"

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "chunk",
                "-i",
                str(input_dir),
                "-o",
                str(output_dir),
                "-e",
                str(extracted_dir),
                "--no-progress",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        captured = capsys.readouterr()
        # Should NOT show progress like "[1/2]"
        assert "[1/" not in captured.out
        assert "[2/" not in captured.out
