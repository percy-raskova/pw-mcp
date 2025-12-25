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
        """Input directory should default to Path('extracted')."""
        parser = _create_parser()
        args = parser.parse_args(["chunk"])
        assert args.input == Path("extracted")

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
        # Content must have at least 10 words to avoid micro-chunk filtering
        content = (
            "== Introduction ==\n"
            "This is the introduction to the article content which provides important "
            "context and background information about the topic being discussed."
        )
        (input_dir / "article.txt").write_text(content)

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


# =============================================================================
# RESUME SUPPORT TESTS
# =============================================================================


class TestChunkResumeSupport:
    """Tests for chunk resume functionality (skip existing outputs)."""

    @pytest.mark.unit
    def test_chunk_skips_existing_jsonl(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should skip files that already have corresponding .jsonl output."""
        input_dir = tmp_path / "extracted"
        input_dir.mkdir()
        # Content must have at least 10 words to produce a chunk
        content = (
            "== Introduction ==\n"
            "This is the introduction to the article content which provides important "
            "context and background information about the topic being discussed."
        )
        (input_dir / "article1.txt").write_text(content)
        (input_dir / "article2.txt").write_text(content)
        (input_dir / "article3.txt").write_text(content)

        extracted_dir = tmp_path / "meta" / "articles"
        extracted_dir.mkdir(parents=True)
        for name in ["article1", "article2", "article3"]:
            (extracted_dir / f"{name}.json").write_text(
                json.dumps({"namespace": "Main", "categories": [], "internal_links": []})
            )

        output_dir = tmp_path / "chunks"
        output_dir.mkdir()
        # Pre-create output for article1 (this should be skipped)
        (output_dir / "article1.jsonl").write_text('{"chunk_id": "existing"}')

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
        # Should report 1 skipped file
        assert "Skipped" in captured.out
        assert "1" in captured.out

        # article1.jsonl should still contain original content (not overwritten)
        content_after = (output_dir / "article1.jsonl").read_text()
        assert "existing" in content_after

    @pytest.mark.unit
    def test_chunk_all_skipped_returns_success(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should return 0 when all files are already processed."""
        input_dir = tmp_path / "extracted"
        input_dir.mkdir()
        (input_dir / "article.txt").write_text("== Section ==\nContent.")

        extracted_dir = tmp_path / "meta" / "articles"
        extracted_dir.mkdir(parents=True)

        output_dir = tmp_path / "chunks"
        output_dir.mkdir()
        # Pre-create output for all input files
        (output_dir / "article.jsonl").write_text('{"chunk_id": "test"}')

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
        assert "No files to process" in captured.out or "all already chunked" in captured.out


# =============================================================================
# DIAGNOSE COMMAND TESTS
# =============================================================================


class TestDiagnoseCommand:
    """Tests for the diagnose subcommand."""

    @pytest.mark.unit
    def test_diagnose_subcommand_exists(self) -> None:
        """'diagnose' should be registered as a valid subcommand."""
        parser = _create_parser()
        args = parser.parse_args(["diagnose"])
        assert args.command == "diagnose"

    @pytest.mark.unit
    def test_diagnose_default_directories(self) -> None:
        """Diagnose should have sensible directory defaults."""
        parser = _create_parser()
        args = parser.parse_args(["diagnose"])
        assert args.extracted_dir == Path("extracted")
        assert args.chunks_dir == Path("chunks")
        assert args.embeddings_dir == Path("embeddings")

    @pytest.mark.unit
    def test_diagnose_reports_statistics(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Should report file counts for each stage."""
        # Create test directories
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()
        (extracted_dir / "file1.txt").write_text("content")
        (extracted_dir / "file2.txt").write_text("content")

        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        (chunks_dir / "file1.jsonl").write_text('{"chunk": 1}')

        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "diagnose",
                "--extracted-dir",
                str(extracted_dir),
                "--chunks-dir",
                str(chunks_dir),
                "--embeddings-dir",
                str(embeddings_dir),
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # Exit code 1 indicates issues found (missing chunks)
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Stage Statistics" in captured.out
        assert "extracted" in captured.out
        assert "chunks" in captured.out

    @pytest.mark.unit
    def test_diagnose_json_format(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """--format json should produce valid JSON output."""
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "diagnose",
                "--extracted-dir",
                str(extracted_dir),
                "--chunks-dir",
                str(tmp_path / "chunks"),
                "--embeddings-dir",
                str(tmp_path / "embeddings"),
                "--format",
                "json",
            ],
        ):
            with pytest.raises(SystemExit):
                main()

        captured = capsys.readouterr()
        # Should be valid JSON
        result = json.loads(captured.out)
        assert "stages" in result
        assert "healthy" in result


# =============================================================================
# REPAIR COMMAND TESTS
# =============================================================================


class TestRepairCommand:
    """Tests for the repair subcommand."""

    @pytest.mark.unit
    def test_repair_subcommand_exists(self) -> None:
        """'repair' should be registered as a valid subcommand."""
        parser = _create_parser()
        args = parser.parse_args(["repair", "--action", "delete-empty"])
        assert args.command == "repair"
        assert args.action == "delete-empty"

    @pytest.mark.unit
    def test_repair_requires_action(self) -> None:
        """repair command should require --action argument."""
        parser = _create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["repair"])

    @pytest.mark.unit
    def test_repair_dry_run_does_not_modify(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--dry-run should not actually delete files."""
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()
        empty_file = extracted_dir / "empty.txt"
        empty_file.write_text("")  # Create empty file

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "repair",
                "--extracted-dir",
                str(extracted_dir),
                "--chunks-dir",
                str(tmp_path / "chunks"),
                "--embeddings-dir",
                str(tmp_path / "embeddings"),
                "--action",
                "delete-empty",
                "--dry-run",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        # File should still exist (dry run)
        assert empty_file.exists()

        captured = capsys.readouterr()
        assert "Would delete" in captured.out
        assert "Dry run complete" in captured.out

    @pytest.mark.unit
    def test_repair_delete_empty_removes_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--action delete-empty should remove 0-byte files."""
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()
        empty_file = extracted_dir / "empty.txt"
        empty_file.write_text("")  # Create empty file
        non_empty = extracted_dir / "content.txt"
        non_empty.write_text("Some content")

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "repair",
                "--extracted-dir",
                str(extracted_dir),
                "--chunks-dir",
                str(tmp_path / "chunks"),
                "--embeddings-dir",
                str(tmp_path / "embeddings"),
                "--action",
                "delete-empty",
                "--stage",
                "extracted",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        # Empty file should be deleted
        assert not empty_file.exists()
        # Non-empty file should remain
        assert non_empty.exists()

        captured = capsys.readouterr()
        assert "Deleted" in captured.out


# =============================================================================
# CONTENT VALIDATION TESTS (TDD Red Phase)
# =============================================================================


class TestDiagnoseContentValidation:
    """Tests for content validation in diagnose command."""

    @pytest.mark.unit
    def test_validate_content_flag_exists(self) -> None:
        """--validate-content should be a valid flag for diagnose."""
        parser = _create_parser()
        args = parser.parse_args(["diagnose", "--validate-content"])
        assert args.validate_content is True

    @pytest.mark.unit
    def test_validate_content_default_false(self) -> None:
        """--validate-content should default to False."""
        parser = _create_parser()
        args = parser.parse_args(["diagnose"])
        assert args.validate_content is False

    @pytest.mark.unit
    def test_duplication_threshold_parameter(self) -> None:
        """--duplication-threshold should accept float value."""
        parser = _create_parser()
        args = parser.parse_args(
            ["diagnose", "--validate-content", "--duplication-threshold", "0.05"]
        )
        assert args.duplication_threshold == 0.05

    @pytest.mark.unit
    def test_duplication_threshold_default(self) -> None:
        """--duplication-threshold should default to 0.1 (10%)."""
        parser = _create_parser()
        args = parser.parse_args(["diagnose"])
        assert args.duplication_threshold == 0.1

    @pytest.mark.unit
    def test_content_validation_detects_duplicates(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Content validation should detect files with excessive duplicates."""
        # Create test directories
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()

        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        # Create JSONL with 50% duplicate chunks
        corrupt_jsonl = chunks_dir / "corrupt.jsonl"
        chunks_data = [
            {"text": "unique chunk 1", "chunk_id": "test#0"},
            {"text": "duplicate text", "chunk_id": "test#1"},
            {"text": "duplicate text", "chunk_id": "test#2"},  # Duplicate
            {"text": "unique chunk 2", "chunk_id": "test#3"},
        ]
        corrupt_jsonl.write_text("\n".join(json.dumps(c) for c in chunks_data))

        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "diagnose",
                "--extracted-dir",
                str(extracted_dir),
                "--chunks-dir",
                str(chunks_dir),
                "--embeddings-dir",
                str(embeddings_dir),
                "--validate-content",
                "--duplication-threshold",
                "0.1",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # Exit code 1 indicates issues found
            assert exc_info.value.code == 1

        captured = capsys.readouterr()
        assert "Content" in captured.out or "Duplicate" in captured.out
        assert "corrupt" in captured.out.lower()

    @pytest.mark.unit
    def test_content_validation_respects_threshold(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Files below threshold should not be flagged."""
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()
        # Create matching extracted file to avoid orphan detection
        (extracted_dir / "mild.txt").write_text("some content")

        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        # Create JSONL with 5% duplicate chunks (1 duplicate in 20)
        mild_jsonl = chunks_dir / "mild.jsonl"
        chunks_data = [{"text": f"unique chunk {i}", "chunk_id": f"test#{i}"} for i in range(19)]
        chunks_data.append({"text": "unique chunk 0", "chunk_id": "test#19"})  # 1 duplicate = 5%
        mild_jsonl.write_text("\n".join(json.dumps(c) for c in chunks_data))

        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()
        # Create matching embedding file to avoid missing embedding detection
        (embeddings_dir / "mild.npy").write_bytes(b"fake embedding")

        # With 10% threshold, 5% duplicates should NOT be flagged
        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "diagnose",
                "--extracted-dir",
                str(extracted_dir),
                "--chunks-dir",
                str(chunks_dir),
                "--embeddings-dir",
                str(embeddings_dir),
                "--validate-content",
                "--duplication-threshold",
                "0.1",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # Should be healthy (5% < 10% threshold)
            assert exc_info.value.code == 0

    @pytest.mark.unit
    def test_content_validation_json_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """JSON output should include content_issues field when validating content."""
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()

        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        # Create JSONL with duplicates
        corrupt_jsonl = chunks_dir / "corrupt.jsonl"
        chunks_data = [
            {"text": "same text", "chunk_id": "test#0"},
            {"text": "same text", "chunk_id": "test#1"},
        ]
        corrupt_jsonl.write_text("\n".join(json.dumps(c) for c in chunks_data))

        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "diagnose",
                "--extracted-dir",
                str(extracted_dir),
                "--chunks-dir",
                str(chunks_dir),
                "--embeddings-dir",
                str(embeddings_dir),
                "--validate-content",
                "--format",
                "json",
            ],
        ):
            with pytest.raises(SystemExit):
                main()

        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "content_issues" in result
        assert "excessive_duplicates" in result["content_issues"]


class TestRepairDeleteCorrupt:
    """Tests for delete-corrupt repair action."""

    @pytest.mark.unit
    def test_delete_corrupt_action_exists(self) -> None:
        """delete-corrupt should be a valid action for repair."""
        parser = _create_parser()
        args = parser.parse_args(["repair", "--action", "delete-corrupt"])
        assert args.action == "delete-corrupt"

    @pytest.mark.unit
    def test_repair_duplication_threshold_parameter(self) -> None:
        """Repair should accept --duplication-threshold for delete-corrupt."""
        parser = _create_parser()
        args = parser.parse_args(
            ["repair", "--action", "delete-corrupt", "--duplication-threshold", "0.2"]
        )
        assert args.duplication_threshold == 0.2

    @pytest.mark.unit
    def test_delete_corrupt_dry_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--dry-run should not delete corrupt files."""
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()

        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        corrupt_jsonl = chunks_dir / "corrupt.jsonl"
        chunks_data = [{"text": "same", "chunk_id": f"test#{i}"} for i in range(10)]
        corrupt_jsonl.write_text("\n".join(json.dumps(c) for c in chunks_data))

        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()
        corrupt_npy = embeddings_dir / "corrupt.npy"
        corrupt_npy.write_bytes(b"fake embedding data")

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "repair",
                "--extracted-dir",
                str(extracted_dir),
                "--chunks-dir",
                str(chunks_dir),
                "--embeddings-dir",
                str(embeddings_dir),
                "--action",
                "delete-corrupt",
                "--dry-run",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        # Files should still exist (dry run)
        assert corrupt_jsonl.exists()
        assert corrupt_npy.exists()

        captured = capsys.readouterr()
        assert "Would delete" in captured.out

    @pytest.mark.unit
    def test_delete_corrupt_removes_both_chunks_and_embeddings(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """delete-corrupt should remove corrupt JSONL and corresponding NPY."""
        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()

        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        corrupt_jsonl = chunks_dir / "corrupt.jsonl"
        # 100% duplicates - definitely corrupt
        chunks_data = [{"text": "same", "chunk_id": f"test#{i}"} for i in range(10)]
        corrupt_jsonl.write_text("\n".join(json.dumps(c) for c in chunks_data))

        # Also create a healthy file that should NOT be deleted
        healthy_jsonl = chunks_dir / "healthy.jsonl"
        healthy_data = [{"text": f"unique {i}", "chunk_id": f"test#{i}"} for i in range(10)]
        healthy_jsonl.write_text("\n".join(json.dumps(c) for c in healthy_data))

        embeddings_dir = tmp_path / "embeddings"
        embeddings_dir.mkdir()
        corrupt_npy = embeddings_dir / "corrupt.npy"
        corrupt_npy.write_bytes(b"fake embedding data")
        healthy_npy = embeddings_dir / "healthy.npy"
        healthy_npy.write_bytes(b"real embedding data")

        with patch(
            "sys.argv",
            [
                "pw-ingest",
                "repair",
                "--extracted-dir",
                str(extracted_dir),
                "--chunks-dir",
                str(chunks_dir),
                "--embeddings-dir",
                str(embeddings_dir),
                "--action",
                "delete-corrupt",
            ],
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

        # Corrupt files should be deleted
        assert not corrupt_jsonl.exists()
        assert not corrupt_npy.exists()

        # Healthy files should remain
        assert healthy_jsonl.exists()
        assert healthy_npy.exists()

        captured = capsys.readouterr()
        assert "Deleted" in captured.out
