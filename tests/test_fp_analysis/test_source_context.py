"""Tests for source context extraction and function body finding."""

from __future__ import annotations

import pytest

from llm_client.fp_analysis.source_context import SourceContextExtractor


@pytest.fixture
def extractor(projects_dir) -> SourceContextExtractor:
    return SourceContextExtractor(project_root=projects_dir / "aria2")


class TestExtractFunctionBody:
    def test_extract_single_line_signature(self, extractor: SourceContextExtractor, wslay_lib):
        """wslay_queue_push has signature and '{' on the same line."""
        body = extractor.extract_function_body(
            str(wslay_lib / "wslay_queue.c"),
            "wslay_queue_push",
        )
        assert body is not None
        assert "wslay_queue_push" in body
        assert "struct wslay_queue_cell" in body
        assert body.rstrip().endswith("}")

    def test_extract_multi_line_signature(self, extractor: SourceContextExtractor, wslay_lib):
        """wslay_event_omsg_fragmented_init has multi-line signature."""
        body = extractor.extract_function_body(
            str(wslay_lib / "wslay_event.c"),
            "wslay_event_omsg_fragmented_init",
        )
        assert body is not None
        assert "wslay_event_omsg_fragmented_init" in body
        assert "omsg" in body
        assert body.rstrip().endswith("}")

    def test_extract_byte_chunk_init(self, extractor: SourceContextExtractor, wslay_lib):
        body = extractor.extract_function_body(
            str(wslay_lib / "wslay_event.c"),
            "wslay_event_byte_chunk_init",
        )
        assert body is not None
        assert "wslay_event_byte_chunk_init" in body
        assert "malloc" in body or "*chunk" in body

    def test_function_not_found(self, extractor: SourceContextExtractor, wslay_lib):
        body = extractor.extract_function_body(
            str(wslay_lib / "wslay_queue.c"),
            "nonexistent_function_xyz",
        )
        assert body is None

    def test_file_not_found(self, extractor: SourceContextExtractor):
        body = extractor.extract_function_body(
            "/nonexistent/path/file.c",
            "any_func",
        )
        assert body is None


class TestFindFunctionDefinition:
    def test_find_in_project(self, extractor: SourceContextExtractor, wslay_lib):
        file_path, body = extractor.find_function_definition("wslay_queue_push")
        assert file_path is not None
        assert "wslay_queue.c" in file_path
        assert body is not None
        assert "wslay_queue_push" in body

    def test_find_missing_function(self, extractor: SourceContextExtractor):
        file_path, body = extractor.find_function_definition(
            "this_function_does_not_exist_xyz"
        )
        assert file_path is None
        assert body is None

    def test_find_with_custom_search_roots(self, extractor: SourceContextExtractor, wslay_lib):
        file_path, body = extractor.find_function_definition(
            "wslay_queue_push",
            search_roots=[wslay_lib],
        )
        assert file_path is not None
        assert body is not None


class TestExtractCallChain:
    def test_no_calling_events_in_simple_report(self):
        """Smoke test: extract_call_chain returns empty for minimal data."""
        from llm_client.fp_analysis.models import ParsedReport, ReportMetadata

        parsed = ParsedReport(
            metadata=ReportMetadata(
                bug_type="memory_leak",
                bug_file="test.c",
                bug_line=10,
                function_name="test_func",
                path_length=3,
            ),
            path_events=[
                {"path_number": 1, "event_kind": "event", "description": "test", "line_number": 5},
            ],
            source_snippets={},
            raw_html_path="/dev/null",
        )
        extractor = SourceContextExtractor()
        result = extractor.extract_call_chain(parsed)
        assert result == {}
