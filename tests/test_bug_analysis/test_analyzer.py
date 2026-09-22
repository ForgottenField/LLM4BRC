"""Tests for the BugAnalyzer orchestrator."""

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from llm_client.base import LLMConfig
from llm_client.bug_analysis.analyzer import BugAnalyzer, _extract_json
from llm_client.bug_analysis.schemas import (
    Severity,
)

MOCK_SARIF_JSON = """{
  "findings": [
    {
      "bug_type": "CWE-120: Buffer Overflow",
      "severity": "high",
      "affected_location": {
        "file": "src/network.c",
        "line": 87,
        "end_line": null,
        "column": 5,
        "end_column": 27,
        "snippet": "memcpy(dst, src, len);"
      },
      "description": "Buffer overflow via memcpy without bounds check.",
      "root_cause": "No bounds checking on destination buffer.",
      "suggested_fix": "Use snprintf or strncpy with size limit."
    }
  ]
}"""

MOCK_EMPTY_JSON = """{"findings": []}"""


class TestBugAnalyzer:
    """Tests for BugAnalyzer core functionality."""

    @patch("llm_client.bug_analysis.analyzer.ProviderFactory")
    def test_analyze_returns_analysis_result(self, MockFactory, sarif_report):
        """Happy path: SARIF input produces a valid AnalysisResult."""
        mock_provider = MockFactory.create.return_value
        mock_provider.send.return_value.content = MOCK_SARIF_JSON
        mock_provider.send.return_value.model = "deepseek-v4-pro"
        mock_provider.send.return_value.provider = "deepseek"

        config = LLMConfig(provider_name="deepseek", api_key="sk-test")
        analyzer = BugAnalyzer(config)

        result = analyzer.analyze(sarif_report)
        assert result.metadata.total_findings == 1
        assert result.findings[0].bug_type == "CWE-120: Buffer Overflow"
        assert result.findings[0].severity == Severity.HIGH
        assert result.findings[0].affected_location is not None
        assert result.findings[0].affected_location.file == "src/network.c"
        # original_report is stripped — compare without trailing whitespace
        assert result.original_report == sarif_report.strip()

    @patch("llm_client.bug_analysis.analyzer.ProviderFactory")
    def test_analyze_empty_report_raises(self, MockFactory):
        config = LLMConfig(provider_name="deepseek", api_key="sk-test")
        analyzer = BugAnalyzer(config)

        with pytest.raises(ValidationError, match="at least 1 character"):
            analyzer.analyze("")

    @patch("llm_client.bug_analysis.analyzer.ProviderFactory")
    def test_analyze_plaintext(self, MockFactory, plaintext_report):
        """Plain text input should still work."""
        mock_provider = MockFactory.create.return_value
        mock_provider.send.return_value.content = MOCK_SARIF_JSON
        mock_provider.send.return_value.model = "deepseek-v4-pro"

        config = LLMConfig(provider_name="deepseek", api_key="sk-test")
        analyzer = BugAnalyzer(config)
        result = analyzer.analyze(plaintext_report)

        assert result.metadata.total_findings == 1
        assert result.metadata.input_format.value == "plain_text"

    @patch("llm_client.bug_analysis.analyzer.ProviderFactory")
    def test_analyze_no_findings(self, MockFactory, sarif_report):
        """Empty findings list from LLM should still produce a valid result."""
        mock_provider = MockFactory.create.return_value
        mock_provider.send.return_value.content = MOCK_EMPTY_JSON
        mock_provider.send.return_value.model = "deepseek-v4-pro"

        config = LLMConfig(provider_name="deepseek", api_key="sk-test")
        analyzer = BugAnalyzer(config)
        result = analyzer.analyze(sarif_report)

        assert result.metadata.total_findings == 0
        assert result.findings == []

    @patch("llm_client.bug_analysis.analyzer.ProviderFactory")
    def test_analyze_invalid_llm_json_raises(self, MockFactory, sarif_report):
        """If LLM returns non-JSON, should raise ValueError."""
        mock_provider = MockFactory.create.return_value
        mock_provider.send.return_value.content = "not json at all"

        config = LLMConfig(provider_name="deepseek", api_key="sk-test")
        analyzer = BugAnalyzer(config)

        with pytest.raises(ValueError, match="invalid JSON"):
            analyzer.analyze(sarif_report)

    @patch("llm_client.bug_analysis.analyzer.ProviderFactory")
    def test_analyze_file_not_found_raises(self, MockFactory):
        config = LLMConfig(provider_name="deepseek", api_key="sk-test")
        analyzer = BugAnalyzer(config)

        with pytest.raises(FileNotFoundError):
            analyzer.analyze_from_file("/nonexistent/report.txt")

    @patch("llm_client.bug_analysis.analyzer.ProviderFactory")
    def test_analyze_batch_all_success(self, MockFactory, sarif_report, plaintext_report):
        """Batch with all valid reports."""
        mock_provider = MockFactory.create.return_value
        mock_provider.send.return_value.content = MOCK_SARIF_JSON
        mock_provider.send.return_value.model = "deepseek-v4-pro"

        config = LLMConfig(provider_name="deepseek", api_key="sk-test")
        analyzer = BugAnalyzer(config)

        batch_result = analyzer.analyze_batch([sarif_report, plaintext_report])
        assert batch_result.summary.total_reports == 2
        assert batch_result.summary.successful == 2
        assert batch_result.summary.failed == 0

    @patch("llm_client.bug_analysis.analyzer.ProviderFactory")
    def test_analyze_batch_partial_failure(self, MockFactory, sarif_report):
        """Batch with one valid and one empty report should handle partial failure."""
        mock_provider = MockFactory.create.return_value
        mock_provider.send.return_value.content = MOCK_SARIF_JSON
        mock_provider.send.return_value.model = "deepseek-v4-pro"

        config = LLMConfig(provider_name="deepseek", api_key="sk-test")
        analyzer = BugAnalyzer(config)

        batch_result = analyzer.analyze_batch([sarif_report, ""])
        assert batch_result.summary.total_reports == 2
        assert batch_result.summary.successful == 1
        assert batch_result.summary.failed == 1
        # Failed report should have error message
        assert batch_result.results[1].status.value == "error"
        assert batch_result.results[1].error is not None

    @patch("llm_client.bug_analysis.analyzer.ProviderFactory")
    def test_analyze_batch_empty_raises(self, MockFactory):
        config = LLMConfig(provider_name="deepseek", api_key="sk-test")
        analyzer = BugAnalyzer(config)

        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze_batch([])


class TestExtractJson:
    """Tests for the internal _extract_json helper."""

    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == '{"a": 1}'

    def test_markdown_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert _extract_json(raw) == '{"a": 1}'

    def test_markdown_fence_no_lang(self):
        raw = '```\n{"a": 1}\n```'
        assert _extract_json(raw) == '{"a": 1}'

    def test_trailing_text(self):
        raw = '```json\n{"a": 1}\n```\nsome explanation'
        assert _extract_json(raw) == '{"a": 1}'
