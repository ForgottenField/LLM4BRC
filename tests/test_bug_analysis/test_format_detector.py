"""Tests for input format detection."""

import pytest

from llm_client.bug_analysis.format_detector import detect_format
from llm_client.bug_analysis.schemas import InputFormat


class TestDetectFormat:
    def test_detects_sarif(self, sarif_report):
        assert detect_format(sarif_report) == InputFormat.SARIF

    def test_detects_plain_text(self, plaintext_report):
        assert detect_format(plaintext_report) == InputFormat.PLAIN_TEXT

    def test_detects_json(self, json_report):
        assert detect_format(json_report) == InputFormat.JSON

    def test_empty_string(self):
        assert detect_format("") == InputFormat.PLAIN_TEXT
        assert detect_format("   ") == InputFormat.PLAIN_TEXT

    def test_invalid_json_falls_to_text(self):
        assert detect_format("{invalid json}") == InputFormat.PLAIN_TEXT

    def test_sarif_with_version_only(self):
        content = '{"version": "2.1.0", "runs": []}'
        assert detect_format(content) == InputFormat.SARIF

    def test_sarif_with_schema_only(self):
        content = '{"$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/2.1.0/schema.json"}'
        assert detect_format(content) == InputFormat.SARIF

    def test_sarif_schema_case_insensitive(self):
        content = '{"$schema": "SARIF/2.1.0/schema.json", "version": "2.1.0"}'
        assert detect_format(content) == InputFormat.SARIF
