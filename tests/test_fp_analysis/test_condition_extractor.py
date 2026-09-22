"""Tests for SourceLineConditionExtractor — Phase 1 of branch extraction fix.

Tests the condition extractor that maps control-flow source lines
(like ``if (r >= 0)``) to the pure condition expression (``r >= 0``).

Test order matches TDD: earliest tests verify minimal API and simplest
cases; later tests cover edge cases and multi-line conditions.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from llm_client.fp_analysis.condition_extractor import (
    ExtractedCondition,
    extract_condition,
    extract_condition_from_text,
)


# ═══════════════════════════════════════════════════════════════════════
# Test 1: extract_condition_from_text — pure string extraction
# ═══════════════════════════════════════════════════════════════════════


class TestExtractConditionFromText:
    """Unit tests for the pure-text extractor (no file I/O)."""

    def test_simple_if(self) -> None:
        """Simple ``if (r >= 0)`` → ``r >= 0``."""
        result = extract_condition_from_text("if (r >= 0)")
        assert result == "r >= 0"

    def test_if_with_braces(self) -> None:
        """``if (r >= 0) {`` → ``r >= 0`` (strip trailing brace)."""
        result = extract_condition_from_text("if (r >= 0) {")
        assert result == "r >= 0"

    def test_if_with_space(self) -> None:
        """``if (  r >= 0  )`` → ``r >= 0`` (strip inner whitespace)."""
        result = extract_condition_from_text("if (  r >= 0  )")
        assert result == "r >= 0"

    def test_while_loop(self) -> None:
        """``while (len > 0)`` → ``len > 0``."""
        result = extract_condition_from_text("while (len > 0)")
        assert result == "len > 0"

    def test_for_loop(self) -> None:
        """``for (int i = 0; i < n; i++)`` → ``int i = 0; i < n; i++``."""
        result = extract_condition_from_text("for (int i = 0; i < n; i++)")
        assert result == "int i = 0; i < n; i++"

    def test_else_if(self) -> None:
        """``} else if (z == 0) {`` → ``z == 0``."""
        result = extract_condition_from_text("} else if (z == 0) {")
        assert result == "z == 0"

    def test_no_condition(self) -> None:
        """Assignment line → None."""
        result = extract_condition_from_text("x = y + 1;")
        assert result is None

    def test_return_statement(self) -> None:
        """``return r;`` → None."""
        result = extract_condition_from_text("return r;")
        assert result is None

    def test_csa_generic_desc(self) -> None:
        """CSA generic event description → None."""
        result = extract_condition_from_text("← Taking true branch →")
        assert result is None

    def test_nested_parentheses(self) -> None:
        """``if (a && (b || c))`` → ``a && (b || c)``."""
        result = extract_condition_from_text("if (a && (b || c))")
        assert result == "a && (b || c)"


# ═══════════════════════════════════════════════════════════════════════
# Test 2: extract_condition — file-based extraction
# ═══════════════════════════════════════════════════════════════════════


class TestExtractCondition:
    """Integration tests with temp source files."""

    @pytest.fixture
    def source_file(self) -> str:
        """Create a temp C++ source file with known control flow."""
        content = """\
void func() {
    int x = 0;
    if (x >= 0) {
        return;
    }
    while (len > 0) {
        len--;
    }
}
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False,
        ) as f:
            f.write(content)
            path = f.name
        yield path
        os.unlink(path)

    def test_extract_if_condition(self, source_file: str) -> None:
        """Read the ``if (x >= 0)`` at line 3 → ``x >= 0``."""
        result = extract_condition(source_file, 3)
        assert result.expression == "x >= 0"
        assert result.source_line == 3
        assert result.confidence == 1.0

    def test_extract_while_condition(self, source_file: str) -> None:
        """Read the ``while (len > 0)`` at line 6 → ``len > 0``."""
        result = extract_condition(source_file, 6)
        assert result.expression == "len > 0"
        assert result.source_line == 6
        assert result.confidence == 1.0

    def test_non_control_line(self, source_file: str) -> None:
        """Line 2 is assignment → empty expression, low confidence."""
        result = extract_condition(source_file, 2)
        assert result.expression == ""
        assert result.confidence == 0.0

    def test_brace_line(self, source_file: str) -> None:
        """Line 4 is just ``{`` → backward scan finds ``if (x >= 0)`` on L3."""
        result = extract_condition(source_file, 4)
        assert result.expression == "x >= 0"
        assert result.source_line == 3
        assert result.confidence == 0.7  # backward scan

    def test_file_not_found(self) -> None:
        """Non-existent file → empty expression, low confidence."""
        result = extract_condition("/nonexistent/file.cpp", 10)
        assert result.expression == ""
        assert result.confidence == 0.0

    def test_line_out_of_range(self, source_file: str) -> None:
        """Line beyond file length → empty expression."""
        result = extract_condition(source_file, 999)
        assert result.expression == ""
        assert result.confidence == 0.0

    def test_extract_with_source_root(self, source_file: str) -> None:
        """Test with source_root parameter (source_root + rel_path)."""
        abs_path = Path(source_file)
        rel_path = abs_path.name
        result = extract_condition(
            rel_path, 3, source_root=str(abs_path.parent),
        )
        assert result.expression == "x >= 0"


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Multi-line conditions
# ═══════════════════════════════════════════════════════════════════════


class TestMultiLineCondition:
    """Multi-line control flow conditions spanning 2+ lines."""

    @pytest.fixture
    def multi_line_file(self) -> str:
        """Source file with multi-line conditions."""
        content = """\
void func() {
    if (a &&
        b &&
        c) {
        return;
    }
    if (very_long_function_name_that_exceeds_limit(
            arg1, arg2)) {
        return;
    }
}
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cpp", delete=False,
        ) as f:
            f.write(content)
            path = f.name
        yield path
        os.unlink(path)

    def test_multi_line_condition_on_first_line(self, multi_line_file: str) -> None:
        """L2 ``if (a &&`` → extract by reading ahead."""
        result = extract_condition(multi_line_file, 2)
        assert result.expression == "a && b && c"
        assert result.confidence > 0

    def test_multi_line_condition_on_last_line(self, multi_line_file: str) -> None:
        """L4 ``c) {`` → extract by scanning backward."""
        result = extract_condition(multi_line_file, 4)
        assert result.expression == "a && b && c"
        assert result.confidence > 0

    def test_function_call_in_condition(self, multi_line_file: str) -> None:
        """L7 `if (very_long_function_name...` → extract full condition."""
        result = extract_condition(multi_line_file, 7)
        assert "arg1" in result.expression
        assert "arg2" in result.expression
        assert result.confidence > 0
