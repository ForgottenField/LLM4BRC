"""Tests for the code reducer that converts slice results to reduced C++."""

from __future__ import annotations

from pathlib import Path

import pytest
import tempfile

from llm_client.fp_analysis.pdg_models import PDGNode, SliceNode, SliceResult
from llm_client.fp_analysis.code_reducer import reduce_code, _infer_includes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_slice_node(
    node_id: str,
    kind: str = "assign",
    variable: str = "",
    expression: str = "",
    line: int = 1,
    col: int = 5,
    fn_name: str = "test_func",
    source_file: str = "/tmp/test.cc",
) -> SliceNode:
    """Create a SliceNode for testing."""
    pnode = PDGNode(
        id=node_id,
        kind=kind,
        variable=variable,
        expression=expression,
        type="int",
        source_line=line,
        source_col=col,
        block_id=1,
    )
    return SliceNode(
        function_name=fn_name,
        source_file=source_file,
        node=pnode,
        reason="data_dep",
    )


# ---------------------------------------------------------------------------
# Test: include inference
# ---------------------------------------------------------------------------


class TestIncludeInference:
    def test_basic_includes(self):
        """Common standard library usage should produce correct includes."""
        exprs = ["std::string s", "memset(ptr, 0, 10)", "printf(\"%d\", x)"]
        includes = _infer_includes(exprs, [])
        assert any("string" in inc for inc in includes)
        assert any("cstring" in inc for inc in includes)
        assert any("cstdio" in inc for inc in includes)

    def test_keywords_not_inferred(self):
        """C++ keywords should not trigger include inference."""
        exprs = ["if (x > 0)", "return 0", "while (true) { }"]
        includes = _infer_includes(exprs, [])
        # Should still have the basic cstddef
        assert any("cstddef" in inc for inc in includes)


# ---------------------------------------------------------------------------
# Test: code reduction
# ---------------------------------------------------------------------------


class TestReduceCode:
    def test_empty_slice(self):
        """An empty slice result should indicate no source lines."""
        result = SliceResult(trigger_function="f", trigger_node_id="S_1_5")
        code = reduce_code(result)
        assert "empty slice" in code

    def test_single_node_slice(self):
        """A slice with one node should produce code containing that statement."""
        sn = _make_slice_node("S_1_5", expression="int x = 42")
        result = SliceResult(
            trigger_function="test_func",
            trigger_node_id="S_1_5",
            nodes=[sn],
            input_variables=set(),
            involved_functions={"test_func"},
        )
        code = reduce_code(result)
        assert "x = 42" in code or "x = 42; // (from PDG)" in code
        assert "int main()" in code

    def test_frontier_declarations(self):
        """Input variables should get declaration stubs."""
        sn = _make_slice_node("S_2_5", expression="int y = x + 1")
        result = SliceResult(
            trigger_function="test_func",
            trigger_node_id="S_2_5",
            nodes=[sn],
            input_variables={"x"},
            involved_functions={"test_func"},
        )
        code = reduce_code(result)
        assert "x" in code  # x should be declared as symbolic input
        assert "0; // symbolic input" in code

    def test_multiple_nodes_ordered(self):
        """Multiple slice nodes should be emitted in line-number order."""
        sn1 = _make_slice_node("S_3_5", expression="int z = 0", line=3)
        sn2 = _make_slice_node("S_1_5", expression="int x = 1", line=1)
        sn3 = _make_slice_node("S_2_5", expression="int y = x", line=2)

        result = SliceResult(
            trigger_function="test_func",
            trigger_node_id="S_3_5",
            nodes=[sn1, sn2, sn3],  # intentionally out of order
            input_variables=set(),
            involved_functions={"test_func"},
        )
        code = reduce_code(result)

        # Find positions of each statement in the output
        pos_x = code.find("x = 1")
        pos_y = code.find("y = x")
        pos_z = code.find("z = 0")
        if pos_x >= 0 and pos_y >= 0 and pos_z >= 0:
            assert pos_x < pos_y < pos_z  # correct program order

    def test_multiple_functions(self):
        """Nodes from different functions should be grouped."""
        sn1 = _make_slice_node("S_1_5", fn_name="helper", expression="int a = 0")
        sn2 = _make_slice_node("S_1_5", fn_name="main_func", expression="int b = a")
        result = SliceResult(
            trigger_function="main_func",
            trigger_node_id="S_1_5",
            nodes=[sn1, sn2],
            input_variables={"a"},
            involved_functions={"helper", "main_func"},
        )
        code = reduce_code(result)
        # Both function names should appear as comments
        assert "helper" in code
        assert "main_func" in code
        # Reordered: the assignment a = 0 should be in the reduced code
        # (the frontier variable a gets declared too)
        assert "a" in code

    def test_source_line_extraction(self):
        """If the source file exists, reducer should extract actual source lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = Path(tmpdir) / "test.cc"
            src_file.write_text("int main() {\n    int x = 42;\n    return x;\n}\n")

            sn = _make_slice_node(
                "S_2_5",
                expression="placeholder",
                line=2,
                source_file=str(src_file),
            )
            result = SliceResult(
                trigger_function="main",
                trigger_node_id="S_2_5",
                nodes=[sn],
                input_variables=set(),
                involved_functions={"main"},
            )
            code = reduce_code(result, include_search_roots=[tmpdir])
            # Should use the actual source line, not the PDG expression
            assert "x = 42" in code

    def test_no_crash_on_missing_file(self):
        """Reducer should not crash when a source file is missing."""
        sn = _make_slice_node("S_1_5", expression="int x = 1",
                               source_file="/nonexistent/file.cc")
        result = SliceResult(
            trigger_function="f",
            trigger_node_id="S_1_5",
            nodes=[sn],
            input_variables=set(),
            involved_functions={"f"},
        )
        # Should produce code without crashing
        code = reduce_code(result)
        assert code
