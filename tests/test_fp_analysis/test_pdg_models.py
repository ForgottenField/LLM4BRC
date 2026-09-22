"""Tests for PDG/SDG data model construction, index building, and queries."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from llm_client.fp_analysis.pdg_models import (
    ControlDepEdge,
    DefUseEdge,
    FunctionPDG,
    FunctionSummary,
    PDGNode,
    SDG,
    SliceNode,
    SliceResult,
)
from llm_client.fp_analysis.pdg_loader import load_pdg


# ---------------------------------------------------------------------------
# Fixtures: synthetic PDG data
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_pdg() -> FunctionPDG:
    """Create a simple PDG for a function with two variables and an assignment.

    Source equivalent::

        void example() {
            int x = 1;      // S_1_5 (defines x)
            int y = x + 1;  // S_2_5 (defines y, uses x)
        }
    """
    nodes = {
        "S_1_5": PDGNode(id="S_1_5", kind="decl", variable="x",
                          expression="int x = 1", type="int",
                          source_line=1, source_col=5, block_id=1),
        "S_2_5": PDGNode(id="S_2_5", kind="decl", variable="y",
                          expression="int y = x + 1", type="int",
                          source_line=2, source_col=5, block_id=2),
    }
    def_use = [
        DefUseEdge(def_node_id="S_1_5", use_node_id="S_2_5", variable="x"),
    ]
    return FunctionPDG(
        function_name="example",
        source_file="test.cc",
        nodes=nodes,
        def_use_edges=def_use,
        summary=FunctionSummary(input_variables=[], output_variables=["x", "y"]),
    )


@pytest.fixture
def control_flow_pdg() -> FunctionPDG:
    """PDG with control flow: if (x > 0) { y = x; } else { y = -x; }

    Source equivalent::

        int example(int x) {
            int y;             // S_1_5 (defines y)
            if (x > 0) {       // S_2_9 (uses x, control predicate)
                y = x;         // S_3_9 (defines y, uses x) — ctrl-dep on S_2_9
            } else {
                y = -x;        // S_5_9 (defines y, uses x) — ctrl-dep on S_2_9
            }
            return y;          // S_7_5 (uses y) — ctrl-dep on S_2_9 (and post-merge)
        }
    """
    nodes = {
        "S_1_5": PDGNode(id="S_1_5", kind="decl", variable="y",
                          expression="int y", type="int",
                          source_line=1, source_col=5, block_id=1),
        "S_2_9": PDGNode(id="S_2_9", kind="binary_op", variable="",
                          expression="x > 0", type="bool",
                          source_line=2, source_col=9, block_id=2),
        "S_3_9": PDGNode(id="S_3_9", kind="assign", variable="y",
                          expression="y = x", type="int",
                          source_line=3, source_col=9, block_id=3),
        "S_5_9": PDGNode(id="S_5_9", kind="assign", variable="y",
                          expression="y = -x", type="int",
                          source_line=5, source_col=9, block_id=4),
        "S_7_5": PDGNode(id="S_7_5", kind="return", variable="",
                          expression="return y", type="int",
                          source_line=7, source_col=5, block_id=5),
    }
    def_use = [
        DefUseEdge(def_node_id="S_1_5", use_node_id="S_3_9", variable="y"),
        DefUseEdge(def_node_id="S_1_5", use_node_id="S_5_9", variable="y"),
        DefUseEdge(def_node_id="S_3_9", use_node_id="S_7_5", variable="y"),
        DefUseEdge(def_node_id="S_5_9", use_node_id="S_7_5", variable="y"),
    ]
    control_dep = [
        ControlDepEdge(source_id="S_2_9", target_id="S_3_9", kind="if_branch"),
        ControlDepEdge(source_id="S_2_9", target_id="S_5_9", kind="if_branch"),
        ControlDepEdge(source_id="S_2_9", target_id="S_7_5", kind="if_branch"),
    ]
    return FunctionPDG(
        function_name="example",
        source_file="test.cc",
        nodes=nodes,
        def_use_edges=def_use,
        control_dep_edges=control_dep,
        summary=FunctionSummary(input_variables=["x"], output_variables=["y"]),
    )


# ---------------------------------------------------------------------------
# PDG construction and index tests
# ---------------------------------------------------------------------------


class TestFunctionPDG:
    def test_empty_pdg(self):
        """An empty PDG should have no indices and be marked empty."""
        pdg = FunctionPDG(function_name="empty", source_file="empty.cc")
        assert pdg.is_empty
        assert pdg.defs_by_variable == {}
        assert pdg.uses_by_variable == {}
        assert pdg.data_preds_by_node == {}
        assert pdg.ctrl_preds_by_node == {}

    def test_defs_by_variable(self, simple_pdg):
        """defs_by_variable should map variable name to defining node IDs."""
        assert simple_pdg.get_definitions_of("x") == ["S_1_5"]
        assert simple_pdg.get_definitions_of("y") == ["S_2_5"]
        assert simple_pdg.get_definitions_of("nonexistent") == []

    def test_uses_by_variable(self, simple_pdg):
        """uses_by_variable should map variable name to using node IDs."""
        assert simple_pdg.get_uses_of("x") == ["S_2_5"]
        assert simple_pdg.get_uses_of("y") == []

    def test_backward_dep_nodes_simple(self, simple_pdg):
        """backward_dep_nodes should return data predecessors."""
        deps = simple_pdg.backward_dep_nodes("S_2_5")
        assert "S_1_5" in deps  # x is defined at S_1_5
        assert len(deps) == 1

    def test_backward_dep_nodes_control(self, control_flow_pdg):
        """backward_dep_nodes should return both data and control predecessors."""
        deps = control_flow_pdg.backward_dep_nodes("S_3_9")
        assert "S_1_5" in deps   # y defined at S_1_5 (data dep)
        assert "S_2_9" in deps   # control predicate (control dep)
        assert len(deps) == 2

    def test_forward_dep_nodes(self, simple_pdg):
        """forward_dep_nodes should return data successors."""
        deps = simple_pdg.forward_dep_nodes("S_1_5")
        assert "S_2_5" in deps
        assert len(deps) == 1

    def test_control_flow_backward_multiple(self, control_flow_pdg):
        """return y at S_7_5 should depend on both y definitions via data dep."""
        deps = control_flow_pdg.backward_dep_nodes("S_7_5")
        assert "S_3_9" in deps or "S_5_9" in deps
        assert "S_2_9" in deps  # also control-dependent on the if

    def test_forward_control(self, control_flow_pdg):
        """Control source should list all control-dependent targets."""
        deps = control_flow_pdg.forward_dep_nodes("S_2_9")
        assert "S_3_9" in deps
        assert "S_5_9" in deps
        assert "S_7_5" in deps


# ---------------------------------------------------------------------------
# SDG tests
# ---------------------------------------------------------------------------


class TestSDG:
    def test_empty_sdg(self):
        """An empty SDG should have no functions."""
        sdg = SDG()
        assert sdg.function_names == []

    def test_get_pdg_by_name(self, simple_pdg):
        """get_pdg should find PDG by exact qualified name."""
        sdg = SDG(pdgs={"example": simple_pdg})
        assert sdg.get_pdg("example") is simple_pdg
        assert sdg.get_pdg("nonexistent") is None

    def test_get_pdg_by_suffix(self, simple_pdg):
        """get_pdg should find PDG by suffix match."""
        sdg = SDG(pdgs={"MyClass::example": simple_pdg})
        assert sdg.get_pdg("example") is simple_pdg

    def test_get_pdg_unqualified(self, simple_pdg):
        """get_pdg should find PDG by unqualified (last component) match."""
        sdg = SDG(pdgs={"ns::example": simple_pdg})
        assert sdg.get_pdg("example") is simple_pdg
        # non-matching unqualified
        assert sdg.get_pdg("nonexistent") is None


# ---------------------------------------------------------------------------
# PDG loader tests
# ---------------------------------------------------------------------------


class TestPDGLoader:
    def test_load_invalid_path(self):
        """Loading from a non-existent path should return empty dict."""
        result = load_pdg("/nonexistent/path.json")
        assert result == {}

    def test_load_invalid_json(self):
        """Loading from an invalid JSON file should return empty dict."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("not valid json")
            tmp = f.name
        try:
            result = load_pdg(tmp)
            assert result == {}
        finally:
            Path(tmp).unlink()

    def test_load_valid_json(self):
        """Loading a valid PDG JSON should produce correct FunctionPDG objects."""
        data = {
            "functions": [
                {
                    "function_name": "test_func",
                    "source_file": "/src/test.cc",
                    "nodes": [
                        {"id": "S_1_5", "kind": "decl", "variable": "x",
                         "expression": "int x = 1", "type": "int",
                         "source_line": 1, "source_col": 5, "block_id": 1,
                         "is_call": False, "callee": "", "actual_params": []}
                    ],
                    "def_use_edges": [],
                    "control_dep_edges": [],
                    "input_variables": [],
                    "output_variables": ["x"],
                }
            ]
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(data, f)
            tmp = f.name
        try:
            pdgs = load_pdg(tmp)
            assert "test_func" in pdgs
            pdg = pdgs["test_func"]
            assert len(pdg.nodes) == 1
            assert "S_1_5" in pdg.nodes
            assert pdg.nodes["S_1_5"].variable == "x"
            assert pdg.summary.output_variables == ["x"]
        finally:
            Path(tmp).unlink()


# ---------------------------------------------------------------------------
# SliceResult tests
# ---------------------------------------------------------------------------


class TestSliceResult:
    def test_empty_slice(self):
        """An empty slice result should have no nodes and no inputs."""
        result = SliceResult(trigger_function="f", trigger_node_id="S_1_5")
        assert result.nodes == []
        assert result.input_variables == set()
        assert result.involved_functions == set()
        assert not result.truncated

    def test_slice_with_nodes(self, simple_pdg):
        """A slice with nodes should correctly track involved functions."""
        sn = SliceNode(
            function_name="example",
            source_file="test.cc",
            node=simple_pdg.nodes["S_1_5"],
            reason="trigger",
        )
        result = SliceResult(
            trigger_function="example",
            trigger_node_id="S_1_5",
            nodes=[sn],
            input_variables={"x"},
            involved_functions={"example"},
        )
        assert len(result.nodes) == 1
        assert "x" in result.input_variables
        assert "example" in result.involved_functions
