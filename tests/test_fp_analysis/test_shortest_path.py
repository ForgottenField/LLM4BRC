"""Tests for ShortestPathComputer + CalleePathCache — Phase 1 of
shortest-path selection + conflict-driven learning.
"""

from __future__ import annotations

import pytest
from llm_client.fp_analysis.cfg_branch_models import (
    BranchTag,
    CFGBranchNode,
    CFGBranchTree,
    Postcondition,
)
from llm_client.fp_analysis.pdg_models import (
    ControlDepEdge,
    DefUseEdge,
    FunctionPDG,
    PDGNode,
    SDG,
)


# ─── Helper factories ───────────────────────────────────────────────────────────


def _make_pdg_node(
    nid: str,
    kind: str = "assign",
    variable: str = "",
    expression: str = "",
    source_line: int = 0,
    block_id: int = 0,
    is_call: bool = False,
    callee: str = "",
) -> PDGNode:
    return PDGNode(
        id=nid,
        kind=kind,
        variable=variable,
        expression=expression,
        source_line=source_line,
        block_id=block_id,
        is_call=is_call,
        callee=callee,
    )


def _make_pdg(
    fn_name: str,
    nodes: list[PDGNode],
    def_use_edges: list[DefUseEdge] | None = None,
    control_dep_edges: list[ControlDepEdge] | None = None,
    source_file: str = "test.c",
) -> FunctionPDG:
    pdg = FunctionPDG(
        function_name=fn_name,
        source_file=source_file,
        nodes={n.id: n for n in nodes},
        def_use_edges=def_use_edges or [],
        control_dep_edges=control_dep_edges or [],
    )
    return pdg


# ─── CalleePathCache tests ──────────────────────────────────────────────────────


class TestCalleePathCache:
    """Tests for CalleePathCache — pre-computing callee shortest paths."""

    def test_single_return_simple_function(self):
        """CalleePathCache computes min_stmts for a function with 1 return.

        PDG layout:
          S_0_0 (param, line=0)  ← entry
          S_10_5 (assign) → data_dep on S_0_0
          S_12_3 (binary_op) → control_dep source for S_15_5
          S_15_5 (return) → data_dep on S_10_5, control_dep on S_12_3
        """
        from llm_client.fp_analysis.shortest_path import CalleePathCache

        nodes = [
            _make_pdg_node("S_0_0", "param", "x", source_line=0),
            _make_pdg_node("S_10_5", "assign", "r", "r = x + 1", source_line=10),
            _make_pdg_node("S_12_3", "binary_op", "", "r > 0", source_line=12),
            _make_pdg_node("S_15_5", "return", "", "r", source_line=15),
        ]
        # Build dependency edges: return depends on assign, assign depends on param
        def_use = [
            DefUseEdge("S_0_0", "S_10_5", "x"),
            DefUseEdge("S_10_5", "S_15_5", "r"),
        ]
        ctrl = [
            ControlDepEdge("S_12_3", "S_15_5", "if_branch"),
        ]
        pdg = _make_pdg("callee_simple", nodes, def_use, ctrl)
        sdg = SDG(pdgs={pdg.function_name: pdg})

        cache = CalleePathCache(sdg)
        cache.precompute("callee_simple")

        min_s = cache.get_min_stmts("callee_simple")
        assert min_s > 0, "Should return positive statement count"

        paths = cache.get_paths("callee_simple")
        assert len(paths) >= 1, "Should have at least one return path"

    def test_multi_return_function(self):
        """CalleePathCache computes paths for each return statement.

        PDG layout designed so paths to different returns have different
        statement counts:

        Short path (return -1):  S_12_5 ← S_10_3 ← S_0_0   = 2 counted nodes
        Long path (return r):    S_22_5 ← S_20_3 ← S_18_3 ← S_15_5 ← S_10_3 ← S_0_0
                                 = 5 counted nodes (extra intermediate assign)

        The key is that the long path goes through intermediate callees that
        must be traversed, not bypassed via direct def-use to the entry.
        """
        from llm_client.fp_analysis.shortest_path import CalleePathCache

        nodes = [
            _make_pdg_node("S_0_0", "param", "x", source_line=0),
            # Branch condition
            _make_pdg_node("S_10_3", "binary_op", "", "x < 0", source_line=10),
            # Short path: direct return
            _make_pdg_node("S_12_5", "return", "", "-1", source_line=12),
            # Long path: extra intermediate steps before return
            _make_pdg_node("S_15_5", "assign", "r", "r = x * 2", source_line=15),
            _make_pdg_node("S_18_3", "call_expr", "", "validate(r)", source_line=18,
                           is_call=True, callee="validate"),
            _make_pdg_node("S_20_3", "binary_op", "", "r > 100", source_line=20),
            _make_pdg_node("S_22_5", "return", "", "r", source_line=22),
        ]
        # S_10_3 uses x from entry. S_15_5 uses x from entry AND is
        # control-dep on S_10_3.  S_18_3 uses r from S_15_5.  S_20_3 uses
        # result from S_18_3.  S_22_5 uses r from S_15_5 AND is control-dep
        # on S_20_3.
        #
        # Critical: S_15_5 does NOT have a direct def-use from S_0_0.
        # Its only backward connection is via control_dep on S_10_3.
        # S_22_5 also does NOT have a direct def-use from S_15_5; instead
        # it is data-dep on S_18_3 and control-dep on S_20_3.
        def_use = [
            DefUseEdge("S_0_0", "S_10_3", "x"),
            # NO S_0_0 → S_15_5 — forces path through S_10_3
            DefUseEdge("S_15_5", "S_18_3", "r"),
            DefUseEdge("S_18_3", "S_20_3", "r"),
            DefUseEdge("S_18_3", "S_22_5", "r"),  # r via validate result
        ]
        ctrl = [
            ControlDepEdge("S_10_3", "S_12_5", "if_branch"),
            ControlDepEdge("S_10_3", "S_15_5", "if_branch"),
            ControlDepEdge("S_20_3", "S_22_5", "if_branch"),
        ]
        pdg = _make_pdg("callee_multi", nodes, def_use, ctrl)
        sdg = SDG(pdgs={pdg.function_name: pdg})

        cache = CalleePathCache(sdg)
        cache.precompute("callee_multi")

        paths = cache.get_paths("callee_multi")
        assert len(paths) >= 2, f"Should have 2 return paths, got {len(paths)}"

        # Short path: return -1 at L12 → binary_op at L10 → param = 2
        # Long path: return r at L22 → binary_op at L20 → call at L18
        #            → assign at L15 → binary_op at L10 → param = 5
        stmt_counts = [p.statement_count for p in paths]
        assert min(stmt_counts) < max(stmt_counts), (
            f"Different paths should have different statement counts, got {stmt_counts}"
        )

    def test_unknown_function_returns_default(self):
        """When callee PDG is not in SDG, get_min_stmts returns conservative default."""
        from llm_client.fp_analysis.shortest_path import CalleePathCache

        sdg = SDG()
        cache = CalleePathCache(sdg)

        # Not precomputed → should return conservative default
        min_s = cache.get_min_stmts("nonexistent")
        assert min_s == 1, "Unknown callee should return conservative default of 1"

    def test_empty_pdg_returns_default(self):
        """When callee PDG has no nodes, get_min_stmts returns 1."""
        from llm_client.fp_analysis.shortest_path import CalleePathCache

        pdg = _make_pdg("empty_fn", [])
        sdg = SDG(pdgs={pdg.function_name: pdg})
        cache = CalleePathCache(sdg)
        cache.precompute("empty_fn")

        min_s = cache.get_min_stmts("empty_fn")
        assert min_s == 1, "Empty PDG should return default of 1"

    def test_auxiliary_nodes_excluded(self):
        """Recovery and param nodes should be excluded from statement count."""
        from llm_client.fp_analysis.shortest_path import CalleePathCache

        nodes = [
            _make_pdg_node("S_0_0", "param", "x", source_line=0),
            _make_pdg_node("S_0_1", "param", "y", source_line=0),
            _make_pdg_node("S_5_3", "recovery", "", source_line=5),  # excluded
            _make_pdg_node("S_10_5", "assign", "r", "r = x + y", source_line=10),
            _make_pdg_node("S_12_3", "return", "", "r", source_line=12),
        ]
        def_use = [
            DefUseEdge("S_0_0", "S_10_5", "x"),
            DefUseEdge("S_0_1", "S_10_5", "y"),
            DefUseEdge("S_10_5", "S_12_3", "r"),
        ]
        pdg = _make_pdg("callee_aux", nodes, def_use)
        sdg = SDG(pdgs={pdg.function_name: pdg})

        cache = CalleePathCache(sdg)
        cache.precompute("callee_aux")

        min_s = cache.get_min_stmts("callee_aux")
        # Should count assign + return = 2, NOT recovery or param
        assert min_s == 2, (
            f"Expected 2 (assign+return), got {min_s} (recovery/param should be excluded)"
        )


# ─── ShortestPathComputer tests ─────────────────────────────────────────────────


class TestShortestPathComputer:
    """Tests for ShortestPathComputer — BFS on CFG branch tree."""

    def test_linear_segment_zero_branches(self):
        """Segment with 0 branches returns single empty candidate."""
        from llm_client.fp_analysis.shortest_path import ShortestPathComputer, CalleePathCache

        tree = CFGBranchTree(
            segment_index=0,
            function_name="linear_fn",
            line_start=10,
            line_end=20,
            postcondition=Postcondition("linear_fn", 20, "r >= 0", True),
        )
        sdg = SDG()
        cache = CalleePathCache(sdg)
        pc = ShortestPathComputer(sdg, cache)

        candidates = pc.compute_topk(tree, k=3)
        assert len(candidates) == 1, "Linear segment should return 1 candidate"
        decisions, metrics = candidates[0]
        assert len(decisions) == 0, "No branch decisions for linear segment"
        assert metrics.S == 0
        assert metrics.C == 0
        assert metrics.V == 0

    def test_contradictory_branch_pruned(self):
        """BFS skips branches tagged as contradictory."""
        from llm_client.fp_analysis.shortest_path import ShortestPathComputer, CalleePathCache

        # Build tree with 1 contradictory branch
        branch = CFGBranchNode(
            node_id="B5_taken",
            source_line=15,
            function_name="fn",
            condition_expr="x < 0",
            is_taken_branch=True,
            tag=BranchTag("B5_taken", "contradictory", layer=1, reason="x >= 0 postcondition"),
        )
        tree = CFGBranchTree(
            segment_index=0,
            function_name="fn",
            line_start=10,
            line_end=20,
            postcondition=Postcondition("fn", 20, "x >= 0", True),
            root_branches=[branch],
        )
        tree.update_stats()

        sdg = SDG()
        cache = CalleePathCache(sdg)
        pc = ShortestPathComputer(sdg, cache)

        candidates = pc.compute_topk(tree, k=3)
        # The contradictory branch should be excluded; single path with no decisions
        assert len(candidates) == 1
        decisions, _metrics = candidates[0]
        assert len(decisions) == 0, "Contradictory branch should be skipped"

    def test_consistent_branch_auto_confirmed(self):
        """BFS auto-confirms consistent-tagged branches (only explores confirmed direction)."""
        from llm_client.fp_analysis.shortest_path import ShortestPathComputer, CalleePathCache

        branch = CFGBranchNode(
            node_id="B10_taken",
            source_line=18,
            function_name="fn",
            condition_expr="x >= 0",
            is_taken_branch=True,
            tag=BranchTag("B10_taken", "consistent", layer=1, reason="matches postcondition"),
        )
        tree = CFGBranchTree(
            segment_index=0,
            function_name="fn",
            line_start=10,
            line_end=20,
            postcondition=Postcondition("fn", 20, "x >= 0", True),
            root_branches=[branch],
        )
        tree.update_stats()

        sdg = SDG()
        cache = CalleePathCache(sdg)
        pc = ShortestPathComputer(sdg, cache)

        candidates = pc.compute_topk(tree, k=3)
        assert len(candidates) == 1
        decisions, _metrics = candidates[0]
        assert len(decisions) == 1
        assert decisions[0].node_id == "B10_taken"
        assert decisions[0].branch_taken == True

    def test_insufficient_info_explores_both(self):
        """For insufficient_info branches, both true/false sides are explored."""
        from llm_client.fp_analysis.shortest_path import ShortestPathComputer, CalleePathCache

        branch = CFGBranchNode(
            node_id="B15_taken",
            source_line=15,
            function_name="fn",
            condition_expr="x > 0",
            is_taken_branch=True,
            tag=BranchTag("B15_taken", "insufficient_info", layer=0, reason="unknown"),
        )
        tree = CFGBranchTree(
            segment_index=0,
            function_name="fn",
            line_start=10,
            line_end=20,
            postcondition=Postcondition("fn", 20, "r >= 0", True),
            root_branches=[branch],
        )
        tree.update_stats()

        sdg = SDG()
        cache = CalleePathCache(sdg)
        pc = ShortestPathComputer(sdg, cache)

        candidates = pc.compute_topk(tree, k=5)
        assert len(candidates) == 2, (
            f"insufficient_info should produce 2 paths (true/false), got {len(candidates)}"
        )

    def test_metric_ordering_s_before_c_before_v(self):
        """Candidates are sorted by S↑ → C↑ → V↑."""
        from llm_client.fp_analysis.shortest_path import ShortestPathComputer, CalleePathCache, PathMetrics

        # We'll verify that PathMetrics ordering works correctly
        m1 = PathMetrics(S=3, C=1, V=2)
        m2 = PathMetrics(S=3, C=1, V=1)
        m3 = PathMetrics(S=1, C=5, V=10)
        m4 = PathMetrics(S=3, C=2, V=1)

        # Sort ascending: S↑ → C↑ → V↑
        sorted_metrics = sorted([m1, m2, m3, m4])
        assert sorted_metrics[0] == m3, "S=1 should come first"
        assert sorted_metrics[1] == m2, "S=3,C=1,V=1 should come before V=2"
        assert sorted_metrics[2] == m1, "S=3,C=1,V=2"
        assert sorted_metrics[3] == m4, "S=3,C=2 should come last"

    def test_cross_procedure_stmts_added(self):
        """Callee statement count from CalleePathCache is added to path metrics."""
        from llm_client.fp_analysis.shortest_path import ShortestPathComputer, CalleePathCache

        # Set up callee with known statement count
        callee_nodes = [
            _make_pdg_node("S_0_0", "param", "x", source_line=0),
            _make_pdg_node("S_10_5", "assign", "r", "r = x + 1", source_line=10),
            _make_pdg_node("S_12_3", "return", "", "r", source_line=12),
        ]
        callee_def_use = [
            DefUseEdge("S_0_0", "S_10_5", "x"),
            DefUseEdge("S_10_5", "S_12_3", "r"),
        ]
        callee_pdg = _make_pdg("callee_fn", callee_nodes, callee_def_use)
        sdg = SDG(pdgs={callee_pdg.function_name: callee_pdg})
        cache = CalleePathCache(sdg)
        cache.precompute("callee_fn")

        # Call site with callee sentinel
        sentinel = CFGBranchNode(
            node_id="call_L15",
            source_line=15,
            function_name="caller_fn",
            condition_expr="→ callee_fn()",
            callee_name="callee_fn",
        )
        tree = CFGBranchTree(
            segment_index=0,
            function_name="caller_fn",
            line_start=10,
            line_end=20,
            postcondition=Postcondition("caller_fn", 20, "r >= 0", True),
            root_branches=[sentinel],
        )
        tree.update_stats()

        pc = ShortestPathComputer(sdg, cache)
        candidates = pc.compute_topk(tree, k=3)
        assert len(candidates) >= 1
        _decisions, metrics = candidates[0]
        # S should include callee's statement count (2 non-auxiliary nodes)
        assert metrics.S >= 2, (
            f"Expected S >= 2 (callee contribution), got {metrics.S}"
        )

    def test_topk_limiting(self):
        """compute_topk returns at most k candidates."""
        from llm_client.fp_analysis.shortest_path import ShortestPathComputer, CalleePathCache

        # Create 2 insufficient_info branches → 2^2 = 4 potential paths
        branches = []
        for i in range(2):
            branches.append(CFGBranchNode(
                node_id=f"B{i}_taken",
                source_line=15 + i,
                function_name="fn",
                condition_expr=f"x_{i} > 0",
                tag=BranchTag(f"B{i}_taken", "insufficient_info", layer=0),
            ))
        tree = CFGBranchTree(
            segment_index=0,
            function_name="fn",
            line_start=10,
            line_end=20,
            postcondition=Postcondition("fn", 20, "r >= 0", True),
            root_branches=branches,
        )
        tree.update_stats()

        sdg = SDG()
        cache = CalleePathCache(sdg)
        pc = ShortestPathComputer(sdg, cache)

        candidates = pc.compute_topk(tree, k=2)
        assert len(candidates) <= 2, (
            f"Should return at most k=2 candidates, got {len(candidates)}"
        )

    def test_untagged_branch_explored_as_insufficient(self):
        """Untagged branches (no BranchTag) are treated as insufficient_info."""
        from llm_client.fp_analysis.shortest_path import ShortestPathComputer, CalleePathCache

        branch = CFGBranchNode(
            node_id="B5_taken",
            source_line=15,
            function_name="fn",
            condition_expr="x > 0",
            # tag is None → treated as insufficient_info
        )
        tree = CFGBranchTree(
            segment_index=0,
            function_name="fn",
            line_start=10,
            line_end=20,
            postcondition=Postcondition("fn", 20, "r >= 0", True),
            root_branches=[branch],
        )
        tree.update_stats()

        sdg = SDG()
        cache = CalleePathCache(sdg)
        pc = ShortestPathComputer(sdg, cache)

        candidates = pc.compute_topk(tree, k=5)
        # Untagged → treated as insufficient → both sides explored
        assert len(candidates) == 2, (
            f"Untagged branch should be explored both ways, got {len(candidates)}"
        )

    def test_nested_branches(self):
        """BFS handles nested branch trees correctly."""
        from llm_client.fp_analysis.shortest_path import ShortestPathComputer, CalleePathCache

        # Outer branch (insufficient) → inner branch (consistent)
        inner = CFGBranchNode(
            node_id="inner_taken",
            source_line=20,
            function_name="fn",
            condition_expr="y == 1",
            tag=BranchTag("inner_taken", "consistent", layer=1),
        )
        outer = CFGBranchNode(
            node_id="outer_taken",
            source_line=15,
            function_name="fn",
            condition_expr="x > 0",
            tag=BranchTag("outer_taken", "insufficient_info", layer=0),
            children=[inner],
        )
        tree = CFGBranchTree(
            segment_index=0,
            function_name="fn",
            line_start=10,
            line_end=25,
            postcondition=Postcondition("fn", 25, "r >= 0", True),
            root_branches=[outer],
        )
        tree.update_stats()

        sdg = SDG()
        cache = CalleePathCache(sdg)
        pc = ShortestPathComputer(sdg, cache)

        candidates = pc.compute_topk(tree, k=5)
        # Outer: insufficient (2 ways) → inner: consistent (1 way)
        assert len(candidates) == 2, (
            f"Nested branches should give 2 paths, got {len(candidates)}"
        )

    def test_all_branches_contradictory(self):
        """When all branches are contradictory, return single empty path."""
        from llm_client.fp_analysis.shortest_path import ShortestPathComputer, CalleePathCache

        branch = CFGBranchNode(
            node_id="B5_taken",
            source_line=15,
            function_name="fn",
            condition_expr="x < 0",
            tag=BranchTag("B5_taken", "contradictory", layer=1),
        )
        tree = CFGBranchTree(
            segment_index=0,
            function_name="fn",
            line_start=10,
            line_end=20,
            postcondition=Postcondition("fn", 20, "x >= 0", True),
            root_branches=[branch],
        )
        tree.update_stats()

        sdg = SDG()
        cache = CalleePathCache(sdg)
        pc = ShortestPathComputer(sdg, cache)

        candidates = pc.compute_topk(tree, k=3)
        assert len(candidates) == 1
        decisions, _metrics = candidates[0]
        assert len(decisions) == 0, "All contradictory → empty path"
