"""Tests for the A/B/C path-space compression in ``CFGBranchExtractor``.

A — path reachability: expand only the code the reported CSA path can reach,
    computed from the function's CFG under the report's branch decisions.
B — slice-retention gating: beyond a shallow depth, only callees the PDG slice
    kept are expanded.
C — structural guards: recursion stack, environment (non-project) code, and
    the serialization cap in ``path_space``.

The contract every one of these must preserve: a callee that is *not* expanded
still leaves a leaf sentinel carrying ``callee_name``, the ``→ name()``
expression and the ``callee_<caller>_<callee>_L<line>`` node id — four
downstream consumers read that shape.
"""

from __future__ import annotations

import pytest

from llm_client.fp_analysis.cfg_branch_extractor import CFGBranchExtractor
from llm_client.fp_analysis.cfg_models import CFGBlock, CFGFunction
from llm_client.fp_analysis.path_space import (
    _MAX_SERIALIZED_DECISIONS,
    _normalize_call_edge,
    _segment_signature,
    ci_branch_set_of,
)
from llm_client.fp_analysis.pdg_models import (
    ControlDepEdge,
    FunctionPDG,
    PDGNode,
)
from llm_client.fp_analysis.slice_mask import SliceMask


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _block(bid, lines=(), succs=(), terminator=None, label=""):
    return CFGBlock(
        block_id=bid,
        statements=[f"  {ln}: stmt" for ln in lines],
        successors=list(succs),
        terminator=terminator,
        label=label,
    )


@pytest.fixture
def if_else_cfg() -> CFGFunction:
    """``if (x > 0) { body } else { alt }`` as Clang numbers it.

    ``B1``'s successors are ``[true arm, false arm]`` — the convention A relies
    on.  ``B2``/``B3`` are the two arms, both rejoining at ``B4``.
    """
    return CFGFunction(
        function_name="faiss::probe",
        blocks={
            "B1": _block("B1", terminator="if [B1.1] (BinaryOperator) x > 0",
                         succs=["B2", "B3"]),
            "B2": _block("B2", lines=[10], succs=["B4"]),
            "B3": _block("B3", lines=[12], succs=["B4"]),
            "B4": _block("B4", lines=[14], label="EXIT"),
        },
        entry_block_id="B1",
        exit_block_id="B4",
    )


def _pdg(fn_name, source_file, nodes, ctrl_edges=()):
    return FunctionPDG(
        function_name=fn_name,
        source_file=source_file,
        nodes={n.id: n for n in nodes},
        control_dep_edges=list(ctrl_edges),
    )


def _branch_node(nid, line, block_id, expr):
    """A PDG node that ``_get_branch_sources`` accepts as a predicate."""
    return PDGNode(id=nid, kind="binary_op", expression=expr,
                   source_line=line, block_id=block_id)


class _SDG:
    def __init__(self, pdgs):
        self.pdgs = pdgs

    def get_pdg(self, name):
        return self.pdgs.get(name)


# ─── A: successor-arm selection ──────────────────────────────────────────────


class TestReachableBlocks:
    """The block→line map comes from the PDG, so each test needs a PDG whose
    nodes sit in the blocks of the CFG under test."""

    def _extractor(self, cfg, decisions, lines_by_block):
        nodes = [
            PDGNode(id=f"N_{bid}_{ln}", kind="event", source_line=ln,
                    block_id=bid)
            for bid, lns in lines_by_block.items() for ln in lns
        ]
        pdg = _pdg(cfg.function_name, "/p/probe.cpp", nodes)
        ex = CFGBranchExtractor({cfg.function_name: cfg},
                                _SDG({cfg.function_name: pdg}), SliceMask())
        ex._decisions_by_fn = {cfg.function_name: decisions}
        return ex

    # B1's predicate line is 1; the two arms and the exit carry 10/12/14.
    _LINES = {1: [1], 2: [10], 3: [12], 4: [14]}

    def test_false_decision_takes_the_else_arm(self, if_else_cfg):
        ex = self._extractor(if_else_cfg, {1: False}, self._LINES)
        assert ex._on_path_lines("faiss::probe") == frozenset({1, 12, 14})

    def test_true_decision_takes_the_then_arm(self, if_else_cfg):
        ex = self._extractor(if_else_cfg, {1: True}, self._LINES)
        assert ex._on_path_lines("faiss::probe") == frozenset({1, 10, 14})

    def test_undecided_branch_keeps_both_arms(self, if_else_cfg):
        ex = self._extractor(if_else_cfg, {}, self._LINES)
        assert ex._on_path_lines("faiss::probe") is None

    def test_unreachable_exit_disables_filtering(self, if_else_cfg):
        """A wrong successor order strands the exit block → fall back, loudly."""
        if_else_cfg.blocks["B2"].successors = ["B3"]
        if_else_cfg.blocks["B3"].successors = ["B2"]  # loop that never exits
        ex = self._extractor(if_else_cfg, {1: True}, self._LINES)
        assert ex._on_path_lines("faiss::probe") is None

    def test_segment_end_off_path_disables_filtering(self, if_else_cfg):
        """The segment ends where the report says; if that line is unreachable
        the walk disagrees with the report, so A must switch itself off."""
        ex = self._extractor(if_else_cfg, {1: True}, self._LINES)
        assert ex._on_path_lines_for_segment("faiss::probe", 12) is None
        assert ex._on_path_lines_for_segment("faiss::probe", 10) is not None

    def test_short_circuit_line_with_conflicting_decisions_is_kept(self):
        """Clang splits ``a || b`` into one block per operand; when two operand
        lines share a branch block and disagree, both arms stay reachable."""
        cfg = CFGFunction(
            function_name="faiss::probe",
            blocks={
                "B1": _block("B1", terminator="if [B1.1]", succs=["B2", "B3"]),
                "B2": _block("B2", lines=[10], succs=["B4"]),
                "B3": _block("B3", lines=[12], succs=["B4"]),
                "B4": _block("B4", lines=[14], label="EXIT"),
            },
            entry_block_id="B1", exit_block_id="B4",
        )
        ex = self._extractor(cfg, {7: True, 8: False}, {1: [7, 8], **{2: [10], 3: [12], 4: [14]}})
        on_path = ex._on_path_lines("faiss::probe")
        assert on_path == frozenset({7, 8, 10, 12, 14})  # undecided → both arms


# ─── A: the alignment gate ───────────────────────────────────────────────────


class TestAlignmentGate:
    """``node.block_id`` and the cache's ``B<n>`` come from two different Clang
    frontends (the cache adds implicit-dtor and scope blocks), so the pairing
    the walk depends on is not guaranteed.  Before pruning anything A checks the
    two numberings against each other *by content* — see ``_alignment_score``."""

    _CHAIN = (("B2", 10), ("B3", 12), ("B4", 14), ("B5", 16), ("B6", 18),
              ("B7", 20))

    def _cfg(self):
        blocks = {"B1": _block("B1", terminator="if [B1.1]", succs=["B2", "B3"])}
        for i, (bid, line) in enumerate(self._CHAIN):
            nxt = self._CHAIN[i + 1][0] if i + 1 < len(self._CHAIN) else bid
            blocks[bid] = _block(bid, lines=[line], succs=[nxt])
        return CFGFunction(function_name="faiss::probe", blocks=blocks,
                           entry_block_id="B1", exit_block_id="B7")

    def _extractor(self, expression):
        cfg = self._cfg()
        nodes = [_branch_node("P_1", 1, 1, "x > 0")] + [
            PDGNode(id=f"N_{line}", kind="event", expression=expression(line),
                    source_line=line, block_id=int(bid[1:]))
            for bid, line in self._CHAIN
        ]
        pdg = _pdg(cfg.function_name, "/p/probe.cpp", nodes)
        ex = CFGBranchExtractor({cfg.function_name: cfg},
                                _SDG({cfg.function_name: pdg}), SliceMask())
        ex._decisions_by_fn = {cfg.function_name: {1: True}}
        return ex, cfg

    def test_content_that_agrees_leaves_filtering_on(self):
        """Each CFG statement contains the PDG's expression for the same id."""
        ex, cfg = self._extractor(lambda line: "stmt")
        assert ex._alignment_score("faiss::probe", cfg) == (1.0, 6)
        assert ex._on_path_lines("faiss::probe") == frozenset(
            {1, 10, 12, 14, 16, 18, 20})

    def test_content_that_disagrees_disables_filtering(self):
        """The cache's block N holds another block's code: the walk would read
        the wrong lines, so A must return ``None`` (prune nothing)."""
        ex, cfg = self._extractor(lambda line: f"unrelated_expression_{line}")
        assert ex._alignment_score("faiss::probe", cfg) == (0.0, 6)
        assert ex._on_path_lines("faiss::probe") is None

    def test_uncomparable_blocks_are_no_evidence(self):
        """A PDG without expressions cannot be checked, so the gate stays open —
        which is how the tests above and the pre-gate behaviour look."""
        ex, _ = self._extractor(lambda line: "")
        assert ex._on_path_lines("faiss::probe") is not None


# ─── A2: the segment scan skips unreachable calls and predicates ─────────────


class TestSegmentFiltering:
    def _setup(self, tmp_path):
        src = tmp_path / "probe.cpp"
        src.write_text("\n".join(f"// line {i}" for i in range(1, 20)))
        caller = _pdg("faiss::probe", str(src), [
            _branch_node("P_1", 1, 1, "x > 0"),
            _branch_node("P_10", 10, 2, "y > 0"),
            _branch_node("P_12", 12, 3, "y < 0"),
            PDGNode(id="C_10", kind="call_expr", expression="leaf()",
                    source_line=10, block_id=2, is_call=True, callee="leaf"),
            PDGNode(id="C_12", kind="call_expr", expression="leaf()",
                    source_line=12, block_id=3, is_call=True, callee="leaf"),
            # The exit block's own line — the segment's ``line_end`` must be
            # reachable, and reachability is read off the PDG's block map.
            PDGNode(id="N_14", kind="event", source_line=14, block_id=4),
        ], [ControlDepEdge(source_id="P_1", target_id="P_10"),
            ControlDepEdge(source_id="P_10", target_id="C_10"),
            ControlDepEdge(source_id="P_12", target_id="C_12")])
        leaf = _pdg("leaf", str(tmp_path / "leaf.cpp"), [
            _branch_node("L_3", 3, 1, "n > 1"),
        ], [ControlDepEdge(source_id="L_3", target_id="L_3")])
        sdg = _SDG({"faiss::probe": caller, "leaf": leaf})
        cfg = CFGFunction(
            function_name="faiss::probe",
            blocks={
                "B1": _block("B1", terminator="if [B1.1]", succs=["B2", "B3"]),
                "B2": _block("B2", lines=[10], succs=["B4"]),
                "B3": _block("B3", lines=[12], succs=["B4"]),
                "B4": _block("B4", lines=[14], label="EXIT"),
            },
            entry_block_id="B1", exit_block_id="B4",
        )
        return sdg, cfg, str(tmp_path)

    def _extract(self, sdg, cfg, annotations):
        ex = CFGBranchExtractor({"faiss::probe": cfg}, sdg, SliceMask(),
                                path_annotations=annotations)
        tree = ex.extract(
            segment_index=0, function_name="faiss::probe",
            line_start=1, line_end=14,
            postcondition=None,
        )
        return tree

    def test_off_path_predicate_and_call_are_dropped(self, tmp_path):
        sdg, cfg, root = self._setup(tmp_path)
        tree = self._extract(sdg, cfg, {("faiss::probe", 1): True})
        # line 10 only: the L12 predicate and its call sit in the arm the
        # report did not take.
        assert {b.source_line for b in tree.root_branches} == {1, 10}

    def test_no_annotations_keeps_everything(self, tmp_path):
        sdg, cfg, root = self._setup(tmp_path)
        tree = self._extract(sdg, cfg, None)
        assert {b.source_line for b in tree.root_branches} == {1, 10, 12}


# ─── B: slice-retention gating keeps the leaf-sentinel contract ──────────────


class TestSliceGating:
    def _tree(self, retained, depth_gate=0):
        callee = _pdg("faiss::deep", "/p/deep.cpp", [
            _branch_node("D_5", 5, 1, "k > 0"),
        ], [ControlDepEdge(source_id="D_5", target_id="D_5")])
        caller = _pdg("faiss::probe", "/p/probe.cpp", [
            _branch_node("P_1", 1, 1, "x"),
            PDGNode(id="C_1", kind="call_expr", expression="deep()",
                    source_line=1, block_id=1, is_call=True, callee="deep"),
        ], [ControlDepEdge(source_id="P_1", target_id="C_1")])
        sdg = _SDG({"faiss::probe": caller, "faiss::deep": callee})
        cfg = CFGFunction(
            function_name="faiss::probe",
            blocks={"B1": _block("B1", lines=[1], label="EXIT")},
            entry_block_id="B1", exit_block_id="B1",
        )
        ex = CFGBranchExtractor({"faiss::probe": cfg}, sdg,
                               SliceMask(pdg_retained_nodes=retained),
                               shallow_callee_depth=depth_gate)
        return ex

    def test_retained_callee_is_expanded(self):
        ex = self._tree({"faiss::deep": {"D_5"}})
        assert ex._expansion_mode(ex._sdg.get_pdg("faiss::deep"), 2) == "full"

    def test_unretained_callee_is_gated(self):
        ex = self._tree({"faiss::other": {"N_1"}})
        assert ex._expansion_mode(ex._sdg.get_pdg("faiss::deep"), 2) == "none"

    def test_shallow_depth_shields_direct_callees(self):
        ex = self._tree({"faiss::other": {"N_1"}}, depth_gate=1)
        assert ex._expansion_mode(ex._sdg.get_pdg("faiss::deep"), 1) == "full"

    def test_gated_callee_still_yields_a_leaf_sentinel(self):
        ex = self._tree({"faiss::other": {"N_1"}})
        node = ex._build_callee_node(
            callee_pdg=ex._sdg.get_pdg("faiss::deep"),
            callee_name="deep", call_line=5,
            parent_fn="faiss::probe", depth=1,
        )
        assert node is not None
        assert node.callee_name == "deep"
        assert node.condition_expr == "→ deep()"
        assert node.node_id == "callee_faiss::probe_deep_L5"
        assert node.children == []
        # The node id is still a call edge for the CI grouping in path_space.
        assert _normalize_call_edge(node.node_id) == "callee_faiss::probe_deep"

    def test_empty_callee_adds_no_node(self):
        empty = _pdg("faiss::trivial", "/p/t.cpp", [])
        ex = self._tree({"faiss::other": {"N_1"}})
        assert ex._build_callee_node(
            callee_pdg=empty, callee_name="trivial", call_line=9,
            parent_fn="faiss::probe", depth=1,
        ) is None


# ─── C(i): the recursion stack ───────────────────────────────────────────────


class TestRecursionStack:
    def test_callee_on_the_stack_is_not_re_entered(self):
        a = _pdg("faiss::read_index", "/p/a.cpp", [
            _branch_node("A_1", 1, 1, "x"),
        ], [ControlDepEdge(source_id="A_1", target_id="A_1")])
        ex = CFGBranchExtractor({}, _SDG({"faiss::read_index": a}), SliceMask(),
                                shallow_callee_depth=99)
        ex._expansion_stack = ("faiss::read_index",)
        assert ex._expansion_mode(a, 2) == "none"
        ex._expansion_stack = ("faiss::read_ivf_header",)
        assert ex._expansion_mode(a, 2) == "full"

    def test_stack_is_restored_after_expansion(self):
        callee = _pdg("faiss::leaf_fn", "/p/l.cpp", [
            _branch_node("L_1", 1, 1, "y"),
        ], [ControlDepEdge(source_id="L_1", target_id="L_1")])
        ex = CFGBranchExtractor({}, _SDG({"faiss::leaf_fn": callee}), SliceMask(),
                                shallow_callee_depth=99)
        ex._build_callee_node(callee_pdg=callee, callee_name="leaf_fn",
                              call_line=1, parent_fn="faiss::probe", depth=0)
        assert ex._expansion_stack == ()


# ─── C(ii): environment code is not walked into ──────────────────────────────


class TestEnvironmentCode:
    def test_system_header_callee_is_shallow(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir()
        system = _pdg("std::vector<int>::resize",
                      "/usr/include/c++/10/bits/stl_vector.h", [
                          _branch_node("V_1", 1, 1, "__new_size > size()"),
                      ], [ControlDepEdge(source_id="V_1", target_id="V_1")])
        ex = CFGBranchExtractor({}, _SDG({}), SliceMask(),
                                shallow_callee_depth=99,
                                source_root=str(project))
        assert ex._expansion_mode(system, 1) == "shallow"

    def test_project_callee_is_not_shallow(self, tmp_path):
        project = tmp_path / "project"
        src = project / "sub" / "read.cpp"
        src.parent.mkdir(parents=True)
        src.write_text("// x")
        callee = _pdg("faiss::read_NSG", str(src), [
            _branch_node("N_1", 1, 1, "z"),
        ], [ControlDepEdge(source_id="N_1", target_id="N_1")])
        ex = CFGBranchExtractor({}, _SDG({}), SliceMask(),
                                shallow_callee_depth=99,
                                source_root=str(project))
        assert ex._expansion_mode(callee, 1) == "full"


# ─── C(iii): serialization cap keeps candidate equivalence intact ────────────


class TestSerializationCap:
    def test_cap_keeps_signature_and_ci_keys_full(self):
        decisions = [
            {"node_id": f"callee_a_b_L{i}", "branch_taken": True,
             "condition_expr": "c", "source_line": i}
            for i in range(1, _MAX_SERIALIZED_DECISIONS + 25)
        ]
        keys = sorted({_normalize_call_edge(d["node_id"]) for d in decisions})
        entry = {
            "ci_call_edges_only": True,
            "ci_branch_keys": keys,
            "candidates": [{
                "branch_decisions": decisions[:_MAX_SERIALIZED_DECISIONS],
                "num_branch_decisions": len(decisions),
                "signature": _segment_signature(decisions),
            }],
        }
        assert ci_branch_set_of(entry) == frozenset(keys)
        assert len(entry["candidates"][0]["branch_decisions"]) == \
            _MAX_SERIALIZED_DECISIONS
        # The signature is computed over the FULL list, so a candidate is still
        # identifiable as itself after the cap.
        assert entry["candidates"][0]["signature"].count("L") == len(decisions)

    def test_own_body_branch_disables_ci_grouping(self):
        entry = {"ci_call_edges_only": False, "ci_branch_keys": ["callee_a_b"]}
        assert ci_branch_set_of(entry) == frozenset()
