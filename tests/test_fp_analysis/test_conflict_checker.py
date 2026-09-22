"""Tests for ConflictChecker — 4-layer branch conflict analysis."""

from __future__ import annotations

import pytest
from llm_client.fp_analysis.cfg_branch_models import (
    BranchTag,
    CFGBranchNode,
    CFGBranchTree,
    CalleeReturnInfo,
    Postcondition,
    ReturnValueMap,
)
from llm_client.fp_analysis.pdg_models import PDGNode, FunctionPDG
from llm_client.fp_analysis.conflict_checker import (
    CFGReachabilityChecker,
    ConflictChecker,
    DirectVariableMatcher,
    IndirectVariableTracker,
    ReturnValueMapper,
)
from llm_client.fp_analysis.cfg_models import CFGBlock, CFGFunction, SegmentInfo
from llm_client.fp_analysis.slice_mask import SliceMask


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def pc_r_ge_0() -> Postcondition:
    return Postcondition("wslay_event_recv", 573, "r >= 0", branch_taken=True)


@pytest.fixture
def branch_r_ge_0() -> CFGBranchNode:
    return CFGBranchNode("B4_taken", 573, condition_expr="r >= 0",
                         is_taken_branch=True)


@pytest.fixture
def branch_r_lt_0() -> CFGBranchNode:
    return CFGBranchNode("B4_not_taken", 573, condition_expr="r < 0",
                         is_taken_branch=False)


@pytest.fixture
def branch_opcode_ping() -> CFGBranchNode:
    return CFGBranchNode("B732_taken", 732,
                         condition_expr="ctx->imsg->opcode == WSLAY_PING",
                         is_taken_branch=True)


@pytest.fixture
def branch_opcode_close() -> CFGBranchNode:
    return CFGBranchNode("B704_taken", 704,
                         condition_expr="ctx->imsg->opcode == WSLAY_CONNECTION_CLOSE",
                         is_taken_branch=True)


class TestDirectVariableMatcher:
    """Layer 1: direct variable matching."""

    def test_consistent_same_condition(self, pc_r_ge_0, branch_r_ge_0):
        matcher = DirectVariableMatcher()
        tag = matcher.check(branch_r_ge_0, pc_r_ge_0, {})
        assert tag.verdict == "consistent"
        assert tag.layer == 1

    def test_contradictory_negated_condition(self, pc_r_ge_0, branch_r_lt_0):
        matcher = DirectVariableMatcher()
        tag = matcher.check(branch_r_lt_0, pc_r_ge_0, {})
        assert tag.verdict == "contradictory"
        assert tag.layer == 1
        assert "contradict" in tag.reason.lower()

    def test_contradictory_enum_mismatch(self, branch_opcode_ping, branch_opcode_close):
        """opcode==PING vs opcode==CLOSE should be contradictory."""
        pc = Postcondition("f", 732, "ctx->imsg->opcode == WSLAY_PING", True)
        matcher = DirectVariableMatcher()
        tag = matcher.check(branch_opcode_close, pc, {})
        assert tag.verdict == "contradictory"

    def test_insufficient_info_different_vars(self, pc_r_ge_0):
        """Different variables → insufficient_info."""
        branch = CFGBranchNode("B_x", 500, condition_expr="data_length <= 0")
        matcher = DirectVariableMatcher()
        tag = matcher.check(branch, pc_r_ge_0, {})
        assert tag.verdict == "insufficient_info"

    def test_consistent_with_established_state(self, pc_r_ge_0):
        """Check branch against established_state, not just postcondition."""
        branch = CFGBranchNode("B_x", 600,
                               condition_expr="ctx->imsg->opcode == WSLAY_PING")
        state = {"ctx->imsg->opcode": "== WSLAY_PING"}
        matcher = DirectVariableMatcher()
        tag = matcher.check(branch, pc_r_ge_0, state)
        assert tag.verdict == "consistent"

    def test_contradictory_with_established_state(self, pc_r_ge_0):
        """Branch contradicts an established state variable."""
        branch = CFGBranchNode("B_x", 600,
                               condition_expr="ctx->imsg->opcode == WSLAY_CLOSE")
        state = {"ctx->imsg->opcode": "== WSLAY_PING"}
        matcher = DirectVariableMatcher()
        tag = matcher.check(branch, pc_r_ge_0, state)
        assert tag.verdict == "contradictory"

    def test_bool_vs_ge_zero(self):
        """'while(read_enabled)' vs 'r>=0' postcondition → insufficient_info."""
        pc = Postcondition("f", 100, "r >= 0", True)
        branch = CFGBranchNode("B1", 100,
                               condition_expr="ctx->read_enabled",
                               is_taken_branch=True)
        matcher = DirectVariableMatcher()
        tag = matcher.check(branch, pc, {})
        assert tag.verdict == "insufficient_info"

    def test_empty_condition(self):
        """Empty branch condition → insufficient_info."""
        pc = Postcondition("f", 10, "x > 0", True)
        branch = CFGBranchNode("b1", 10, condition_expr="")
        matcher = DirectVariableMatcher()
        tag = matcher.check(branch, pc, {})
        assert tag.verdict == "insufficient_info"


class TestDirectVariableMatcherResolvedExpr:
    """Layer 1 matching against resolved_expr (Clang element-reference form)."""

    @pytest.fixture
    def matcher(self):
        return DirectVariableMatcher()

    def test_matches_resolved_expr(self, matcher):
        """When branch has resolved_expr, match against it instead of raw condition.

        Uses a simple resolved expression that the _compare_variable
        logic can handle: same variable and same relation.
        """
        pc = Postcondition("f", 100, "r >= 0", True)
        branch = CFGBranchNode(
            "b1", 100,
            condition_expr="[B6.15] != [B6.11]",  # raw element-reference
            resolved_expr="r >= 0",  # human-readable, matches postcondition
        )
        tag = matcher.check(branch, pc, {})
        assert tag.verdict == "consistent", f"expected consistent, got {tag}"

    def test_resolved_expr_contradictory(self, matcher):
        """Resolved expression contradicts postcondition (negated relation)."""
        pc = Postcondition("f", 100, "r >= 0", True)
        branch = CFGBranchNode(
            "b1", 100,
            condition_expr="[B6.15] != [B6.11]",
            resolved_expr="r < 0",  # negated relation vs postcondition
        )
        tag = matcher.check(branch, pc, {})
        assert tag.verdict == "contradictory", f"expected contradictory, got {tag}"

    def test_resolved_expr_vs_established_state_consistent(self, matcher):
        """Resolved expression matches established state."""
        pc = Postcondition("f", 100, "r >= 0", True)  # unrelated to state var
        branch = CFGBranchNode(
            "b1", 200,
            condition_expr="[B6.14] != [B6.11]",
            resolved_expr="ctx->imsg->opcode == WSLAY_PING",
        )
        state = {"ctx->imsg->opcode": "== WSLAY_PING"}
        tag = matcher.check(branch, pc, state)
        assert tag.verdict == "consistent", f"expected consistent, got {tag}"

    def test_resolved_expr_vs_established_state_contradictory(self, matcher):
        """Resolved expression contradicts established state."""
        pc = Postcondition("f", 100, "r >= 0", True)
        branch = CFGBranchNode(
            "b1", 200,
            condition_expr="[B6.14] != [B6.11]",
            resolved_expr="ctx->imsg->opcode == WSLAY_CLOSE",
        )
        state = {"ctx->imsg->opcode": "== WSLAY_PING"}
        tag = matcher.check(branch, pc, state)
        assert tag.verdict == "contradictory", f"expected contradictory, got {tag}"

    def test_no_resolved_expr_falls_back_to_condition(self, matcher):
        """When resolved_expr is empty, use condition_expr as before."""
        pc = Postcondition("f", 100, "r >= 0", True)
        branch = CFGBranchNode(
            "b1", 100,
            condition_expr="r >= 0",
            resolved_expr="",
        )
        tag = matcher.check(branch, pc, {})
        assert tag.verdict == "consistent", f"expected consistent, got {tag}"


class TestReturnValueMapper:
    """Layer 2: return value → callee path mapping."""

    def test_constant_compatible(self):
        """A callee return value that satisfies postcondition is compatible."""
        mapper = ReturnValueMapper()
        pc = Postcondition("caller", 10, "r >= 0", True)
        assert mapper.eval_constants_compat(0, pc) is True
        assert mapper.eval_constants_compat(42, pc) is True
        assert mapper.eval_constants_compat(-1, pc) is False
        assert mapper.eval_constants_compat(-2, pc) is False

    def test_constant_compat_equal(self):
        """Postcondition 'r == WSLAY_PING' → only matching value is compatible."""
        mapper = ReturnValueMapper()
        pc = Postcondition("f", 10, "opcode == WSLAY_PING", True)
        assert mapper.eval_constants_compat("WSLAY_PING", pc) is True
        assert mapper.eval_constants_compat("WSLAY_CLOSE", pc) is False

    def test_constant_compat_not_equal(self):
        """Postcondition 'opcode != WSLAY_CLOSE' → matching value is incompatible."""
        mapper = ReturnValueMapper()
        pc = Postcondition("f", 10, "opcode != WSLAY_CONNECTION_CLOSE", True)
        assert mapper.eval_constants_compat("WSLAY_CONNECTION_CLOSE", pc) is False
        assert mapper.eval_constants_compat("WSLAY_PING", pc) is True


class TestIndirectVariableTracker:
    """Layer 3: indirect variable tracking through pointer/ref parameters."""

    def test_insufficient_info_no_indirect_vars(self, pc_r_ge_0):
        """No indirect var info → insufficient_info."""
        tracker = IndirectVariableTracker()
        branch = CFGBranchNode("B_x", 600, condition_expr="iocb.opcode == TEXT")
        tag = tracker.check(branch, pc_r_ge_0, SliceMask(), [])
        assert tag.verdict == "insufficient_info"

    def test_consistent_when_guarded_by_postcondition(self):
        """Postcondition r>=0 → indirect vars are guarded."""
        tracker = IndirectVariableTracker()
        pc = Postcondition("f", 100, "r >= 0", True)
        branch = CFGBranchNode("B_x", 200,
                               condition_expr="iocb.data_length == 0")
        tag = tracker.check(branch, pc, SliceMask(), [])
        assert tag.verdict == "insufficient_info"  # No SDG data


class TestCFGReachabilityChecker:
    """Layer 4: structural CFG reachability (belt-and-suspenders)."""

    def test_insufficient_info_by_default(self):
        """Without CFG data, reachability is undetermined."""
        checker = CFGReachabilityChecker({})
        branch = CFGBranchNode("b1", 10)
        tag = checker.check(branch, CFGBranchTree(0, "f", 1, 10,
                            Postcondition("f", 5, "x", True)))
        assert tag.verdict == "insufficient_info"


class TestCFGReachabilityCheckerFull:
    """Layer 4: CFG structural reachability tests."""

    @staticmethod
    def _make_block(bid: str, stmts: list[str],
                    terminator: str | None = None,
                    succs: list[str] | None = None) -> CFGBlock:
        return CFGBlock(block_id=bid, statements=stmts,
                        terminator=terminator, successors=succs or [])

    @pytest.fixture
    def chain_cfg(self) -> dict[str, CFGFunction]:
        """Simple chain: B1->B2->B3->EXIT. Branch at B2."""
        blocks = {
            "B1": self._make_block("B1", [], succs=["B2"]),
            "B2": self._make_block("B2", [],
                                    terminator="if condition",
                                    succs=["B3", "B0"]),
            "B3": self._make_block("B3", [], succs=["B0"]),
            "B0": CFGBlock(block_id="B0", statements=[""], label="EXIT"),
        }
        cfg = CFGFunction(function_name="f", blocks=blocks,
                          entry_block_id="B1")
        return {"f": cfg}

    def test_branch_reachable(self, chain_cfg):
        """Branch at B2 is reachable from entry B1 → insufficient_info."""
        checker = CFGReachabilityChecker(chain_cfg)
        tree = CFGBranchTree(0, "f", 1, 10,
                             Postcondition("f", 5, "x > 0", True))
        branch = CFGBranchNode("seg0_f_B2", 5,
                               blocks_taken=["B3"], blocks_not_taken=["B0"])
        tree.add_branch(branch)
        result = checker.check(branch, tree)
        # Reachable → need deeper analysis → insufficient_info
        assert result.verdict == "insufficient_info"

    def test_nonexistent_block(self, chain_cfg):
        """Branch successor that doesn't exist in CFG → contradictory."""
        checker = CFGReachabilityChecker(chain_cfg)
        tree = CFGBranchTree(0, "f", 1, 10,
                             Postcondition("f", 5, "x > 0", True))
        branch = CFGBranchNode("seg0_f_B2", 5,
                               blocks_taken=["B99"],  # doesn't exist
                               blocks_not_taken=["B0"])
        tree.add_branch(branch)
        result = checker.check(branch, tree)
        assert result.verdict == "contradictory"

    def test_exit_block_contradicts_continue(self, chain_cfg):
        """If postcondition says 'continue' but branch goes to EXIT → contradictory.

        For a branch where both successors are EXIT (only block B0),
        and the postcondition implies 'not exiting' → contradictory.
        """
        checker = CFGReachabilityChecker(chain_cfg)
        tree = CFGBranchTree(0, "f", 1, 10,
                             Postcondition("f", 5, "r >= 0", True))
        branch = CFGBranchNode("seg0_f_B2", 5,
                               blocks_taken=["B0"], blocks_not_taken=[])
        tree.add_branch(branch)
        result = checker.check(branch, tree)
        assert result.verdict == "contradictory"


class TestVariableInvariance:
    """Variable invariance check using PDG def-use information.

    If a variable in established_state has its last definition BEFORE
    segment entry, it is invariant (cannot change) within the segment,
    so the established constraint continues to hold.
    """

    def _make_pdg(self, fn: str, defs: dict[str, int]) -> FunctionPDG:
        """Build a minimal FunctionPDG with defs_by_variable.

        *defs* maps variable_name → last_definition_line.
        """
        nodes = {}
        for var, line in defs.items():
            nid = f"S_{line}_1"
            nodes[nid] = PDGNode(id=nid, variable=var, source_line=line,
                                  kind="assign", expression=f"{var} = ...")
        pdg = FunctionPDG(function_name=fn, source_file="test.c", nodes=nodes)
        # Manually set defs_by_variable since __post_init__ already built it
        return pdg

    def _make_sdg(self, pdg: FunctionPDG):
        from types import SimpleNamespace
        return SimpleNamespace(get_pdg=lambda fn: pdg if fn == pdg.function_name else None)

    def _make_segment_info(self, fn: str, line_start: int) -> SegmentInfo:
        return SegmentInfo(
            segment_index=0, event_range=(1, 2), line_start=line_start,
            line_end=line_start + 20, function_name=fn,
        )

    def test_variable_invariant_before_segment(self):
        """Variable defined at L5, segment starts at L20 → invariant → consistent."""
        pdg = self._make_pdg("f", {"r": 5})
        sdg = self._make_sdg(pdg)
        checker = ConflictChecker(sdg, {}, SliceMask())
        pc = Postcondition("f", 30, "x > 0", True)
        tree = CFGBranchTree(0, "f", 20, 40, pc)
        branch = CFGBranchNode("b1", 25, condition_expr="r < 0", function_name="f")
        state = {"r": "should be >= 0"}
        # Layer 1 returns insufficient_info (var 'r' in state but value comparison inconclusive)
        # Variable invariance check should find 'r' is invariant → consistent
        tag = checker._check_single(branch, pc, state, [], seg_line_start=20)
        assert tag.verdict == "consistent"

    def test_variable_defined_after_segment_entry(self):
        """Variable defined at L25, segment starts at L20 → may change → insufficient_info."""
        pdg = self._make_pdg("f", {"r": 25})
        sdg = self._make_sdg(pdg)
        checker = ConflictChecker(sdg, {}, SliceMask())
        pc = Postcondition("f", 40, "x > 0", True)
        tree = CFGBranchTree(0, "f", 20, 50, pc)
        branch = CFGBranchNode("b1", 35, condition_expr="r < 0", function_name="f")
        state = {"r": "should be >= 0"}
        tag = checker._check_single(branch, pc, state, [], seg_line_start=20)
        assert tag.verdict == "insufficient_info"

    def test_variable_not_in_pdg(self):
        """Variable not found in PDG → insufficient_info (conservative)."""
        pdg = self._make_pdg("f", {"x": 5})  # 'r' not in PDG
        sdg = self._make_sdg(pdg)
        checker = ConflictChecker(sdg, {}, SliceMask())
        pc = Postcondition("f", 40, "x > 0", True)
        tree = CFGBranchTree(0, "f", 20, 50, pc)
        branch = CFGBranchNode("b1", 35, condition_expr="r < 0", function_name="f")
        state = {"r": "should be >= 0"}
        tag = checker._check_single(branch, pc, state, [], seg_line_start=20)
        assert tag.verdict == "insufficient_info"

    def test_qualified_var_short_name_matches_pdg(self):
        """Qualified name 'ctx->imsg' - PDG stores 'ctx->imsg' as variable → consistent."""
        pdg = self._make_pdg("f", {"ctx->imsg": 5})
        sdg = self._make_sdg(pdg)
        checker = ConflictChecker(sdg, {}, SliceMask())
        pc = Postcondition("f", 30, "x > 0", True)
        tree = CFGBranchTree(0, "f", 20, 40, pc)
        branch = CFGBranchNode("b1", 25, condition_expr="ctx->imsg == WSLAY_PING", function_name="f")
        state = {"ctx->imsg": "== WSLAY_PING"}
        tag = checker._check_single(branch, pc, state, [], seg_line_start=20)
        assert tag.verdict == "consistent"

    def test_qualified_var_short_name_defined_after(self):
        """Qualified var 'ctx->imsg' defined after segment start → insufficient_info."""
        pdg = self._make_pdg("f", {"ctx->imsg": 30})
        sdg = self._make_sdg(pdg)
        checker = ConflictChecker(sdg, {}, SliceMask())
        pc = Postcondition("f", 40, "x > 0", True)
        tree = CFGBranchTree(0, "f", 20, 50, pc)
        branch = CFGBranchNode("b1", 35, condition_expr="ctx->imsg == WSLAY_PING")
        state = {"ctx->imsg": "== WSLAY_PING"}
        tag = checker._check_single(branch, pc, state, [], seg_line_start=20)
        assert tag.verdict == "insufficient_info"


class TestConflictChecker:
    """Full ConflictChecker — 4-layer orchestration."""

    def test_empty_tree_no_crash(self, pc_r_ge_0):
        """Check segment with empty branch tree returns clean."""
        tree = CFGBranchTree(0, "f", 1, 10, pc_r_ge_0)
        checker = ConflictChecker(None, {}, SliceMask())
        result = checker.check_segment(tree, None, {})
        assert result.num_branches == 0

    def test_layer1_consistent_skips_other_layers(self, pc_r_ge_0, branch_r_ge_0):
        """Layer 1 consistent → no need for deeper layers."""
        tree = CFGBranchTree(0, "f", 1, 10, pc_r_ge_0)
        tree.add_branch(branch_r_ge_0)
        checker = ConflictChecker(None, {}, SliceMask())
        result = checker.check_segment(tree, None, {})
        assert result.num_consistent == 1
        assert result.num_contradictory == 0

    def test_layer1_contradictory_marked(self, pc_r_ge_0, branch_r_lt_0):
        """Layer 1 contradictory → branch marked contradictory."""
        tree = CFGBranchTree(0, "f", 1, 10, pc_r_ge_0)
        tree.add_branch(branch_r_lt_0)
        checker = ConflictChecker(None, {}, SliceMask())
        result = checker.check_segment(tree, None, {})
        assert result.num_contradictory == 1
        assert result.num_consistent == 0

    def test_insufficient_info_no_tag(self, pc_r_ge_0):
        """Insufficient info → tagged but not consistent/contradictory."""
        tree = CFGBranchTree(0, "f", 1, 10, pc_r_ge_0)
        branch = CFGBranchNode("b_x", 500, condition_expr="data_length <= 0")
        tree.add_branch(branch)
        checker = ConflictChecker(None, {}, SliceMask())
        result = checker.check_segment(tree, None, {})
        assert result.num_insufficient == 1

    def test_multiple_branches_mixed_verdicts(self, pc_r_ge_0):
        """Several branches with different verdicts all checked correctly."""
        tree = CFGBranchTree(0, "f", 1, 10, pc_r_ge_0)
        tree.add_branch(CFGBranchNode("b1", 573, condition_expr="r >= 0"))
        tree.add_branch(CFGBranchNode("b2", 573, condition_expr="r < 0"))
        tree.add_branch(CFGBranchNode("b3", 500, condition_expr="x == 0"))
        checker = ConflictChecker(None, {}, SliceMask())
        result = checker.check_segment(tree, None, {})
        assert result.num_consistent == 1
        assert result.num_contradictory == 1
        assert result.num_insufficient == 1

    def test_tagged_nodes_skipped(self, pc_r_ge_0):
        """Already-tagged nodes are not re-checked."""
        tree = CFGBranchTree(0, "f", 1, 10, pc_r_ge_0)
        b = CFGBranchNode("b1", 573, condition_expr="r >= 0")
        b.tag = BranchTag("b1", "consistent", 1, "pre-tagged")
        tree.add_branch(b)
        checker = ConflictChecker(None, {}, SliceMask())
        result = checker.check_segment(tree, None, {})
        assert result.num_consistent == 1

    def test_check_segment_called_with_segment_info(self, pc_r_ge_0):
        """check_segment accepts a segment_info parameter."""
        tree = CFGBranchTree(0, "f", 1, 10, pc_r_ge_0)
        tree.add_branch(CFGBranchNode("b1", 573, condition_expr="r >= 0"))
        checker = ConflictChecker(None, {}, SliceMask())
        # Should not crash when segment_info is a dict
        result = checker.check_segment(tree, {"segment_index": 0}, {})
        assert result.num_consistent == 1


class TestReturnValueMapperFull:
    """Layer 2: callee return path analysis — walks callee CFG for return values."""

    @staticmethod
    def _make_block(bid: str, stmts: list[str],
                    terminator: str | None = None,
                    succs: list[str] | None = None) -> CFGBlock:
        return CFGBlock(block_id=bid, statements=stmts,
                        terminator=terminator, successors=succs or [])

    @pytest.fixture
    def mapper(self):
        return ReturnValueMapper()

    def _make_cfg(self, name: str,
                  returns: list[tuple[str, str, str]]) -> CFGFunction:
        """Build CFGFunction with return statements.

        Each entry is (block_id, return_stmt_text, condition_str_or_none).
        """
        blocks = {}
        for bid, ret_stmt, cond in returns:
            stmts = [ret_stmt] if ret_stmt else []
            if cond:
                blocks[bid] = self._make_block(
                    bid, stmts, terminator=cond,
                    succs=[f"{bid}_t", f"{bid}_f"],
                )
            else:
                blocks[bid] = self._make_block(bid, stmts, succs=["EXIT"])
        blocks["EXIT"] = CFGBlock(block_id="EXIT", statements=[""], label="EXIT")
        return CFGFunction(function_name=name, blocks=blocks,
                           entry_block_id=returns[0][0])

    def test_return_value_check_contradictory(self, mapper):
        """Callee returns error constants that contradict r >= 0."""
        pc = Postcondition("caller", 572, "r >= 0", True)
        wf_cfg = self._make_cfg("wslay_frame_recv", [
            ("wf_B1", "", "if condition"),
            ("wf_B2", "[wf_B2.1] (ReturnStmt) return -1", ""),
            ("wf_B3", "", "if r < 0"),
            ("wf_B4", "[wf_B4.1] (ReturnStmt) return -1", ""),
            ("wf_B5", "", "if r == 0"),
            ("wf_B6", "[wf_B6.1] (ReturnStmt) return -2", ""),
        ])
        cfg_cache = {"wslay_frame_recv": wf_cfg}
        branch = CFGBranchNode("call_wf", 572, callee_name="wslay_frame_recv",
                               function_name="caller",
                               condition_expr="call wslay_frame_recv")
        tag = mapper.check(branch, pc, None, SliceMask(), [], cfg_cache=cfg_cache)
        assert tag.verdict == "contradictory", f"expected contradictory, got {tag}"

    def test_return_value_check_consistent(self, mapper):
        """Callee returns 0 which is compatible with r >= 0."""
        pc = Postcondition("caller", 572, "r >= 0", True)
        wf_cfg = self._make_cfg("wslay_frame_recv", [
            ("wf_B1", "", ""),
            ("wf_B2", "[wf_B2.1] (ReturnStmt) return 0", ""),
        ])
        cfg_cache = {"wslay_frame_recv": wf_cfg}
        branch = CFGBranchNode("call_wf", 572, callee_name="wslay_frame_recv",
                               condition_expr="call wslay_frame_recv")
        tag = mapper.check(branch, pc, None, SliceMask(), [], cfg_cache=cfg_cache)
        assert tag.verdict == "consistent", f"expected consistent, got {tag}"

    def test_non_callee_node_insufficient_info(self, mapper):
        """Non-callee branch nodes skip Layer 2."""
        pc = Postcondition("f", 100, "r >= 0", True)
        branch = CFGBranchNode("b1", 100, condition_expr="x > 0")
        tag = mapper.check(branch, pc, None, SliceMask(), [])
        assert tag.verdict == "insufficient_info"
