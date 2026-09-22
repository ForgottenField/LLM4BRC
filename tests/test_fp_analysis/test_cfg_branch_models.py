"""Tests for CFG branch data models — Postcondition, CFGBranchNode, BranchTag, etc."""

from __future__ import annotations

from llm_client.fp_analysis.cfg_branch_models import (
    BranchTag,
    CFGBranchNode,
    CFGBranchTree,
    CalleeReturnInfo,
    Postcondition,
    ReturnValueMap,
    VerdictType,
)


class TestPostcondition:
    """Postcondition — from control events to structured constraints."""

    def test_construct_from_fields(self):
        """Postcondition can be constructed with explicit field values."""
        pc = Postcondition(
            function_name="wslay_event_recv",
            source_line=573,
            condition_expr="r >= 0",
            branch_taken=True,
        )
        assert pc.function_name == "wslay_event_recv"
        assert pc.source_line == 573
        assert pc.condition_expr == "r >= 0"
        assert pc.branch_taken is True

    def test_from_control_event(self):
        """Postcondition.from_control_event parses a CSA control event dict."""
        event = {
            "event_kind": "control",
            "description": "← Taking true branch →",
            "line_number": 573,
            "branch_condition": True,
            "function_name": "wslay_event_recv",
        }
        pc = Postcondition.from_control_event(event)
        assert pc.function_name == "wslay_event_recv"
        assert pc.source_line == 573
        assert pc.branch_taken is True
        assert pc.condition_expr != ""

    def test_negate_branch(self):
        """negate() flips the branch direction."""
        pc = Postcondition("f", 10, "r >= 0", True)
        neg = pc.negate()
        assert neg.branch_taken is False
        assert neg.source_line == 10

    def test_default_constructor(self):
        """Postcondition can be created with minimal fields."""
        pc = Postcondition(function_name="f", source_line=1)
        assert pc.condition_expr == ""
        assert pc.branch_taken is True


class TestBranchTag:
    """BranchTag — 3-state verdict after conflict checking."""

    def test_consistent_verdict(self):
        """A consistent verdict has the correct state and layer."""
        tag = BranchTag("b1", "consistent", layer=1, reason="r>=0 == r>=0")
        assert tag.verdict == "consistent"
        assert tag.layer == 1

    def test_contradictory_verdict(self):
        """A contradictory verdict marks the branch as impossible."""
        tag = BranchTag("b1", "contradictory", layer=2,
                        reason="return -1 violates r>=0")
        assert tag.verdict == "contradictory"

    def test_insufficient_info(self):
        """Insufficient_info means the branch needs LLM judgement."""
        tag = BranchTag.info(CFGBranchNode("b1", 10), "variable not in scope")
        assert tag.verdict == "insufficient_info"
        assert tag.layer == 0

    def test_verdict_type_is_literal(self):
        """VerdictType must be one of the three allowed values."""
        v: VerdictType = "consistent"
        assert v in ("consistent", "contradictory", "insufficient_info")


class TestCFGBranchNode:
    """CFGBranchNode — a single branch in the CFG tree."""

    def test_minimal_construction(self):
        """A branch node can be created with minimal fields."""
        node = CFGBranchNode("seg0_B4_taken", 573)
        assert node.node_id == "seg0_B4_taken"
        assert node.source_line == 573
        assert node.condition_expr == ""
        assert node.children == []
        assert node.tag is None

    def test_full_construction(self):
        """A branch node with all fields populated."""
        node = CFGBranchNode(
            node_id="seg0_B4_taken",
            source_line=573,
            function_name="wslay_event_recv",
            condition_expr="r >= 0",
            is_taken_branch=True,
            call_depth=0,
            blocks_taken=["B5", "B6"],
            blocks_not_taken=["B12"],
        )
        assert node.is_taken_branch is True
        assert node.call_depth == 0
        assert "B5" in node.blocks_taken
        assert "B12" in node.blocks_not_taken

    def test_with_child_branches(self):
        """A branch node can have nested children (callee branches)."""
        parent = CFGBranchNode("seg0_B3_call", 572)
        child = CFGBranchNode(
            "seg0_wf_B1_taken", 42,
            function_name="wslay_frame_recv",
            call_depth=1,
            parent_call_site="L572 wslay_frame_recv",
        )
        parent.children.append(child)
        assert len(parent.children) == 1
        assert parent.children[0].function_name == "wslay_frame_recv"
        assert parent.children[0].call_depth == 1

    def test_with_tag(self):
        """A branch node can carry a BranchTag after checking."""
        node = CFGBranchNode("b1", 10)
        tag = BranchTag("b1", "consistent", 1, "direct match")
        node.tag = tag
        assert node.tag.verdict == "consistent"
        assert node.tag.layer == 1

    def test_branch_label_mapping(self):
        """is_taken_branch maps to human-readable label."""
        node_true = CFGBranchNode("b1", 10, is_taken_branch=True, branch_label="true")
        node_false = CFGBranchNode("b2", 20, is_taken_branch=False, branch_label="false")
        assert node_true.is_taken_branch is True
        assert node_false.is_taken_branch is False


class TestCFGBranchTree:
    """CFGBranchTree — the complete branch tree for one segment."""

    def test_empty_tree(self):
        """A tree with no branches has zero counts."""
        pc = Postcondition("f", 10, "x > 0", True)
        tree = CFGBranchTree(segment_index=0, function_name="f",
                             line_start=1, line_end=10, postcondition=pc)
        assert tree.segment_index == 0
        assert tree.num_branches == 0
        assert tree.num_consistent == 0
        assert tree.num_contradictory == 0
        assert tree.num_insufficient == 0

    def test_add_branch(self):
        """Adding a branch increments the counter."""
        pc = Postcondition("f", 10, "x > 0", True)
        tree = CFGBranchTree(0, "f", 1, 10, pc)
        tree.add_branch(CFGBranchNode("b1", 5))
        assert tree.num_branches == 1
        assert len(tree.root_branches) == 1

    def test_update_stats_with_tags(self):
        """update_stats counts consistent/contradictory/insufficient across all nodes."""
        pc = Postcondition("f", 10, "x > 0", True)
        tree = CFGBranchTree(0, "f", 1, 10, pc)
        b1 = CFGBranchNode("b1", 5)
        b1.tag = BranchTag("b1", "consistent", 1, "ok")
        b2 = CFGBranchNode("b2", 6)
        b2.tag = BranchTag("b2", "contradictory", 2, "bad")
        b3 = CFGBranchNode("b3", 7)
        b3.tag = BranchTag("b3", "insufficient_info", 0, "?")
        tree.add_branch(b1)
        tree.add_branch(b2)
        tree.add_branch(b3)
        tree.update_stats()
        assert tree.num_branches == 3
        assert tree.num_consistent == 1
        assert tree.num_contradictory == 1
        assert tree.num_insufficient == 1

    def test_update_stats_includes_children(self):
        """update_stats recurses into child branches."""
        pc = Postcondition("f", 10, "x > 0", True)
        tree = CFGBranchTree(0, "f", 1, 10, pc)
        parent = CFGBranchNode("b1", 5)
        parent.tag = BranchTag("b1", "consistent", 1, "ok")
        child = CFGBranchNode("b1_child", 6)
        child.tag = BranchTag("b1_child", "contradictory", 2, "bad")
        parent.children.append(child)
        tree.add_branch(parent)
        tree.update_stats()
        assert tree.num_branches == 2
        assert tree.num_consistent == 1
        assert tree.num_contradictory == 1

    def test_tree_with_no_tags(self):
        """update_stats on untagged tree: all counts remain zero."""
        pc = Postcondition("f", 10, "x > 0", True)
        tree = CFGBranchTree(0, "f", 1, 10, pc)
        tree.add_branch(CFGBranchNode("b1", 5))
        tree.add_branch(CFGBranchNode("b2", 6))
        tree.update_stats()
        assert tree.num_consistent == 0
        assert tree.num_contradictory == 0


class TestReturnValueMap:
    """ReturnValueMap — postcondition-to-callee return path mapping."""

    def test_construct_return_map(self):
        """ReturnValueMap can store callee return mappings."""
        ret_info = CalleeReturnInfo(
            return_value="WSLAY_ERR_WANT_READ",
            return_line=88,
            blocks=["B_wf4"],
            compatible=False,
        )
        rmap = ReturnValueMap(
            post_var="r",
            post_cond="r >= 0",
            call_site=("wslay_event_recv", 572),
            callee="wslay_frame_recv",
            callee_returns=[ret_info],
        )
        assert rmap.post_var == "r"
        assert rmap.callee == "wslay_frame_recv"
        assert len(rmap.callee_returns) == 1
        assert rmap.callee_returns[0].compatible is False

    def test_multiple_returns(self):
        """A callee can have multiple return paths with different compatibility."""
        returns = [
            CalleeReturnInfo(-1, 42, ["B1"], compatible=False),
            CalleeReturnInfo(-2, 88, ["B4"], compatible=False),
            CalleeReturnInfo("nread", 100, ["B5"], compatible=True),
        ]
        rmap = ReturnValueMap("r", "r >= 0", ("f", 10), "g", returns)
        compatible = [r for r in rmap.callee_returns if r.compatible]
        incompatible = [r for r in rmap.callee_returns if not r.compatible]
        assert len(compatible) == 1
        assert len(incompatible) == 2


class TestCalleeReturnInfo:
    """CalleeReturnInfo — one return path in a callee function."""

    def test_default_compatible(self):
        """compatible defaults to True."""
        info = CalleeReturnInfo(0, 10, ["B5"])
        assert info.compatible is True

    def test_with_branch_decisions(self):
        """CalleeReturnInfo can carry CFG path information."""
        info = CalleeReturnInfo(
            return_value="nread",
            return_line=100,
            blocks=["B_wf2", "B_wf4", "B_wf5"],
        )
        assert "B_wf4" in info.blocks
        assert info.return_line == 100


class TestCFGBranchNodeEdgeCases:
    """Edge cases for CFGBranchNode."""

    def test_empty_blocks(self):
        """blocks_taken and blocks_not_taken can be empty."""
        node = CFGBranchNode("b1", 10)
        assert node.blocks_taken == []
        assert node.blocks_not_taken == []

    def test_no_children(self):
        """A leaf node has no children by default."""
        node = CFGBranchNode("b1", 10)
        assert node.children == []

    def test_callee_fields(self):
        """Callee-related fields are optional."""
        node = CFGBranchNode("b1", 10)
        assert node.callee_name is None
        assert node.parent_call_site is None

        node.callee_name = "wslay_frame_recv"
        node.parent_call_site = "L572 wslay_frame_recv"
        assert node.callee_name == "wslay_frame_recv"

    def test_resolved_expr_default_empty(self):
        """resolved_expr defaults to empty string."""
        node = CFGBranchNode("b1", 10)
        assert node.resolved_expr == ""

    def test_resolved_expr_set(self):
        """resolved_expr holds human-readable condition separately from condition_expr."""
        node = CFGBranchNode("b1", 10, condition_expr="[B6.15] != [B6.17]",
                             resolved_expr="ctx->imsg != ctx->imsgs")
        assert node.resolved_expr == "ctx->imsg != ctx->imsgs"
        assert node.condition_expr == "[B6.15] != [B6.17]"
        # existing condition_expr should be preserved
        assert node.condition_expr != node.resolved_expr
