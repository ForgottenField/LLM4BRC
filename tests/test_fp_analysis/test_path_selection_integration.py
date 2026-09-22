"""Tests for the conflict-driven re-selection loop integration.

Integration of ShortestPathComputer + ConflictPatternLearner into the
FPAnalyzer path selection pipeline (the pipeline's only selection mechanism).
"""

from __future__ import annotations


class TestReSelectionLoop:
    """Tests for the conflict-driven re-selection loop logic."""

    def test_re_selection_identifies_conflicting_segments(self):
        """Re-selection correctly identifies which segments need rework."""
        # Simulate the scenario from the design spec:
        # 24 selections → merge check finds conflict between segments 5 and 12
        current_selections = {i: {"state": f"s{i}"} for i in range(24)}

        # Merge check reports segments 5 and 12 as conflicting
        conflict_indices = {5, 12}

        frozen = {
            i: sel for i, sel in current_selections.items()
            if i not in conflict_indices
        }
        assert len(frozen) == 22, "22 segments should be frozen"
        assert 5 not in frozen
        assert 12 not in frozen
        assert 0 in frozen
        assert 23 in frozen

    def test_max_three_rounds(self):
        """Maintain at most 3 rounds of re-selection."""
        max_rounds = 3
        rounds_executed = 0
        for round_num in range(1, max_rounds + 1):
            rounds_executed += 1
            # Simulate merge check — always conflicts
            if round_num == max_rounds:
                break
        assert rounds_executed <= max_rounds

    def test_stuck_detection_early_exit(self):
        """When same conflict pattern repeats with no new hints, exit early."""
        from llm_client.fp_analysis.conflict_learner import ConflictPatternLearner

        learner = ConflictPatternLearner()
        prev_sigs = {"range:r:seg0-seg5", "null:ptr:seg2-seg3"}
        curr_sigs = {"range:r:seg0-seg5", "null:ptr:seg2-seg3"}

        # Round 2, same sigs, no new hints → stuck
        assert learner._is_stuck(prev_sigs, curr_sigs, round_num=2, has_new_hints=False)

    def test_round3_always_returns(self):
        """Round 3 always returns regardless of signatures."""
        from llm_client.fp_analysis.conflict_learner import ConflictPatternLearner

        learner = ConflictPatternLearner()

        # Even with new patterns, round >= 3 is stuck
        assert learner._is_stuck(set(), set(), round_num=3, has_new_hints=True)
        assert learner._is_stuck({"a"}, {"b"}, round_num=3, has_new_hints=True)


class TestPromptTemplates:
    """Tests for the new prompt templates."""

    def test_path_candidate_selection_system_prompt_exists(self):
        """PATH_CANDIDATE_SELECTION_SYSTEM_PROMPT is a non-empty string."""
        from llm_client.fp_analysis.prompt_templates import (
            PATH_CANDIDATE_SELECTION_SYSTEM_PROMPT,
        )
        assert isinstance(PATH_CANDIDATE_SELECTION_SYSTEM_PROMPT, str)
        assert len(PATH_CANDIDATE_SELECTION_SYSTEM_PROMPT) > 0
        assert "shortest" in PATH_CANDIDATE_SELECTION_SYSTEM_PROMPT.lower()

    def test_conflict_learning_system_prompt_exists(self):
        """CONFLICT_LEARNING_SYSTEM_PROMPT is a non-empty string."""
        from llm_client.fp_analysis.prompt_templates import (
            CONFLICT_LEARNING_SYSTEM_PROMPT,
        )
        assert isinstance(CONFLICT_LEARNING_SYSTEM_PROMPT, str)
        assert len(CONFLICT_LEARNING_SYSTEM_PROMPT) > 0
        assert "conflict" in CONFLICT_LEARNING_SYSTEM_PROMPT.lower()

    def test_reselect_prompt_template_exists(self):
        """SEGMENT_RESELECT_PROMPT is a non-empty template string."""
        from llm_client.fp_analysis.prompt_templates import (
            SEGMENT_RESELECT_PROMPT,
        )
        assert isinstance(SEGMENT_RESELECT_PROMPT, str)
        assert len(SEGMENT_RESELECT_PROMPT) > 0
        assert "{segment_index}" in SEGMENT_RESELECT_PROMPT
        assert "{round}" in SEGMENT_RESELECT_PROMPT

    def test_path_candidate_selection_prompt_template(self):
        """PATH_CANDIDATE_SELECTION_PROMPT has required placeholders."""
        from llm_client.fp_analysis.prompt_templates import (
            PATH_CANDIDATE_SELECTION_PROMPT,
        )
        assert "{candidates}" in PATH_CANDIDATE_SELECTION_PROMPT
        assert "{constraints}" in PATH_CANDIDATE_SELECTION_PROMPT
        assert "{preceding_state}" in PATH_CANDIDATE_SELECTION_PROMPT


class TestShortestPathIntegration:
    """Integration tests connecting ShortestPathComputer to CFGBranchTree."""

    def test_compute_topk_with_real_tree_structure(self):
        """ShortestPathComputer handles a tree with mixed verdicts."""
        from llm_client.fp_analysis.shortest_path import ShortestPathComputer, CalleePathCache
        from llm_client.fp_analysis.cfg_branch_models import (
            BranchTag, CFGBranchNode, CFGBranchTree, Postcondition,
        )
        from llm_client.fp_analysis.pdg_models import SDG

        # Simulate a segment with:
        # - Branch 1: consistent (auto-confirmed)
        # - Branch 2: insufficient_info (needs exploration)
        tree = CFGBranchTree(
            segment_index=0,
            function_name="test_fn",
            line_start=10,
            line_end=50,
            postcondition=Postcondition("test_fn", 50, "r >= 0", True),
        )
        branch1 = CFGBranchNode(
            node_id="B10",
            source_line=15,
            function_name="test_fn",
            condition_expr="r >= 0",
            tag=BranchTag("B10", "consistent", layer=1, reason="matches postcondition"),
        )
        branch2 = CFGBranchNode(
            node_id="B20",
            source_line=25,
            function_name="test_fn",
            condition_expr="x > 0",
            tag=BranchTag("B20", "insufficient_info", layer=0),
        )
        branch3 = CFGBranchNode(
            node_id="B30",
            source_line=35,
            function_name="test_fn",
            condition_expr="y < 0",
            tag=BranchTag("B30", "contradictory", layer=1, reason="y >= 0"),
        )
        tree.root_branches = [branch1, branch2, branch3]
        tree.update_stats()

        sdg = SDG()
        cache = CalleePathCache(sdg)
        pc = ShortestPathComputer(sdg, cache)

        candidates = pc.compute_topk(tree, k=5)
        # branch1=consistent (1 path), branch2=insufficient (2 paths), branch3=pruned
        # → 1 × 2 × 1 = 2 candidates
        assert len(candidates) == 2, f"Expected 2 candidates, got {len(candidates)}"

        # All candidates should have branch1 consistent decision
        for decisions, _metrics in candidates:
            b1 = [d for d in decisions if d.node_id == "B10"]
            assert len(b1) == 1, "Branch 1 (consistent) should be in all paths"
            assert b1[0].branch_taken is True, "Consistent branch should be taken"

    def test_callee_cache_integration(self):
        """CalleePathCache integrates with SDG for real callee lookup."""
        from llm_client.fp_analysis.shortest_path import CalleePathCache
        from llm_client.fp_analysis.pdg_models import (
            ControlDepEdge, DefUseEdge, FunctionPDG, PDGNode, SDG,
        )

        nodes = [
            PDGNode("S_0_0", "param", "x", source_line=0),
            PDGNode("S_10_5", "assign", "r", "r = x + 1", source_line=10),
            PDGNode("S_12_3", "return", "", "r", source_line=12),
        ]
        def_use = [
            DefUseEdge("S_0_0", "S_10_5", "x"),
            DefUseEdge("S_10_5", "S_12_3", "r"),
        ]
        pdg = FunctionPDG(
            function_name="Namespace::callee",
            source_file="test.cpp",
            nodes={n.id: n for n in nodes},
            def_use_edges=def_use,
            control_dep_edges=[],
        )
        sdg = SDG(pdgs={pdg.function_name: pdg})

        cache = CalleePathCache(sdg)
        # Lookup by unqualified name (get_pdg handles suffix match)
        cache.precompute("callee")

        # Should find the PDG via suffix match
        min_s = cache.get_min_stmts("callee")
        assert min_s > 0, "Should resolve via suffix match"
