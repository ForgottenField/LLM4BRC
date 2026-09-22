"""Tests for ConflictPatternLearner + conflict data models.

Phase 2 of the shortest-path selection + conflict-driven learning design.
"""

from __future__ import annotations

import pytest
from llm_client.fp_analysis.cfg_branch_models import BranchDecision


# ─── Conflict Models tests ─────────────────────────────────────────────────────


class TestDataflowContext:
    """Tests for DataflowContext — prevents stale conflict pruning."""

    def test_valid_when_no_reassignment(self):
        """DataflowContext.is_still_valid returns True when def sites unchanged."""
        from llm_client.fp_analysis.conflict_models import DataflowContext

        ctx = DataflowContext(
            var_def_sites={"r": ("callee", 572, "r = wslay_frame_recv(...)")},
            def_use_chains={"r": ["S_572_5", "S_573_3"]},
            reassignment_checks={"r": False},
        )

        # Same selections — valid
        current = {
            0: {"established_state": {"r": ">= 0"}},
        }
        assert ctx.is_still_valid(current) is True

    def test_stale_after_reassignment(self):
        """DataflowContext.is_still_valid returns False when a variable was
        reassigned, changing the dataflow context."""
        from llm_client.fp_analysis.conflict_models import DataflowContext

        ctx = DataflowContext(
            var_def_sites={"r": ("callee", 572, "r = wslay_frame_recv(...)")},
            def_use_chains={"r": ["S_572_5"]},
            reassignment_checks={"r": True},  # was reassigned
        )

        current = {
            1: {"established_state": {"r": ">= 0"}},
        }
        # Variable was reassigned — context is stale
        assert ctx.is_still_valid(current) is False

    def test_context_hash_changes_with_data(self):
        """context_hash changes when any field changes."""
        from llm_client.fp_analysis.conflict_models import DataflowContext

        ctx1 = DataflowContext(
            var_def_sites={"r": ("f", 10, "r = ...")},
        )
        ctx2 = DataflowContext(
            var_def_sites={"r": ("f", 20, "r = ...")},  # different line
        )
        assert ctx1.context_hash != ctx2.context_hash

    def test_empty_context_valid(self):
        """Empty DataflowContext is always valid."""
        from llm_client.fp_analysis.conflict_models import DataflowContext

        ctx = DataflowContext()
        assert ctx.is_still_valid({}) is True
        assert ctx.is_still_valid({0: {}}) is True


class TestConstraintConflict:
    """Tests for ConstraintConflict — deterministic constraint clash."""

    def test_range_contradiction_type(self):
        from llm_client.fp_analysis.conflict_models import ConstraintConflict

        cc = ConstraintConflict(
            variable="r",
            segment_a=0,
            segment_a_constraint="r >= 0",
            segment_b=5,
            segment_b_constraint="r < 0",
            conflict_type="range_contradiction",
        )
        assert cc.variable == "r"
        assert cc.conflict_type == "range_contradiction"
        assert cc.segment_a == 0
        assert cc.segment_b == 5

    def test_null_check_type(self):
        from llm_client.fp_analysis.conflict_models import ConstraintConflict

        cc = ConstraintConflict(
            variable="ptr",
            segment_a=2,
            segment_a_constraint="ptr == NULL",
            segment_b=3,
            segment_b_constraint="ptr != NULL",
            conflict_type="null_check",
        )
        assert cc.conflict_type == "null_check"


class TestHierarchicalConflictPattern:
    """Tests for the 8-layer HierarchicalConflictPattern."""

    def test_minimal_construction(self):
        from llm_client.fp_analysis.conflict_models import (
            ConstraintConflict,
            HierarchicalConflictPattern,
            ConflictSemanticType,
        )

        cc = ConstraintConflict(
            variable="r",
            segment_a=0,
            segment_a_constraint="r >= 0",
            segment_b=5,
            segment_b_constraint="r < 0",
            conflict_type="range_contradiction",
        )
        pattern = HierarchicalConflictPattern(
            conflict_id="conflict-001",
            conflicting_segments=[0, 5],
            violated_constraints=[cc],
            semantic_type=ConflictSemanticType.RETURN_VALUE_CONTRADICTION,
            conflict_objects=["r"],
            generalized_pattern="Postcondition mismatch on return value",
        )
        assert pattern.conflict_id == "conflict-001"
        assert pattern.conflicting_segments == [0, 5]
        assert pattern.semantic_type == ConflictSemanticType.RETURN_VALUE_CONTRADICTION
        assert len(pattern.violated_constraints) == 1

    def test_full_8_layer_construction(self):
        from llm_client.fp_analysis.conflict_models import (
            ConstraintConflict,
            DataflowContext,
            HierarchicalConflictPattern,
            PropagationStep,
            RepairHint,
            ConflictSemanticType,
        )

        pattern = HierarchicalConflictPattern(
            conflict_id="conflict-002",
            conflicting_segments=[1, 3],
            # Layer 1-2: Location + Constraint
            violated_constraints=[
                ConstraintConflict(
                    variable="ptr",
                    segment_a=1,
                    segment_a_constraint="ptr != NULL",
                    segment_b=3,
                    segment_b_constraint="ptr == NULL",
                    conflict_type="null_check",
                ),
            ],
            # Layer 3: Semantic Object
            conflict_objects=["ptr", "ctx"],
            object_roles={"ptr": "allocated_buffer", "ctx": "context_struct"},
            # Layer 4: Semantic Type
            semantic_type=ConflictSemanticType.NULL_CHECK_CONTRADICTION,
            # Layer 5: Root Cause (MUS)
            minimal_unsat_core=[
                ConstraintConflict(
                    variable="ptr",
                    segment_a=1,
                    segment_a_constraint="ptr != NULL",
                    segment_b=3,
                    segment_b_constraint="ptr == NULL",
                    conflict_type="null_check",
                ),
            ],
            # Layer 6: Propagation Chain
            propagation_chain=[
                PropagationStep(segment_index=1, source_line=100,
                               description="ptr allocated via malloc"),
                PropagationStep(segment_index=2, source_line=120,
                               description="ptr passed through function call"),
                PropagationStep(segment_index=3, source_line=140,
                               description="ptr checked for NULL — conflict"),
            ],
            # Layer 7: Repair Hint
            repair_hints=[
                RepairHint(
                    target_segment=3,
                    target_variable="ptr",
                    suggested_action="re_constrain",
                    reasoning="segment 3 may not reach NULL path",
                ),
            ],
            # Layer 8: Generalization
            generalized_pattern="NULL check after allocation — likely false path",
            pattern_signature="null_check:ptr:seg1-seg3",
            # Context
            dataflow_context=DataflowContext(
                var_def_sites={"ptr": ("alloc_fn", 90, "ptr = malloc(n)")},
            ),
        )
        assert pattern.conflict_id == "conflict-002"
        assert len(pattern.propagation_chain) == 3
        assert len(pattern.repair_hints) == 1
        assert pattern.generalized_pattern != ""
        assert pattern.pattern_signature == "null_check:ptr:seg1-seg3"

    def test_pattern_signature_different_for_different_conflicts(self):
        from llm_client.fp_analysis.conflict_models import (
            ConstraintConflict,
            HierarchicalConflictPattern,
            ConflictSemanticType,
        )

        p1 = HierarchicalConflictPattern(
            conflict_id="c1",
            conflicting_segments=[0, 5],
            violated_constraints=[
                ConstraintConflict("r", 0, "r >= 0", 5, "r < 0", "range_contradiction"),
            ],
            semantic_type=ConflictSemanticType.RETURN_VALUE_CONTRADICTION,
            conflict_objects=["r"],
            generalized_pattern="Return value mismatch",
            pattern_signature="range:r:seg0-seg5",
        )
        p2 = HierarchicalConflictPattern(
            conflict_id="c2",
            conflicting_segments=[2, 4],
            violated_constraints=[
                ConstraintConflict("ptr", 2, "ptr != NULL", 4, "ptr == NULL", "null_check"),
            ],
            semantic_type=ConflictSemanticType.NULL_CHECK_CONTRADICTION,
            conflict_objects=["ptr"],
            generalized_pattern="Null check contradiction",
            pattern_signature="null:ptr:seg2-seg4",
        )
        assert p1.pattern_signature != p2.pattern_signature


class TestRepairHint:
    """Tests for RepairHint — guides re-selection."""

    def test_change_branch_action(self):
        from llm_client.fp_analysis.conflict_models import RepairHint

        hint = RepairHint(
            target_segment=3,
            target_variable="r",
            suggested_action="change_branch",
            suggested_branch="False",
            reasoning="Taking false branch avoids the NULL deref path",
            impact="changes postcondition to r < 0",
        )
        assert hint.suggested_action == "change_branch"
        assert hint.suggested_branch == "False"
        assert hint.target_segment == 3

    def test_re_constrain_action(self):
        from llm_client.fp_analysis.conflict_models import RepairHint

        hint = RepairHint(
            target_segment=2,
            target_variable="ptr",
            suggested_action="re_constrain",
            reasoning="Loosen constraint to allow NULL",
            impact="may require upstream segment adjustment",
        )
        assert hint.suggested_action == "re_constrain"
        assert hint.suggested_branch is None


# ─── ConflictPatternLearner tests ──────────────────────────────────────────────


class TestConflictPatternLearner:
    """Tests for ConflictPatternLearner — deterministic L1-L2 extraction."""

    def test_l1_l2_deterministic_extraction(self):
        """Layers 1-2 (Location + Constraint) are extracted deterministically
        from conflicting segment states."""
        from llm_client.fp_analysis.conflict_models import ConstraintConflict
        from llm_client.fp_analysis.conflict_learner import ConflictPatternLearner

        learner = ConflictPatternLearner(provider=None, model="test")

        # Simulated conflict: segment 0 says r >= 0, segment 5 says r < 0
        segment_states = {
            0: {"postcondition": "r >= 0"},
            5: {"postcondition": "r < 0"},
        }
        conflict_segments = [0, 5]
        conflict_details = "Constraint clash: r >= 0 vs r < 0"

        patterns = learner._extract_l1_l2(
            conflict_segments=conflict_segments,
            segment_states=segment_states,
            conflict_details=conflict_details,
        )
        assert len(patterns) > 0
        # Should find at least one ConstraintConflict for variable "r"
        found_r = False
        for p in patterns:
            for cc in p.violated_constraints:
                if cc.variable == "r":
                    found_r = True
        assert found_r, "Should detect constraint conflict on variable 'r'"

    def test_l1_l2_no_conflict_when_compatible(self):
        """L1-L2 extraction returns nothing when constraints are compatible."""
        from llm_client.fp_analysis.conflict_learner import ConflictPatternLearner

        learner = ConflictPatternLearner(provider=None, model="test")

        segment_states = {
            0: {"postcondition": "r >= 0"},
            5: {"postcondition": "r > -1"},
        }
        patterns = learner._extract_l1_l2(
            conflict_segments=[0, 5],
            segment_states=segment_states,
            conflict_details="r >= 0 and r > -1 are compatible",
        )
        # Compatible constraints should produce no deterministic conflict
        # (LLM would decide if there's still a semantic conflict)
        assert len(patterns) == 0

    def test_generate_avoidance_rules(self):
        """generate_avoidance_rules produces injectable prompt text."""
        from llm_client.fp_analysis.conflict_models import (
            ConstraintConflict,
            DataflowContext,
            HierarchicalConflictPattern,
            RepairHint,
            ConflictSemanticType,
        )
        from llm_client.fp_analysis.conflict_learner import ConflictPatternLearner

        learner = ConflictPatternLearner(provider=None, model="test")

        pattern = HierarchicalConflictPattern(
            conflict_id="conflict-001",
            conflicting_segments=[0, 5],
            violated_constraints=[
                ConstraintConflict(
                    variable="r", segment_a=0, segment_a_constraint="r >= 0",
                    segment_b=5, segment_b_constraint="r < 0",
                    conflict_type="range_contradiction",
                ),
            ],
            semantic_type=ConflictSemanticType.RETURN_VALUE_CONTRADICTION,
            conflict_objects=["r"],
            generalized_pattern="Return value postcondition mismatch",
            pattern_signature="range:r:seg0-seg5",
            repair_hints=[
                RepairHint(
                    target_segment=0,
                    target_variable="r",
                    suggested_action="change_branch",
                    reasoning="Try alternative branch in segment 0",
                ),
            ],
            dataflow_context=DataflowContext(),
        )

        rules_text = learner.generate_avoidance_rules(
            patterns=[pattern],
            current_selections={0: {}, 5: {}},
        )
        assert "conflict-001" in rules_text
        assert "RETURN_VALUE_CONTRADICTION" in rules_text or "return_value_contradiction" in rules_text
        assert "Segment" in rules_text or "segment" in rules_text

    def test_generate_avoidance_rules_stale_context(self):
        """When dataflow context is stale, output includes warning marker."""
        from llm_client.fp_analysis.conflict_models import (
            ConstraintConflict,
            DataflowContext,
            HierarchicalConflictPattern,
            ConflictSemanticType,
        )
        from llm_client.fp_analysis.conflict_learner import ConflictPatternLearner

        learner = ConflictPatternLearner(provider=None, model="test")

        ctx = DataflowContext(
            var_def_sites={"r": ("fn", 10, "r = ...")},
            reassignment_checks={"r": True},  # stale!
        )
        pattern = HierarchicalConflictPattern(
            conflict_id="conflict-stale",
            conflicting_segments=[1],
            violated_constraints=[
                ConstraintConflict("r", 1, "r >= 0", 2, "r < 0", "range_contradiction"),
            ],
            semantic_type=ConflictSemanticType.RETURN_VALUE_CONTRADICTION,
            conflict_objects=["r"],
            generalized_pattern="Stale pattern test",
            pattern_signature="test:stale",
            dataflow_context=ctx,
        )

        rules_text = learner.generate_avoidance_rules(
            patterns=[pattern],
            current_selections={1: {}, 2: {}},
        )
        assert "STALE" in rules_text.upper() or "⚠" in rules_text


# ─── Stuck detection tests ─────────────────────────────────────────────────────


class TestStuckDetection:
    """Tests for round-limit and stuck detection logic."""

    def test_different_signatures_not_stuck(self):
        """Two rounds with different pattern signatures → not stuck."""
        from llm_client.fp_analysis.conflict_learner import ConflictPatternLearner

        learner = ConflictPatternLearner(provider=None, model="test")

        prev_sigs = {"range:r:seg0-seg5"}
        curr_sigs = {"null:ptr:seg2-seg4"}
        assert not learner._is_stuck(prev_sigs, curr_sigs, round_num=2)

    def test_same_signatures_is_stuck(self):
        """Two rounds with identical pattern signatures and no new hints → stuck."""
        from llm_client.fp_analysis.conflict_learner import ConflictPatternLearner

        learner = ConflictPatternLearner(provider=None, model="test")

        sigs = {"range:r:seg0-seg5"}
        # no new repair hints available → stuck
        assert learner._is_stuck(sigs, sigs, round_num=2, has_new_hints=False)

    def test_same_signatures_not_stuck_if_new_hints(self):
        """Same signatures but new repair hints available → not stuck."""
        from llm_client.fp_analysis.conflict_learner import ConflictPatternLearner

        learner = ConflictPatternLearner(provider=None, model="test")

        sigs = {"range:r:seg0-seg5"}
        assert not learner._is_stuck(sigs, sigs, round_num=2, has_new_hints=True)

    def test_round_limit_exceeded(self):
        """Round >= 3 should be considered stuck regardless."""
        from llm_client.fp_analysis.conflict_learner import ConflictPatternLearner

        learner = ConflictPatternLearner(provider=None, model="test")

        assert learner._is_stuck(set(), set(), round_num=4, has_new_hints=True)


# ─── Frozen segments tests ─────────────────────────────────────────────────────


class TestFrozenSegments:
    """Tests for frozen segment logic during re-selection."""

    def test_conflicting_segments_identified(self):
        """Only conflicting segment indices are re-selected."""
        current_selections = {0: {"state": "a"}, 1: {"state": "b"}, 2: {"state": "c"}}
        conflict_indices = {1}

        frozen = {
            i: sel for i, sel in current_selections.items()
            if i not in conflict_indices
        }
        assert frozen == {0: {"state": "a"}, 2: {"state": "c"}}
        assert 1 not in frozen

    def test_all_segments_conflicting(self):
        """When all segments conflict, nothing is frozen."""
        current_selections = {0: {"state": "a"}, 1: {"state": "b"}}
        conflict_indices = {0, 1}

        frozen = {
            i: sel for i, sel in current_selections.items()
            if i not in conflict_indices
        }
        assert frozen == {}

    def test_empty_selections(self):
        """Empty selections produce empty frozen set."""
        current_selections = {}
        frozen = {
            i: sel for i, sel in current_selections.items()
            if i not in {0}
        }
        assert frozen == {}
