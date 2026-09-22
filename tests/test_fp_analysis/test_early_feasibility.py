"""Tests for early feasibility check — Step 3.5 + Step 4 cumulative state."""

from __future__ import annotations

import pytest

from llm_client.fp_analysis.path_analyzer import (
    _check_anchor_against_constraints,
    _evaluate_constraint_against_branch,
    _find_variable_in_condition,
)


class TestFindVariableInCondition:
    """_find_variable_in_condition — extract variable from condition text."""

    def test_simple_variable(self):
        assert _find_variable_in_condition("blockLength_ <= 0") == "blockLength_"

    def test_member_field(self):
        assert _find_variable_in_condition("ptr != NULL") == "ptr"

    def test_dereference(self):
        assert _find_variable_in_condition("*ptr == 0") == "ptr"

    def test_function_call(self):
        """If the condition is a function call, return None (can't extract simply)."""
        result = _find_variable_in_condition("wslay_is_ctrl_frame(arg) == true")
        assert result is None

    def test_empty_condition(self):
        assert _find_variable_in_condition("") is None

    def test_literal_only(self):
        result = _find_variable_in_condition("true")
        assert result is None

    def test_qualified_name(self):
        assert _find_variable_in_condition("obj->field > 5") == "obj->field"

    def test_boolean_var(self):
        assert _find_variable_in_condition("flag") is None  # too simple, no operator


class TestEvaluateConstraintAgainstBranch:
    """_evaluate_constraint_against_branch — core deterministic check."""

    def test_false_branch_on_positive_range_contradiction(self):
        """FALSE on x > 0 means x <= 0, contradicts constraint x > 0."""
        result, msg = _evaluate_constraint_against_branch(
            anchor_branch=False,  # FALSE branch: condition is NOT true
            condition="x > 0",
            constraint_str="> 0",
        )
        assert result == "contradictory"
        assert "x" in msg or "> 0" in msg

    def test_true_branch_matches_constraint(self):
        """TRUE on x > 0 with constraint x > 0 → consistent."""
        result, msg = _evaluate_constraint_against_branch(
            anchor_branch=True,
            condition="x > 0",
            constraint_str="> 0",
        )
        assert result == "consistent"

    def test_false_on_not_null_means_null_consistent(self):
        """FALSE on ptr != NULL means ptr IS NULL, constraint says NULL → consistent."""
        result, msg = _evaluate_constraint_against_branch(
            anchor_branch=False,  # FALSE branch: ptr != NULL is FALSE → ptr is NULL
            condition="ptr != NULL",
            constraint_str="NULL",
        )
        assert result == "consistent"

    def test_true_on_null_vs_nonnull_contradictory(self):
        """TRUE on ptr == NULL requires ptr NULL, but constraint says non-null → contradict."""
        result, msg = _evaluate_constraint_against_branch(
            anchor_branch=True,
            condition="ptr == NULL",
            constraint_str="non-null allocated pointer",
        )
        assert result == "contradictory"

    def test_non_numeric_non_null_constraint(self):
        """'allocated' implies non-null → consistent with TRUE on ptr != NULL."""
        result, msg = _evaluate_constraint_against_branch(
            anchor_branch=True,
            condition="ptr != NULL",
            constraint_str="allocated",
        )
        assert result == "consistent"

    def test_no_variable_in_condition(self):
        """Function-call conditions return insufficient_info."""
        result, msg = _evaluate_constraint_against_branch(
            anchor_branch=True,
            condition="wslay_is_ctrl_frame(arg) == true",
            constraint_str="true",
        )
        assert result == "insufficient_info"


class TestCheckAnchorAgainstConstraints:
    """_check_anchor_against_constraints — full anchor vs constraints match."""

    def test_direct_match_contradiction(self):
        """TRUE on x <= 0 with constraint x > 0 → contradictory."""
        result, msg = _check_anchor_against_constraints(
            anchor_branch=True,
            condition_text="x <= 0",
            variable="x",
            concretized_values={"x": "> 0"},
        )
        assert result == "contradictory"

    def test_direct_match_consistent(self):
        """Direct variable match with supporting constraints."""
        result, msg = _check_anchor_against_constraints(
            anchor_branch=True,
            condition_text="x > 5",
            variable="x",
            concretized_values={"x": "10"},
        )
        assert result == "consistent"

    def test_no_constraint_for_variable(self):
        """Variable not in concretized_values → insufficient_info."""
        result, msg = _check_anchor_against_constraints(
            anchor_branch=True,
            condition_text="x > 5",
            variable="x",
            concretized_values={"y": "10"},
        )
        assert result == "insufficient_info"

    def test_empty_concretized_values(self):
        """Empty concretized_values dict → insufficient_info."""
        result, msg = _check_anchor_against_constraints(
            anchor_branch=True,
            condition_text="x > 5",
            variable="x",
            concretized_values={},
        )
        assert result == "insufficient_info"

    def test_no_variable_provided(self):
        """No variable → insufficient_info (LLM fallback needed)."""
        result, msg = _check_anchor_against_constraints(
            anchor_branch=True,
            condition_text="wslay_is_ctrl_frame(arg) == true",
            variable=None,
            concretized_values={"arg": "0x9 (PING)"},
        )
        assert result == "insufficient_info"

    def test_range_consistent(self):
        """Range constraint that overlaps with branch condition."""
        result, msg = _check_anchor_against_constraints(
            anchor_branch=True,
            condition_text="x > 0",
            variable="x",
            concretized_values={"x": "> 0, default 1M"},
        )
        assert result == "consistent"

    def test_range_contradictory(self):
        """Range constraint that contradicts branch condition."""
        result, msg = _check_anchor_against_constraints(
            anchor_branch=True,
            condition_text="blockLength_ <= 0",
            variable="blockLength_",
            concretized_values={"blockLength_": "> 0 (from config option)"},
        )
        assert result == "contradictory"


class TestMergeState:
    """_merge_state — parse LLM free-text into structured {var: constraint}."""

    def test_simple_merge(self):
        """Basic variable assignments are extracted."""
        from llm_client.fp_analysis.path_analyzer import _merge_state
        cumulative = {}
        _merge_state(cumulative, "x >= 10, ptr == NULL, flag == true")
        # Values include the operator for downstream prompt injection
        assert cumulative == {"x": ">= 10", "ptr": "== NULL", "flag": "== true"}

    def test_merge_overwrites_existing(self):
        """Later value overwrites earlier value for same variable."""
        from llm_client.fp_analysis.path_analyzer import _merge_state
        cumulative = {"x": "< 0"}
        _merge_state(cumulative, "x >= 10")
        assert cumulative["x"] == ">= 10"

    def test_empty_state_no_change(self):
        """Empty text doesn't alter cumulative state."""
        from llm_client.fp_analysis.path_analyzer import _merge_state
        cumulative = {"x": "> 0"}
        _merge_state(cumulative, "")
        assert cumulative == {"x": "> 0"}

    def test_merge_preserves_unrelated_vars(self):
        """Variables not mentioned in new state survive."""
        from llm_client.fp_analysis.path_analyzer import _merge_state
        cumulative = {"a": "1", "b": "2"}
        _merge_state(cumulative, "c == 3")
        # Value includes the operator
        assert cumulative == {"a": "1", "b": "2", "c": "== 3"}
