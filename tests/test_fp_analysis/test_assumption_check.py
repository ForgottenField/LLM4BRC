"""Tests for layered assumption feasibility checking in path_analyzer.

Tests that extracted CSA assumptions are converted into **hard constraints**
for the path selection pipeline, enabling path selection to discover conflicts
and return FP on its own terms.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_client.fp_analysis.cfg_feasibility import extract_assumptions_from_events
from llm_client.fp_analysis.cfg_models import AssumptionFact, AssumptionVerdict
from llm_client.fp_analysis.path_analyzer import FPAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_analyzer():
    """FPAnalyzer with minimal setup — no LLM provider (tests Tier-1 only)."""
    analyzer = FPAnalyzer.__new__(FPAnalyzer)
    analyzer._source_root = "/tmp/test_project"
    analyzer._project_name = "test"
    analyzer._provider = None
    analyzer._model = "test-model"
    analyzer._sdg = None
    return analyzer


@pytest.fixture
def blocklength_fact():
    """AssumptionFact matching the DefaultPieceStorage Error Start."""
    return AssumptionFact(
        event_number=5,
        line_number=62,
        segment_index=1,
        variable="blockLength_",
        operator="<=",
        value="0",
        raw_description="Assuming field 'blockLength_' is <= 0",
        is_error_start=True,
    )


# ---------------------------------------------------------------------------
# Tier-1: Rule-based check
# ---------------------------------------------------------------------------


class TestLayer1RuleCheck:
    """_layer1_rule_check() — deterministic constraint-cache-based checks."""

    def test_returns_uncertain_when_no_global_constraints(self, mock_analyzer, blocklength_fact):
        result = mock_analyzer._layer1_rule_check(blocklength_fact, "")
        assert result == "UNCERTAIN"

    def test_returns_uncertain_when_no_match_in_constraints(self, mock_analyzer, blocklength_fact):
        constraints = "class Foo: field 'bar_' initial_value '42' (confidence: high)"
        result = mock_analyzer._layer1_rule_check(blocklength_fact, constraints)
        assert result == "UNCERTAIN"

    def test_returns_violated_when_high_confidence_contradiction(
        self, mock_analyzer, blocklength_fact,
    ):
        constraints = (
            "class BitfieldMan: field 'blockLength_' initial_value '128' "
            "  (confidence: high)"
        )
        result = mock_analyzer._layer1_rule_check(blocklength_fact, constraints)
        assert result == "VIOLATED"

    def test_returns_uncertain_when_low_confidence_match(
        self, mock_analyzer, blocklength_fact,
    ):
        constraints = (
            "class BitfieldMan: field 'blockLength_' initial_value '128' "
            "  (confidence: low)"
        )
        result = mock_analyzer._layer1_rule_check(blocklength_fact, constraints)
        assert result == "UNCERTAIN"

    def test_returns_consistent_when_assumption_matches_constraint(self, mock_analyzer):
        fact = AssumptionFact(
            event_number=1, line_number=10, segment_index=0,
            variable="blockLength_", operator="==", value="128",
            raw_description="blockLength_ is == 128",
            is_error_start=False,
        )
        constraints = (
            "class BitfieldMan: field 'blockLength_' initial_value '128' "
            "  (confidence: high)"
        )
        result = mock_analyzer._layer1_rule_check(fact, constraints)
        assert result == "CONSISTENT"

    def test_violated_when_equality_conflicts(self, mock_analyzer):
        fact = AssumptionFact(
            event_number=1, line_number=10, segment_index=0,
            variable="blockLength_", operator="==", value="0",
            raw_description="blockLength_ is == 0",
            is_error_start=False,
        )
        constraints = (
            "class BitfieldMan: field 'blockLength_' initial_value '128' "
            "  (confidence: high)"
        )
        result = mock_analyzer._layer1_rule_check(fact, constraints)
        assert result == "VIOLATED"


# ---------------------------------------------------------------------------
# Constraint conversion from assumption check results
# ---------------------------------------------------------------------------


class TestBuildHardConstraints:
    """_build_hard_constraints_from_assumptions() — converts VIOLATED/CONSISTENT
    results into hard constraints for path selection."""

    def test_violated_produces_negation_constraint(self, mock_analyzer):
        """A Tier-1 VIOLATED assumption (blockLength_ <= 0 vs known=128)
        produces a hard constraint asserting the opposite."""
        fact = AssumptionFact(
            event_number=5, line_number=62, segment_index=1,
            variable="blockLength_", operator="<=", value="0",
            raw_description="blockLength_ <= 0",
            is_error_start=True,
        )
        # Mark it as VIOLATED
        fact.layer1_result = "VIOLATED"

        hc = mock_analyzer._build_hard_constraints_from_assumptions(
            [fact], tier2_results=None,
        )
        assert len(hc) >= 1
        # The hard constraint should assert that the value is NOT <= 0, i.e. > 0
        assert any("blockLength_" in c for c in hc)
        assert any(">" in c or "positive" in c.lower() for c in hc)

    def test_consistent_equality_produces_equality_constraint(self, mock_analyzer):
        """A CONSISTENT == fact produces a binding constraint."""
        fact = AssumptionFact(
            event_number=1, line_number=10, segment_index=0,
            variable="count", operator="==", value="128",
            raw_description="count is == 128",
            is_error_start=False,
        )
        fact.layer1_result = "CONSISTENT"
        hc = mock_analyzer._build_hard_constraints_from_assumptions([fact], None)
        assert len(hc) == 1
        assert hc[0] == "count == 128"

    def test_uncertain_produces_no_constraint(self, mock_analyzer):
        """UNCERTAIN facts don't produce hard constraints (defer to LLM)."""
        fact = AssumptionFact(
            event_number=5, line_number=62, segment_index=1,
            variable="blockLength_", operator="<=", value="0",
            raw_description="blockLength_ <= 0",
            is_error_start=True,
        )
        fact.layer1_result = "UNCERTAIN"
        hc = mock_analyzer._build_hard_constraints_from_assumptions(
            [fact], tier2_results=None,
        )
        assert hc == []

    def test_tier2_results_merged(self, mock_analyzer):
        """When tier2_results provides constraints, they're included."""
        fact = AssumptionFact(
            event_number=5, line_number=62, segment_index=1,
            variable="blockLength_", operator="<=", value="0",
            raw_description="blockLength_ <= 0",
            is_error_start=True,
        )
        fact.layer1_result = "UNCERTAIN"
        tier2 = {
            "is_infeasible": True,
            "hard_constraints": ["blockLength_ > 0 (piece length always positive)"],
            "state_bindings": [],
        }
        hc = mock_analyzer._build_hard_constraints_from_assumptions(
            [fact], tier2_results=tier2,
        )
        assert len(hc) == 1
        assert "blockLength_ > 0" in hc[0]


# ---------------------------------------------------------------------------
# Constraint conflict checking (within path selection)
# ---------------------------------------------------------------------------


class TestCheckConstraintConflict:
    """_check_constraint_conflict() — checks if a segment's CSA assumptions
    conflict with hard constraints derived from assumption checking."""

    def test_no_conflict_when_no_hard_constraints(self, mock_analyzer):
        """Empty hard constraints → no conflict."""
        conflict = mock_analyzer._check_constraint_conflict(
            segment=None, parsed=None, hard_constraints=[],
        )
        assert conflict is None

    def test_no_conflict_when_no_assumptions_in_segment(self, mock_analyzer):
        """Segment with no extractable assumptions → no conflict."""
        from llm_client.fp_analysis.cfg_models import SegmentInfo

        segment = SegmentInfo(
            segment_index=0, event_range=(1, 1), function_name="test_fn",
        )
        hard_constraints = ["blockLength_ > 0"]
        # Mock extraction to return empty
        with patch(
            "llm_client.fp_analysis.path_analyzer.extract_assumptions_from_events",
            return_value=[],
        ):
            conflict = mock_analyzer._check_constraint_conflict(
                segment, MagicMock(), hard_constraints,
            )
        assert conflict is None

    def test_conflict_detected_when_constraints_contradict(self, mock_analyzer):
        """blockLength_ <= 0 conflicts with blockLength_ > 0 → conflict found."""
        from llm_client.fp_analysis.cfg_models import SegmentInfo

        segment = SegmentInfo(
            segment_index=1, event_range=(4, 5), function_name="test_fn",
        )

        fact = AssumptionFact(
            event_number=5, line_number=62, segment_index=1,
            variable="blockLength_", operator="<=", value="0",
            raw_description="Assuming field 'blockLength_' is <= 0",
            is_error_start=True,
        )

        hard_constraints = ["blockLength_ > 0"]
        with patch(
            "llm_client.fp_analysis.path_analyzer.extract_assumptions_from_events",
            return_value=[fact],
        ):
            conflict = mock_analyzer._check_constraint_conflict(
                segment, MagicMock(), hard_constraints,
            )
        assert conflict is not None
        assert "blockLength_" in conflict
        assert "<= 0" in conflict

    def test_no_conflict_when_constraints_are_compatible(self, mock_analyzer):
        """blockLength_ > 100 is compatible with blockLength_ > 0 → no conflict."""
        from llm_client.fp_analysis.cfg_models import SegmentInfo

        segment = SegmentInfo(
            segment_index=1, event_range=(4, 5), function_name="test_fn",
        )
        fact = AssumptionFact(
            event_number=5, line_number=62, segment_index=1,
            variable="blockLength_", operator=">", value="100",
            raw_description="blockLength_ > 100",
            is_error_start=False,
        )

        hard_constraints = ["blockLength_ > 0"]
        with patch(
            "llm_client.fp_analysis.path_analyzer.extract_assumptions_from_events",
            return_value=[fact],
        ):
            conflict = mock_analyzer._check_constraint_conflict(
                segment, MagicMock(), hard_constraints,
            )
        assert conflict is None
