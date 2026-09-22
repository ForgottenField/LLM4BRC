"""Tests for cfg_models — SegmentInfo anchor events, state tracking, and assumption checking."""

from __future__ import annotations

from llm_client.fp_analysis.cfg_models import (
    AnchorEvent,
    AssumptionFact,
    AssumptionVerdict,
    BugReportPath,
    NodeType,
    PathNode,
    SegmentInfo,
)


class TestAnchorEvent:
    """AnchorEvent model — CSA report branch event anchoring."""

    def test_anchor_event_creation(self):
        """An AnchorEvent can be created with all fields."""
        event = AnchorEvent(
            event_number=5,
            line=42,
            branch=True,
            condition="blockLength_ <= 0",
            variable="blockLength_",
            status="pending",
        )
        assert event.event_number == 5
        assert event.line == 42
        assert event.branch is True
        assert event.condition == "blockLength_ <= 0"
        assert event.variable == "blockLength_"
        assert event.status == "pending"

    def test_anchor_event_default_variable(self):
        """variable defaults to None when condition has no clear variable."""
        event = AnchorEvent(
            event_number=3,
            line=100,
            branch=False,
            condition="ptr != NULL",
            status="consistent",
        )
        assert event.variable is None
        assert event.status == "consistent"

    def test_anchor_event_all_statuses(self):
        """status field accepts all three valid values."""
        for status in ("pending", "consistent", "contradictory"):
            event = AnchorEvent(
                event_number=1,
                line=10,
                branch=True,
                condition="x > 0",
                status=status,
            )
            assert event.status == status


class TestSegmentInfoAnchor:
    """SegmentInfo enhancements — anchor_events and pruned_event_indices."""

    def test_segment_with_anchors(self):
        """SegmentInfo carries anchor events from the CSA report."""
        segment = SegmentInfo(
            segment_index=0,
            label="L100-L110 [branch=TRUE]",
            event_range=(1, 3),
            line_start=100,
            line_end=110,
            function_name="test_func",
            anchor_events=[
                AnchorEvent(
                    event_number=2,
                    line=105,
                    branch=True,
                    condition="x > 0",
                    status="pending",
                ),
            ],
        )
        assert len(segment.anchor_events) == 1
        assert segment.anchor_events[0].line == 105
        assert segment.anchor_events[0].branch is True

    def test_segment_empty_anchors_default(self):
        """SegmentInfo defaults to empty anchor_events list."""
        segment = SegmentInfo(
            segment_index=0,
            event_range=(1, 1),
            function_name="f",
        )
        assert segment.anchor_events == []
        assert segment.pruned_event_indices == []

    def test_segment_pruned_events(self):
        """pruned_event_indices tracks events pruned by prior segment state."""
        segment = SegmentInfo(
            segment_index=1,
            label="L200-L210",
            event_range=(4, 6),
            line_start=200,
            line_end=210,
            function_name="f",
            pruned_event_indices=[4],
        )
        assert segment.pruned_event_indices == [4]

    def test_segment_preserves_existing_fields(self):
        """Existing SegmentInfo fields are preserved unchanged."""
        segment = SegmentInfo(
            segment_index=0,
            event_range=(1, 2),
            line_start=10,
            line_end=20,
            function_name="f",
            precondition_facts=["x > 0"],
            if_line_numbers=[15],
            called_functions=["g"],
        )
        assert segment.precondition_facts == ["x > 0"]
        assert segment.if_line_numbers == [15]
        assert segment.called_functions == ["g"]
        assert segment.anchor_events == []
        assert segment.pruned_event_indices == []


class TestSegmentFormatMethods:
    """SegmentInfo.format_postcondition / format_precondition still work."""

    def test_format_postcondition_with_anchor(self):
        """format_postcondition works when control events come from anchors."""
        segment = SegmentInfo(
            segment_index=0,
            event_range=(1, 2),
            line_start=10,
            line_end=20,
            function_name="f",
            control_event_number=2,
            control_branch_taken=True,
            anchor_events=[
                AnchorEvent(
                    event_number=2, line=20, branch=True,
                    condition="flag", status="pending",
                ),
            ],
        )
        text = segment.format_postcondition()
        assert "TRUE" in text
        assert "#2" in text

    def test_format_precondition_empty(self):
        """format_precondition returns default text when empty."""
        segment = SegmentInfo(segment_index=0, event_range=(1, 1), function_name="f")
        text = segment.format_precondition()
        assert "Function entry" in text or "no prior events" in text


# ---------------------------------------------------------------------------
# AssumptionFact — CSA assumption extraction from path events
# ---------------------------------------------------------------------------


class TestAssumptionFact:
    """AssumptionFact dataclass — extracted CSA assumption from path events."""

    def test_create_basic_assumption(self):
        """Can create an AssumptionFact with all fields."""
        fact = AssumptionFact(
            event_number=5,
            line_number=62,
            segment_index=1,
            variable="blockLength_",
            operator="<=",
            value="0",
            raw_description="Assuming field 'blockLength_' is <= 0",
            is_error_start=True,
        )
        assert fact.event_number == 5
        assert fact.line_number == 62
        assert fact.segment_index == 1
        assert fact.variable == "blockLength_"
        assert fact.operator == "<="
        assert fact.value == "0"
        assert fact.raw_description == "Assuming field 'blockLength_' is <= 0"
        assert fact.is_error_start is True

    def test_null_pointer_assumption(self):
        """Assumption tracking null pointer storage."""
        fact = AssumptionFact(
            event_number=4,
            line_number=52,
            segment_index=1,
            variable="tempBitfield.bitfield_",
            operator="==",
            value="null",
            raw_description="Null pointer value stored to 'tempBitfield.bitfield_'",
            is_error_start=True,
        )
        assert fact.variable == "tempBitfield.bitfield_"
        assert fact.value == "null"
        assert fact.operator == "=="

    def test_equality_assumption(self):
        """Assumption tracking field equality."""
        fact = AssumptionFact(
            event_number=8,
            line_number=665,
            segment_index=2,
            variable="bitfieldLength",
            operator="==",
            value="bitfieldLength_",
            raw_description="Assuming 'bitfieldLength' is equal to field 'bitfieldLength_'",
            is_error_start=False,
        )
        assert fact.variable == "bitfieldLength"
        assert fact.value == "bitfieldLength_"
        assert fact.is_error_start is False

    def test_is_feasible_helper(self):
        """Can check feasibility when method is available."""
        fact = AssumptionFact(
            event_number=1, line_number=10, segment_index=0,
            variable="x", operator=">", value="0",
            raw_description="Assuming 'x' is > 0",
            is_error_start=False,
        )
        # Default: no layer1_result → feasible unless contradicted
        assert fact.is_feasible() is True


class TestAssumptionVerdict:
    """AssumptionVerdict enum — outcome of assumption feasibility check."""

    def test_verdict_values(self):
        """Verdict has FEASIBLE and INFEASIBLE values."""
        assert AssumptionVerdict.FEASIBLE.value == "feasible"
        assert AssumptionVerdict.INFEASIBLE.value == "infeasible"

    def test_verdict_truthiness(self):
        """FEASIBLE is truthy, INFEASIBLE is falsy."""
        assert AssumptionVerdict.FEASIBLE
        assert not AssumptionVerdict.INFEASIBLE
