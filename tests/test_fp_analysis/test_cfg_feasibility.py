"""Tests for cfg_feasibility — segmentation, assumption extraction, and path building."""

from __future__ import annotations

import pytest

from llm_client.fp_analysis.cfg_feasibility import extract_assumptions_from_events


# ---------------------------------------------------------------------------
# Test data: CSA path events matching real DefaultPieceStorage report patterns
# ---------------------------------------------------------------------------

MOCK_PATH_EVENTS_BLOCKLENGTH = [
    # Event #1: CALL event (should be skipped — call type)
    {
        "path_number": 1,
        "event_kind": "call",
        "description": "Calling 'DefaultPieceStorage::getMissingPiece'",
        "line_number": 341,
        "branch_condition": None,
        "is_error_start": False,
        "is_calling_event": True,
        "called_function": "DefaultPieceStorage::getMissingPiece",
        "is_control_flow": False,
    },
    # Event #2: event type — constructor call (not an assumption pattern)
    {
        "path_number": 2,
        "event_kind": "event",
        "description": "Calling constructor for 'BitfieldMan'",
        "line_number": 285,
        "branch_condition": None,
        "is_error_start": False,
        "is_calling_event": False,
        "is_control_flow": False,
    },
    # Event #3: error_start — Error Start marker
    {
        "path_number": 3,
        "event_kind": "error_start",
        "description": "Error Start",
        "line_number": 52,
        "branch_condition": None,
        "is_error_start": True,
        "is_calling_event": False,
        "is_control_flow": False,
    },
    # Event #4: event — null pointer stored
    {
        "path_number": 4,
        "event_kind": "event",
        "description": "Null pointer value stored to 'tempBitfield.bitfield_'",
        "line_number": 52,
        "branch_condition": None,
        "is_error_start": True,
        "is_calling_event": False,
        "is_control_flow": False,
    },
    # Event #5: event — Assuming field 'blockLength_' is <= 0
    {
        "path_number": 5,
        "event_kind": "event",
        "description": "Assuming field 'blockLength_' is <= 0",
        "line_number": 62,
        "branch_condition": None,
        "is_error_start": True,
        "is_calling_event": False,
        "is_control_flow": False,
    },
    # Event #6: event — returning from constructor
    {
        "path_number": 6,
        "event_kind": "event",
        "description": "Returning from constructor for 'BitfieldMan'",
        "line_number": 285,
        "branch_condition": None,
        "is_error_start": False,
        "is_calling_event": False,
        "is_control_flow": False,
    },
    # Event #7: call event (should be skipped)
    {
        "path_number": 7,
        "event_kind": "call",
        "description": "Calling 'BitfieldMan::setBitfield'",
        "line_number": 287,
        "branch_condition": None,
        "is_error_start": False,
        "is_calling_event": True,
        "called_function": "BitfieldMan::setBitfield",
        "is_control_flow": False,
    },
    # Event #8: event — Assuming 'bitfieldLength' is equal to field 'bitfieldLength_'
    {
        "path_number": 8,
        "event_kind": "event",
        "description": "Assuming 'bitfieldLength' is equal to field 'bitfieldLength_'",
        "line_number": 665,
        "branch_condition": None,
        "is_error_start": False,
        "is_calling_event": False,
        "is_control_flow": False,
    },
    # Event #9: control event (should be skipped — control type)
    {
        "path_number": 9,
        "event_kind": "control",
        "description": "Taking false branch",
        "line_number": 665,
        "branch_condition": False,
        "is_error_start": False,
        "is_calling_event": False,
        "is_control_flow": True,
    },
    # Event #9999: report event (should be skipped — report type)
    {
        "path_number": 9999,
        "event_kind": "report",
        "description": "Null pointer passed to 1st parameter expecting 'nonnull'",
        "line_number": 668,
        "branch_condition": None,
        "is_error_start": False,
        "is_calling_event": False,
        "is_control_flow": False,
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractAssumptionsFromEvents:
    """extract_assumptions_from_events() — parses CSA event-type path events."""

    def test_extracts_blocklength_leq_zero(self):
        """Pattern: 'Assuming field X is <= Y' → AssumptionFact with operator '<='."""
        # Event range covering the Error Start segment (events 2-7)
        event_range = (2, 7)
        facts = extract_assumptions_from_events(
            MOCK_PATH_EVENTS_BLOCKLENGTH, event_range, segment_index=1,
        )
        assert len(facts) == 2  # events #4 (null stored) and #5 (assuming <=)

        # Fact #1: null pointer stored to 'tempBitfield.bitfield_'
        null_fact = facts[0]
        assert null_fact.event_number == 4
        assert null_fact.line_number == 52
        assert null_fact.variable == "tempBitfield.bitfield_"
        assert null_fact.operator == "=="
        assert null_fact.value == "null"
        assert null_fact.is_error_start is True

        # Fact #2: blockLength_ <= 0
        leq_fact = facts[1]
        assert leq_fact.event_number == 5
        assert leq_fact.line_number == 62
        assert leq_fact.variable == "blockLength_"
        assert leq_fact.operator == "<="
        assert leq_fact.value == "0"
        assert leq_fact.is_error_start is True

    def test_extracts_equality_assumption(self):
        """Pattern: 'Assuming X is equal to field Y' → AssumptionFact with '=='."""
        event_range = (8, 9)
        facts = extract_assumptions_from_events(
            MOCK_PATH_EVENTS_BLOCKLENGTH, event_range, segment_index=2,
        )
        assert len(facts) == 1

        fact = facts[0]
        assert fact.event_number == 8
        assert fact.line_number == 665
        assert fact.variable == "bitfieldLength"
        assert fact.operator == "=="
        assert fact.value == "bitfieldLength_"
        assert fact.is_error_start is False

    def test_skips_control_and_call_events(self):
        """Only event-type and error_start events are extracted."""
        # Full range but the function should filter to only event/error_start
        event_range = (1, 9999)
        facts = extract_assumptions_from_events(
            MOCK_PATH_EVENTS_BLOCKLENGTH, event_range, segment_index=0,
        )
        # Should only extract events #4 and #5 (inside event_type), plus #8
        # Event #3 is error_start but has no extractable assumption pattern
        # Events #1, #7, #9, #9999 are call/control/report → skipped
        assert len(facts) == 3  # #4, #5, #8

        event_numbers = {f.event_number for f in facts}
        assert event_numbers == {4, 5, 8}

    def test_event_range_filtering(self):
        """Only events within the given event_range are extracted."""
        event_range = (4, 5)  # Only events #4 and #5
        facts = extract_assumptions_from_events(
            MOCK_PATH_EVENTS_BLOCKLENGTH, event_range, segment_index=0,
        )
        assert len(facts) == 2
        assert {f.event_number for f in facts} == {4, 5}

    def test_no_matching_events_returns_empty(self):
        """Returns empty list when no event-type assumptions match patterns."""
        events = [
            {
                "path_number": 1,
                "event_kind": "control",
                "description": "Taking true branch",
                "line_number": 10,
                "branch_condition": True,
                "is_error_start": False,
                "is_calling_event": False,
                "is_control_flow": True,
            },
        ]
        facts = extract_assumptions_from_events(events, (1, 1), segment_index=0)
        assert facts == []

    def test_unknown_event_description_is_skipped(self):
        """Event-type events with unrecognized descriptions are skipped gracefully."""
        events = [
            {
                "path_number": 1,
                "event_kind": "event",
                "description": "Some unrecognized event description",
                "line_number": 10,
                "branch_condition": None,
                "is_error_start": False,
                "is_calling_event": False,
                "is_control_flow": False,
            },
        ]
        facts = extract_assumptions_from_events(events, (1, 1), segment_index=0)
        assert facts == []


class TestAssumptionExtractionVariants:
    """Additional CSA assumption patterns."""

    def test_less_than_operator(self):
        """Pattern: 'Assuming field X is < Y'."""
        events = [
            {
                "path_number": 1,
                "event_kind": "event",
                "description": "Assuming field 'size' is < 0",
                "line_number": 42,
                "is_error_start": False,
                "is_calling_event": False,
                "is_control_flow": False,
            },
        ]
        facts = extract_assumptions_from_events(events, (1, 1), segment_index=0)
        assert len(facts) == 1
        assert facts[0].variable == "size"
        assert facts[0].operator == "<"
        assert facts[0].value == "0"

    def test_not_equal_operator(self):
        """Pattern: 'Assuming field X is != Y'."""
        events = [
            {
                "path_number": 1,
                "event_kind": "event",
                "description": "Assuming field 'count' is != 0",
                "line_number": 42,
                "is_error_start": False,
                "is_calling_event": False,
                "is_control_flow": False,
            },
        ]
        facts = extract_assumptions_from_events(events, (1, 1), segment_index=0)
        assert len(facts) == 1
        assert facts[0].variable == "count"
        assert facts[0].operator == "!="
        assert facts[0].value == "0"

    def test_value_stored_to_field(self):
        """Pattern: 'X stored to Y' without 'Null pointer' prefix."""
        events = [
            {
                "path_number": 1,
                "event_kind": "event",
                "description": "Value '0' stored to 'result'",
                "line_number": 42,
                "is_error_start": False,
                "is_calling_event": False,
                "is_control_flow": False,
            },
        ]
        facts = extract_assumptions_from_events(events, (1, 1), segment_index=0)
        assert len(facts) == 1
        assert facts[0].variable == "result"
        assert facts[0].operator == "=="
        assert facts[0].value == "0"

    def test_trailing_arrow_is_stripped(self):
        """Trailing '→' in CSA descriptions is stripped from the value."""
        events = [
            {
                "path_number": 1,
                "event_kind": "event",
                "description": "Assuming field 'blockLength_' is <= 0 →",
                "line_number": 62,
                "is_error_start": True,
                "is_calling_event": False,
                "is_control_flow": False,
            },
        ]
        facts = extract_assumptions_from_events(events, (1, 1), segment_index=0)
        assert len(facts) == 1
        assert facts[0].value == "0"
        assert facts[0].operator == "<="
        assert facts[0].variable == "blockLength_"
