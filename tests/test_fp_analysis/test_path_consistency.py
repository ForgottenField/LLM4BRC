"""Deterministic selected-path versus CSA control-event checks."""

from types import SimpleNamespace

from fp_analysis.models import ParsedReport, ReportMetadata
from fp_analysis.cfg_models import SegmentInfo
from fp_analysis.path_analyzer import FPAnalyzer, PathSelectionUnavailable
from fp_analysis.path_semantics import selected_call_return_violations
from fp_analysis.report_consistency import evaluate_control_direction_conflicts
from pipeline.path_consistency import selected_path_violations
from pipeline.selection import run_path_selection


def _segment(*constraints):
    return SegmentInfo(
        segment_index=7,
        event_range=(60, 64),
        line_start=694,
        line_end=694,
        function_name="wslay_event_recv",
        function_file="lib/wslay_event.c",
        control_constraints=list(constraints),
    )


def test_same_line_call_and_control_uses_the_control_event():
    segment = _segment({
        "event_number": 63,
        "line": 694,
        "branch_taken": False,
        "file": "lib/wslay_event.c",
        "function": "wslay_event_recv",
    })
    selections = [{
        "segment_index": 7,
        "branch_decisions": [{
            "node_id": "pdg_wslay_event_recv_ts_123",
            "source_line": 694,
            "branch_taken": True,
        }],
    }]

    violations = selected_path_violations(selections, [segment])

    assert violations == [{
        "segment_index": 7,
        "selection_segment_index": 7,
        "event_number": 63,
        "line": 694,
        "function": "wslay_event_recv",
        "file": "lib/wslay_event.c",
        "reported_branch_taken": False,
        "selected_branch_taken": True,
        "node_id": "pdg_wslay_event_recv_ts_123",
    }]


def test_missing_branch_mapping_is_not_an_explicit_violation():
    segment = _segment({
        "event_number": 63,
        "line": 694,
        "branch_taken": False,
        "file": "lib/wslay_event.c",
        "function": "wslay_event_recv",
    })
    selections = [{"segment_index": 7, "branch_decisions": []}]

    assert selected_path_violations(selections, [segment]) == []


def test_matching_direction_at_the_same_location_is_consistent():
    segment = _segment({
        "event_number": 63,
        "line": 694,
        "branch_taken": False,
        "file": "lib/wslay_event.c",
        "function": "wslay_event_recv",
    })
    selections = [{
        "segment_index": 7,
        "branch_decisions": [{"source_line": 694, "branch_taken": False}],
    }]

    assert selected_path_violations(selections, [segment]) == []


def test_cross_segment_same_line_control_is_still_checked():
    call_segment = SegmentInfo(
        segment_index=23,
        event_range=(54, 62),
        line_start=686,
        line_end=694,
        function_name="wslay_event_recv",
        function_file="lib/wslay_event.c",
    )
    control_segment = _segment({
        "event_number": 63,
        "line": 694,
        "branch_taken": False,
        "file": "lib/wslay_event.c",
        "function": "wslay_event_recv",
    })
    selections = [{
        "segment_index": 23,
        "branch_decisions": [{
            "source_line": 694,
            "branch_taken": True,
            "node_id": "pdg_wslay_event_recv_ts_123",
        }],
    }]

    violations = selected_path_violations(
        selections, [call_segment, control_segment],
    )

    assert len(violations) == 1
    assert violations[0]["event_number"] == 63
    assert violations[0]["segment_index"] == 7
    assert violations[0]["selection_segment_index"] == 23


def test_report_control_direction_is_evaluated_from_macros_and_facts(tmp_path):
    source = tmp_path / "event.c"
    source.write_text(
        "\n" * 19
        + "if(!config_enabled(ctx) ||\n"
        + "   is_control_frame(opcode)) {\n"
        + "}\n",
        encoding="utf-8",
    )
    (tmp_path / "project.h").write_text(
        "#define is_control_frame(opcode) ((opcode >> 3) & 1)\n"
        "enum {\n"
        "  CONTROL_PING = 0x9u,\n"
        "};\n",
        encoding="utf-8",
    )
    parsed = ParsedReport(
        metadata=ReportMetadata(
            bug_type="null dereference",
            bug_file=str(source),
            bug_line=20,
            function_name="recv",
            path_evidence="trace",
        ),
        source_snippets={},
        raw_html_path=str(tmp_path / "report.html"),
        path_events=[
            {
                "path_number": 1,
                "event_kind": "event",
                "description": "Calling 'config_enabled'",
                "line_number": 20,
                "file_path": str(source),
            },
            {
                "path_number": 2,
                "event_kind": "event",
                "description": "Returning the value 1",
                "line_number": 20,
                "file_path": str(source),
            },
            {
                "path_number": 3,
                "event_kind": "control",
                "description": "Taking false branch",
                "line_number": 20,
                "branch_condition": False,
                "is_control_flow": True,
                "file_path": str(source),
            },
            {
                "path_number": 4,
                "event_kind": "event",
                "description": "Assuming field 'opcode' is equal to CONTROL_PING",
                "line_number": 20,
                "file_path": str(source),
            },
        ],
    )

    conflicts = evaluate_control_direction_conflicts(parsed, tmp_path)

    assert conflicts == [{
        "event_number": 3,
        "line": 20,
        "reported_branch_taken": False,
        "derived_branch_taken": True,
        "condition": "!config_enabled(ctx) || is_control_frame(opcode)",
        "supporting_events": [2, 3, 4],
    }]


def test_selected_call_must_match_the_callee_return_value(tmp_path):
    source = tmp_path / "cursor.cpp"
    source.write_text(
        "bool tryAdvanceBuffer() {\n"
        "  int remaining = 1;\n"
        "  if (remaining == 0) {\n"
        "    return false;\n"
        "  }\n"
        "  return true;\n"
        "}\n"
        "\n"
        "bool pullAtMostSlow() {\n"
        "  int length = 0;\n"
        "  if (length == 0 && !tryAdvanceBuffer()) {\n"
        "    return false;\n"
        "  }\n"
        "  return true;\n"
        "}\n",
        encoding="utf-8",
    )
    segment = SegmentInfo(
        segment_index=5,
        event_range=(1, 3),
        line_start=9,
        line_end=14,
        function_name="pullAtMostSlow",
        function_file=str(source),
    )
    selections = [{
        "segment_index": 5,
        "function": "pullAtMostSlow",
        "file": str(source),
        "branch_decisions": [
            {
                "kind": "control",
                "source_line": 11,
                "branch_taken": True,
                "condition": "length == 0 && !tryAdvanceBuffer()",
                "function_name": "pullAtMostSlow",
                "file": str(source),
            },
            {
                "kind": "call",
                "source_line": 11,
                "branch_taken": True,
                "condition": "→ tryAdvanceBuffer()",
                "callee_name": "tryAdvanceBuffer",
                "function_name": "pullAtMostSlow",
                "file": str(source),
            },
            {
                "kind": "control",
                "source_line": 3,
                "branch_taken": False,
                "condition": "remaining == 0",
                "function_name": "tryAdvanceBuffer",
                "file": str(source),
            },
        ],
    }]

    violations = selected_call_return_violations(selections, [segment])

    assert len(violations) == 1
    assert violations[0]["callee"] == "tryAdvanceBuffer"
    assert violations[0]["required_return"] is False
    assert violations[0]["selected_return"] is True


def test_selected_call_return_match_is_consistent(tmp_path):
    source = tmp_path / "cursor.cpp"
    source.write_text(
        "bool tryAdvanceBuffer() {\n"
        "  int remaining = 0;\n"
        "  if (remaining == 0) {\n"
        "    return false;\n"
        "  }\n"
        "  return true;\n"
        "}\n"
        "\n"
        "bool pullAtMostSlow() {\n"
        "  int length = 0;\n"
        "  if (length == 0 && !tryAdvanceBuffer()) {\n"
        "    return false;\n"
        "  }\n"
        "  return true;\n"
        "}\n",
        encoding="utf-8",
    )
    segment = SegmentInfo(
        segment_index=5,
        event_range=(1, 3),
        line_start=9,
        line_end=14,
        function_name="pullAtMostSlow",
        function_file=str(source),
    )
    selections = [{
        "segment_index": 5,
        "function": "pullAtMostSlow",
        "file": str(source),
        "branch_decisions": [
            {
                "kind": "control",
                "source_line": 11,
                "branch_taken": True,
                "condition": "length == 0 && !tryAdvanceBuffer()",
                "function_name": "pullAtMostSlow",
                "file": str(source),
            },
            {
                "kind": "call",
                "source_line": 11,
                "branch_taken": True,
                "condition": "→ tryAdvanceBuffer()",
                "callee_name": "tryAdvanceBuffer",
                "function_name": "pullAtMostSlow",
                "file": str(source),
            },
            {
                "kind": "control",
                "source_line": 3,
                "branch_taken": True,
                "condition": "remaining == 0",
                "function_name": "tryAdvanceBuffer",
                "file": str(source),
            },
        ],
    }]

    assert selected_call_return_violations(selections, [segment]) == []


def test_global_merge_rejects_call_return_conflict_before_llm(tmp_path):
    source = tmp_path / "cursor.cpp"
    source.write_text(
        "bool tryAdvanceBuffer() {\n"
        "  int remaining = 0;\n"
        "  if (remaining == 0) {\n"
        "    return false;\n"
        "  }\n"
        "  return true;\n"
        "}\n"
        "\n"
        "bool pullAtMostSlow() {\n"
        "  int length = 0;\n"
        "  if (length == 0 && !tryAdvanceBuffer()) {\n"
        "    return false;\n"
        "  }\n"
        "  return true;\n"
        "}\n",
        encoding="utf-8",
    )
    segment = SegmentInfo(
        segment_index=5,
        event_range=(1, 3),
        line_start=9,
        line_end=14,
        function_name="pullAtMostSlow",
        function_file=str(source),
    )
    selections = [{
        "segment_index": 5,
        "function": "pullAtMostSlow",
        "file": str(source),
        "branch_decisions": [
            {
                "kind": "control",
                "source_line": 11,
                "branch_taken": True,
                "condition": "length == 0 && !tryAdvanceBuffer()",
                "function_name": "pullAtMostSlow",
                "file": str(source),
            },
            {
                "kind": "call",
                "source_line": 11,
                "branch_taken": True,
                "condition": "→ tryAdvanceBuffer()",
                "callee_name": "tryAdvanceBuffer",
                "function_name": "pullAtMostSlow",
                "file": str(source),
            },
            {
                "kind": "control",
                "source_line": 3,
                "branch_taken": False,
                "condition": "remaining == 0",
                "function_name": "tryAdvanceBuffer",
                "file": str(source),
            },
        ],
    }]
    analyzer = FPAnalyzer.__new__(FPAnalyzer)
    analyzer._source_root = str(tmp_path)
    analysis = SimpleNamespace(
        parsed_report=SimpleNamespace(),
        source_root=str(tmp_path),
    )

    result = analyzer._check_global_path_consistency(
        analysis, None, selections, "", "", segments=[segment],
    )

    assert result["is_globally_feasible"] is False
    assert result["conflicting_segments"] == [5]
    assert "tryAdvanceBuffer() == False" in result["reason"]


def test_path_selection_model_failure_is_not_reported_as_fp(tmp_path):
    class Analyzer:
        def select_feasible_path_segments(self, *_args, **_kwargs):
            raise PathSelectionUnavailable("invalid model response")

    selections, stats = run_path_selection(
        Analyzer(),
        parsed_report=object(),
        segments=[],
        slice_mask=None,
        source_root=tmp_path,
    )

    assert selections is None
    assert stats["selection_error"] == "invalid model response"
    assert "not a report-level FP" in stats["result"]
