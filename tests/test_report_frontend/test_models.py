"""The canonical model: enumerations, locations, steps, findings, documents.

These tests pin the *contract* other layers read, not the implementation: which
step kinds segment a path, that severity never defaults upward, that a finding's
selectors resolve by id / location / rule, and that the model round-trips
through ``to_dict``/``from_dict`` — the property a serialised bundle needs.
"""

from __future__ import annotations

import pytest

from report_frontend.report_model import (
    FRONTEND_MODEL_VERSION,
    CONSTRAINT_STEP_KINDS,
    AnalyzerKind,
    BugFinding,
    ReportDocument,
    ReportFormat,
    Severity,
    SourceLocation,
    StepKind,
    TraceStep,
)


class TestEnumerations:
    def test_analyzer_coerce_is_case_insensitive_and_total(self) -> None:
        assert AnalyzerKind.coerce("CodeQL") is AnalyzerKind.CODEQL
        assert AnalyzerKind.coerce("  infer ") is AnalyzerKind.INFER
        # Anything unknown — including None — must land on UNKNOWN rather than
        # raising: adapters feed this from third-party files.
        assert AnalyzerKind.coerce("coverity") is AnalyzerKind.UNKNOWN
        assert AnalyzerKind.coerce(None) is AnalyzerKind.UNKNOWN

    def test_severity_never_defaults_upward(self) -> None:
        # The conservative default is the whole point of this mapping: a wrong
        # "this is serious" is worse than an honest "unknown".
        assert Severity.coerce("critical") is Severity.ERROR
        assert Severity.coerce("medium") is Severity.WARNING
        assert Severity.coerce("low") is Severity.NOTE
        assert Severity.coerce("Memory error") is Severity.ERROR
        assert Severity.coerce("Logic error") is Severity.ERROR
        assert Severity.coerce("flarp") is Severity.UNKNOWN
        assert Severity.coerce(None) is Severity.UNKNOWN

    def test_only_boundary_kinds_segment_a_path(self) -> None:
        # cfg_feasibility splits segments on exactly these; adding one here
        # silently re-segments every report.
        assert {k for k in StepKind if k.is_boundary} == {
            StepKind.CONTROL, StepKind.CALL, StepKind.REPORT,
        }

    def test_constraint_kinds_are_the_ones_stages_4_and_6b_scan(self) -> None:
        assert CONSTRAINT_STEP_KINDS == frozenset(
            {StepKind.ASSUME, StepKind.CONTROL}
        )


class TestSourceLocation:
    def test_validity_and_description(self) -> None:
        loc = SourceLocation(file="faiss/impl/index_read.cpp", line=12, column=7)
        assert loc.is_valid
        assert not SourceLocation().is_valid
        assert "index_read.cpp:12:7" in loc.describe()

    def test_at_line_keeps_the_file(self) -> None:
        loc = SourceLocation(file="a.cpp", line=12, column=7, end_line=14)
        moved = loc.at_line(30)
        assert (moved.file, moved.line) == ("a.cpp", 30)

    def test_round_trip(self) -> None:
        loc = SourceLocation(file="a.cpp", line=3, column=4, end_line=5, end_column=6)
        assert SourceLocation.from_dict(loc.to_dict()) == loc


class TestTraceStep:
    def test_accessors_read_the_location(self) -> None:
        step = TraceStep(
            index=2, kind=StepKind.CALL,
            location=SourceLocation(file="a.cpp", line=9, column=1),
            message="Calling 'fourcc'", callee="fourcc",
        )
        assert step.file == "a.cpp"
        assert step.line == 9
        assert step.is_call and not step.is_control
        assert not step.is_error_start
        assert step.is_constraint is False

    def test_assume_is_a_constraint(self) -> None:
        step = TraceStep(index=1, kind=StepKind.ASSUME,
                         message="Assuming the condition is true")
        assert step.is_constraint

    def test_round_trip_keeps_raw(self) -> None:
        step = TraceStep(
            index=4, kind=StepKind.CONTROL, message="Taking true branch",
            direction=True, condition="x > 0",
            raw={"path_number": 4, "legacy_branch_condition": True},
        )
        back = TraceStep.from_dict(step.to_dict())
        assert back.kind is StepKind.CONTROL
        assert back.direction is True
        assert back.condition == "x > 0"
        assert back.raw == step.raw


class TestBugFinding:
    @pytest.fixture
    def finding(self) -> BugFinding:
        return BugFinding(
            finding_id="cpp/demo@a.cpp:12",
            analyzer=AnalyzerKind.CODEQL,
            rule_id="cpp/demo",
            category="Security",
            severity=Severity.WARNING,
            message="demo bug",
            location=SourceLocation(file="a.cpp", line=12),
            steps=[
                TraceStep(index=1, kind=StepKind.ERROR_START,
                          location=SourceLocation(file="a.cpp", line=3),
                          message="origin"),
                TraceStep(index=2, kind=StepKind.CALL,
                          location=SourceLocation(file="b.h", line=5),
                          message="Calling 'helper'", callee="helper"),
                TraceStep(index=3, kind=StepKind.CONTROL,
                          location=SourceLocation(file="a.cpp", line=9),
                          message="Taking true branch", direction=True),
                TraceStep(index=4, kind=StepKind.REPORT,
                          location=SourceLocation(file="a.cpp", line=12),
                          message="demo bug"),
            ],
        )

    def test_path_length_counts_parsed_steps(self, finding: BugFinding) -> None:
        assert finding.path_length == 4

    def test_error_start_and_calls(self, finding: BugFinding) -> None:
        assert finding.error_start is not None
        assert finding.error_start.message == "origin"
        assert finding.call_chain == ["helper"]

    def test_files_are_deduplicated_in_path_order(self, finding: BugFinding) -> None:
        assert finding.files() == ["a.cpp", "b.h"]

    def test_steps_of(self, finding: BugFinding) -> None:
        assert len(finding.steps_of(StepKind.CONTROL)) == 1

    def test_round_trip(self, finding: BugFinding) -> None:
        back = BugFinding.from_dict(finding.to_dict())
        assert back.finding_id == finding.finding_id
        assert back.analyzer is AnalyzerKind.CODEQL
        assert back.severity is Severity.WARNING
        assert [s.kind for s in back.steps] == [s.kind for s in finding.steps]
        assert back.location == finding.location


class TestReportDocument:
    @pytest.fixture
    def document(self) -> ReportDocument:
        def mk(fid: str, rule: str, file: str, line: int) -> BugFinding:
            return BugFinding(
                finding_id=fid, analyzer=AnalyzerKind.CODEQL, rule_id=rule,
                message="m", location=SourceLocation(file=file, line=line),
            )

        return ReportDocument(
            path="x.sarif",
            format=ReportFormat.SARIF,
            analyzer=AnalyzerKind.CODEQL,
            findings=[
                mk("cpp/a@faiss/impl/index_read.cpp:12",
                   "cpp/a", "faiss/impl/index_read.cpp", 12),
                mk("cpp/b@faiss/utils/AlignedTable.h:66",
                   "cpp/b", "faiss/utils/AlignedTable.h", 66),
            ],
        )

    def test_is_multi_finding(self, document: ReportDocument) -> None:
        assert document.is_multi_finding

    @pytest.mark.parametrize("selector", [
        "cpp/a@faiss/impl/index_read.cpp:12",   # exact finding id
        "12",                                    # bare line
        "faiss/impl/index_read.cpp:12",          # path:line
        "index_read.cpp:12",                     # basename:line
        "cpp/a@index_read.cpp:12",               # rule@basename:line
    ])
    def test_selectors(self, document: ReportDocument, selector: str) -> None:
        found = document.finding(selector)
        assert found is not None, selector
        assert found.rule_id == "cpp/a"

    def test_unmatched_selector_returns_none(self, document: ReportDocument) -> None:
        assert document.finding("nope") is None

    def test_round_trip(self, document: ReportDocument) -> None:
        back = ReportDocument.from_dict(document.to_dict())
        assert back.format is ReportFormat.SARIF
        assert [f.finding_id for f in back.findings] == [
            f.finding_id for f in document.findings
        ]

    def test_model_version_is_declared(self) -> None:
        # A serialised bundle that cannot say which model shape it is has
        # already cost this repository one full evaluation.
        assert FRONTEND_MODEL_VERSION
