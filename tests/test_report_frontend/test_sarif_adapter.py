"""The SARIF adapter: CodeQL today, Infer and Cppcheck through the same door.

SARIF is the container every one of them can write, so this adapter is the
reason the frontend exists.  What it must *not* do is just as important as what
it must: it never invents a CONTROL step, never relabels the first flow
location as the origin of the bug, and never drops a document's claims —
``code_flow_count`` survives in ``raw`` because a result with three flows
analysed as one is a data-loss event, not a simplification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from report_frontend.report_model import (
    AnalyzerKind,
    ReportFormat,
    Severity,
    StepKind,
)
from report_frontend.registry import load_document


@pytest.fixture
def document(sarif_path: Path):
    return load_document(sarif_path)


class TestDocument:
    def test_tool_and_format(self, document) -> None:
        assert document.format is ReportFormat.SARIF
        assert document.analyzer is AnalyzerKind.CODEQL
        assert document.tool_version == "2.16.1"

    def test_every_result_is_a_finding(self, document) -> None:
        assert [f.rule_id for f in document.findings] == [
            "cpp/uncontrolled-allocation-size", "cpp/leap-year",
        ]

    def test_source_root_hint_comes_from_original_uri_base_ids(
        self, document
    ) -> None:
        # ``file:///build/faiss/`` is the build machine's root.  Keeping it is
        # what lets the CLI look the project up by name instead of guessing.
        assert "build/faiss" in (document.source_root_hint or "")


class TestFinding:
    def test_location_and_message(self, document) -> None:
        finding = document.findings[0]
        assert finding.location.file.endswith("faiss/impl/index_read.cpp")
        assert finding.location.line == 12
        assert finding.location.column == 7
        assert finding.location.end_line == 12
        assert "unvalidated value" in finding.message

    def test_severity_from_security_severity(self, document) -> None:
        # 8.1 ≥ 7.0 → error.  An unrecognised severity must not become ERROR,
        # but a stated one must not be thrown away either.
        assert document.findings[0].severity is Severity.ERROR

    def test_severity_from_level_when_no_score(self, document) -> None:
        assert document.findings[1].severity is Severity.NOTE

    def test_category_from_tags(self, document) -> None:
        assert document.findings[0].category

    def test_fingerprint_is_kept(self, document) -> None:
        assert document.findings[0].fingerprint == "abc123:1"

    def test_finding_ids_are_unique_and_readable(self, document) -> None:
        ids = [f.finding_id for f in document.findings]
        assert len(set(ids)) == len(ids)
        assert ids[0].startswith("cpp/uncontrolled-allocation-size@")
        assert ":12" in ids[0]

    def test_duplicate_locations_get_distinct_ids(self, tmp_path: Path) -> None:
        # Two results of the same rule on the same line are two findings; a
        # shared id would make one of them unreachable by --report.
        import json

        doc = {
            "version": "2.1.0",
            "runs": [{
                "tool": {"driver": {"name": "CodeQL", "rules": []}},
                "results": [
                    {"ruleId": "cpp/x", "message": {"text": "a"},
                     "locations": [{"physicalLocation": {
                         "artifactLocation": {"uri": "a.cpp"},
                         "region": {"startLine": 3}}}]},
                    {"ruleId": "cpp/x", "message": {"text": "b"},
                     "locations": [{"physicalLocation": {
                         "artifactLocation": {"uri": "a.cpp"},
                         "region": {"startLine": 3}}}]},
                ],
            }],
        }
        path = tmp_path / "dup.sarif"
        path.write_text(json.dumps(doc))
        ids = [f.finding_id for f in load_document(path).findings]
        assert len(set(ids)) == 2

    def test_no_function_is_claimed(self, document) -> None:
        # SARIF states no enclosing function.  Claiming one would be an
        # invention, and the pipeline fills it from the PDG instead.
        assert document.findings[0].function == ""


class TestCodeFlow:
    def test_flow_becomes_steps_ending_at_the_report(self, document) -> None:
        steps = document.findings[0].steps
        assert steps[-1].kind is StepKind.REPORT
        assert steps[-1].location.line == 12

    def test_flow_locations_are_in_order(self, document) -> None:
        lines = [s.location.line for s in document.findings[0].steps]
        assert lines == [5, 9, 12]

    def test_messages_survive(self, document) -> None:
        messages = [s.message for s in document.findings[0].steps]
        assert "value originates here" in messages[0]

    def test_no_control_steps_are_invented(self, document) -> None:
        # A SARIF flow carries no branch direction.  Emitting CONTROL steps
        # would hand stage 6a/6b and the branch filter evidence the analyzer
        # never gave.
        kinds = {s.kind for s in document.findings[0].steps}
        assert StepKind.CONTROL not in kinds
        assert StepKind.ASSUME not in kinds

    def test_flow_counts_are_recorded(self, document) -> None:
        raw = document.findings[0].raw
        assert raw["code_flow_count"] == 1
        assert raw["thread_flow_count"] == 1

    def test_result_without_a_flow_is_a_location_plus_a_report_step(
        self, document
    ) -> None:
        finding = document.findings[1]
        assert [s.kind for s in finding.steps] == [StepKind.REPORT]
        assert finding.steps[0].location == finding.location

    def test_return_marker_becomes_a_return_step(self, tmp_path: Path) -> None:
        import json

        doc = {
            "version": "2.1.0",
            "runs": [{
                "tool": {"driver": {"name": "CodeQL", "rules": []}},
                "results": [{
                    "ruleId": "cpp/x",
                    "message": {"text": "m"},
                    "locations": [{"physicalLocation": {
                        "artifactLocation": {"uri": "a.cpp"},
                        "region": {"startLine": 9}}}],
                    "codeFlows": [{"threadFlows": [{"locations": [
                        {"location": {"physicalLocation": {
                            "artifactLocation": {"uri": "a.cpp"},
                            "region": {"startLine": 3}},
                            "message": {"text": "v"}}},
                        {"location": {"physicalLocation": {
                            "artifactLocation": {"uri": "a.cpp"},
                            "region": {"startLine": 5}},
                            "message": {"text": "v [Return]"}}},
                        {"location": {"physicalLocation": {
                            "artifactLocation": {"uri": "a.cpp"},
                            "region": {"startLine": 9}},
                            "message": {"text": "v"}}},
                    ]}]}],
                }],
            }],
        }
        path = tmp_path / "return.sarif"
        path.write_text(json.dumps(doc))
        kinds = [s.kind for s in load_document(path).findings[0].steps]
        assert StepKind.RETURN in kinds
        assert kinds[-1] is StepKind.REPORT

    def test_the_longest_flow_wins_and_the_rest_are_counted(
        self, tmp_path: Path
    ) -> None:
        import json

        def loc(line: int) -> dict:
            return {"location": {"physicalLocation": {
                "artifactLocation": {"uri": "a.cpp"},
                "region": {"startLine": line}}}}

        doc = {
            "version": "2.1.0",
            "runs": [{
                "tool": {"driver": {"name": "CodeQL", "rules": []}},
                "results": [{
                    "ruleId": "cpp/x",
                    "message": {"text": "m"},
                    "locations": [{"physicalLocation": {
                        "artifactLocation": {"uri": "a.cpp"},
                        "region": {"startLine": 30}}}],
                    "codeFlows": [
                        {"threadFlows": [{"locations": [loc(1), loc(30)]}]},
                        {"threadFlows": [{"locations": [loc(2), loc(3), loc(30)]}]},
                    ],
                }],
            }],
        }
        path = tmp_path / "flows.sarif"
        path.write_text(json.dumps(doc))
        finding = load_document(path).findings[0]
        assert [s.location.line for s in finding.steps] == [2, 3, 30]
        # The discarded flow is still visible as a count.
        assert finding.raw["code_flow_count"] == 2


class TestMalformedInput:
    def test_not_json(self, tmp_path: Path) -> None:
        from report_frontend.errors import ReportParseError

        path = tmp_path / "broken.sarif"
        from report_frontend.registry import get_adapter
        path.write_text("{not json")
        with pytest.raises(ReportParseError):
            get_adapter("sarif").load(path, path.read_text())

    def test_json_without_runs(self, tmp_path: Path) -> None:
        from report_frontend.errors import ReportParseError
        from report_frontend.registry import get_adapter

        path = tmp_path / "other.json"
        path.write_text('{"findings": []}')
        with pytest.raises(ReportParseError):
            get_adapter("sarif").load(path, path.read_text())

    def test_suppressions_are_preserved(self, document) -> None:
        # A suppressed finding is still a finding; the pipeline decides what to
        # do with it, and that needs the fact.
        assert document.findings[1].raw["suppressions"] == [{"kind": "inSource"}]


class TestNoSourceFile:
    """``file:///`` is "no file here", not the filesystem root.

    Found in the real faiss export: two steps of a 32-step flow point at the
    bare ``file:///`` artefact (CodeQL's entry for locations it cannot place).
    Read as ``/`` it became a *file the report named* — ``SourceCodeModel``
    listed it under ``unresolved`` and the frontend warned that ``/`` does not
    exist under the source root, which reads exactly like a failed resolution.
    """

    def test_the_bare_root_is_no_file(self) -> None:
        from report_frontend.adapters.sarif import (
            _location_from,
            _uri_to_path,
        )

        assert _uri_to_path("file:///") == ""
        assert _uri_to_path("file://") == ""
        loc = _location_from(
            {"artifactLocation": {"uri": "file:///"}}, {"artifacts": []}
        )
        assert loc.file == "" and loc.line == 0

    def test_a_real_root_level_file_survives(self) -> None:
        from report_frontend.adapters.sarif import _uri_to_path

        assert _uri_to_path("file:///build/faiss/clone_index.cpp") == (
            "/build/faiss/clone_index.cpp"
        )

    def test_a_system_header_is_still_placed_by_its_absolute_path(self) -> None:
        # The other absolute spelling in the same export: those resolve by
        # `direct` to the real file, which is faithful — the report named it.
        # (A system header is outside the source root, so the *pipeline* is
        # what must decline to expand into it; see rule C in
        # `cfg_branch_extractor`.)  Only the bare root is empty.
        from report_frontend.adapters.sarif import _uri_to_path

        assert _uri_to_path("file:///usr/include/c++/9/bits/shared_ptr.h") == (
            "/usr/include/c++/9/bits/shared_ptr.h"
        )
