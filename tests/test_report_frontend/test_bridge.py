"""The bridge: canonical model → the legacy views the pipeline stages read.

The bridge is the seam that lets a formats refactor land without rewriting the
26k lines below it, so its contract has two halves and both are load-bearing:

* **CSA in, unchanged out.**  Covered by ``test_csa_fidelity`` against a
  golden captured from the parser this replaced.
* **Anything else in, nothing invented out.**  That is what this file tests.
  A SARIF finding has no ``error start`` step, no ``Calling 'X'`` wording and no
  branch direction; the rendered dicts must simply not carry those fields, so
  the stages that look for them find nothing and stay conservative.  Inventing
  values so CodeQL input "looks like" a CSA path is how a false TP gets
  certified, and CLAUDE.md is explicit that a missing signal must produce the
  conservative default, never a confident one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_client.fp_analysis.report_bridge import (
    _SNIPPET_RADIUS,
    source_snippets,
    to_bug_path,
    to_legacy,
    to_parsed_report,
)
from report_frontend import ReportFrontend, ReportRef
from csa_analysis import EventKind


@pytest.fixture
def csa_loaded(csa_report: Path):
    return ReportFrontend().load(csa_report)


@pytest.fixture
def sarif_loaded(sarif_path: Path):
    return ReportFrontend().load(
        ReportRef(sarif_path, "cpp/uncontrolled-allocation-size@index_read.cpp:12")
    )


class TestLegacyEventShape:
    """The exact keys ``cfg_feasibility`` and its neighbours read."""

    LEGACY_KEYS = {
        "path_number", "event_kind", "description", "line_number",
        "branch_condition", "file_path", "is_error_start", "is_calling_event",
        "called_function", "is_control_flow",
    }

    def test_csa_events_have_every_key(self, csa_loaded) -> None:
        parsed = to_parsed_report(csa_loaded)
        for evt in parsed.path_events:
            assert self.LEGACY_KEYS <= set(evt)

    def test_sarif_events_have_every_key_too(self, sarif_loaded) -> None:
        # Same shape, different *values*: a stage that does ``evt["is_control_flow"]``
        # must not get a KeyError for a CodeQL report.
        parsed = to_parsed_report(sarif_loaded)
        for evt in parsed.path_events:
            assert self.LEGACY_KEYS <= set(evt)

    def test_event_kinds_are_the_three_the_stages_know(self, sarif_loaded) -> None:
        parsed = to_parsed_report(sarif_loaded)
        assert {e["event_kind"] for e in parsed.path_events} <= {
            "event", "control", "report",
        }


class TestNothingIsInventedForNonCsa:
    def test_no_error_start(self, sarif_loaded) -> None:
        parsed = to_parsed_report(sarif_loaded)
        assert not any(e["is_error_start"] for e in parsed.path_events)
        assert parsed.error_start_event is None

    def test_no_phantom_call_events(self, sarif_loaded) -> None:
        parsed = to_parsed_report(sarif_loaded)
        assert parsed.calling_events == []
        assert not any(e["is_calling_event"] for e in parsed.path_events)
        # The *steps* are there — a flow is a flow — they are just not labelled
        # as calls, because SARIF does not say they are.
        assert len(parsed.path_events) == 3

    def test_no_branch_directions(self, sarif_loaded) -> None:
        parsed = to_parsed_report(sarif_loaded)
        assert all(e["branch_condition"] is None for e in parsed.path_events)
        assert not any(e["is_control_flow"] for e in parsed.path_events)

    def test_no_function_is_claimed(self, sarif_loaded) -> None:
        # The CLI fills this from the PDG; the bridge must not fabricate one.
        assert to_parsed_report(sarif_loaded).metadata.function_name == ""

    def test_the_bug_is_still_reported(self, sarif_loaded) -> None:
        # Conservative is not the same as empty: the location, the rule and the
        # message are all there, and they are what the gates read.
        meta = to_parsed_report(sarif_loaded).metadata
        assert meta.bug_type == "cpp/uncontrolled-allocation-size"
        assert meta.bug_line == 12
        assert meta.bug_file.endswith("faiss/impl/index_read.cpp")
        assert meta.bug_description

    def test_the_path_ends_at_the_report(self, sarif_loaded) -> None:
        parsed = to_parsed_report(sarif_loaded)
        assert parsed.path_events[-1]["event_kind"] == "report"
        assert parsed.path_events[-1]["line_number"] == 12


class TestPathLength:
    def test_csa_keeps_the_analyzers_own_count(self, csa_loaded) -> None:
        # CSA's BUGPATHLENGTH is what the prompts quote and the result JSON
        # carries; it is not len(events) (25 vs 25 here, 40 vs 38 on 484-1).
        parsed = to_parsed_report(csa_loaded)
        assert parsed.metadata.path_length == csa_loaded.finding.reported_path_length

    def test_falls_back_to_the_parsed_count_when_the_analyzer_states_none(
        self, sarif_loaded
    ) -> None:
        assert sarif_loaded.finding.reported_path_length == 0
        parsed = to_parsed_report(sarif_loaded)
        assert parsed.metadata.path_length == len(parsed.path_events)


class TestSnippets:
    def test_embedded_wins(self, csa_loaded) -> None:
        """The report's own copy wins — the one keyed by the *flagged file*.

        Asserting equality with ``raw["source_snippets"]`` used to name the
        wrong half of the report's embedded source: that key is a
        whole-document window filtered by line number alone, so 6 of its 61
        lines here (387, 395, 404, 410, 417, 432) belong to the report's other
        file sections.  The claim is unchanged — embedded beats the local tree —
        and it is the file's own listing that is embedded.
        """
        finding = csa_loaded.finding
        own = finding.raw["embedded_files"][finding.location.file]
        expected = {
            n: text for n, text in own.items()
            if abs(n - finding.location.line) <= _SNIPPET_RADIUS
        }
        assert expected
        assert source_snippets(csa_loaded) == expected

    def test_disk_fallback_when_the_report_embeds_nothing(
        self, sarif_path: Path, source_tree: Path
    ) -> None:
        # A SARIF report carries no source.  Reading the tree here is what lets
        # the semantic gates see the flagged code for a CodeQL finding at all.
        loaded = ReportFrontend(source_root=source_tree).load(
            ReportRef(sarif_path, "cpp/leap-year@AlignedTable.h:66")
        )
        snippets = source_snippets(loaded)
        assert snippets
        assert 66 in snippets

    def test_empty_when_neither_source_exists(self, sarif_loaded) -> None:
        assert source_snippets(sarif_loaded) == {}


class TestBugPathRendering:
    def test_path_id_is_prefixed_with_the_ref_key(self, csa_loaded) -> None:
        assert to_bug_path(csa_loaded).path_id == f"fp_{csa_loaded.ref.key}"

    def test_path_id_is_unique_per_finding(self, sarif_path: Path) -> None:
        frontend = ReportFrontend()
        ids = set()
        for finding_id, _ in frontend.iter_findings(sarif_path):
            loaded = frontend.load(ReportRef(sarif_path, finding_id))
            ids.add(to_bug_path(loaded).path_id)
        assert len(ids) == 2

    def test_last_event_is_always_the_report(self, sarif_loaded) -> None:
        # ``BugPath`` validates this; a finding whose flow does not end at the
        # bug gets a REPORT event appended rather than an exception.
        assert to_bug_path(sarif_loaded).events[-1].event_kind is EventKind.REPORT

    def test_a_synthesized_report_event_does_not_duplicate_an_existing_one(
        self, csa_loaded
    ) -> None:
        events = to_bug_path(csa_loaded).events
        assert sum(1 for e in events if e.event_kind is EventKind.REPORT) == 1

    def test_bug_type_and_category_come_from_the_finding(self, csa_loaded) -> None:
        bp = to_bug_path(csa_loaded)
        assert bp.bug_type == csa_loaded.finding.rule_id
        assert bp.category == csa_loaded.finding.category
        assert bp.issue_context == csa_loaded.finding.function

    def test_to_legacy_returns_both(self, csa_loaded) -> None:
        parsed, bug_path = to_legacy(csa_loaded)
        assert len(parsed.path_events) == len(bug_path.events)


class TestStageOneConsumers:
    """The bridge's output must be *usable* by stage 2, not merely well-shaped."""

    def test_segmentation_accepts_a_sarif_finding(self, sarif_loaded) -> None:
        from llm_client.fp_analysis.cfg_feasibility import segment_path

        parsed = to_parsed_report(sarif_loaded)
        segments = segment_path(parsed)
        # One segment per boundary plus the tail; a SARIF flow of three EVENT
        # steps is a single linear segment.
        assert len(segments) >= 1
        assert all(s.line_start and s.line_end for s in segments)

    def test_segmentation_accepts_a_control_bearing_finding(
        self, csa_loaded
    ) -> None:
        from llm_client.fp_analysis.cfg_feasibility import segment_path

        segments = segment_path(to_parsed_report(csa_loaded))
        assert len(segments) > 1
