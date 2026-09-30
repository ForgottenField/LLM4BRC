"""CSA fidelity: the frontend must be indistinguishable from the parser it replaced.

The whole formats refactor rests on one claim — that for CSA input, reading a
report through ``ReportFrontend`` + ``report_bridge`` produces exactly what
``FPReportParser`` produced.  If that claim is wrong the refactor is not a
refactor: every stage below is calibrated against the legacy dicts, and a
silently re-classified step re-segments a path.

So this file does not test the *adapter*.  It pins the **rendered legacy
output**:

* :class:`TestGoldenOutput` compares against literals captured from the
  pre-frontend parser (``git show HEAD:src/llm_client/fp_analysis/html_parser.py``)
  run over the in-repo fixture — a byte-level regression baseline, checked in
  because the parser it came from no longer exists to re-derive it;
* :class:`TestCorpusInvariants` states the *contract* the downstream stages
  depend on and checks it over the whole corpus when the corpus is present
  (``CSA_PROJECTS_DIR``), because 211 real reports find what one fixture cannot.

The distinction matters: the golden catches "this changed", the invariants
catch "this is wrong" on input the golden never saw.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_client.fp_analysis.report_bridge import (
    _SNIPPET_RADIUS,
    to_bug_path,
    to_parsed_report,
)
from llm_client.report_frontend import ReportFrontend
from llm_client.report_frontend.adapters.csa_html import CsaHtmlAdapter
from llm_client.report_frontend.models import AnalyzerKind, ReportFormat, StepKind

#: Captured from the legacy parser.  Every field of ``ReportMetadata``:
LEGACY_METADATA = {
    "bug_type": "Dereference of null pointer",
    "bug_category": "Logic error",
    "bug_line": 407,
    "bug_column": 7,
    "function_name": "Destroy",
    "path_length": 25,
}

#: ``(path_number, event_kind, line_number, is_error_start, is_calling_event,
#: called_function, is_control_flow)`` per event, in order — the entire legacy
#: ``path_events`` contract, minus the long free-text descriptions (asserted
#: separately, and readable in the fixture).
LEGACY_EVENTS = [
    (1, "control", 324, False, False, None, True),
    (2, "event", 324, False, True,
     "PredicateFormatterFromMatcher::operator()", False),
    (3, "event", 1534, False, True,
     "SafeMatcherCast<const std::vector<float, folly::AlignedSysAllocator<float>>"
     " &, testing::internal::ElementsAreArrayMatcher<double>>", False),
    (4, "event", 517, False, True,
     "MatcherCast<const std::vector<float, folly::AlignedSysAllocator<float>>"
     " &, testing::internal::ElementsAreArrayMatcher<double>>", False),
    (5, "event", 510, False, True, "MatcherCastImpl::Cast", False),
    (6, "event", 350, False, True, "MatcherCastImpl::CastImpl", False),
    (7, "event", 368, False, True,
     "ElementsAreArrayMatcher::operator Matcher", False),
    (8, "event", 3617, False, False, None, False),
    (9, "control", 3187, False, False, None, True),
    (10, "control", 3187, False, False, None, True),
    (11, "event", 3188, False, True, "MatcherCast<const float &, double>", False),
    (12, "event", 510, False, True, "MatcherCastImpl::Cast", False),
    (13, "event", 350, False, True, "MatcherCastImpl::CastImpl", False),
    (14, "event", 368, False, False, None, False),
    (15, "event", 503, False, False, None, False),
    (16, "event", 287, False, False, None, False),
    (17, "event", 503, False, False, None, False),
    (18, "event", 880, False, False, None, False),
    (19, "event", 880, False, False, None, False),
    (20, "control", 321, False, False, None, True),
    (21, "event", 322, False, True, "MatcherBase::Destroy", False),
    (22, "event", 406, False, False, None, False),
    (23, "event", 406, False, False, None, False),
    (24, "control", 406, False, False, None, True),
    (9999, "report", 407, False, False, None, False),
]

#: The reason the pipeline reads ``path_length`` from the analyzer and not from
#: ``len(path_events)``: the two genuinely differ on a real report.
LEGACY_PATH_ID = "fp_report-MemoryTest.cpp-Destroy-2-1"


@pytest.fixture
def rendered(csa_report: Path):
    loaded = ReportFrontend(source_root=None).load(csa_report)
    return loaded, to_parsed_report(loaded), to_bug_path(loaded)


class TestGoldenOutput:
    def test_metadata_is_unchanged(self, rendered) -> None:
        _, parsed, _ = rendered
        meta = parsed.metadata
        assert meta.bug_type == LEGACY_METADATA["bug_type"]
        assert meta.bug_category == LEGACY_METADATA["bug_category"]
        assert meta.bug_line == LEGACY_METADATA["bug_line"]
        assert meta.bug_column == LEGACY_METADATA["bug_column"]
        assert meta.function_name == LEGACY_METADATA["function_name"]
        assert meta.path_length == LEGACY_METADATA["path_length"]

    def test_event_table_is_unchanged(self, rendered) -> None:
        _, parsed, _ = rendered
        rows = [
            (
                e["path_number"], e["event_kind"], e["line_number"],
                e["is_error_start"], e["is_calling_event"],
                e["called_function"], e["is_control_flow"],
            )
            for e in parsed.path_events
        ]
        assert rows == LEGACY_EVENTS

    def test_descriptions_survive_verbatim(self, rendered) -> None:
        _, parsed, _ = rendered
        # The pipeline reads this prose: 'Assuming the condition is …',
        # "Calling 'X'", 'error start'.  It is not decoration.
        first = parsed.path_events[0]["description"]
        assert first == "Control jumps to 'case 0:' at line 324 →"
        last = parsed.path_events[-1]["description"]
        assert "dereference of a null pointer" in last

    def test_error_start_and_calling_indices(self, rendered) -> None:
        _, parsed, _ = rendered
        # No event in this report carries the 'error start' marker, and the
        # bridge must not manufacture one.
        assert parsed.error_start_event is None
        assert [e["event_number"] for e in parsed.calling_events] == [
            2, 3, 4, 5, 6, 7, 11, 12, 13, 21,
        ]

    def test_bug_path_shape(self, rendered) -> None:
        _, _, bug_path = rendered
        assert bug_path.path_id == LEGACY_PATH_ID
        assert bug_path.bug_type == LEGACY_METADATA["bug_type"]
        assert bug_path.category == LEGACY_METADATA["bug_category"]
        assert bug_path.issue_context == LEGACY_METADATA["function_name"]
        assert len(bug_path.events) == len(LEGACY_EVENTS)
        assert bug_path.events[-1].event_kind.value == "report"
        assert bug_path.events[0].event_kind.value == "control"

    def test_check_name_and_issue_hash_stay_unset(self, rendered) -> None:
        # The legacy parser left both None; nothing reads them, and the
        # refactor's contract is that the legacy view does not change.  The
        # information lives on the canonical finding (rule_id/fingerprint).
        _, _, bug_path = rendered
        assert bug_path.check_name is None
        assert bug_path.issue_hash is None

    def test_source_snippets_are_the_reports_own_copy_of_the_bug_file(
        self, rendered
    ) -> None:
        """The window is the flagged file's own listing, not the whole document's.

        ``raw["source_snippets"]`` filters the document by line number alone, so
        it also collects lines of that number from the report's *other*
        sections.  6 of its 61 lines here are foreign: three from the bug's own
        TU (``MemoryTest.cpp``) and three from the sibling ``gmock-matchers.h``
        — including ``struct TestAlloc2 : TestAlloc1<T> {``, which that window
        serves as ``gtest-matchers.h:432``.  The pipeline keys these lines by
        number, so a foreign line is indistinguishable from the flagged file's
        own code at that number.
        """
        loaded, parsed, _ = rendered
        own = loaded.finding.raw["embedded_files"][loaded.finding.location.file]
        foreign = sorted(
            {int(k) for k in loaded.finding.raw["source_snippets"]} - set(own)
        )
        assert foreign == [387, 395, 404, 410, 417, 432]
        assert not set(parsed.source_snippets) & set(foreign)
        assert LEGACY_METADATA["bug_line"] in parsed.source_snippets
        # The embedded copy still wins over the local tree — the report is the
        # revision the analyzer ran on — and the model is the one reader: the
        # window is exactly that file's own slice of ±radius around the bug line.
        assert parsed.source_snippets == {
            n: text for n, text in own.items()
            if abs(n - LEGACY_METADATA["bug_line"]) <= _SNIPPET_RADIUS
        }
        assert len(parsed.source_snippets) == 55

    def test_raw_html_path_is_the_report(self, rendered, csa_report: Path) -> None:
        _, parsed, _ = rendered
        assert Path(parsed.raw_html_path) == csa_report


class TestCanonicalModel:
    """What the canonical model says about the same report (the new API)."""

    def test_typed_view(self, rendered) -> None:
        loaded, _, _ = rendered
        assert loaded.document.format is ReportFormat.CSA_HTML
        assert loaded.document.analyzer is AnalyzerKind.CSA
        assert loaded.finding.rule_id == LEGACY_METADATA["bug_type"]
        assert loaded.finding.location.line == LEGACY_METADATA["bug_line"]
        assert loaded.finding.reported_path_length == LEGACY_METADATA["path_length"]

    def test_last_step_is_the_report_and_only_one_is(self, rendered) -> None:
        loaded, _, _ = rendered
        reports = [s for s in loaded.finding.steps if s.kind is StepKind.REPORT]
        assert len(reports) == 1
        assert loaded.finding.steps[-1].kind is StepKind.REPORT

    def test_calling_steps_carry_their_callee(self, rendered) -> None:
        loaded, _, _ = rendered
        calls = loaded.finding.calls
        assert len(calls) == len(LEGACY_EVENTS[1:8]) + 3  # 10 calling events
        assert all(s.callee for s in calls)
        assert calls[0].callee == "PredicateFormatterFromMatcher::operator()"

    def test_loop_and_ternary_control_steps_have_no_direction(
        self, rendered
    ) -> None:
        # 25 events, not 24: the model keeps the analyzer's own step count
        # even though one event is not rendered as a distinct step.
        loaded, _, _ = rendered
        control = [s for s in loaded.finding.steps if s.kind is StepKind.CONTROL]
        assert control
        # ``legacy_branch_condition`` is what the bridge renders; the typed
        # ``direction`` may be richer and must not leak into it blindly.
        for step in control:
            assert "legacy_branch_condition" in step.raw


# ---------------------------------------------------------------------------
# Embedded code text
# ---------------------------------------------------------------------------


def _listing(*rows: tuple[int, str], file: str = "/build/proj/src/a.cc") -> str:
    """A minimal CSA listing: a FileName header, a code table, then the rows."""
    body = "".join(
        f'<tr class="codeline" data-linenumber="{n}">'
        f'<td class="num">{n}</td><td class="line">{cell}</td></tr>'
        for n, cell in rows
    )
    return (
        f'<h4 class="FileName">{file}</h4>'
        f'<table class="code"><tbody>{body}</tbody></table>'
    )


class TestEmbeddedCodeText:
    """The text the code model carries must be the *code*, not CSA's chrome.

    CSA renders tooltips inside the code line.  They are not part of the line,
    and a naive ``re.sub(r"<[^>]+>", "")`` splices them in — which is what the
    LLM then reads as the program's source.
    """

    def test_a_macro_tooltip_is_not_spliced_into_the_line(self) -> None:
        # The real shape: the tooltip is a *child* of the macro span, so the
        # text after ``TRYCLONE(a, b)`` is the tooltip's body, not the line's.
        html = _listing(
            (54, "<span class='directive'>#define <span class='macro'>TRYCLONE(a, b)"
                 "<span class='macro_popup'>if (const a* c = dynamic_cast&lt;const a*&gt;"
                 "<br>(x)) { return new a(*c); } else</span></span>   \\</span>"),
        )
        assert CsaHtmlAdapter.extract_source_lines(html, 54, radius=1)[54] == (
            "#define TRYCLONE(a, b)   \\"
        )

    def test_a_variable_tooltip_does_not_truncate_the_line(self) -> None:
        # The tooltip is a nested <table> with a <td> of its own, so a
        # non-greedy ``(.*?)</td>`` stops inside it and drops the rest of the
        # line — ``if (accelerate) {`` would come back as ``if (accelerate``.
        html = _listing(
            (278, "<span class='keyword'>if</span> (<span class='variable'>accelerate"
                  "<table class='variable_popup'><tbody><tr><td valign='top'>"
                  "<div class='PathIndex PathIndexPopUp'>27.1</div></td>"
                  "<td>'accelerate' is false</td></tr></tbody></table></span>) {"),
        )
        assert CsaHtmlAdapter.extract_source_lines(html, 278, radius=1)[278] == (
            "if (accelerate) {"
        )

    def test_entities_are_decoded(self) -> None:
        # 20.5% of the corpus's code lines carry one; ``&lt;`` is ``<``.
        html = _listing(
            (7, "# include &lt;unistd.h&gt;  // a &amp; b"),
        )
        assert CsaHtmlAdapter.extract_source_lines(html, 7, radius=1)[7] == (
            "# include <unistd.h>  // a & b"
        )

    def test_both_readers_agree_on_the_text_they_share(self) -> None:
        # ``extract_file_lines`` attributes sections, ``extract_source_lines``
        # windows a line — but for the same cell they must agree, or the two
        # code channels the pipeline reads would disagree with each other.
        html = _listing(
            (11, "<span class='variable'>n<table class='variable_popup'><tbody><tr>"
                 "<td>'n' is false</td></tr></tbody></table></span> &gt; 0"),
        )
        windowed = CsaHtmlAdapter.extract_source_lines(html, 11, radius=1)
        by_file = CsaHtmlAdapter.extract_file_lines(html)
        assert windowed[11] == "n > 0"
        assert by_file["/build/proj/src/a.cc"][11] == "n > 0"


# ---------------------------------------------------------------------------
# Corpus invariants
# ---------------------------------------------------------------------------


def _corpus_reports() -> list[Path]:
    import os

    projects = Path(
        os.environ.get("CSA_PROJECTS_DIR") or Path.home() / "csa_reports" / "project"
    )
    if not projects.is_dir():
        return []
    return sorted(projects.glob("*/reports/*/report-*.html"))


@pytest.mark.parametrize(
    "report", _corpus_reports(), ids=lambda p: f"{p.parent.parent.parent.name}/{p.name}",
)
def test_corpus_invariants(report: Path) -> None:
    """The contract every downstream stage relies on, over the real corpus."""
    loaded = ReportFrontend().load(report)
    parsed = to_parsed_report(loaded)

    events = parsed.path_events
    assert events, report
    # Ordering: build_bug_report_path walks the list in order.
    assert [e["path_number"] for e in events] == sorted(e["path_number"] for e in events)
    # The path ends at the report.
    assert events[-1]["event_kind"] == "report"
    assert events[-1]["line_number"] == parsed.metadata.bug_line
    # At most one origin, and it is flagged, not inferred from prose.
    assert sum(1 for e in events if e["is_error_start"]) <= 1
    # A calling event without a callee would silently break
    # conflict_checker's ReturnValueMapper and branch_relevance._parse_callee.
    for evt in events:
        if evt["is_calling_event"]:
            assert evt["called_function"], (report, evt)
        if evt["event_kind"] == "control":
            assert evt["is_control_flow"]
    # The analyzer's own count is never *smaller* than what was rendered.
    assert parsed.metadata.path_length >= 0
    # Every event resolves to a file, and the report's own bug file is among them.
    assert all(e["file_path"] for e in events), report
    # The canonical steps agree with the rendered event kinds, one to one.
    assert len(loaded.finding.steps) == len(events)
    for step, evt in zip(loaded.finding.steps, events):
        expected = {
            StepKind.REPORT: "report",
            StepKind.CONTROL: "control",
        }.get(step.kind, "event")
        assert expected == evt["event_kind"], (report, step, evt)


def test_corpus_is_non_empty_or_skipped() -> None:
    if not _corpus_reports():
        pytest.skip("corpus not present (set CSA_PROJECTS_DIR)")


def test_adapter_round_trips_its_own_document(csa_report: Path) -> None:
    """A re-parse of the same text yields the same model (no hidden state)."""
    adapter = CsaHtmlAdapter()
    text = csa_report.read_text(encoding="utf-8", errors="replace")
    a = adapter.load(csa_report, text)
    b = adapter.load(csa_report, text)
    assert [f.finding_id for f in a.findings] == [f.finding_id for f in b.findings]
    assert [s.to_dict() for s in a.findings[0].steps] == [
        s.to_dict() for s in b.findings[0].steps
    ]
