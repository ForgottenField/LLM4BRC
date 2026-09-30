"""Canonical report model → the pipeline's legacy ``ParsedReport``/``BugPath``.

The path-selection pipeline is 26k lines that read ``ParsedReport``: a
pydantic object whose ``path_events`` is a list of dicts keyed
``path_number``/``event_kind``/``description``/``line_number``/``branch_condition``/
``file_path``/``is_error_start``/``is_calling_event``/``called_function``/
``is_control_flow``.  That shape is CSA's, and about a dozen stages decode it.

Rewriting those stages to read the canonical model would be the *right* end
state and is far too large a change to land at once.  So the pipeline keeps its
interface and this module is the seam: it renders the canonical finding back
into exactly the dicts the stages expect.  Two consequences worth stating
plainly, because they are the whole reason this file is short:

* **For CSA input the rendering is lossless.**  Every legacy field is
  reconstructed from the typed step, and the three that the model refines
  (``branch_condition`` for loop/ternary control steps) are taken from the
  step's ``raw`` — the value the old parser produced — so a refactor that
  changed nothing about the analysis changes nothing about its inputs either.
* **For non-CSA input it is the honest best available.**  A SARIF finding has
  no ``error start`` step, no ``Calling 'X'`` wording and no branch direction;
  the rendered dicts simply do not carry those fields, and the stages that look
  for them find nothing and stay conservative.  Inventing values so that
  CodeQL input "looks like" a CSA path is how a false TP gets certified, and
  the calibration note in ``CLAUDE.md`` is explicit that the conservative
  default is what a missing signal must produce.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from llm_client.csa_analysis.models import BugPath, EventKind, PathEvent
from llm_client.fp_analysis.models import ParsedReport, ReportMetadata
from llm_client.report_frontend.frontend import LoadedReport
from llm_client.report_frontend.models import StepKind, TraceStep

logger = logging.getLogger(__name__)

#: Radius of the bug-file window, matching what the CSA HTML parser has always
#: used (``CsaHtmlAdapter.extract_source_lines``).  A report that embeds no
#: source (SARIF) gets the same window read off the local tree, so the two
#: paths hand the prompts comparable context.
_SNIPPET_RADIUS = 30


def _step_event_kind(step: TraceStep) -> str:
    """The legacy ``event_kind`` value for a canonical step.

    Three values, not eight: ``build_bug_report_path`` and everything under it
    know CONTROL, REPORT and "everything else", and the finer kinds reach them
    through ``is_error_start``/``is_calling_event`` exactly as before.
    """
    if step.kind is StepKind.REPORT:
        return "report"
    if step.kind is StepKind.CONTROL:
        return "control"
    return "event"


def _step_branch_condition(step: TraceStep) -> Optional[bool]:
    """The legacy ``branch_condition`` value for a canonical step.

    CSA steps carry the value the pre-frontend parser produced (see
    ``csa_html`` module note); a format that has no such notion yields ``None``.
    The model's own ``direction`` is deliberately *not* the fallback here for
    CSA steps — it is more precise than the legacy value, and the pipeline's
    behaviour is calibrated against the legacy one.
    """
    if "legacy_branch_condition" in step.raw:
        return step.raw["legacy_branch_condition"]
    return step.direction


#: CSA's EndPath marker.  The pipeline reads it in exactly one place
#: (``cfg_feasibility.build_bug_report_path``) to recognise the report step, and
#: a rendered step that does not carry it leaves the path with no REPORT node —
#: which segments a SARIF finding into zero segments.  Reused rather than
#: replaced because it *is* the legacy encoding of "this is the bug"; stage 1
#: now also accepts ``event_kind == "report"``, so the two agree.
_LEGACY_REPORT_PATH_NUMBER = 9999


def _step_path_number(step: TraceStep) -> int:
    """The legacy ``path_number`` for a canonical step.

    The analyzer's own number when it stated one (CSA always does — it is the
    step's position in the rendered path, and stage-level code and prompts both
    echo it), else the step's index, with the report step carrying the EndPath
    sentinel so a non-CSA path still terminates at a REPORT node.
    """
    raw = step.raw.get("path_number")
    if raw:
        return int(raw)
    if step.kind is StepKind.REPORT:
        return _LEGACY_REPORT_PATH_NUMBER
    return int(step.index)


def _step_to_event(step: TraceStep, fallback_file: str) -> dict[str, Any]:
    """One canonical step → the legacy path-event dict."""
    event_kind = _step_event_kind(step)
    is_calling = step.kind is StepKind.CALL
    return {
        "path_number": _step_path_number(step),
        "event_kind": event_kind,
        "description": step.message,
        "line_number": step.location.line,
        "branch_condition": _step_branch_condition(step),
        "file_path": step.location.file or fallback_file,
        # -- the annotation keys the pre-frontend parser wrote; consumers still
        # -- read them, so the bridge has to reproduce them exactly --
        "is_error_start": step.kind is StepKind.ERROR_START,
        "is_calling_event": is_calling,
        "called_function": (step.callee or None) if is_calling else None,
        "is_control_flow": event_kind == "control",
    }


def _error_start_event(events: list[dict]) -> Optional[dict]:
    for evt in events:
        if evt.get("is_error_start"):
            return {
                "event_number": evt.get("path_number", 0),
                "line_number": evt.get("line_number", 0),
                "description": evt.get("description", ""),
                "file_path": evt.get("file_path", ""),
            }
    return None


def _calling_events(events: list[dict]) -> list[dict]:
    return [
        {
            "event_number": evt.get("path_number", 0),
            "line_number": evt.get("line_number", 0),
            "description": evt.get("description", ""),
            "called_function": evt.get("called_function", ""),
            "file_path": evt.get("file_path", ""),
        }
        for evt in events
        if evt.get("is_calling_event")
    ]


def source_snippets(loaded: LoadedReport) -> dict[int, str]:
    """The bug file's source window, read through the code model.

    A CSA report embeds the region it flagged and that copy wins: it is the
    revision the analyzer actually ran on, and ``build_project_deps`` compares
    the local tree *against* it to detect a stale checkout.  A report that
    embeds nothing (SARIF) has only the local tree to read, and reading it here
    rather than leaving the window empty is what lets the domain-fact and
    semantic gates see the flagged code for a CodeQL finding at all.

    *Which* embedded copy is the one thing this function must not decide on its
    own.  The report carries two: ``raw["source_snippets"]`` is a window over
    the **whole document** filtered by line number alone, and
    ``extract_file_lines`` attributes each line to its own section.  The first
    therefore also collects lines of that number from the report's *other* file
    sections — or, where the bug file has a blank line, plain fills it with
    another file's code.  Measured over the 211-report corpus: **1828 lines in
    62 reports**, every one of them a line the bug file does not have at that
    number.  Handing that to the LLM means showing it another file's code at
    the line it is being asked about, and the consumer that keys by line number
    (``SimulationVerifier``) cannot tell.

    So the window comes from ``SourceCodeModel.snippet``, which serves the
    embedded copy *for that reported file* first and the local tree otherwise —
    one reader, one rule, the same one the rest of the pipeline uses.
    ``raw["source_snippets"]`` remains only as the degenerate fallback: it is
    today's text, so a report where the model somehow yields nothing keeps the
    behaviour it has always had rather than silently losing its window.
    """
    window = loaded.code.snippet(
        loaded.finding.location.file,
        loaded.finding.location.line,
        radius=_SNIPPET_RADIUS,
    )
    if window:
        return window
    embedded = loaded.finding.raw.get("source_snippets")
    if embedded:
        return {int(k): v for k, v in embedded.items()}
    return {}


def to_parsed_report(loaded: LoadedReport) -> ParsedReport:
    """Render a loaded finding as the pipeline's ``ParsedReport``."""
    finding = loaded.finding
    events = [
        _step_to_event(step, fallback_file=finding.location.file)
        for step in finding.steps
    ]
    metadata = ReportMetadata(
        bug_type=finding.rule_id,
        bug_category=finding.category,
        bug_file=finding.location.file,
        bug_line=finding.location.line,
        bug_column=finding.location.column,
        bug_description=finding.message,
        function_name=finding.function,
        # The analyzer's own step count when it states one (CSA's
        # ``BUGPATHLENGTH``), else what was parsed.  The pipeline reads this
        # field — prompts quote it and the result JSON carries it — so it keeps
        # the analyzer's number rather than silently becoming ``len(events)``.
        path_length=finding.reported_path_length or len(events),
        # Same reasoning one level out: the *analyzer's* statement about
        # whether there is a path, not a reading of the rendered event list.
        path_evidence=finding.path_evidence.value,
    )
    return ParsedReport(
        metadata=metadata,
        path_events=events,
        source_snippets=source_snippets(loaded),
        raw_html_path=str(loaded.ref.path),
        error_start_event=_error_start_event(events),
        calling_events=_calling_events(events),
        # The report's own code model rides along: it is the one reader of
        # project source for the LLM-facing stages, and it is per-report
        # (``--scan-dir`` runs many findings in one process, so a shared model
        # would carry one report's embedded copy into another's).
        code_model=loaded.code,
    )



def to_bug_path(loaded: LoadedReport) -> BugPath:
    """Render a loaded finding as the CSA-shaped ``BugPath``."""
    finding = loaded.finding
    snippets = source_snippets(loaded)
    events: list[PathEvent] = []
    for step in finding.steps:
        if step.kind is StepKind.REPORT:
            kind = EventKind.REPORT
        elif step.kind is StepKind.CONTROL:
            kind = EventKind.CONTROL
        else:
            kind = EventKind.EVENT
        line = step.location.line
        events.append(
            PathEvent(
                event_kind=kind,
                file_path=step.location.file or finding.location.file,
                line_number=int(line or 0),
                column_number=step.location.column or None,
                description=step.message,
                code_snippet=snippets.get(int(line)) if line else None,
                branch_condition=_step_branch_condition(step),
            )
        )

    # ``BugPath`` validates that the last event is the report; a finding whose
    # path the analyzer truncated gets one appended rather than raising.
    if not events or events[-1].event_kind is not EventKind.REPORT:
        events.append(
            PathEvent(
                event_kind=EventKind.REPORT,
                file_path=finding.location.file,
                line_number=int(finding.location.line or 0),
                description=finding.message,
            )
        )

    return BugPath(
        path_id=f"fp_{loaded.ref.key}",
        bug_type=finding.rule_id,
        category=finding.category or "unknown",
        issue_context=finding.function or None,
        events=events,
        # ``check_name``/``issue_hash`` are left unset exactly as the
        # pre-frontend parser left them: nothing in the repository reads them,
        # and the refactor's contract is that the legacy views are *unchanged*.
        # The information is on the canonical finding (``rule_id``,
        # ``fingerprint``) for whoever wants it next.
    )


def to_legacy(loaded: LoadedReport) -> tuple[ParsedReport, BugPath]:
    """Both legacy views at once — what stage 1 of the pipeline consumes."""
    return to_parsed_report(loaded), to_bug_path(loaded)


__all__ = [
    "source_snippets",
    "to_bug_path",
    "to_legacy",
    "to_parsed_report",
]
