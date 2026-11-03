"""SARIF 2.1.0 reports → canonical model (CodeQL, Infer, Cppcheck, …).

SARIF is the one document shape several analyzers agree on, and it is the
shape the ones still to come can already produce: CodeQL emits it natively,
Infer emits it with ``--sarif``, Cppcheck with ``--output-format=sarif``.  So
this adapter is the reason the frontend is worth having — one parser covers
four analyzers, and each of them is identified from ``tool.driver.name``
rather than from the file extension.

What the document gives us
--------------------------
``runs[]``            one analysis run per language/database; findings from all
                      runs are concatenated.
``tool.driver``       the analyzer's name/version and its rule table
                      (``rules[]``: id, shortDescription, tags, severity).
``artifacts[]``       the file table the results' ``index`` fields point into.
``originalUriBaseIds`` the root each ``uriBaseId`` resolves against.
``results[]``         a finding: ruleId, message, ``locations[0]`` (the bug),
                      ``relatedLocations[]`` (secondary context), and — for
                      queries that track a value — ``codeFlows``: the ordered
                      data-flow path.
``partialFingerprints`` the analyzer's own stable id for a finding.

What it does *not* give, and why nothing here invents it
--------------------------------------------------------
A CodeQL result is a **location**, not a CSA path.  56 of the 62 faiss results
carry no ``codeFlows`` at all, and those that do describe a *data-flow* trace
("fread output argument" → "*ptr [Return]" → "operator() output argument"),
not a sequence of branch decisions.  There is no branch direction anywhere in
a SARIF document, so:

* a step is never classified ``CONTROL`` — the traces are data-flow steps, and
  calling one a branch decision would hand stage 6a a direction it was never
  told and stage 5 a condition it cannot check;
* the first flow location is **not** relabelled ``ERROR_START``.  For a taint
  query it is the *source*, which is not the same claim; ``ERROR_START`` stays
  reserved for an analyzer that says the flagged value originates there.  A
  consumer that wants "where did the tainted value come from" reads
  ``steps[0]`` and knows which analyzer it is looking at.

The consequence downstream is the intended one: stages keyed on CSA wording
find no claims, and the conservative default (``insufficient`` → enumerate →
LLM) carries the analysis, rather than a fabricated path.  See
``docs/report-frontend.md`` §"What a non-CSA report can and cannot feed".
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..errors import ReportParseError
from ..report_model import (
    AnalyzerKind,
    BugFinding,
    PathEvidence,
    ReportModel,
    ReportFormat,
    Severity,
    SourceLocation,
    StepKind,
    TraceStep,
)
from ..registry import ReportAdapter, register

logger = logging.getLogger(__name__)

#: ``tool.driver.name`` → analyzer.  Matched case-insensitively as a prefix,
#: because the driver name carries a version suffix in some builds
#: (``CodeQL``, ``codeql-cli``; ``Infer``/``infer-analyze``; ``Cppcheck``).
_DRIVER_NAMES: tuple[tuple[str, AnalyzerKind], ...] = (
    ("codeql", AnalyzerKind.CODEQL),
    ("infer", AnalyzerKind.INFER),
    ("cppcheck", AnalyzerKind.CPPCHECK),
)

#: SARIF ``properties.tags`` entries that name a category.  Checked in this
#: order: CodeQL tags most of its correctness queries ``security`` too (a
#: multiplication that can overflow is both), and the more specific family is
#: the more useful label.
_CATEGORY_TAGS: tuple[tuple[str, str], ...] = (
    ("security", "Security"),
    ("memory", "Memory error"),
    ("resource", "Resource error"),
    ("reliability", "Reliability"),
    ("correctness", "Correctness"),
    ("maintainability", "Maintainability"),
    ("readability", "Readability"),
    ("performance", "Performance"),
)

#: ``cpp/uncontrolled-allocation-size`` → ``uncontrolled allocation size``.
_RULE_ID_PRETTY_RE = re.compile(r"[-_/]")


def _coerce_text(value) -> str:
    """SARIF text fields are ``{"text": …}`` or ``{"markdown": …}`` or a string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("text") or value.get("markdown") or "").strip()
    return str(value).strip()


def _uri_to_path(uri: str) -> str:
    """A SARIF ``artifactLocation.uri`` as a filesystem-ish path string.

    URIs are percent-encoded and may be absolute ``file://`` URLs; the rest of
    the pipeline keys on plain paths.  Relative URIs (``faiss/impl/io.cpp``
    under ``%SRCROOT%``) are returned unchanged — the code model is what joins
    them to a source root, and rewriting them here would destroy the
    information that they are relative.

    ``file:///`` — the root with *no* path — is CodeQL's spelling for a
    location with no source file at all (its own artefact table carries one,
    referenced by steps that sit in code it cannot place; the real faiss
    export has two such steps in a 32-step flow).  It must come back as "no
    file", not as ``"/"``: the root is a path the report never named, and
    returning it made ``SourceCodeModel`` report an unresolved *file* and warn
    that ``/`` does not exist under the source root — noise that reads exactly
    like a failed resolution.  ``file:///foo`` (a real root-level file) is
    untouched; only the bare root is empty.
    """
    if not uri:
        return ""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        path = unquote(parsed.path)
        return "" if path == "/" else path
    if parsed.scheme:
        # Some producer's custom scheme (e.g. ``codeql://``): keep the path part.
        return unquote(parsed.path) or uri
    return unquote(uri)


def _location_from(physical: dict, run: dict) -> SourceLocation:
    """One SARIF ``physicalLocation`` → :class:`SourceLocation`."""
    if not isinstance(physical, dict):
        return SourceLocation()
    artifact = physical.get("artifactLocation") or {}
    uri = artifact.get("uri", "")
    index = artifact.get("index")
    if not uri and isinstance(index, int):
        # The artefact index is a hop into the run's file table; used when a
        # producer elides the uri on repeat references.
        artifacts = run.get("artifacts") or []
        if 0 <= index < len(artifacts):
            uri = (artifacts[index].get("location") or {}).get("uri", "")
    region = physical.get("region") or {}
    start = int(region.get("startLine", 0) or 0)
    return SourceLocation(
        file=_uri_to_path(uri),
        line=start,
        column=int(region.get("startColumn", 0) or 0),
        end_line=int(region.get("endLine", start) or start),
        end_column=int(region.get("endColumn", 0) or 0),
    )


def _category_from_rule(rule: dict) -> str:
    props = rule.get("properties") or {}
    tags = [str(t).lower() for t in (props.get("tags") or [])]
    for tag, name in _CATEGORY_TAGS:
        if tag in tags:
            return name
    kind = str(props.get("kind", "") or "")
    return kind.replace("problem", "Correctness").replace("path-problem", "Correctness")


def _severity_from_rule(rule: dict, result: dict) -> Severity:
    """SARIF severity, most specific source first.

    ``result.level`` is the producer's own override; ``defaultConfiguration``
    is the rule's; CodeQL then adds ``properties.problem.severity`` (which is
    what the query author wrote) and ``security-severity`` (a 0–10 CVSS-style
    score, where ≥7 is high).  The last two are checked first because they are
    the only ones a CodeQL query actually populates on every rule.
    """
    props = rule.get("properties") or {}
    problem = props.get("problem.severity")
    if problem:
        return Severity.coerce(problem)
    security = props.get("security-severity")
    if security:
        try:
            score = float(security)
        except (TypeError, ValueError):
            score = 0.0
        if score >= 7.0:
            return Severity.ERROR
        if score >= 4.0:
            return Severity.WARNING
        return Severity.NOTE
    level = result.get("level") or (rule.get("defaultConfiguration") or {}).get("level")
    return Severity.coerce(level)


def _same_place(a: SourceLocation, b: SourceLocation) -> bool:
    """True when two locations name the same source position.

    File and line, deliberately not the full location: a SARIF ``threadFlow``
    location may name only ``startLine`` while the result's location also gives
    the column and end position.  Comparing structurally would report a
    duplicate step at the bug's own line.
    """
    if not a.line or not b.line:
        return False
    return Path(a.file).name == Path(b.file).name and a.line == b.line


def _steps_from_code_flow(flow: dict, finding_location: SourceLocation, run: dict) -> list[TraceStep]:
    """One ``codeFlow`` → ordered ``TraceStep``s, ending at the report step.

    Threads are concatenated in order: a ``codeFlow`` may model several threads
    (the same value crossing a thread boundary) and the ordered reading of the
    document is the whole flow.  The *last* flow location becomes the REPORT
    step when it is the finding's own location, and a REPORT step is appended
    at the finding location otherwise — either way the path ends where the bug
    is, which is the invariant ``BugPath``'s validator and every stage assume.
    """
    locations: list[dict] = []
    for thread in flow.get("threadFlows") or []:
        locations.extend(thread.get("locations") or [])
    steps: list[TraceStep] = []
    for raw_loc in locations:
        loc = raw_loc.get("location") or raw_loc
        message = _coerce_text(loc.get("message"))
        location = _location_from(
            loc.get("physicalLocation") or {}, run,
        )
        kind = StepKind.EVENT
        # CodeQL marks the end of a callee's contribution with a ``[Return]``
        # suffix on the value name.  That is the only call-boundary signal in a
        # SARIF flow, and it is a reliable one — it is emitted by the query
        # library, not by the model.
        if message.endswith("[Return]"):
            kind = StepKind.RETURN
        steps.append(
            TraceStep(
                index=len(steps) + 1,
                kind=kind,
                location=location,
                message=message,
                raw={"nesting_level": raw_loc.get("nestingLevel"),
                     "kinds": raw_loc.get("kinds")},
            )
        )
    # "the last flow location *is* the finding" is compared on (file, line), not
    # on the whole location: a threadFlow location routinely pins only
    # ``startLine`` (no column, no ``endLine``) while the result's location
    # carries all of them, so a structural comparison would call the same place
    # two places and emit a duplicated REPORT step with no message.  The
    # finding's own location wins where it is the richer of the two — it is the
    # one the pipeline reads as ``bug_file``/``bug_line``.
    if steps and _same_place(steps[-1].location, finding_location):
        steps[-1].kind = StepKind.REPORT
        steps[-1].location = finding_location
    else:
        steps.append(
            TraceStep(
                index=len(steps) + 1,
                kind=StepKind.REPORT,
                location=finding_location,
                message="",
                raw={"synthesized": "report"},
            )
        )
    return steps


@register
class SarifAdapter(ReportAdapter):
    """Parser for SARIF 2.1.0 — CodeQL, Infer, Cppcheck and friends."""

    name = "sarif"
    format = ReportFormat.CODEQL_SARIF
    analyzer = AnalyzerKind.UNKNOWN  # refined per document from tool.driver.name
    suffixes = (".sarif", ".sarif.json")

    def sniff(self, path: Path, head: str) -> float:
        stripped = head.lstrip()
        if not stripped.startswith("{"):
            return 0.0
        # ``version`` + ``runs`` is the SARIF signature; either alone is not
        # enough (Infer's native report.json also has an array of findings).
        has_version = '"version"' in head and "2.1.0" in head
        has_runs = '"runs"' in head
        if has_version and has_runs:
            return 1.0
        if has_runs and '"sarif"' in head.lower():
            return 0.95
        if path.name.lower().endswith(".sarif"):
            return 0.9
        return 0.0

    # -- adapter interface -------------------------------------------------

    def load(self, path: Path, text: str) -> ReportModel:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ReportParseError(f"{Path(path).name}: not valid JSON: {exc}") from exc
        if not isinstance(data, dict) or "runs" not in data:
            raise ReportParseError(
                f"{Path(path).name}: not a SARIF document (no 'runs' array)"
            )
        version = str(data.get("version", ""))
        if version and not version.startswith("2."):
            logger.warning(
                "%s declares SARIF version %s; this adapter implements 2.1.0 — "
                "parsing best-effort", Path(path).name, version,
            )

        findings: list[BugFinding] = []
        seen_ids: dict[str, int] = {}
        analyzer = AnalyzerKind.UNKNOWN
        tool_version = ""
        hint: str | None = None
        run_meta: list[dict] = []

        for run_index, run in enumerate(data.get("runs") or []):
            driver = (run.get("tool") or {}).get("driver") or {}
            if analyzer is AnalyzerKind.UNKNOWN:
                analyzer = self._analyzer_for(str(driver.get("name", "")))
                tool_version = str(driver.get("version", "") or "")
            if hint is None:
                hint = self._source_root_hint(run, path)
            run_meta.append({
                "run_index": run_index,
                "driver": driver.get("name", ""),
                "driver_version": driver.get("version", ""),
                "results": len(run.get("results") or []),
                "properties": run.get("properties") or {},
            })
            for result in run.get("results") or []:
                finding = self._finding_from(
                    result, run, driver, analyzer, path, seen_ids,
                )
                if finding is not None:
                    findings.append(finding)

        if not findings:
            raise ReportParseError(
                f"{Path(path).name}: SARIF document with no results"
            )
        return ReportModel(
            path=path,
            format=ReportFormat.CODEQL_SARIF,
            analyzer=analyzer,
            findings=findings,
            source_root_hint=hint,
            tool_version=tool_version,
            metadata={"sarif_version": version, "runs": run_meta},
        )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _analyzer_for(driver_name: str) -> AnalyzerKind:
        lowered = (driver_name or "").strip().lower()
        for prefix, kind in _DRIVER_NAMES:
            if lowered.startswith(prefix):
                return kind
        return AnalyzerKind.UNKNOWN

    @staticmethod
    def _source_root_hint(run: dict, path: Path) -> str | None:
        """The root this run's relative URIs resolve against, when it says so.

        ``originalUriBaseIds`` is where a producer records the absolute root a
        ``uriBaseId`` stands for (CodeQL leaves it empty and writes URIs
        relative to the *database's* source root, which is the tree the user
        passes as ``--source-root``).  A ``file://`` URI is unwrapped to a path;
        anything else is returned as-is and the code model decides whether it
        exists.
        """
        bases = run.get("originalUriBaseIds") or {}
        for key in ("%SRCROOT%", "SRCROOT", "%ROOT%"):
            entry = bases.get(key)
            if isinstance(entry, dict) and entry.get("uri"):
                return _uri_to_path(str(entry["uri"])).rstrip("/") or None
        return None

    def _finding_from(
        self,
        result: dict,
        run: dict,
        driver: dict,
        analyzer: AnalyzerKind,
        path: Path,
        seen_ids: dict[str, int],
    ) -> BugFinding | None:
        rules = driver.get("rules") or []
        rule_ref = result.get("rule") or {}
        rule_id = str(result.get("ruleId") or rule_ref.get("id") or "").strip()
        rule: dict = {}
        index = result.get("ruleIndex", rule_ref.get("index"))
        if isinstance(index, int) and 0 <= index < len(rules):
            rule = rules[index]
        elif rule_id:
            rule = next(
                (r for r in rules if str(r.get("id", "")) == rule_id), {},
            )

        locations = result.get("locations") or []
        if not locations:
            logger.debug("SARIF result without a location dropped: %s", rule_id)
            return None
        finding_location = _location_from(
            (locations[0] or {}).get("physicalLocation") or {}, run,
        )

        message = _coerce_text(result.get("message"))
        if not message:
            message = _coerce_text(rule.get("shortDescription")) or rule_id

        # Path: the longest codeFlow is the primary one (a query that tracks
        # several sources emits one flow per source and the pipeline analyses
        # one path at a time).  The count is kept so nothing is silently lost.
        flows = result.get("codeFlows") or []
        best_flow = max(
            flows,
            key=lambda f: sum(len(t.get("locations") or [])
                              for t in (f.get("threadFlows") or [])),
            default=None,
        )
        steps = (
            _steps_from_code_flow(best_flow, finding_location, run)
            if best_flow else
            [TraceStep(index=1, kind=StepKind.REPORT,
                       location=finding_location, message=message,
                       raw={"synthesized": "report"})]
        )

        related = [
            _location_from((rl or {}).get("physicalLocation") or {}, run)
            for rl in (result.get("relatedLocations") or [])
        ]

        fingerprint = ""
        prints = result.get("partialFingerprints") or {}
        if prints:
            fingerprint = str(
                prints.get("primaryLocationLineHash")
                or next(iter(sorted(prints.values())), "")
            )

        short_file = Path(finding_location.file).name or "?"
        base_id = f"{rule_id or 'finding'}@{short_file}:{finding_location.line}"
        seen_ids[base_id] = seen_ids.get(base_id, 0) + 1
        finding_id = base_id if seen_ids[base_id] == 1 else f"{base_id}#{seen_ids[base_id]}"

        return BugFinding(
            finding_id=finding_id,
            analyzer=analyzer,
            rule_id=rule_id or _coerce_text(rule.get("shortDescription")),
            rule_name=_coerce_text(rule.get("shortDescription")) or rule_id,
            category=_category_from_rule(rule),
            severity=_severity_from_rule(rule, result),
            message=message,
            location=finding_location,
            function="",  # SARIF states no enclosing function; the code model derives it
            steps=steps,
            # A codeFlow is the analyzer walking a path; its absence is the
            # analyzer stating a location and a claim.  127 of the 176 results
            # in the real corpus are the latter, and a flow that contributes
            # nothing beyond the report step is the same statement — it tells
            # us *where*, not *how*.  Everything below reads this instead of
            # counting steps, because a location-only finding still has one.
            path_evidence=(
                PathEvidence.TRACE if best_flow and len(steps) > 1
                else PathEvidence.LOCATION_ONLY
            ),
            related=related,
            fingerprint=fingerprint,
            raw={
                "rule_tags": (rule.get("properties") or {}).get("tags") or [],
                "code_flow_count": len(flows),
                "thread_flow_count": sum(
                    len(f.get("threadFlows") or []) for f in flows
                ),
                "suppressions": result.get("suppressions") or [],
                "baseline_state": result.get("baselineState"),
                "help_uri": rule.get("helpUri", ""),
                "rule_id_pretty": _RULE_ID_PRETTY_RE.sub(
                    " ", rule_id
                ).strip(),
            },
        )


def parse_sarif(path: Path, text: str | None = None) -> ReportModel:
    """Convenience: parse a SARIF file without going through detection."""
    if text is None:
        if not path.exists():
            raise ReportParseError(f"SARIF report not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
    return SarifAdapter().load(path, text)


__all__ = ["SarifAdapter", "parse_sarif"]
