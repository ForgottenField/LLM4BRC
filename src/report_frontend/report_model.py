"""Canonical model of a static-analysis bug report and of the code it names.

This module is the *interchange format* of the report frontend.  Every adapter
(CSA HTML, SARIF/CodeQL, and the ones still to come) normalises its own document
shape into these dataclasses, and every consumer — the path-selection pipeline,
the dependency builder, the eval tooling — reads them instead of a format.

Why a canonical model at all
----------------------------
The pipeline used to read CSA HTML directly: ``ParsedReport.path_events`` is a
list of *dicts* whose keys (``path_number``/``event_kind``/``description``/…)
carry CSA's wording, and a dozen downstream stages decode that wording with
regexes (``"Calling 'X'"``, ``"Assuming the condition is true"``,
``"error start"``).  That is a fine internal representation for one analyzer and
a dead end for two: a CodeQL finding has no "← Taking false branch →" text to
parse, and inventing one would be fiction.

So the model separates three things that CSA's HTML conflates:

* **what the analyzer asserted** — a rule, a location, a message (``BugFinding``);
* **the ordered steps it walked** — typed (``TraceStep.kind``) *and* carrying the
  analyzer's verbatim text (``TraceStep.message``), so a consumer that needs
  semantics reads the kind and one that needs evidence reads the text;
* **the code the report points at** — every file/line it names, resolved against
  a local source tree (``code_model.SourceCodeModel``), which is what lets a
  snippet be read from disk when the report does not embed source (SARIF).

Nothing in here is analyzer-specific.  ``AnalyzerKind``/``rule_id`` say who
produced the finding; the wording of a step lives in ``message`` and is never
parsed by this layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

#: Bumped whenever the *shape* of the canonical model changes incompatibly.
#: Serialised bundles carry it so a consumer can refuse an artifact built by a
#: different revision, the same way ``pdg_models.PDG_ARTIFACT_VERSION`` guards
#: the SDG artifacts.  A stale model that parses fine and says nothing about
#: itself is how a whole evaluation once ran on a 42%-incomplete graph.
FRONTEND_MODEL_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class AnalyzerKind(str, Enum):
    """Which static analyzer produced a report."""

    CSA = "csa"
    CPPCHECK = "cppcheck"
    INFER = "infer"
    CODEQL = "codeql"
    JOERN = "joern"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: str | AnalyzerKind | None) -> AnalyzerKind:
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        try:
            # if input value is "Csa", "CSA", or "  csa  ", convert it to "csa"
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.UNKNOWN


class ReportFormat(str, Enum):
    """On-disk shape of the report document (the *container*, not the tool)."""

    CSA_HTML = "csa-html"
    CODEQL_SARIF = "codeql_sarif"
    INFER_JSON = "infer-json"
    CPPCHECK_XML = "cppcheck-xml"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Normalised severity.  ``UNKNOWN`` is the conservative default."""

    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: str | Severity | None) -> Severity:
        """Map an analyzer's own vocabulary onto the four levels.

        CSA's categories, SARIF's ``level`` (``error``/``warning``/``note``/
        ``none``), CodeQL's ``problem.severity`` and Infer's ``bug_type``
        qualifiers all land here.  Anything unrecognised becomes ``UNKNOWN``.
        """
        if isinstance(value, Severity):
            return value
        if value is None:
            return cls.UNKNOWN

        severity = str(value).strip().lower()
        if severity in ("error", "high", "critical", "severe", "fatal"):
            return cls.ERROR
        if severity in ("warning", "medium", "moderate"):
            return cls.WARNING
        if severity in ("note", "info", "low", "none", "remark"):
            return cls.NOTE

        # CSA emits free-text categories ("Memory error", "Logic error",
        # "Security", "Dead store"); "error" as a substring is the only signal
        # they carry.  Order matters: check "error" before nothing matches.
        if "error" in severity:
            return cls.ERROR
        if "warning" in severity:
            return cls.WARNING
        return cls.UNKNOWN


class StepKind(str, Enum):
    """What one step of a reported path *is*.

    The vocabulary is the union of what the analyzers express, deliberately
    small and deliberately not CSA's: ``ASSUME`` is any constraint the analyzer
    applied at that point (CSA's ``Assuming …``), ``CONTROL`` any branch
    decision, ``CALL``/``RETURN`` a procedure boundary.  ``EVENT`` is the
    catch-all and is always safe — a consumer that does not recognise a kind
    must fall back to reading ``message``.
    """

    ERROR_START = "error_start"   # where the flagged value/state originates
    EVENT = "event"               # plain data-flow / observation step
    CONTROL = "control"           # branch decision (see direction)
    ASSUME = "assume"             # constraint the analyzer applied
    CALL = "call"                 # entering a callee
    RETURN = "return"             # leaving a callee
    NOTE = "note"                 # analyzer commentary, not part of the flow
    REPORT = "report"             # the finding itself (last step)

    @property
    def is_boundary(self) -> bool:
        """True for the kinds that segment a path (see ``cfg_feasibility``)."""
        return self in (StepKind.CONTROL, StepKind.CALL, StepKind.RETURN, StepKind.REPORT)


#: Step kinds that assert a constraint the report relies on.  CSA's path
#: wording for these is ``Assuming …``; stages 4 and 6b scan exactly this set.
CONSTRAINT_STEP_KINDS: frozenset[StepKind] = frozenset(
    {StepKind.ASSUME, StepKind.CONTROL}
)


class PathEvidence(str, Enum):
    """How much of a *path* the analyzer actually gave us.

    This is not a property of the finding's severity or of how many steps
    happen to survive parsing — it is what the analyzer stated, and the two
    analyzers state it differently:

    * CSA walks a symbolic path and prints every step of it;
    * a CodeQL query may report only *where*, with no ``codeFlow`` at all, which is a location and a
      claim, not a path.

    The **adapter** sets it, and the adapter is allowed to look at the whole
    document while doing so (that is where the analyzer's statement lives).
    Consumers may not re-derive it from ``len(steps)``: the finding that says
    ``TRACE`` and holds two steps is a different thing from the one that says
    ``LOCATION_ONLY`` and holds two, and only the adapter knows which happened.
    """

    TRACE = "trace"              # the analyzer supplied a step trace
    LOCATION_ONLY = "location"   # only the flagged location and the claim
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: str | PathEvidence | None) -> PathEvidence:
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.UNKNOWN
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            return cls.UNKNOWN


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SourceLocation:
    """A position in a source file, exactly as the report spells it.

    ``file`` is the *reported* path — an absolute build path for CSA
    (``/home/me/projects/cpp-project/faiss/faiss/impl/io.cpp``), a ``%SRCROOT%``
    -relative URI for CodeQL (``faiss/impl/io.cpp``).  It is never rewritten
    here: resolution to a local path is the code model's job, and keeping the
    raw string means a report can be re-serialised byte-faithfully.
    """

    file: str = ""
    line: int = 0
    column: int = 0
    end_line: int = 0
    end_column: int = 0

    @property
    def is_valid(self) -> bool:
        return bool(self.file) and self.line > 0

    def at_line(self, line: int) -> SourceLocation:
        return SourceLocation(
            file=self.file, line=line, column=0,
            end_line=line, end_column=0,
        )

    def describe(self) -> str:
        """``file:line:col`` — the form every log line and prompt wants."""
        out = self.file or "<unknown>"
        if self.line:
            out += f":{self.line}"
            if self.column:
                out += f":{self.column}"
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceLocation:
        return cls(
            file=str(data.get("file", "")),
            line=int(data.get("line", 0) or 0),
            column=int(data.get("column", 0) or 0),
            end_line=int(data.get("end_line", 0) or 0),
            end_column=int(data.get("end_column", 0) or 0),
        )


# ---------------------------------------------------------------------------
# CSA Report Path steps
# ---------------------------------------------------------------------------

@dataclass
class TraceStep:
    """One ordered step of a reported path.

    ``message`` is the analyzer's own text, verbatim: it is the evidence a
    human reads and the string the legacy bridge renders back into CSA's
    vocabulary (``description``).  The typed fields beside it are the *same*
    information in machine-readable form and are what new consumers should use.
    A step that carries a typed field but no message is a step this layer
    synthesised — the bridge renders one from the typed fields for it.
    """

    index: int = 0
    kind: StepKind = StepKind.EVENT
    location: SourceLocation = field(default_factory=SourceLocation)
    message: str = ""
    function: str = ""
    condition: str = ""
    #: For CONTROL: True = the branch was taken, False = not taken, None = the analyzer recorded a branch decision without a direction.
    direction: Optional[bool] = None
    #: For ASSUME: the constraint text up to the analyzer's arrow, e.g. ``"'ret' is equal to 1"``.
    assumption: str = ""
    callee: str = ""
    #: Analyzer-specific extras, preserved so an adapter's information is never
    #: silently dropped (e.g. a SARIF threadFlow location's nesting level).
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def file(self) -> str:
        return self.location.file

    @property
    def line(self) -> int:
        return self.location.line

    @property
    def is_error_start_step(self) -> bool:
        return self.kind is StepKind.ERROR_START

    @property
    def is_call_step(self) -> bool:
        return self.kind is StepKind.CALL

    @property
    def is_control_step(self) -> bool:
        return self.kind is StepKind.CONTROL

    @property
    def is_constraint(self) -> bool:
        """True when this step asserts a condition the report depends on."""
        return self.kind in CONSTRAINT_STEP_KINDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind.value,
            "location": self.location.to_dict(),
            "message": self.message,
            "function": self.function,
            "condition": self.condition,
            "direction": self.direction,
            "assumption": self.assumption,
            "callee": self.callee,
            "raw": dict(self.raw),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceStep":
        return cls(
            index=int(data.get("index", 0) or 0),
            kind=StepKind(data.get("kind", "event")),
            location=SourceLocation.from_dict(data.get("location", {}) or {}),
            message=str(data.get("message", "")),
            function=str(data.get("function", "")),
            condition=str(data.get("condition", "")),
            direction=data.get("direction"),
            assumption=str(data.get("assumption", "")),
            callee=str(data.get("callee", "")),
            raw=dict(data.get("raw", {}) or {}),
        )


# ---------------------------------------------------------------------------
# Bug finding related Info
# ---------------------------------------------------------------------------

@dataclass
class BugFinding:
    """One reported bug: what the analyzer claims, where, and how it got there.

    A finding is *self-contained*: two findings from the same SARIF document
    share nothing but the document they came from, so everything a consumer
    needs is here (this is what makes a multi-result SARIF file splittable into
    independent pipeline work items — see ``ReportRef``).
    """

    #: Stable identifier, unique within the source document.  Prefix it with the
    #: analyzer when it is synthesised from a location, so it stays readable and
    #: so the same finding keeps its id across re-parses.
    finding_id: str = ""
    analyzer: AnalyzerKind = AnalyzerKind.UNKNOWN
    #: The analyzer's rule identifier: CSA's bug type ("Memory leak"), CodeQL's
    #: rule id ("cpp/integer-multiplication-cast-to-long"), Infer's bug type.
    rule_id: str = ""
    #: Human-readable rule name, when the analyzer has one distinct from the id.
    rule_name: str = ""
    #: Coarse family, normalised where the analyzer's own taxonomy allows it
    #: ("Memory error", "Security", "Correctness", …).  Free text on purpose:
    #: the leak verdict layer keys on the rule id, not on this.
    category: str = ""
    severity: Severity = Severity.UNKNOWN
    #: The finding's headline message.
    message: str = ""
    #: Where the bug is — the location the pipeline calls ``bug_file``/``bug_line``.
    location: SourceLocation = field(default_factory=SourceLocation)
    #: Enclosing function at ``location``, when the report states it or the code
    #: model could derive it.
    function: str = ""
    #: Ordered path steps, ending at the REPORT step.  Empty is legal — a
    #: finding without a trace (most CodeQL results) is a location and a claim.
    steps: list[TraceStep] = field(default_factory=list)
    #: What the analyzer said about the path — see :class:`PathEvidence`.  The
    #: adapters set it; nothing downstream may infer it from ``len(steps)``.
    path_evidence: PathEvidence = PathEvidence.LOCATION_ONLY
    reported_path_length: int = 0
    #: Secondary locations the analyzer attached ("This catch block", …).
    related: list[SourceLocation] = field(default_factory=list)
    #: The analyzer's stable fingerprint, when it emits one (SARIF
    #: ``partialFingerprints``).  Used for de-duplication and result naming.
    fingerprint: str = ""
    #: Everything analyzer-specific this model does not name (SARIF
    #: ``properties``, CSA metadata comments, …).
    raw: dict[str, Any] = field(default_factory=dict)

    # -- derived views -----------------------------------------------------

    @property
    def path_length(self) -> int:
        return len(self.steps)

    @property
    def has_path(self) -> bool:
        """True only when the analyzer reported an actual step trace.

        A location-only finding still has the report step, so this is *not*
        ``path_length > 1`` — the two answers coincide on today's corpus, but
        they answer different questions ("did it walk a path?" vs "how many
        steps are in the list?"), and the free ones are the ones that drift.
        """
        return self.path_evidence is PathEvidence.TRACE

    @property
    def error_start(self) -> Optional[StepKind]:
        """The step where the flagged value/state originates, if reported."""
        for step in self.steps:
            if step.kind is StepKind.ERROR_START:
                return step
        return None

    @property
    def calls(self) -> list[TraceStep]:
        """Every ``Calling …`` step, in path order."""
        return [s for s in self.steps if s.kind is TraceStep.CALL]

    @property
    def call_chain(self) -> list[str]:
        """Callee names in call order (``error_start``-independent)."""
        return [s.callee for s in self.calls if s.callee]

    def steps_of(self, *kinds: StepKind) -> list[StepKind]:
        wanted = set(kinds)
        return [s for s in self.steps if s.kind in wanted]

    def files(self) -> list[str]:
        """Every distinct reported file this finding mentions, in path order.

        The first entry is the bug file; the rest are the files the path
        crosses.  This is the seed set for the dependency builder, and it is
        deliberately *reported* strings — resolving them is the code model's job.
        """
        seen: list[str] = []
        for candidate in [self.location.file] + [s.file for s in self.steps]:
            if candidate and candidate not in seen:
                seen.append(candidate)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "analyzer": self.analyzer.value,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "category": self.category,
            "severity": self.severity.value,
            "message": self.message,
            "location": self.location.to_dict(),
            "function": self.function,
            "steps": [s.to_dict() for s in self.steps],
            "path_evidence": self.path_evidence.value,
            "reported_path_length": self.reported_path_length,
            "related": [r.to_dict() for r in self.related],
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BugFinding:
        return cls(
            finding_id=str(data.get("finding_id", "")),
            analyzer=AnalyzerKind.coerce(data.get("analyzer")),
            rule_id=str(data.get("rule_id", "")),
            rule_name=str(data.get("rule_name", "")),
            category=str(data.get("category", "")),
            severity=Severity.coerce(data.get("severity")),
            message=str(data.get("message", "")),
            location=SourceLocation.from_dict(data.get("location", {}) or {}),
            function=str(data.get("function", "")),
            steps=[TraceStep.from_dict(s) for s in data.get("steps", []) or []],
            path_evidence=PathEvidence.coerce(data.get("path_evidence")),
            reported_path_length=int(data.get("reported_path_length", 0) or 0),
            related=[
                SourceLocation.from_dict(r) for r in data.get("related", []) or []
            ],
            fingerprint=str(data.get("fingerprint", "")),
        )


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------

@dataclass
class ReportModel:
    """A parsed report *file* — the container of one or more findings.

    A CSA HTML file holds exactly one finding; a SARIF file holds every result
    of one analysis run.  Keeping both under one type is what lets the CLI
    treat "a report" uniformly: ``--report <file>`` for CSA means the document,
    and for SARIF the document is expanded into one work item per finding.
    """

    path: Path
    format: ReportFormat = ReportFormat.UNKNOWN
    analyzer: AnalyzerKind = AnalyzerKind.UNKNOWN
    findings: list[BugFinding] = field(default_factory=list)
    #: The root the report's relative paths are anchored at, when the document
    #: states one (SARIF ``uriBaseId``/``originalUriBaseIds``, a clang command
    #: line in a CSA report's header).  A *hint*: the code model still tries it
    #: against the real tree before trusting it, and ``--source-root`` wins.
    source_root_hint: Optional[str] = None
    #: Analyzer version string, when the document records one.
    tool_version: str = ""
    #: Document-level extras (SARIF ``run.properties``, CSA report header).
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_multi_finding(self) -> bool:
        return len(self.findings) > 1

    def finding(self, selector: str) -> Optional[BugFinding]:
        """Look a finding up by id, fingerprint, or ``file:line`` shorthand.

        ``selector`` is what a user types after ``::`` on ``--report``.  The
        fallbacks exist because a fingerprint is unguessable and the location
        is what a human actually has in front of them.
        """
        if not selector:
            return self.findings[0] if len(self.findings) == 1 else None
        want = selector.strip()
        for f in self.findings:
            if f.finding_id == want:
                return f
        for f in self.findings:
            if f.fingerprint and f.fingerprint == want:
                return f
        # ``file:line`` / ``line`` / ``rule-id@file:line`` shorthands.
        for f in self.findings:
            loc = f.location
            if want in (str(loc.line), f"{loc.file}:{loc.line}",
                        f"{Path(loc.file).name}:{loc.line}"):
                return f
            if want == f"{f.rule_id}@{Path(loc.file).name}:{loc.line}":
                return f
        # Last resort: a unique rule-id match.
        by_rule = [f for f in self.findings if f.rule_id == want]
        if len(by_rule) == 1:
            return by_rule[0]
        return None

    def iter_findings(self) -> Iterator[BugFinding]:
        return iter(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": FRONTEND_MODEL_VERSION,
            "path": str(self.path),
            "format": self.format.value,
            "analyzer": self.analyzer.value,
            "tool_version": self.tool_version,
            "source_root_hint": self.source_root_hint,
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReportModel:
        return cls(
            path=Path(data.get("path", "")),
            format=ReportFormat(data.get("format", "unknown")),
            analyzer=AnalyzerKind.coerce(data.get("analyzer")),
            findings=[BugFinding.from_dict(f) for f in data.get("findings", []) or []],
            source_root_hint=data.get("source_root_hint"),
            tool_version=str(data.get("tool_version", "")),
            metadata=dict(data.get("metadata", {}) or {}),
        )
