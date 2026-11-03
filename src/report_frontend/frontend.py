"""The frontend facade: one entry point that turns a report into a model.

Everything above this line is a format detail; everything below it reads the
canonical model.  The facade adds the two things a *pipeline* needs that a
parser does not have:

* :class:`ReportRef` — an address for **one finding**, not one file.  A CSA
  report is a file, a SARIF report is 62 findings in a file, and the pipeline's
  unit of work is a finding (it writes one result JSON, generates one POC, and
  names both after the report).  A ref is ``<path>`` or ``<path>::<selector>``,
  and because it is a :class:`os.PathLike` the CLI can keep passing it around
  where it used to pass a :class:`~pathlib.Path`.
* :class:`LoadedReport` — the finding *plus* the code model that resolved every
  file it names, built with the same source root the pipeline will analyse
  against.  That pairing is the point: a report analysed against a tree it does
  not belong to is the failure mode this repository has hit hardest, and it is
  detectable here, early, before any LLM call.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

from .adapters import load_adapters
from .code_model import SourceCodeModel
from .errors import ReportNotFoundError, ReportParseError, UnsupportedReportFormat
from .report_model import (
    FRONTEND_MODEL_VERSION,
    AnalyzerKind,
    BugFinding,
    ReportModel,
    ReportFormat,
)
from .registry import detect_adapter, finding_for

logger = logging.getLogger(__name__)

#: ``<path>::<selector>``.  ``::`` cannot appear in a POSIX filename component
#: and does not appear in any of the report formats' own path spellings.
_SELECTOR_SEP = "::"

#: Characters kept when a selector becomes part of an output filename.
_UNSAFE_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_KEY_LEN = 80


@dataclass(frozen=True)
class ReportRef:
    """An address for one finding inside one report document.

    Behaves like the :class:`~pathlib.Path` it wraps for everything the CLI
    does with it (``str()``, ``Path(ref)``, ``ref.name``, ``ref.stem``,
    ``ref.parent``, ``ref.exists()``) so the integration is a substitution, not
    a rewrite — and adds the two things a path cannot express: which finding is
    meant, and a filesystem-safe ``key`` to name the results after.
    """

    path: Path
    selector: str = ""

    # -- construction ------------------------------------------------------

    @classmethod
    def parse(cls, spec: "str | os.PathLike | ReportRef") -> "ReportRef":
        """Parse ``<path>`` or ``<path>::<selector>``.

        The selector is split off the *right*: a path may legitimately contain
        ``::`` on some filesystems, a selector may not contain one at all.
        """
        if isinstance(spec, ReportRef):
            return spec
        text = os.fspath(spec)
        if _SELECTOR_SEP in text:
            path_text, _, selector = text.rpartition(_SELECTOR_SEP)
            return cls(Path(path_text), selector.strip())
        return cls(Path(text), "")

    # -- path-likeness -----------------------------------------------------

    def __fspath__(self) -> str:
        return str(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return (f"ReportRef({str(self.path)!r}"
                + (f", selector={self.selector!r}" if self.selector else "") + ")")

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def suffix(self) -> str:
        return self.path.suffix

    @property
    def parent(self) -> Path:
        return self.path.parent

    def exists(self) -> bool:
        return self.path.exists()

    # -- identity ----------------------------------------------------------

    @property
    def key(self) -> str:
        """Filesystem-safe name for this ref's outputs (result JSON, POC).

        A report *file* keeps its own stem, so every existing output name is
        unchanged; a finding inside a multi-finding document gets the selector
        appended, because otherwise all 62 results of a SARIF run would write
        to one file and each would overwrite the last.
        """
        if not self.selector:
            return self.path.stem
        safe = _UNSAFE_KEY_RE.sub("_", self.selector).strip("_")
        if len(safe) > _MAX_KEY_LEN:
            safe = safe[:_MAX_KEY_LEN].rstrip("_")
        return f"{self.path.stem}__{safe}" if safe else self.path.stem

    def with_selector(self, selector: str) -> "ReportRef":
        return ReportRef(self.path, selector)


@dataclass
class LoadedReport:
    """A finding, the document it came from, and the code it points at."""

    ref: ReportRef
    document: ReportDocument
    finding: BugFinding
    code: SourceCodeModel
    #: Every finding in the document, so a caller can report that it analysed
    #: one of sixty-two without re-parsing.
    sibling_findings: int = 1
    warnings: list[str] = field(default_factory=list)

    @property
    def analyzer(self) -> AnalyzerKind:
        return self.finding.analyzer

    @property
    def format(self) -> ReportFormat:
        return self.document.format

    @property
    def model_version(self) -> str:
        return FRONTEND_MODEL_VERSION

    def unresolved_files(self) -> list[str]:
        return [r.reported for r in self.code.unresolved(self.finding)]

    def summary(self) -> dict:
        return {
            "model_version": FRONTEND_MODEL_VERSION,
            "report": str(self.ref.path),
            "selector": self.ref.selector,
            "format": self.document.format.value,
            "analyzer": self.finding.analyzer.value,
            "tool_version": self.document.tool_version,
            "rule_id": self.finding.rule_id,
            "severity": self.finding.severity.value,
            "finding": self.finding.location.describe(),
            "steps": self.finding.path_length,
            "sibling_findings": self.sibling_findings,
            "code": self.code.summary(self.finding),
        }


class ReportFrontend:
    """Loads any supported report into the canonical model + code model."""

    def __init__(
        self,
        source_root: "str | Path | None" = None,
        report_format: str | None = None,
    ) -> None:
        load_adapters()
        self.source_root = Path(source_root).resolve() if source_root else None
        self.report_format = report_format

    # -- single report -----------------------------------------------------

    def load_document(self, ref: "ReportRef | str | Path") -> ReportDocument:
        """Parse the whole document (every finding) — no source tree needed."""
        ref = ReportRef.parse(ref)
        if not ref.path.exists():
            raise ReportNotFoundError(f"Report not found: {ref.path}")
        return self._load_document(ref)

    def load(self, ref: "ReportRef | str | Path") -> LoadedReport:
        """Parse, select the finding, and resolve the code it points at."""
        ref = ReportRef.parse(ref)
        document = self.load_document(ref)
        finding = finding_for(document, ref.selector)

        model = SourceCodeModel(
            source_root=self.source_root,
            uri_bases=self._uri_bases(document),
        )
        # A CSA report carries its own copy of the flagged region.  Registering
        # it means snippets come from the report (line alignment stays
        # meaningful) while resolution still finds the local file for the
        # PDG/CFG side.
        self._register_embedded(document, model)
        # The report is an index, the model is its reader: read the code the
        # index points at *now*, so no stage below has to decide for itself
        # which file to open.  (See ``SourceCodeModel.materialize`` — it is the
        # per-report model on purpose, never ``shared_model``.)
        model.materialize(finding)

        warnings: list[str] = []
        unreadable = [
            f["reported"] for f in (model.code_set or {}).get("files", [])
            if f["source"] == "unreadable"
        ]
        if unreadable:
            warnings.append(
                f"{len(unreadable)} file(s) the report names resolved to a local "
                f"path that cannot be read: " + ", ".join(unreadable[:3])
            )
        unresolved = model.unresolved(finding)
        if unresolved:
            warnings.append(
                f"{len(unresolved)} file(s) the report names do not exist under "
                f"{self.source_root or '<unset source root>'}: "
                + ", ".join(r.reported for r in unresolved[:3])
            )
        return LoadedReport(
            ref=ref,
            document=document,
            finding=finding,
            code=model,
            sibling_findings=len(document.findings),
            warnings=warnings,
        )

    # -- batch -------------------------------------------------------------

    def expand(self, refs: Sequence["ReportRef | str | Path"]) -> list[ReportRef]:
        """One ref per finding: files unchanged, multi-finding documents split.

        A CSA HTML file expands to itself.  A SARIF file expands to
        ``<path>::<finding id>`` per result, which is what makes ``--scan-dir``
        over a results directory do the right thing instead of analysing
        whichever finding the parser happened to return first.
        """
        out: list[ReportRef] = []
        for raw in refs:
            ref = ReportRef.parse(raw)
            if ref.selector:
                out.append(ref)
                continue
            try:
                document = self._load_document(ref)
            except (ReportParseError, UnsupportedReportFormat, OSError) as exc:
                logger.error("cannot expand %s: %s", ref.path, exc)
                out.append(ref)  # let the pipeline report the real error
                continue
            if len(document.findings) == 1:
                out.append(ref)
            else:
                logger.info(
                    "%s holds %d findings — expanding into %d work items",
                    ref.path.name, len(document.findings), len(document.findings),
                )
                out.extend(
                    ref.with_selector(f.finding_id) for f in document.findings
                )
        return out

    def iter_findings(
        self, ref: "ReportRef | str | Path"
    ) -> Iterator[tuple[str, BugFinding]]:
        """``(selector, finding)`` for every finding — the ``--list-findings`` view."""
        document = self.load_document(ref)
        for finding in document.findings:
            yield finding.finding_id, finding

    def detect_format(self, path: "str | Path") -> ReportFormat:
        return detect_adapter(Path(path), override=self.report_format).format

    # -- internals ---------------------------------------------------------

    def _load_document(self, ref: ReportRef) -> ReportDocument:
        path = ref.path
        adapter = detect_adapter(path, override=self.report_format)
        logger.debug("parsing %s as %s", path.name, adapter.name)
        return self.parse_with(adapter.name, path)

    def parse_with(self, format_name: str, path: Path) -> ReportDocument:
        """Parse *path* with a named adapter (used by ``load_document``/tests)."""
        from .registry import get_adapter

        adapter = get_adapter(format_name)
        text = path.read_text(encoding="utf-8", errors="replace")
        document = adapter.load(path, text)
        document.path = path
        if not document.findings:
            raise ReportParseError(f"{path.name}: no findings")
        return document

    @staticmethod
    def _uri_bases(document: ReportDocument) -> list[str]:
        """Extra roots a document names for its relative paths (SARIF)."""
        bases: list[str] = []
        hint = document.source_root_hint
        if hint:
            bases.append(hint)
        for run in document.metadata.get("runs", []) or []:
            base = run.get("original_uri_base_id") or run.get("uri_base")
            if base:
                bases.append(str(base))
        return bases

    @staticmethod
    def _register_embedded(document: ReportDocument, model: SourceCodeModel) -> None:
        """Hand every file's embedded source in the document to the code model.

        Two spellings of the same idea, and the order is the rule:

        * ``embedded_files`` — every *listing section* of the document, keyed by
          the path that section's heading spells.  Each line is attributed to
          its own section, so this is the copy that says what line 641 of *this
          file* is.
        * ``finding.raw["source_snippets"]`` — the bug file's window, filtered
          by **line number alone over the whole document**.  It is what CSA
          reports have always been read through, and it also collects lines of
          that number from the report's other file sections — or, where the bug
          file has a blank line, plain fills it with a neighbouring file's code.

        Registering both under one key merged them, and a merge can only
        *overwrite* a wrong line: where the bug file has a blank, the foreign
        line stayed.  So a file the document lists gets its section copy and
        *only* that — ``add_embedded`` is not called twice for one path in the
        common case, and where it is (a file with a listing that nothing else
        names) the section lines simply fill in around the window.

        For the files the document does **not** list, the window is the only
        copy the report carries, and registering it under the reported path is
        what lets ``snippet()`` serve the analyzer's own revision rather than
        the local checkout — otherwise the "embedded copy wins" rule silently
        did not apply, which is a change of revision, invisible in the text.
        """
        for finding in document.findings:
            for reported, lines in (finding.raw.get("embedded_files") or {}).items():
                model.add_embedded(reported, lines)
        for reported, lines in (document.metadata.get("embedded_files") or {}).items():
            model.add_embedded(reported, lines)
        listed = set(model.embedded_files)
        for finding in document.findings:
            reported = finding.location.file
            if reported and reported not in listed:
                model.add_embedded(
                    reported, finding.raw.get("source_snippets") or {},
                )


def load_report(
    ref: "ReportRef | str | Path",
    source_root: "str | Path | None" = None,
    report_format: str | None = None,
) -> LoadedReport:
    """Module-level convenience wrapper."""
    return ReportFrontend(
        source_root=source_root, report_format=report_format,
    ).load(ref)


__all__ = [
    "LoadedReport",
    "ReportFrontend",
    "ReportRef",
    "load_report",
]
