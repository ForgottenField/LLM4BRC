"""CSA HTML reports — compatibility shim over the report frontend.

This module used to *be* the report parser: the regexes that index a CSA
document, the doc-index that attributes an event to the file section it sits in,
the annotation pass, and the two public entry points (``FPReportParser.parse``
and ``parse_to_bug_path``).  That code now lives in
``llm_client.report_frontend.adapters.csa_html`` and is joined to the pipeline
by ``llm_client.fp_analysis.report_bridge``.

What remains here is the interface the rest of the repository imports.  It is a
shim on purpose rather than a rename: ``FPReportParser`` is referenced by the
dependency builder, several tools, and the whole test suite, and a formats
refactor is not a good moment to also churn every call site.  New code should
use ``llm_client.report_frontend`` directly.

``parse()``/``parse_to_bug_path()`` keep working for a report path, and *also*
accept a ``ReportRef`` (``<path>[::<finding>]``) so the CLI can hand the same
object to the CSA and SARIF paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

from csa_analysis import BugPath
from fp_analysis.models import ParsedReport
from fp_analysis.report_bridge import to_bug_path, to_parsed_report
from report_frontend.adapters.csa_html import CsaHtmlAdapter
from report_frontend.code_model import SourceCodeModel
from report_frontend.frontend import ReportFrontend, ReportRef
from report_frontend.models import ReportFormat

logger = logging.getLogger(__name__)

__all__ = [
    "FPReportParser",
    "resolve_report_source_file",
]


class FPReportParser:
    """Parse CSA HTML reports into ``ParsedReport``/``BugPath`` (legacy API).

    Thin wrapper: the parsing is ``CsaHtmlAdapter``, the rendering back into the
    legacy dataclasses is ``report_bridge``.  Constructing one keeps the old
    signature and the old behaviour for a bare path; the optional
    ``source_root`` is what the code model needs to resolve the report's paths
    (it is optional because a caller that only wants the embedded snippets —
    which is every pre-frontend caller — never had to supply one).
    """

    def __init__(self, source_root: str | Path | None = None) -> None:
        self.source_root = source_root
        self._frontend = ReportFrontend(source_root=source_root)
        self._adapter = CsaHtmlAdapter()

    # -- legacy parsing API ------------------------------------------------

    def parse(self, report: str | Path | ReportRef) -> ParsedReport:
        """Parse a CSA report into the pipeline's ``ParsedReport``.

        Raises:
            FileNotFoundError: the report does not exist.
            MissingReportMetadata/ReportParseError: required metadata absent.
        """
        return to_parsed_report(self.load(report))

    def parse_to_bug_path(self, report: str | Path | ReportRef) -> BugPath:
        """Parse a CSA report into a CSA-shaped ``BugPath``."""
        return to_bug_path(self.load(report))

    # -- new API -----------------------------------------------------------

    def load(self, report: str | Path | ReportRef):
        """Parse into the canonical model (``LoadedReport``).

        Delegates to ``ReportFrontend`` so there is exactly one embedded-
        registration rule: this shim used to build its own model, and it
        registered the bug file's ``source_snippets`` under the reported path
        while the frontend did not — two entry paths, two different models for
        the same report.
        """
        return ReportFrontend(
            source_root=self.source_root, report_format=ReportFormat.CSA_HTML,
        ).load(ReportRef.parse(report))

    # -- kept for callers that reached for the private helper --------------

    @staticmethod
    def _clean_html_text(text: str) -> str:
        """Strip tags/entities and the leading path index (legacy helper)."""
        return CsaHtmlAdapter.clean_html_text(text)


def resolve_report_source_file(
    bug_file: str,
    project_root: str | Path | None = None,
    file_name_hint: str | None = None,
) -> Path | None:
    """Resolve a report's source path to a local file (legacy helper).

    Delegates to :class:`~llm_client.report_frontend.code_model.SourceCodeModel`,
    whose ladder is the union of the strategies this function used to try
    (direct, known build prefix, basename) plus relative-path and path-suffix
    matching.  ``file_name_hint`` keeps its old meaning: try the hint's
    basename *before* the report's own basename.

    Prefer the code model directly — it records *which* strategy matched, which
    is the difference between "the report's file is here" and "a file of that
    name is somewhere in the tree".
    """
    model = SourceCodeModel(source_root=project_root)

    for candidate in (file_name_hint, bug_file):
        if not candidate:
            continue
        if file_name_hint and candidate == file_name_hint:
            candidate = Path(candidate).name
            if project_root:
                hits = model.resolve(candidate)
                if hits.local:
                    return hits.local
            continue
        resolved = model.resolve(str(candidate))
        if resolved.local:
            return resolved.local
    return None
