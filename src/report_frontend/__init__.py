"""Report frontend: one model, one interface, every static analyzer.

The pipeline used to read one report format (CSA HTML) with one parser whose
output doubled as the pipeline's internal representation.  Adding a second
analyzer meant either teaching every downstream stage a second vocabulary or
converting the second analyzer's output into the first one's HTML — which is
what was actually done for the CodeQL findings (see
``docs/report-frontend.md``).  Neither scales to Infer and Cppcheck.

This package is the seam.  It has three layers, each replaceable on its own:

``models``
    The canonical model of a *report*: :class:`BugFinding` (what was claimed),
    :class:`TraceStep` (how the analyzer got there), :class:`SourceLocation`
    (where).  Nothing analyzer-specific, nothing parsed from prose.
``adapters``
    One module per container format, each normalising into the canonical
    model: ``csa-html``, ``sarif`` (CodeQL / Infer / Cppcheck).  Adding another
    is a single file — see ``adapters/__init__.py``.
``code_model``
    The other half of the model: the *code* a report names, resolved against a
    local source tree with the resolution strategy recorded, and read from the
    report's embedded copy when it has one.

Usage::

    from report_frontend import ReportRef, ReportFrontend

    frontend = ReportFrontend(source_root="~/csa_reports/project/faiss")
    loaded = frontend.load("report-index_read.cpp-read_index-484-2.html")
    loaded.finding.rule_id          # 'Memory leak'
    loaded.finding.steps[-1].kind   # StepKind.REPORT
    loaded.unresolved_files()       # [] — every path the report names resolved

The pipeline consumes it through ``fp_analysis.report_bridge``, which renders
the canonical finding back into the ``ParsedReport``/``BugPath`` the stages
already speak; a SARIF document is addressed as ``<path>::<finding id>`` so one
file of 62 findings becomes 62 work items (see :class:`ReportRef`).
"""

from __future__ import annotations

from .code_model import (
    CodeReference,
    ResolvedFile,
    SourceCodeModel,
    clear_index_cache,
)
from .errors import (
    MissingReportMetadata,
    ReportFrontendError,
    ReportNotFoundError,
    ReportParseError,
    UnsupportedReportFormat,
)
from .frontend import LoadedReport, ReportFrontend, ReportRef, load_report
from .models import (
    FRONTEND_MODEL_VERSION,
    AnalyzerKind,
    BugFinding,
    ReportDocument,
    ReportFormat,
    Severity,
    SourceLocation,
    StepKind,
    TraceStep,
)
from .registry import (
    detect_adapter,
    get_adapter,
    registered_adapters,
    register,
)

__all__ = [
    "FRONTEND_MODEL_VERSION",
    "AnalyzerKind",
    "BugFinding",
    "CodeReference",
    "LoadedReport",
    "MissingReportMetadata",
    "ReportDocument",
    "ReportFormat",
    "ReportFrontend",
    "ReportFrontendError",
    "ReportNotFoundError",
    "ReportParseError",
    "ReportRef",
    "ResolvedFile",
    "Severity",
    "SourceCodeModel",
    "SourceLocation",
    "StepKind",
    "TraceStep",
    "UnsupportedReportFormat",
    "clear_index_cache",
    "detect_adapter",
    "get_adapter",
    "load_report",
    "register",
    "registered_adapters",
]
