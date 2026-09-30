"""Exception hierarchy of the report frontend.

One rule, mirroring ``llm_client.errors``: a failure raises *these*, never a
raw ``json.JSONDecodeError``/``re.error``/``IndexError`` from inside an
adapter.  The caller is ``run_path_selection``, whose contract is fail-fast on
a bad report *before* the first LLM call — a mis-formatted report must surface
as a named error with the fix in the message, in seconds, not as a per-report
ERROR row after tens of minutes.
"""

from __future__ import annotations


class ReportFrontendError(Exception):
    """Base class for every frontend failure."""


class ReportNotFoundError(ReportFrontendError, FileNotFoundError):
    """The report path does not exist (also a ``FileNotFoundError``, so the
    legacy callers that catch that keep working)."""


class UnsupportedReportFormat(ReportFrontendError):
    """No adapter recognises the document.

    Raised with the recognised-format list and the ``--report-format`` fix:
    an unrecognised report is nearly always a *new* analyzer's output, and the
    answer is to add an adapter, not to guess one.
    """


class ReportParseError(ReportFrontendError):
    """The document is the right format but cannot be parsed."""


class MissingReportMetadata(ReportParseError):
    """The document parsed, but lacks a field the pipeline cannot work without.

    Currently that is the finding location (file + line): every stage keys on
    ``(file, line)`` — segment resolution, PDG node lookup, snippet extraction —
    so a report without one is refused rather than analysed against line 0.
    """


__all__ = [
    "MissingReportMetadata",
    "ReportFrontendError",
    "ReportNotFoundError",
    "ReportParseError",
    "UnsupportedReportFormat",
]
