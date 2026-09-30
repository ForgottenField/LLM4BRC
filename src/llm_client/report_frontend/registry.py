"""Adapter interface + registry + format detection.

Adding an analyzer is meant to be a one-file change: write a
:class:`ReportAdapter` subclass, decorate it with :func:`register`, and import
the module from ``adapters/__init__.py``.  Nothing else in the pipeline knows
the format exists — the frontend looks up an adapter by *sniffing the document*
and every consumer reads the canonical model.

Ambiguity is resolved loudly, never silently
--------------------------------------------
Detection returns a ranked list, and :func:`select_adapter` raises
:class:`UnsupportedReportFormat` when the best score is not strictly better
than the runner-up.  Two adapters claiming one document is a real possibility
(an analyzer that emits SARIF *and* whose native JSON also has a ``runs`` key);
picking one by registration order would make the choice depend on import order,
which is exactly the class of silent wrong-answer this repository keeps
getting bitten by.  ``--report-format`` overrides detection entirely.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from .errors import ReportFrontendError, UnsupportedReportFormat
from .models import AnalyzerKind, BugFinding, ReportDocument, ReportFormat

logger = logging.getLogger(__name__)

#: How many bytes of a document a sniffer sees.  Enough for a JSON header, an
#: XML declaration and the CSA comment block (which sits at the very top of the
#: file); a sniffer that needs more than this should key on the suffix instead.
SNIFF_BYTES = 8192


class ReportAdapter(ABC):
    """Turns one report *document* into the canonical model.

    Implementations must be stateless and cheap to construct: the registry
    holds one instance per format for the process lifetime and calls
    :meth:`load` once per document.
    """

    #: Registry key, also the value ``--report-format`` accepts.
    name: str = ""
    format: ReportFormat = ReportFormat.UNKNOWN
    #: The analyzer this adapter is written for.  An adapter that can serve
    #: several analyzers (SARIF: CodeQL, Infer, Cppcheck) declares the one it
    #: will *label* results with and refines that per document in :meth:`load`.
    analyzer: AnalyzerKind = AnalyzerKind.UNKNOWN
    #: File suffixes worth trying, lowercase including the dot.
    suffixes: tuple[str, ...] = ()

    # -- detection ---------------------------------------------------------

    def sniff(self, path: Path, head: str) -> float:
        """Confidence in ``[0, 1]`` that *head* (the first SNIFF_BYTES of the
        document, decoded leniently) is this adapter's format.

        The default scores on the suffix alone, which is right for formats with
        a canonical extension and wrong for the ones that share one (``.json``
        is Infer, SARIF and Cppcheck alike).  Override for those.
        """
        if not self.suffixes:
            return 0.0
        return 0.9 if path.suffix.lower() in self.suffixes else 0.0

    # -- parsing -----------------------------------------------------------

    @abstractmethod
    def load(self, path: Path, text: str) -> ReportDocument:
        """Parse *text* (the whole document) into canonical findings.

        ``text`` is passed in rather than re-read so a document is decoded once
        and so tests can drive an adapter without touching the filesystem.

        Raises:
            ReportParseError: the document is the right format but malformed.
            MissingReportMetadata: it parsed but has no usable finding location.
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ADAPTERS: dict[str, ReportAdapter] = {}


def register(adapter_cls: type[ReportAdapter]) -> type[ReportAdapter]:
    """Class decorator: instantiate and register an adapter."""
    adapter = adapter_cls()
    if not adapter.name:
        raise ValueError(f"{adapter_cls.__name__} has no name")
    if adapter.name in _ADAPTERS:
        raise ValueError(f"duplicate report adapter name: {adapter.name!r}")
    _ADAPTERS[adapter.name] = adapter
    return adapter_cls


def registered_adapters() -> dict[str, ReportAdapter]:
    """The registry, keyed by ``--report-format`` value (a copy)."""
    return dict(_ADAPTERS)


def get_adapter(name: str) -> ReportAdapter:
    """Look an adapter up by name (``--report-format``)."""
    key = (name or "").strip().lower()
    if key not in _ADAPTERS:
        raise UnsupportedReportFormat(
            f"unknown report format {name!r}; known formats: "
            f"{', '.join(sorted(_ADAPTERS)) or '(none registered)'}"
        )
    return _ADAPTERS[key]


def read_head(path: Path, limit: int = SNIFF_BYTES) -> str:
    """First *limit* bytes of *path*, decoded leniently (never raises on bytes)."""
    with path.open("rb") as fh:
        return fh.read(limit).decode("utf-8", errors="replace")


def rank_adapters(path: Path, head: str | None = None) -> list[tuple[float, ReportAdapter]]:
    """Every adapter that recognises *path*, best score first."""
    if head is None:
        head = read_head(path)
    scored: list[tuple[float, ReportAdapter]] = []
    for adapter in _ADAPTERS.values():
        try:
            score = float(adapter.sniff(path, head))
        except Exception as exc:  # a sniffer must never take the run down
            logger.debug("sniffer %s failed on %s: %s", adapter.name, path, exc)
            continue
        if score > 0:
            scored.append((score, adapter))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def detect_adapter(path: Path, override: str | None = None) -> ReportAdapter:
    """Pick the adapter for *path*, or raise :class:`UnsupportedReportFormat`.

    *override* (``--report-format``) bypasses detection; it also disambiguates,
    which is the escape hatch for a document two sniffers both claim.
    """
    if override:
        return get_adapter(override)

    ranked = rank_adapters(path)
    if not ranked:
        raise UnsupportedReportFormat(
            f"cannot detect the format of {path.name}: no registered adapter "
            f"recognises it. Known formats: {', '.join(sorted(_ADAPTERS))}. "
            f"If this is a new analyzer, add an adapter under "
            f"report_frontend/adapters/; otherwise pass --report-format."
        )
    best_score, best = ranked[0]
    if len(ranked) > 1 and ranked[1][0] >= best_score:
        names = ", ".join(f"{a.name} ({s:.2f})" for s, a in ranked)
        raise UnsupportedReportFormat(
            f"ambiguous report format for {path.name}: {names}. "
            f"Pass --report-format <name> to choose."
        )
    logger.debug("report %s detected as %s (score %.2f)", path.name, best.name, best_score)
    return best


def load_document(
    path: Path, override: str | None = None, text: str | None = None
) -> ReportDocument:
    """Detect + parse.  The frontend's single entry point for one document."""
    adapter = detect_adapter(path, override=override)
    if text is None:
        text = path.read_text(encoding="utf-8", errors="replace")
    document = adapter.load(path, text)
    document.path = path
    if document.format is ReportFormat.UNKNOWN:
        document.format = adapter.format
    if document.analyzer is AnalyzerKind.UNKNOWN:
        document.analyzer = adapter.analyzer
    if not document.findings:
        raise UnsupportedReportFormat(
            f"{path.name} parsed as {adapter.name} but contains no findings"
        )
    return document


def finding_for(document: ReportDocument, selector: str = "") -> BugFinding:
    """Resolve a ``<path>::<selector>`` finding, raising with the real options.

    A missing selector on a multi-finding document is a *usage* error, not a
    silent "first one": for a 62-result SARIF file, analysing result 0 because
    the user forgot a selector is precisely the plausible-looking wrong run
    ``_resolve_paths`` already refuses to do for the wrong project.
    """
    if not selector:
        if len(document.findings) == 1:
            return document.findings[0]
        raise UnsupportedReportFormat(
            f"{Path(document.path).name} holds {len(document.findings)} findings; "
            f"select one with --report {document.path}::<id>, where <id> is a "
            f"finding id, a rule id, or file:line (see --list-findings)."
        )
    finding = document.finding(selector)
    if finding is None:
        options = ", ".join(f.finding_id for f in document.findings[:5])
        raise ReportFrontendError(
            f"no finding {selector!r} in {Path(document.path).name}"
            + (f"; first ids: {options}" if options else "")
        )
    return finding


__all__ = [
    "ReportAdapter",
    "SNIFF_BYTES",
    "detect_adapter",
    "finding_for",
    "get_adapter",
    "load_document",
    "rank_adapters",
    "read_head",
    "register",
    "registered_adapters",
]
