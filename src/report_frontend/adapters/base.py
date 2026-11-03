"""Adapter introspection helpers.

``--list-formats`` is the answer to "why did the frontend refuse my report?",
and it is worth its few lines: the failure mode it prevents is someone
re-running the same misinvocation, or worse, hand-converting a report into
another analyzer's HTML shape because the tool "only reads CSA" (which is
exactly what was done for the CodeQL findings before this package existed —
see ``docs/report-frontend.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..report_model import AnalyzerKind, ReportFormat
from ..registry import ReportAdapter, registered_adapters


@dataclass(frozen=True)
class AdapterInfo:
    """A registered adapter, as a table row."""

    name: str
    format: ReportFormat
    analyzer: AnalyzerKind
    suffixes: tuple[str, ...]
    doc: str

    def to_row(self) -> str:
        suffix = ", ".join(self.suffixes) or "(any)"
        return f"{self.name:12s} {self.format.value:14s} {self.analyzer.value:10s} {suffix}"


def describe_adapters() -> list[AdapterInfo]:
    """Every registered adapter, sorted by name."""
    out: list[AdapterInfo] = []
    for name, adapter in sorted(registered_adapters().items()):
        out.append(
            AdapterInfo(
                name=name,
                format=adapter.format,
                analyzer=adapter.analyzer,
                suffixes=tuple(adapter.suffixes),
                doc=_first_doc_line(type(adapter)),
            )
        )
    return out


def describe_adapters_table() -> str:
    """``--list-formats`` output: one row per adapter plus its first doc line."""
    rows = describe_adapters()
    if not rows:
        return "(no report adapters registered)"
    header = f"{'format':12s} {'container':14s} {'default':10s} suffixes"
    return "\n".join(
        [header, "-" * len(header)]
        + [f"{r.to_row()}\n    {r.doc}" for r in rows]
    )


def _first_doc_line(cls: type[ReportAdapter]) -> str:
    doc = (cls.__doc__ or "").strip().splitlines()
    return doc[0] if doc else ""


__all__ = ["AdapterInfo", "describe_adapters", "describe_adapters_table"]
