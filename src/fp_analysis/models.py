"""Data models for FP path completion analysis."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class CompletedStep(BaseModel):
    """A step added or modified during path completion."""

    new_event_kind: str = Field(description="Type of the new step: event, control, assume")
    description: str = Field(description="What happens in this step")
    file_path: str = Field(description="Source file path")
    line_number: int = Field(ge=0, description="Line number (0 = synthetic)")
    concretization: Optional[dict[str, str | int | float | bool]] = Field(
        None, description="Concretized values, e.g. {'ptr': 'nullptr', 'len': 128}"
    )


class CompletedPath(BaseModel):
    """The result of LLM path completion."""

    completed_steps: list[CompletedStep] = Field(description="Additional completed steps")
    final_path_summary: str = Field(description="LLM description of the full completed path")
    preconditions: list[str] = Field(description="Conditions needed for the bug to trigger")
    concretized_values: dict[str, str | int | float | bool] = Field(
        description="Concrete values for symbolic variables"
    )


# ---------------------------------------------------------------------------
# Parsed report
# ---------------------------------------------------------------------------


class ReportMetadata(BaseModel):
    """Metadata extracted from an FP HTML report."""

    bug_type: str = Field(description="Bug type from CSA")
    bug_category: str = Field(default="", description="Bug category")
    bug_file: str = Field(description="Source file path from report")
    bug_line: int = Field(ge=0, description="Line number of the bug")
    bug_column: int = Field(default=0, description="Column number")
    bug_description: str = Field(default="", description="Bug description")
    function_name: str = Field(default="", description="Function where bug occurs")
    path_length: int = Field(default=0, description="Number of path steps")
    #: ``"trace"`` when the analyzer walked a path, ``"location"`` when it
    #: stated only *where* (most CodeQL results).  A location-only report still
    #: renders its report step, so ``path_length`` alone cannot tell the two
    #: apart.  Defaults to the weaker reading: a caller that needs a path must
    #: not be handed one by a report that never said so.
    path_evidence: str = Field(
        default="location", description="'trace' | 'location' (see report_frontend)"
    )

    @property
    def has_path(self) -> bool:
        """True only when the analyzer reported a step trace, not just a place."""
        return self.path_evidence == "trace"


class ParsedReport(BaseModel):
    """Structured representation of a parsed CSA FP report."""

    metadata: ReportMetadata = Field(description="Report metadata")
    path_events: list[dict] = Field(description="Ordered path events with descriptions")
    source_snippets: dict[int, str] = Field(
        description="Source code lines keyed by line number"
    )
    raw_html_path: str = Field(description="Path to the original HTML file")

    # Enhanced analysis fields (populated during parsing)
    error_start_event: Optional[dict] = Field(
        default=None,
        description="The 'Error Start' event info (event number, line, description) if present",
    )
    calling_events: list[dict] = Field(
        default_factory=list,
        description="List of 'Calling' events with function name, event number, line",
    )

    #: The per-report code model — the one reader of project source for every
    #: stage that hands code to the LLM.  Set by the bridge from the frontend's
    #: ``LoadedReport.code``; excluded from every dump because it is a live
    #: object holding caches, not data.  ``None`` for a report built by hand
    #: (tests, tools) — see :func:`code_model_of`.
    code_model: Any = Field(default=None, exclude=True, repr=False)


def code_model_of(
    parsed: "ParsedReport", source_root: str | None = None
):
    """The code model that serves *parsed*.

    The report's own model is the right one where it exists: it carries the
    copy the report *embedded*, which is the revision the analyzer ran on and
    the only reason its line numbers mean anything.  The fallback is the
    process-wide :func:`shared_model`, which never holds embedded source — the
    correct answer for a report assembled by hand, but **not** a substitute for
    the real model, so callers that can thread the report's own should.

    ``getattr`` rather than attribute access: a test double standing in for
    ``ParsedReport`` has no ``code_model`` at all.
    """
    from report_frontend.code_model import shared_model

    model = getattr(parsed, "code_model", None)
    if model is not None:
        return model
    if source_root is None:
        source_root = getattr(parsed, "source_root", None)
    return shared_model(source_root)


# ---------------------------------------------------------------------------
# Analysis result (combines parsed report + LLM analysis)
# ---------------------------------------------------------------------------


class AnalysisResult(BaseModel):
    """Complete result of FP analysis (parsing + constraint completion + feasibility check)."""

    parsed_report: ParsedReport = Field(description="Parsed HTML report data")
    source_root: str = Field(default="", description="Project root for source resolution")
