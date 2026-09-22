"""Data models for FP path completion analysis."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from llm_client.csa_analysis.models import BugPath, EventKind


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Classification(str, Enum):
    """Whether the report is a true positive (real bug) or false positive."""

    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"
    UNCERTAIN = "uncertain"


class FPGapType(str, Enum):
    """Type of gap in a CSA path that could cause a false positive."""

    OVERLY_SIMPLE_ASSUMPTION = "overly_simple_assumption"
    MISSED_FUNCTION_BODY = "missed_function_body"
    UNCONSTRAINED_VALUE = "unconstrained_value"
    MISSED_BRANCH = "missed_branch"


class GapSeverity(str, Enum):
    """How likely this gap causes the FP."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Confidence(str, Enum):
    """LLM confidence level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class FPGap(BaseModel):
    """A single gap/weak-point in a CSA path that could cause a false positive."""

    gap_type: FPGapType = Field(description="Type of gap")
    location: str = Field(description="Where the gap occurs (event # or function:line)")
    description: str = Field(description="What the analyzer missed")
    severity: GapSeverity = Field(description="How likely this gap causes FP")
    source_context: Optional[str] = Field(None, description="Relevant code snippet")
    suggestion: str = Field(description="How to fix or investigate")


class PathAnalysis(BaseModel):
    """Result of LLM analyzing a CSA path for completeness."""

    path_id: str = Field(description="Identifier for the analyzed path")
    is_likely_fp: bool = Field(description="Whether the path is likely a false positive")
    gaps: list[FPGap] = Field(description="Identified gaps in the path")
    confidence: Confidence = Field(description="LLM confidence in the analysis")
    reasoning: str = Field(description="LLM's overall analysis of the path")


class CompletedStep(BaseModel):
    """A step added or modified during path completion."""

    original_event_index: Optional[int] = Field(
        None, description="Index of original event, None = entirely new step"
    )
    new_event_kind: str = Field(description="Type of the new step: event, control, assume")
    description: str = Field(description="What happens in this step")
    file_path: str = Field(description="Source file path")
    line_number: int = Field(ge=0, description="Line number (0 = synthetic)")
    code_snippet: Optional[str] = Field(None, description="Relevant code")
    concretization: Optional[dict[str, str | int | float | bool]] = Field(
        None, description="Concretized values, e.g. {'ptr': 'nullptr', 'len': 128}"
    )


class CompletedPath(BaseModel):
    """The result of LLM path completion."""

    original_path_summary: str = Field(description="LLM summary of the original path")
    analysis: PathAnalysis = Field(description="Gap analysis")
    completed_steps: list[CompletedStep] = Field(description="Additional completed steps")
    final_path_summary: str = Field(description="LLM description of the full completed path")
    preconditions: list[str] = Field(description="Conditions needed for the bug to trigger")
    concretized_values: dict[str, str | int | float | bool] = Field(
        description="Concrete values for symbolic variables"
    )
    error_start_is_real: bool = Field(
        True,
        description="Is the Error Start value TRULY uninitialized/invalid per C++ semantics? "
                    "False = the value IS actually initialized (via memcpy, assignment, loop init, "
                    "etc.) and the CSA report is a false positive due to an analysis limitation.",
    )


class POCPlan(BaseModel):
    """Plan for generating a POC test case from the completed path."""

    target_file: str = Field(description="Source file containing the bug")
    target_function: str = Field(description="Function to invoke")
    includes: list[str] = Field(description="Required headers")
    setup_code: str = Field(description="State initialization code")
    call_sequence: list[str] = Field(description="Concrete function call sequence")
    expected_crash: str = Field(description="What crash to expect")
    test_framework: str = Field(description="e.g. gtest, standalone")


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


# ---------------------------------------------------------------------------
# Analysis result (combines parsed report + LLM analysis)
# ---------------------------------------------------------------------------


class AnalysisResult(BaseModel):
    """Complete result of FP analysis (parsing + constraint completion + feasibility check)."""

    parsed_report: ParsedReport = Field(description="Parsed HTML report data")
    bug_path: BugPath = Field(description="Structured CSA bug path")
    analysis: PathAnalysis = Field(description="LLM gap analysis")
    source_root: str = Field(default="", description="Project root for source resolution")

    # Enhanced analysis intermediate data (for debugging)
    annotated_path: str = Field(
        default="", description="Path events with data flow annotations for LLM"
    )
    call_chain_context: str = Field(
        default="",
        description="Source code context around each call site in the chain",
    )

    # Classification result
    classification: Classification = Field(
        default=Classification.UNCERTAIN,
        description="Whether the bug is TP (real) or FP (false positive)",
    )
    fp_explanation: str = Field(
        default="",
        description="Human-readable explanation if confirmed false positive",
    )
    verification_history: list[dict] = Field(
        default_factory=list,
        description="Log of verification iterations in the completion loop",
    )
    poc_iteration_log: list[dict] = Field(
        default_factory=list,
        description="Log of compile-fix iterations during POC generation",
    )
    poc_code: str = Field(
        default="",
        description="Final POC source code after compile-fix loop",
    )

    # Feasibility results (new flow)
    is_feasible: bool = Field(
        default=True,
        description="Whether the completed path has no contradictory constraints",
    )
    feasibility_explanation: str = Field(
        default="",
        description="Explanation of feasibility or infeasibility from constraint checking",
    )
    feasibility_conflicts: list[str] = Field(
        default_factory=list,
        description="Specific contradictions found if path is infeasible",
    )

    # PDG-based backward slicing (new flow)
    reduced_code: str = Field(
        default="",
        description="PDG-slice-reduced C++ code containing only statements relevant to this bug path",
    )
    slice_info: Optional[dict] = Field(
        default=None,
        description="Metadata about the slice: node count, reduction ratio, input variables",
    )
    slice_mask: Optional[dict] = Field(
        default=None,
        description="Serialized SliceMask: tracks which PDG nodes / CFG blocks were "
        "retained vs removed by slicing. Keys: pdg_retained_nodes, "
        "pdg_removed_nodes, pdg_retained_branches, pdg_removed_branches, "
        "cfg_retained_blocks, cfg_removed_blocks.",
    )


# ---------------------------------------------------------------------------
# State Lifecycle Analysis (Step 2.7)
# ---------------------------------------------------------------------------


class CallChainEntry(BaseModel):
    """A single function call entry extracted from CSA path events.

    Each ``Calling ...`` or ``← Returning from ...`` event in the CSA path
    becomes one entry.  The ordered sequence forms the *call chain* used
    by Step 2.7 to analyze state variable lifecycles.
    """

    event_index: int = Field(description="Index in original path_events list")
    function_name: str = Field(description="Caller function name")
    callee: Optional[str] = Field(
        default=None,
        description="Callee function name (for Calling entries; None for Return)",
    )
    source_line: int = Field(
        default=0, ge=0,
        description="Source line number in the caller",
    )
    is_entry: bool = Field(
        description="True = 'Calling ...', False = '← Returning from ...'",
    )
    state_params: list[str] = Field(
        default_factory=list,
        description="Pointer/ref parameters that may carry state",
    )


class LifecycleAssessment(BaseModel):
    """Result of Step 2.7 state lifecycle analysis.

    Combines static analysis (call chain, state variable identification)
    with LLM analysis (whether a state lifecycle guarantees initialization
    of the Error Start variable that the CSA flagged).
    """

    call_chain: list[CallChainEntry] = Field(
        default_factory=list,
        description="Ordered call chain extracted from CSA path events",
    )
    state_variables: list[str] = Field(
        default_factory=list,
        description="State variables identified (e.g. ['utf8state', 'ctx->imsg->utf8state'])",
    )
    reset_functions: list[str] = Field(
        default_factory=list,
        description="Functions that SET state to a known initial value",
    )
    non_reset_functions: list[str] = Field(
        default_factory=list,
        description="Functions that READ but do not SET state",
    )
    error_start_initialized_by_lifecycle: bool = Field(
        default=False,
        description=(
            "True = the Error Start variable IS initialized via a state "
            "lifecycle pattern that the CSA does not model."
        ),
    )
    lifecycle_type: Optional[str] = Field(
        default=None,
        description=(
            "Pattern identifier: 'state_reset' | 'loop_carried_init' "
            "| 'union_type_punning' | None"
        ),
    )
    explanation: str = Field(
        default="",
        description="Human-readable analysis explanation",
    )
