"""Data models for Clang CFG analysis — blocks, paths, and feasibility results."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# Forward-reference imports for CFG branch filter integration.
# These are resolved lazily — no circular dependency because
# cfg_branch_models.py does not import from cfg_models.py.
from llm_client.fp_analysis.cfg_branch_models import (  # noqa: E402 (no circular dep)
    CFGBranchTree,
    Postcondition,
)


# ---------------------------------------------------------------------------
# CFG block model
# ---------------------------------------------------------------------------


class CFGBlock(BaseModel):
    """A single basic block in a Clang CFG."""

    block_id: str = Field(description="e.g. 'B1', 'B5'")
    label: str = Field(default="", description="Optional annotation: ENTRY, EXIT, or ''")
    statements: list[str] = Field(
        default_factory=list,
        description="Ordered statement expressions (the [B1.1] / [B1.2] lines)",
    )
    terminator: Optional[str] = Field(
        default=None, description="T: line content, e.g. 'if [B4.6]'"
    )
    predecessors: list[str] = Field(
        default_factory=list, description="Predecessor block IDs"
    )
    successors: list[str] = Field(
        default_factory=list, description="Successor block IDs"
    )
    raw_text: str = Field(default="", description="Raw dump text for debugging")


# ---------------------------------------------------------------------------
# Branch decisions and path variants
# ---------------------------------------------------------------------------


class BranchDecision(BaseModel):
    """A single branch choice at a terminator with two successors."""

    block_id: str = Field(description="Block where the branch occurs")
    terminator_text: str = Field(description="Condition text from T: line")
    source_line: int = Field(default=0, description="Source line if extractable")
    taken_successor: str = Field(description="Successor block that was taken")
    alternative_successor: str = Field(description="Successor block NOT taken")
    taken_is_true_branch: bool = Field(
        default=True, description="True = first successor (true br), False = second (false br)"
    )


class CFGPathVariant(BaseModel):
    """One enumerated path through a function's CFG."""

    variant_id: int = Field(description="0 = CSA original path, 1..N = alternatives")
    branch_decisions: list[BranchDecision] = Field(
        default_factory=list, description="Ordered list of branch choices"
    )
    block_sequence: list[str] = Field(
        default_factory=list, description="Ordered block IDs traversed"
    )
    is_original_path: bool = Field(default=False)


# ---------------------------------------------------------------------------
# Function-level CFG
# ---------------------------------------------------------------------------


class CFGFunction(BaseModel):
    """A complete CFG for one function."""

    function_name: str = Field(description="Function name from dump header")
    source_file: Optional[str] = Field(default=None, description="Source file path")
    blocks: dict[str, CFGBlock] = Field(description="block_id → CFGBlock")
    entry_block_id: str = Field(default="", description="First executable block")
    exit_block_id: str = Field(default="B0", description="EXIT block")
    raw_dump: str = Field(default="", description="Full raw dump for debugging")


# ---------------------------------------------------------------------------
# Bug-Report Path nodes — structured representation of a CSA execution path
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    """Classification of a path node in the CSA execution trace."""

    ENTRY = "entry"                 # Function entry point
    CONTROL = "control"             # Branch decision (if/else/switch)
    CALL = "call"                   # Function call site
    ERROR_START = "error_start"     # Where the undefined/null value originates
    ERROR_SITE = "error_site"       # Where the bug is reported
    ASSUME = "assume"               # CSA assumption about a value
    EVENT = "event"                 # Generic event
    REPORT = "report"               # End of path (bug report)


class PathNode(BaseModel):
    """A single node in the structured bug-report path."""

    event_number: int = Field(description="CSA event number")
    node_type: NodeType = Field(description="Classification of this node")
    line_number: int = Field(default=0, description="Source line number")
    source_file: str = Field(
        default="",
        description="Source file this event belongs to (CSA reports interleave "
                    "path events with the code listing of their own file, so a "
                    "path can span several files)",
    )
    description: str = Field(default="", description="CSA event description")
    function_name: str = Field(default="", description="Function containing this line")
    function_line_start: int = Field(default=0, description="First line of the function")
    function_line_end: int = Field(default=0, description="Last line of the function")
    source_code: str = Field(default="", description="Source code at this line")
    branch_condition: Optional[bool] = Field(
        default=None, description="For CONTROL nodes: True/False branch taken"
    )


class BugReportPath(BaseModel):
    """Structured representation of a CSA execution path with function context.

    Each path event is resolved to a ``PathNode`` annotated with the
    containing function's name, line range, and source code.  This enables
    precise code extraction without heuristic line-range guessing.
    """

    nodes: list[PathNode] = Field(description="Ordered path nodes")
    bug_file: str = Field(default="", description="Source file")
    bug_type: str = Field(default="")
    bug_description: str = Field(default="")

    def function_for_line(self, line: int) -> str:
        """Return the function name containing *line*."""
        for n in self.nodes:
            if n.function_line_start <= line <= n.function_line_end:
                return n.function_name
        return ""

    def function_body(self, function_name: str) -> str:
        """Return the source code of *function_name* from entry to end."""
        lines: list[str] = []
        for n in self.nodes:
            if n.function_name == function_name:
                if n.node_type == NodeType.ENTRY:
                    # Will accumulate lines from all nodes in this function
                    pass
        return ""

    def source_between(
        self, event_from: int, event_to: int
    ) -> str:
        """Return source code between two event numbers.

        Reads from the containing function's entry line to *event_to*'s line,
        then extends past that line to capture any CONTINUATION LINES
        (e.g., multi-line ``if (a ||\n    b ||\n    c)`` conditions).

        Lines within the requested range are marked ``>>>``.
        """
        # Find the function containing event_to
        target = next((n for n in self.nodes if n.event_number == event_to), None)
        if not target:
            return "(event not found)"
        fn_start = target.function_line_start
        fn_end = target.function_line_end
        line_to = target.line_number

        # If event_from exists and is in the same function, start from its line
        source = next((n for n in self.nodes if n.event_number == event_from), None)
        line_from = source.line_number if source else fn_start

        # Read the source file
        bug_file = Path(self.bug_file)
        if not bug_file.exists():
            local = Path("project")
            for f in local.rglob(bug_file.name):
                bug_file = f
                break

        try:
            all_lines = bug_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            return "(source not available)"

        # Determine end line: extend past line_to to capture continuation lines
        # (multi-line conditions like ``if (a ||\n    b ||\n    c)``)
        end_line = min(line_to, len(all_lines))
        continuation_patterns = (",", "(", "||", "&&", "+", "-", "*", "/", "\\")
        for i in range(line_to, min(line_to + 8, len(all_lines))):
            if all_lines[i - 1].rstrip().endswith(continuation_patterns):
                end_line = i + 1
            else:
                break

        result: list[str] = []
        for i in range(fn_start - 1, min(end_line, len(all_lines))):
            ln = i + 1
            mark = ">>>" if (line_from <= ln <= line_to) else "   "
            result.append(f"{mark} {ln:>5}: {all_lines[i]}")

        return "\n".join(result)


# ---------------------------------------------------------------------------
# Anchor events — CSA report branch decisions used for path pruning
# ---------------------------------------------------------------------------


class AnchorEvent(BaseModel):
    """A single branch decision from the CSA report used as an anchoring point.

    Each ``AnchorEvent`` represents a ``control`` or ``assume`` event in the
    CSA execution path where the analyzer made a concrete branch decision
    (TRUE / FALSE).  These events serve as anchoring points during iterative
    slicing: already-traversed nodes from the report are used to prune
    infeasible branches in subsequent segments.

    The ``status`` field tracks the anchor's lifecycle:

    * ``pending`` — not yet checked against completed constraints.
    * ``consistent`` — the branch direction is compatible with constraints.
    * ``contradictory`` — the branch direction conflicts with constraints,
      indicating this anchor's segment is infeasible.
    """

    event_number: int = Field(description="CSA event sequence number")
    line: int = Field(description="Source line number")
    branch: bool = Field(description="True=TRUE branch taken, False=FALSE branch taken")
    condition: str = Field(description="Condition expression text, e.g. 'blockLength_ <= 0'")
    variable: Optional[str] = Field(
        default=None,
        description="Variable name extracted from the condition, if identifiable",
    )
    status: str = Field(
        default="pending",
        description="Lifecycle status: pending | consistent | contradictory",
    )


# ---------------------------------------------------------------------------
# Segment-level analysis
# ---------------------------------------------------------------------------


class SegmentInfo(BaseModel):
    """A segment of the CSA path between two control events.

    Stores all information needed for LLM feasibility checking:
    - **Precondition**: the state facts established before this segment
    - **Postcondition**: the control event outcome terminating this segment
    - **Code**: the source code from function entry to the postcondition line
    - **If-branches**: all ``if`` statements in the code (with line numbers)
    - **Called-functions**: names of functions called within the code
    """

    segment_index: int = Field(description="Index in the segment list")
    label: str = Field(default="")

    # Event range
    event_range: tuple[int, int] = Field(
        description="(first_event_number, last_event_number)"
    )
    line_start: int = Field(default=0, description="First source line in segment")
    line_end: int = Field(default=0, description="Last source line in segment")

    # Precondition: the state facts before this segment
    precondition_facts: list[str] = Field(
        default_factory=list,
        description="Established state facts (branch outcomes, field values)",
    )

    # Postcondition: the control event terminating this segment
    control_event_number: Optional[int] = Field(
        default=None,
        description="Event number of the terminating control event",
    )
    control_branch_taken: Optional[bool] = Field(
        default=None,
        description="Which branch CSA took: True/False",
    )

    # Code: source code from function entry to postcondition line
    function_name: str = Field(default="", description="Containing function")
    function_file: str = Field(
        default="",
        description="File containing ``function_name``.  Needed because a path "
                    "can cross files while function names repeat across them; "
                    "the CFG cache is keyed per translation unit.",
    )
    code_from_entry: str = Field(
        default="",
        description="Source code from function entry to postcondition line",
    )

    # If-branches in the code (line numbers only, not the full condition)
    if_line_numbers: list[int] = Field(
        default_factory=list,
        description="Line numbers of if-statements in the code",
    )

    # Called-functions: names only (bodies are looked up at prompt-build time)
    called_functions: list[str] = Field(
        default_factory=list,
        description="Function names called within this segment's code",
    )

    # ⭐ Phase 1 enhancement: anchor events and pruning metadata
    anchor_events: list[AnchorEvent] = Field(
        default_factory=list,
        description="CSA report branch events within this segment (anchoring points)",
    )
    pruned_event_indices: list[int] = Field(
        default_factory=list,
        description="Event indices pruned by prior segment state (maintains audit trail)",
    )

    # ⭐ CFG branch filter integration: postcondition + branch tree
    postcondition: Optional["Postcondition"] = Field(
        default=None,
        description="Postcondition derived from the segment's boundary control event. "
                    "Used by ConflictChecker for branch filtering.",
    )
    cfg_branch_tree: Optional["CFGBranchTree"] = Field(
        default=None,
        description="CFG branch tree (Step 3 output) containing all extracted "
                    "branch nodes with ConflictChecker verdict tags (Step 4).",
    )

    def format_postcondition(self) -> str:
        """Return human-readable postcondition description."""
        direction = "TRUE" if self.control_branch_taken else "FALSE"
        return (
            f"CSA took the {direction} branch at event "
            f"#{self.control_event_number} (L{self.line_end})."
        )

    def format_precondition(self) -> str:
        """Return human-readable precondition description."""
        return "\n".join(self.precondition_facts) if self.precondition_facts \
            else "Function entry / no prior events."


class SegmentPathAnalysis(BaseModel):
    """Analysis of one segment of the CSA path."""

    segment_index: int = Field(description="Index in segment list")
    segment_label: str = Field(default="")
    function_name: str = Field(default="", description="Function analyzed for this segment")

    original_path_variant: Optional[CFGPathVariant] = Field(default=None)
    alternative_variants: list[CFGPathVariant] = Field(default_factory=list)

    # variant_id → feasibility status
    variant_feasibility: dict[int, Optional[bool]] = Field(
        default_factory=dict, description="True=feasible, False=infeasible, None=unchecked"
    )
    variant_reasoning: dict[int, str] = Field(
        default_factory=dict, description="LLM reasoning per variant"
    )

    has_infeasible_segment: bool = Field(
        default=False,
        description="True if NO variant is feasible → confirmed FP",
    )

    # Established state after this segment (for chaining to next segment)
    established_state: str = Field(
        default="",
        description="LLM's description of program state after this segment executes",
    )


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


class CFGAnalysisResult(BaseModel):
    """Top-level result of CFG-based feasibility analysis."""

    segments: list[SegmentPathAnalysis] = Field(
        default_factory=list, description="Analysis per segment"
    )
    overall_infeasible: bool = Field(
        default=False,
        description="True = at least one segment has no feasible path → confirmed FP",
    )
    cfg_functions: dict[str, CFGFunction] = Field(
        default_factory=dict, description="function_name → parsed CFG"
    )
    original_path_feasible: bool = Field(
        default=True,
        description="Whether the original CSA path is feasible",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Timing, errors, fallback info",
    )


# ---------------------------------------------------------------------------
# Assumption checking — CSA symbolic-execution assumptions extracted from
# event-type path events (NOT control events).
# ---------------------------------------------------------------------------


class AssumptionVerdict(str, Enum):
    """Outcome of an assumption feasibility check.

    ``FEASIBLE`` — the assumption can hold under real-world constraints.
    ``INFEASIBLE`` — the assumption contradicts real-world constraints;
    the CSA path is a false positive.
    """

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"

    def __bool__(self) -> bool:
        return self is AssumptionVerdict.FEASIBLE


class AssumptionFact(BaseModel):
    """A single CSA symbolic-execution assumption extracted from a path event.

    Represents an assumption the CSA made during its symbolic execution
    (e.g.  ``"Assuming field 'blockLength_' is <= 0"``).  These assumptions
    are event-type path events, not control events; they define the
    preconditions under which the CSA path reaches the bug.

    Each fact is annotated with the segment it belongs to and whether it
    is part of the Error Start chain.
    """

    event_number: int = Field(description="CSA event number this came from")
    line_number: int = Field(description="Source line number")
    segment_index: int = Field(description="Which segment this assumption belongs to")
    variable: str = Field(description="Variable name, e.g. 'blockLength_'")
    operator: str = Field(
        description="Relational operator: '<=' | '==' | '!=' | '>=' | '>' | '<'",
    )
    value: str = Field(
        description="RHS value, e.g. '0', 'null', 'bitfieldLength_'",
    )
    raw_description: str = Field(
        default="",
        description="Original CSA event description (for LLM fallback)",
    )
    is_error_start: bool = Field(
        default=False,
        description="Whether this assumption belongs to the Error Start chain",
    )
    # Tier-1 result: set by _layer1_rule_check(); None until checked
    layer1_result: Optional[str] = Field(
        default=None,
        description="Tier-1 rule check outcome: 'VIOLATED' | 'CONSISTENT' | 'UNCERTAIN'",
    )

    def is_feasible(self) -> bool:
        """Return True unless Tier-1 has already proven this infeasible."""
        if self.layer1_result == "VIOLATED":
            return False
        return True
