"""Data models for CFG branch extraction and postcondition-based filtering.

These are the core data structures shared between
:mod:`cfg_branch_extractor` (Step 3) and :mod:`conflict_checker` (Step 4).

All models are plain ``@dataclass`` objects — no ORM, no Pydantic.  They
are designed to be immutable once constructed (the caller is expected to
use ``replace()`` or construct new instances rather than mutate in place).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# ─── Public type alias ────────────────────────────────────────────────────────

VerdictType = Literal["consistent", "contradictory", "insufficient_info"]
"""Three possible outcomes of conflict checking a branch node."""


# ─── Postcondition ────────────────────────────────────────────────────────────


@dataclass
class Postcondition:
    """A segment's postcondition — derived from its boundary CONTROL event.

    This is the "anchor" for CFG branch filtering.  Each segment's terminal
    control event defines a postcondition that must hold for the segment to
    be feasible.  All branches inside the segment are checked against this.

    Attributes:
        function_name: Function containing the control event.
        source_line: Source line of the control event.
        condition_expr: Human-readable condition text, e.g. ``"r >= 0"``.
        branch_taken: ``True`` if the CSA trace took the true branch.
    """

    function_name: str
    source_line: int
    condition_expr: str = ""
    branch_taken: bool = True

    @classmethod
    def from_control_event(cls, event: dict) -> Postcondition:
        """Construct from a CSA control event dict.

        The dict is expected to have keys ``description``, ``line_number``,
        ``branch_condition`` and ``function_name``, as produced by the
        HTML report parser.
        """
        desc = event.get("description", "")
        line = event.get("line_number", 0)
        fn = event.get("function_name", "") or event.get("function", "")
        bc = event.get("branch_condition")
        return cls(
            function_name=fn,
            source_line=int(line) if line else 0,
            condition_expr=_infer_condition_from_event(desc, event),
            branch_taken=bc if bc is not None else True,
        )

    def negate(self) -> Postcondition:
        """Return the inverse postcondition (opposite branch)."""
        return Postcondition(
            function_name=self.function_name,
            source_line=self.source_line,
            condition_expr=self.condition_expr,
            branch_taken=not self.branch_taken,
        )


def _infer_condition_from_event(desc: str, event: dict) -> str:
    """Best-effort extraction of a condition expression from event text.

    Handles common CSA event description patterns:
    - ``← Taking true branch →`` — returns the reported branch_condition text
    - ``← Taking false branch →``
    - ``← Assuming field 'x' is 0 →`` — extracts the assumption
    """
    import re as _re
    # Try to extract a quoted assumption
    m = _re.search(r"'(?P<field>[^']+)'\s+is\s+(?P<val>.+)", desc)
    if m:
        return f"{m.group('field')} == {m.group('val')}"
    # Fall back to branch_kind text
    bc = event.get("branch_condition")
    if bc is not None:
        return f"branch={bc}"
    return desc


# ─── Branch verdict tag ──────────────────────────────────────────────────────


@dataclass
class BranchTag:
    """Result of conflict-checking a single CFG branch node.

    The ``verdict`` field encodes the three possible outcomes from the
    four-layer conflict checking pipeline.

    Attributes:
        branch_id: Matches ``CFGBranchNode.node_id``.
        verdict: One of ``consistent``, ``contradictory``, ``insufficient_info``.
        layer: Which layer produced this verdict (1-4, or 0 for default).
        reason: Human-readable explanation.
        return_chain: Cross-procedure call chain for Layer-2 verdicts.
    """

    branch_id: str
    verdict: VerdictType
    layer: int = 0
    reason: str = ""
    return_chain: list[ReturnChainLink] = field(default_factory=list)

    @classmethod
    def info(cls, branch: CFGBranchNode, reason: str = "") -> BranchTag:
        """Shorthand for an *insufficient_info* tag.

        This is the fallback verdict when no layer can make a deterministic
        judgement — the branch is left for LLM-level path selection.
        """
        return cls(
            branch_id=branch.node_id,
            verdict="insufficient_info",
            layer=0,
            reason=reason or "insufficient static information",
        )


# ─── CFG branch tree ─────────────────────────────────────────────────────────


@dataclass
class CFGBranchNode:
    """A single node in a segment's CFG branch tree.

    Each node corresponds to one branch decision in the CFG (if/while/for
    terminator, or an implicit call-return branch).  Nodes form a tree via
    the ``children`` field, which captures both nested ``if`` statements
    in the same function and callee-internal branches extracted through
    cross-procedural analysis.

    Attributes:
        node_id: Unique identifier, e.g. ``"seg0_B4_taken"``.
        source_line: Source line number of the branch.
        source_file: Source file path.
        function_name: Containing function name.
        condition_expr: Branch condition expression text.
        is_taken_branch: ``True`` for the taken (true) branch side.
        branch_label: Human label, e.g. ``"true"`` / ``"false"``.
        call_depth: 0 = main function, 1 = direct callee, 2+ = indirect.
        parent_call_site: Description of parent call, e.g. ``"L572 wslay_frame_recv"``.
        callee_name: If this is a CALL-type branch, the callee function name.
        blocks_taken: CFG block IDs on the taken side.
        blocks_not_taken: CFG block IDs on the non-taken side.
        children: Nested sub-branches.
        tag: Verdict tag (set by ConflictChecker, ``None`` before checking).
    """

    node_id: str
    source_line: int
    source_file: str = ""
    function_name: str = ""
    condition_expr: str = ""
    is_taken_branch: bool = True
    branch_label: str = ""
    call_depth: int = 0
    parent_call_site: Optional[str] = None
    callee_name: Optional[str] = None
    blocks_taken: list[str] = field(default_factory=list)
    blocks_not_taken: list[str] = field(default_factory=list)
    children: list[CFGBranchNode] = field(default_factory=list)
    resolved_expr: str = ""
    tag: Optional[BranchTag] = None


@dataclass
class CFGBranchTree:
    """The complete branch tree for one segment.

    Contains all CFG-level branch nodes within the segment's source-line
    range, including cross-procedural nodes up to the configured depth.

    ``postcondition`` is the root constraint for the segment — all branch
    nodes in this tree are checked for consistency against it.

    Attributes:
        segment_index: Index in the segment list.
        function_name: Main function for this segment.
        line_start / line_end: Source line range.
        postcondition: The segment's postcondition constraint.
        root_branches: Top-level branch nodes (no parent).
        num_branches / num_consistent / num_contradictory / num_insufficient:
            Statistics updated by :meth:`update_stats`.
    """

    segment_index: int
    function_name: str
    line_start: int
    line_end: int
    postcondition: Postcondition
    root_branches: list[CFGBranchNode] = field(default_factory=list)
    num_branches: int = 0
    num_consistent: int = 0
    num_contradictory: int = 0
    num_insufficient: int = 0

    def add_branch(self, node: CFGBranchNode) -> None:
        """Append a root-level branch node and increment the counter."""
        self.root_branches.append(node)
        self.num_branches += 1

    def update_stats(self) -> None:
        """Walk the entire tree and recount verdict statistics.

        Call this after the :class:`ConflictChecker` has filled ``tag``
        on every node.  Children are included in the counts.
        """
        total, con, ct, ins = _walk_branches(self.root_branches)
        self.num_branches = total
        self.num_consistent = con
        self.num_contradictory = ct
        self.num_insufficient = ins


def _walk_branches(
    nodes: list[CFGBranchNode],
) -> tuple[int, int, int, int]:
    """Recursively count branches and verdicts in a node list."""
    total = 0
    consistent = 0
    contradictory = 0
    insufficient = 0
    for n in nodes:
        total += 1
        if n.tag:
            if n.tag.verdict == "consistent":
                consistent += 1
            elif n.tag.verdict == "contradictory":
                contradictory += 1
            else:
                insufficient += 1
        sub_t, sub_c, sub_ct, sub_i = _walk_branches(n.children)
        total += sub_t
        consistent += sub_c
        contradictory += sub_ct
        insufficient += sub_i
    return total, consistent, contradictory, insufficient


# ─── Return value mapping (Layer 2) ──────────────────────────────────────────


@dataclass
class ReturnChainLink:
    """One level in a cross-procedural call chain.

    Records how the postcondition variable was assigned through a function
    call at a specific call site.
    """

    caller_fn: str
    call_line: int
    assignment: str
    callee: str
    ret_var: str


@dataclass
class CalleeReturnInfo:
    """One return path in a callee function's CFG.

    Each distinct ``return`` statement in the callee gets its own entry.
    The ``blocks`` field records which CFG basic blocks are traversed
    from function entry to this return statement.
    """

    return_value: int | str | None
    return_line: int
    blocks: list[str] = field(default_factory=list)
    compatible: bool = True
    branch_decisions_to_return: list[BranchDecision] = field(default_factory=list)


@dataclass
class ReturnValueMap:
    """Maps a postcondition variable to callee return paths.

    When the postcondition variable (e.g. ``r``) is the return value of a
    function call (e.g. ``r = wslay_frame_recv(...)``), this structure
    records which return paths in the callee are compatible with the
    postcondition and which are not.
    """

    post_var: str
    post_cond: str
    call_site: tuple[str, int]
    callee: str
    callee_returns: list[CalleeReturnInfo] = field(default_factory=list)


# ─── Re-export for convenience ───────────────────────────────────────────────


@dataclass
class BranchDecision:
    """A single branch decision (used by CalleeReturnInfo)."""

    block_id: str
    terminator_text: str = ""
    source_line: int = 0
    taken_successor: str = ""
    alternative_successor: str = ""
    taken_is_true_branch: bool = True
