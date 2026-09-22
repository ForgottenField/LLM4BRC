"""Conflict data models — 8-layer Hierarchical Conflict Pattern and helpers.

Phase 2 of the shortest-path selection + conflict-driven learning design.
These are plain ``@dataclass`` objects shared between
:mod:`conflict_learner` and :mod:`path_analyzer`.

The 8-layer hierarchy (CDCL-inspired):
  L1: Location — conflicting segments + branch decisions
  L2: Constraint — the actual constraint clash
  L3: Semantic Object — program objects involved + their roles
  L4: Semantic Type — classification (null_check, return_value, etc.)
  L5: Root Cause (MUS) — minimal unsatisfiable core
  L6: Propagation Chain — how conflict propagated across segments
  L7: Repair Hint — concrete suggestions for re-selection
  L8: Generalization — reusable pattern + signature
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum


# ─── Conflict Semantic Type (Layer 4) ───────────────────────────────────────────


class ConflictSemanticType(str, Enum):
    """Classification of constraint conflict by program semantic category."""

    RETURN_VALUE_CONTRADICTION = "return_value_contradiction"
    NULL_CHECK_CONTRADICTION = "null_check_contradiction"
    USE_AFTER_FREE = "use_after_free"
    DOUBLE_FREE = "double_free"
    STATE_VARIABLE_CONFLICT = "state_variable_conflict"
    API_SEMANTIC_CONFLICT = "api_semantic_conflict"
    LIFECYCLE_CONFLICT = "lifecycle_conflict"
    TYPE_PUNNING_FALSE_CONFLICT = "type_punning_false_conflict"
    LOOP_CARRIED_STATE_CONFLICT = "loop_carried_state_conflict"


# ─── Constraint Conflict (Layers 1-2) ───────────────────────────────────────────


@dataclass
class ConstraintConflict:
    """A single constraint clash between two segments (Layers 1-2).

    Attributes:
        variable: The variable whose constraint is violated.
        segment_a: First conflicting segment index.
        segment_a_constraint: Constraint on the variable in segment A.
        segment_b: Second conflicting segment index.
        segment_b_constraint: Constraint on the variable in segment B.
        conflict_type: Category — ``"range_contradiction"``,
            ``"null_check"``, ``"lifetime"``, etc.
    """

    variable: str
    segment_a: int
    segment_a_constraint: str
    segment_b: int
    segment_b_constraint: str
    conflict_type: str = "range_contradiction"


# ─── Propagation Step (Layer 6) ─────────────────────────────────────────────────


@dataclass
class PropagationStep:
    """One step in the conflict propagation chain.

    Attributes:
        segment_index: Which segment this step occurs in.
        source_line: Source line number.
        description: Human-readable description of the propagation step.
    """

    segment_index: int
    source_line: int = 0
    description: str = ""


# ─── Repair Hint (Layer 7) ──────────────────────────────────────────────────────


@dataclass
class RepairHint:
    """A concrete suggestion for resolving a conflict.

    Attributes:
        target_segment: Which segment should be re-selected.
        target_variable: The variable whose constraint should change.
        suggested_action: ``"change_branch"``, ``"re_constrain"``, or
            ``"add_assumption"``.
        suggested_branch: The branch direction to try (for ``change_branch``).
        reasoning: Why this change should resolve the conflict.
        impact: What effect this change would have on other segments.
    """

    target_segment: int
    target_variable: str
    suggested_action: str = "change_branch"  # change_branch | re_constrain | add_assumption
    suggested_branch: str | None = None
    reasoning: str = ""
    impact: str = ""


# ─── Dataflow Context ───────────────────────────────────────────────────────────


@dataclass
class DataflowContext:
    """Records dataflow state at the time a conflict pattern was learned.

    Prevents stale conflict patterns from incorrectly pruning valid paths
    when the dataflow context has changed across rounds.

    Attributes:
        var_def_sites: ``{variable: (function, line, expression)}`` —
            where each variable was last defined.
        def_use_chains: ``{variable: [node_id, ...]}`` — PDG node chain.
        reassignment_checks: ``{variable: has_been_reassigned}`` — whether
            the variable was reassigned since the pattern was learned.
        context_hash: Fast validity check — changes when any field changes.
    """

    var_def_sites: dict[str, tuple[str, int, str]] = field(default_factory=dict)
    def_use_chains: dict[str, list[str]] = field(default_factory=dict)
    reassignment_checks: dict[str, bool] = field(default_factory=dict)
    context_hash: str = ""

    def __post_init__(self) -> None:
        if not self.context_hash:
            self.context_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute a stable hash from all dataflow fields."""
        raw = json.dumps({
            "defs": {k: list(v) for k, v in self.var_def_sites.items()},
            "chains": self.def_use_chains,
            "reassign": self.reassignment_checks,
        }, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def is_still_valid(self, current_selections: dict) -> bool:
        """Check whether this context is still applicable.

        A context is considered stale when any tracked variable has been
        reassigned (``reassignment_checks[v] == True``), indicating the
        dataflow has changed and the learned pattern may no longer apply.

        Args:
            current_selections: ``{segment_idx: {established_state: ...}}``
                from the current round's selections.
        """
        # If any tracked variable was reassigned, the context is stale
        for var, was_reassigned in self.reassignment_checks.items():
            if was_reassigned:
                return False
        # Empty context (no tracked variables) is always valid
        return True


# ─── Hierarchical Conflict Pattern ──────────────────────────────────────────────


@dataclass
class HierarchicalConflictPattern:
    """A complete 8-layer conflict pattern learned from a merge failure.

    Attributes:
        conflict_id: Unique identifier for this conflict instance.
        conflicting_segments: Indices of segments involved in the conflict.
        conflict_paths: Per-segment branch decisions at time of conflict.
        violated_constraints: Layer 1-2 — constraint clashes.
        conflict_objects: Layer 3 — program objects involved.
        object_roles: Layer 3 — semantic roles of each object.
        semantic_type: Layer 4 — conflict classification.
        minimal_unsat_core: Layer 5 — minimal unsatisfiable core.
        propagation_chain: Layer 6 — step-by-step conflict propagation.
        repair_hints: Layer 7 — suggested fixes.
        generalized_pattern: Layer 8 — human-readable pattern description.
        pattern_signature: Layer 8 — compact signature for dedup/stuck detection.
        dataflow_context: Context protection against stale pruning.
    """

    conflict_id: str
    conflicting_segments: list[int] = field(default_factory=list)
    conflict_paths: dict[int, list] = field(default_factory=dict)

    # Layer 1-2: Location + Constraint
    violated_constraints: list[ConstraintConflict] = field(default_factory=list)

    # Layer 3: Semantic Object
    conflict_objects: list[str] = field(default_factory=list)
    object_roles: dict[str, str] = field(default_factory=dict)

    # Layer 4: Semantic Type
    semantic_type: ConflictSemanticType = ConflictSemanticType.STATE_VARIABLE_CONFLICT

    # Layer 5: Root Cause (MUS)
    minimal_unsat_core: list[ConstraintConflict] = field(default_factory=list)

    # Layer 6: Propagation Chain
    propagation_chain: list[PropagationStep] = field(default_factory=list)

    # Layer 7: Repair Hint
    repair_hints: list[RepairHint] = field(default_factory=list)

    # Layer 8: Generalization
    generalized_pattern: str = ""
    pattern_signature: str = ""

    # Context protection
    dataflow_context: DataflowContext = field(default_factory=DataflowContext)
