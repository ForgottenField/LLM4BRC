"""Conflict Pattern Learner — CDCL-inspired semantic conflict learning.

Phase 2 of the shortest-path selection + conflict-driven learning design.
Extracts 8-layer Hierarchical Conflict Patterns from inter-segment merge
failures and generates avoidance rules for re-selection prompts.

Layers 1-2 (Location + Constraint) are extracted deterministically via
range/constraint analysis.  Layers 3-8 are extracted via LLM reasoning.
"""

from __future__ import annotations

import logging
from typing import Optional

from llm_client.fp_analysis.conflict_models import (
    ConflictSemanticType,
    ConstraintConflict,
    DataflowContext,
    HierarchicalConflictPattern,
    PropagationStep,
    RepairHint,
)

logger = logging.getLogger(__name__)


class ConflictPatternLearner:
    """Extracts and manages hierarchical conflict patterns across rounds.

    Maintains a session-internal knowledge base of learned patterns and
    provides methods for stuck detection and avoidance rule generation.

    Args:
        provider: LLM provider for L3-L8 extraction (can be ``None`` for
            deterministic-only operation).
        model: LLM model name.
    """

    def __init__(
        self,
        provider: object | None = None,
        model: str = "",
    ) -> None:
        self._provider = provider
        self._model = model
        # Session-internal accumulated patterns
        self._accumulated_patterns: list[HierarchicalConflictPattern] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def learn(
        self,
        merge_result: object,
        segment_selections: dict[int, dict],
        segment_states: dict[int, dict],
    ) -> list[HierarchicalConflictPattern]:
        """Extract conflict patterns from a merge failure.

        Args:
            merge_result: The MergeCheckResult from global merge checking.
            segment_selections: ``{seg_idx: {established_state: ...}}``.
            segment_states: ``{seg_idx: {postcondition: ...}}``.

        Returns:
            List of ``HierarchicalConflictPattern`` (at least L1-L2 filled).
        """
        # Extract conflicting segments from the merge result
        conflict_segs = getattr(merge_result, "conflicting_segments", [])
        if not conflict_segs:
            # Try to infer from conflict details
            conflict_segs = list(segment_selections.keys())

        conflict_details = getattr(merge_result, "conflict_details", "")
        if not conflict_details:
            conflict_details = getattr(merge_result, "reason", "")

        # L1-L2: Deterministic extraction
        patterns = self._extract_l1_l2(
            conflict_segments=conflict_segs,
            segment_states=segment_states,
            conflict_details=str(conflict_details),
        )

        # L3-L8: LLM-driven extraction (when provider is available)
        if patterns and self._provider is not None:
            for p in patterns:
                try:
                    self._extract_l3_l8(p, segment_selections, conflict_details)
                except Exception as exc:
                    logger.warning(
                        "LLM extraction failed for pattern %s: %s",
                        p.conflict_id, exc,
                    )

        self._accumulated_patterns.extend(patterns)
        return patterns

    @property
    def accumulated_patterns(self) -> list[HierarchicalConflictPattern]:
        """Return all patterns learned so far in this session."""
        return list(self._accumulated_patterns)

    # ── Avoidance Rule Generation ───────────────────────────────────────────

    def generate_avoidance_rules(
        self,
        patterns: list[HierarchicalConflictPattern],
        current_selections: dict[int, dict],
    ) -> str:
        """Generate avoidance rules text for LLM prompt injection.

        Args:
            patterns: The conflict patterns to convert to rules.
            current_selections: Current round selections (for staleness check).

        Returns:
            Formatted string for injection into re-selection prompts.
        """
        if not patterns:
            return ""

        rules: list[str] = []
        for p in patterns:
            # Staleness check
            freshness = ""
            if not p.dataflow_context.is_still_valid(current_selections):
                freshness = " ⚠ MAY BE STALE — dataflow context has changed"

            hints_text = ""
            if p.repair_hints:
                hints_text = p.repair_hints[0].suggested_action
            elif p.repair_hints:
                hints_text = "re-select with alternative branch"

            rules.append(
                f"### Avoidance Rule {p.conflict_id}{freshness}\n"
                f"- **Semantic Type**: {p.semantic_type.value}\n"
                f"- **Conflicting Segments**: {p.conflicting_segments}\n"
                f"- **Conflict Objects**: {', '.join(p.conflict_objects) if p.conflict_objects else 'N/A'}\n"
                f"- **Root Cause**: {self._format_mus(p.minimal_unsat_core)}\n"
                f"- **Propagation Chain**: {' → '.join(str(s.segment_index) for s in p.propagation_chain) if p.propagation_chain else 'N/A'}\n"
                f"- **Repair Hint**: {hints_text}\n"
                f"- **Generalized Pattern**: {p.generalized_pattern}"
            )

        return "\n---\n".join(rules)

    @staticmethod
    def _format_mus(mus: list[ConstraintConflict]) -> str:
        """Format a minimal unsatisfiable core for display."""
        if not mus:
            return "N/A"
        parts = []
        for cc in mus:
            parts.append(
                f"{cc.variable}: [{cc.segment_a}] {cc.segment_a_constraint} "
                f"vs [{cc.segment_b}] {cc.segment_b_constraint}"
            )
        return "; ".join(parts)

    # ── Stuck Detection ─────────────────────────────────────────────────────

    def _is_stuck(
        self,
        prev_signatures: set[str],
        curr_signatures: set[str],
        round_num: int,
        has_new_hints: bool = False,
    ) -> bool:
        """Check if the re-selection loop is stuck.

        Stuck = same signatures as previous round AND no new repair hints
        available.  Also stuck after round 3 regardless.

        Args:
            prev_signatures: Pattern signatures from the previous round.
            curr_signatures: Pattern signatures from the current round.
            round_num: Current round number (1-indexed).
            has_new_hints: Whether any new repair hints were generated.

        Returns:
            ``True`` if the loop should exit early.
        """
        # Hard round limit
        if round_num >= 3:
            return True

        # Same signatures with no new hints → stuck
        if prev_signatures and prev_signatures == curr_signatures and not has_new_hints:
            return True

        return False

    # ── L1-L2: Deterministic Extraction ─────────────────────────────────────

    def _extract_l1_l2(
        self,
        conflict_segments: list[int],
        segment_states: dict[int, dict],
        conflict_details: str,
    ) -> list[HierarchicalConflictPattern]:
        """Extract Layers 1-2 deterministically from conflicting states.

        Runs ``_range_intersects()`` / ``_check_non_numeric_constraint()``
        on ``established_state`` vs ``postcondition`` pairs across all
        conflicting segments.

        Args:
            conflict_segments: Indices of conflicting segments.
            segment_states: ``{seg_idx: {postcondition: ..., ...}}``.
            conflict_details: Human-readable conflict description.

        Returns:
            List of ``HierarchicalConflictPattern`` with L1-L2 filled.
        """
        patterns: list[HierarchicalConflictPattern] = []

        # Compare constraints pairwise between conflicting segments
        for i, seg_a in enumerate(conflict_segments):
            for seg_b in conflict_segments[i + 1:]:
                state_a = segment_states.get(seg_a, {})
                state_b = segment_states.get(seg_b, {})

                # Extract postcondition constraints
                pc_a = state_a.get("postcondition", "")
                pc_b = state_b.get("postcondition", "")

                if not pc_a or not pc_b:
                    continue

                # Parse constraints from postcondition text
                constraints_a = self._parse_constraints(pc_a)
                constraints_b = self._parse_constraints(pc_b)

                # Check for conflicts
                violated: list[ConstraintConflict] = []
                for var, (op_a, val_a) in constraints_a.items():
                    if var in constraints_b:
                        op_b, val_b = constraints_b[var]
                        if self._constraints_conflict(op_a, val_a, op_b, val_b):
                            violated.append(ConstraintConflict(
                                variable=var,
                                segment_a=seg_a,
                                segment_a_constraint=pc_a,
                                segment_b=seg_b,
                                segment_b_constraint=pc_b,
                                conflict_type=self._classify_conflict_type(
                                    op_a, val_a, op_b, val_b,
                                ),
                            ))

                if violated:
                    pattern_id = f"conflict-{len(self._accumulated_patterns) + len(patterns):03d}"
                    patterns.append(HierarchicalConflictPattern(
                        conflict_id=pattern_id,
                        conflicting_segments=[seg_a, seg_b],
                        violated_constraints=violated,
                        conflict_objects=[cc.variable for cc in violated],
                        # Default semantic type based on first conflict
                        semantic_type=self._infer_semantic_type(violated),
                    ))

        return patterns

    @staticmethod
    def _parse_constraints(pc_text: str) -> dict[str, tuple[str, str]]:
        """Parse constraint text like ``"r >= 0"`` into ``{var: (op, val)}``.

        Handles simple forms:
        - ``"r >= 0"`` → ``{"r": (">=", "0")}``
        - ``"ptr != NULL"`` → ``{"ptr": ("!=", "NULL")}``
        - ``"r >= 0, msg_length > 0"`` → multiple
        """
        import re
        result: dict[str, tuple[str, str]] = {}
        # Match: variable op value
        pattern = re.compile(
            r'([a-zA-Z_]\w*(?:\s*(?:->|\.)\s*[a-zA-Z_]\w*)*)\s*'
            r'(>=|<=|!=|==|>|<)\s*'
            r'(\S+)'
        )
        for match in pattern.finditer(pc_text):
            var = match.group(1).strip()
            op = match.group(2)
            val = match.group(3).rstrip(',;')
            result[var] = (op, val)
        return result

    @staticmethod
    def _constraints_conflict(
        op_a: str, val_a: str,
        op_b: str, val_b: str,
    ) -> bool:
        """Check if two constraints on the same variable conflict.

        Simple numeric comparison for range contradictions.
        Handles: >= / <= / > / < / == / != on numeric values.
        """
        # NULL check conflicts
        if val_a.upper() == "NULL" or val_b.upper() == "NULL":
            if val_a.upper() == "NULL" and val_b.upper() == "NULL":
                return False  # both NULL
            # One NULL, one non-NULL
            if (op_a == "==" and val_a.upper() == "NULL" and
                    op_b == "!=" and val_b.upper() == "NULL"):
                return True
            if (op_a == "!=" and val_a.upper() == "NULL" and
                    op_b == "==" and val_b.upper() == "NULL"):
                return True
            return False

        # Try numeric comparison
        try:
            n_a = int(val_a)
            n_b = int(val_b)
        except (ValueError, TypeError):
            # Non-numeric values (e.g. enum constants) — can't deterministically
            # check, so assume NO conflict
            return False

        # Range contradiction logic
        # "r >= 0" and "r < 0" → conflict (intersection is empty)
        if op_a == ">=" and op_b == "<":
            return n_a > n_b - 1
        if op_a == ">" and op_b == "<=":
            return n_a >= n_b
        if op_a == ">=" and op_b == "<=":
            return n_a > n_b
        if op_a == ">" and op_b == "<":
            return n_a >= n_b - 1

        # Symmetric
        if op_b == ">=" and op_a == "<":
            return n_b > n_a - 1
        if op_b == ">" and op_a == "<=":
            return n_b >= n_a
        if op_b == ">=" and op_a == "<=":
            return n_b > n_a
        if op_b == ">" and op_a == "<":
            return n_b >= n_a - 1

        # == conflicts
        if op_a == "==" and op_b == "==":
            return n_a != n_b
        if op_a == "==" and op_b == "!=":
            return n_a == n_b
        if op_b == "==" and op_a == "!=":
            return n_b == n_a

        return False

    @staticmethod
    def _classify_conflict_type(
        op_a: str, val_a: str,
        op_b: str, val_b: str,
    ) -> str:
        """Classify the conflict type for a constraint pair."""
        if val_a.upper() == "NULL" or val_b.upper() == "NULL":
            return "null_check"
        return "range_contradiction"

    @staticmethod
    def _infer_semantic_type(
        violated: list[ConstraintConflict],
    ) -> ConflictSemanticType:
        """Infer the semantic type from constraint conflicts."""
        for cc in violated:
            if cc.conflict_type == "null_check":
                return ConflictSemanticType.NULL_CHECK_CONTRADICTION
        return ConflictSemanticType.RETURN_VALUE_CONTRADICTION

    # ── L3-L8: LLM-driven Extraction ────────────────────────────────────────

    def _extract_l3_l8(
        self,
        pattern: HierarchicalConflictPattern,
        segment_selections: dict[int, dict],
        conflict_details: str,
    ) -> None:
        """Extract Layers 3-8 via LLM reasoning.

        Populates ``pattern`` in-place with:
        - L3: object_roles (semantic interpretation of conflict objects)
        - L4: semantic_type (validated/refined)
        - L5: minimal_unsat_core (MUS)
        - L6: propagation_chain
        - L7: repair_hints
        - L8: generalized_pattern + pattern_signature
        """
        if self._provider is None:
            return

        # Build a focused prompt for the LLM
        prompt = self._build_l3_l8_prompt(pattern, conflict_details)

        try:
            response = self._provider.complete(
                prompt,
                system=(
                    "You are a program analysis expert specializing in "
                    "conflict diagnosis for C/C++ static analysis bug reports. "
                    "Extract the requested layers concisely as JSON."
                ),
                model=self._model,
            )
            # Best-effort parsing — failures leave fields at defaults
            if response:
                self._parse_l3_l8_response(pattern, response)
        except Exception as exc:
            logger.warning("L3-L8 LLM extraction failed: %s", exc)

    def _build_l3_l8_prompt(
        self,
        pattern: HierarchicalConflictPattern,
        conflict_details: str,
    ) -> str:
        """Build the L3-L8 extraction prompt."""
        parts = [
            "## Global Merge Failure",
            "",
            "The following inter-segment constraints conflict:",
            conflict_details,
            "",
            "### Deterministic Analysis (L1-L2)",
            f"- Conflicting Segments: {pattern.conflicting_segments}",
            f"- Conflict Object(s): {', '.join(pattern.conflict_objects)}",
        ]
        for cc in pattern.violated_constraints:
            parts.append(
                f"- {cc.variable}: seg{cc.segment_a}[{cc.segment_a_constraint}] "
                f"vs seg{cc.segment_b}[{cc.segment_b_constraint}] "
                f"({cc.conflict_type})"
            )

        parts.extend([
            "",
            "### Task",
            "Extract Layers 3-8 of the Hierarchical Conflict Pattern as JSON:",
            '{"object_roles": {"var": "role"}, "semantic_type": "...", ',
            ' "minimal_unsat_core": "...", "propagation_chain": [...], ',
            ' "repair_hints": [...], "generalized_pattern": "...", ',
            ' "pattern_signature": "..."}',
        ])
        return "\n".join(parts)

    def _parse_l3_l8_response(
        self,
        pattern: HierarchicalConflictPattern,
        response: str,
    ) -> None:
        """Parse LLM response into pattern fields."""
        import json as _json
        try:
            # Try to extract JSON from response
            # Find JSON object in the response
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                data = _json.loads(response[start:end + 1])
                if isinstance(data, dict):
                    # L3
                    if "object_roles" in data:
                        pattern.object_roles = data["object_roles"]
                    # L4
                    if "semantic_type" in data:
                        try:
                            pattern.semantic_type = ConflictSemanticType(
                                data["semantic_type"]
                            )
                        except ValueError:
                            pass
                    # L5
                    if "minimal_unsat_core" in data:
                        mus = data["minimal_unsat_core"]
                        if isinstance(mus, str):
                            pattern.minimal_unsat_core = pattern.violated_constraints
                        elif isinstance(mus, list):
                            # List of constraint dicts
                            pattern.minimal_unsat_core = pattern.violated_constraints
                    # L6
                    if "propagation_chain" in data:
                        chain = data["propagation_chain"]
                        if isinstance(chain, list):
                            pattern.propagation_chain = [
                                PropagationStep(
                                    segment_index=s.get("segment_index", 0),
                                    source_line=s.get("source_line", 0),
                                    description=s.get("description", ""),
                                )
                                for s in chain if isinstance(s, dict)
                            ]
                    # L7
                    if "repair_hints" in data:
                        hints = data["repair_hints"]
                        if isinstance(hints, list):
                            pattern.repair_hints = [
                                RepairHint(
                                    target_segment=h.get("target_segment", 0),
                                    target_variable=h.get("target_variable", ""),
                                    suggested_action=h.get(
                                        "suggested_action", "change_branch",
                                    ),
                                    suggested_branch=h.get("suggested_branch"),
                                    reasoning=h.get("reasoning", ""),
                                    impact=h.get("impact", ""),
                                )
                                for h in hints if isinstance(h, dict)
                            ]
                    # L8
                    if "generalized_pattern" in data:
                        pattern.generalized_pattern = data["generalized_pattern"]
                    if "pattern_signature" in data:
                        pattern.pattern_signature = data["pattern_signature"]
        except Exception as exc:
            logger.debug("Failed to parse L3-L8 response: %s", exc)
