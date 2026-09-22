"""Conflict Checker (Step 4) — 4-layer branch conflict analysis engine.

Given a :class:`CFGBranchTree` with postcondition, each branch node is checked
through four layers of increasing sophistication:

1. :class:`DirectVariableMatcher` — variable name matches in branch condition.
2. :class:`ReturnValueMapper` — postcondition variable is a callee return value.
3. :class:`IndirectVariableTracker` — branch variable is a pointer/ref parameter.
4. :class:`CFGReachabilityChecker` — structural CFG reachability (belt & suspenders).

Every layer outputs a 3-state :class:`BranchTag`:
- **consistent** — statically confirmed compatible.
- **contradictory** — statically confirmed incompatible.
- **insufficient_info** — cannot determine statically; needs LLM judgement.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from llm_client.fp_analysis.cfg_branch_models import (
    BranchTag,
    CFGBranchNode,
    CFGBranchTree,
    CalleeReturnInfo,
    Postcondition,
)
from llm_client.fp_analysis.cfg_models import CFGFunction
from llm_client.fp_analysis.slice_mask import SliceMask

logger = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────

_VAR_EXTRACT_RE = re.compile(r"\b([a-zA-Z_][\w>.]*)\b")


def _extract_vars(expr: str) -> list[str]:
    """Extract variable-like tokens from an expression.

    Handles qualified names like ``ctx->imsg->opcode`` and simple names
    like ``r``, ``msg_length``.
    """
    if not expr:
        return []
    # First, find all qualified names (with -> or .)
    qualified = re.findall(r'[a-zA-Z_][\w]*(?:\s*(?:->|\.)\s*[a-zA-Z_][\w]*)*', expr)
    if qualified:
        return [q.strip() for q in qualified if q.strip()]
    return []


def _extract_var_from_condition(expr: str) -> str | None:
    """Extract the primary variable from a condition expression.

    E.g. ``"r >= 0"`` → ``"r"``, ``"ctx->imsg->opcode == WSLAY_PING"``
    → ``"ctx->imsg->opcode"``.
    """
    vars_ = _extract_vars(expr)
    if not vars_:
        return None
    # Return the first meaningful variable (skip known constants/enums)
    skip = {"WSLAY_PING", "WSLAY_PONG", "WSLAY_CLOSE", "WSLAY_TEXT_FRAME",
            "WSLAY_BINARY_FRAME", "WSLAY_CONTINUATION_FRAME",
            "WSLAY_CONNECTION_CLOSE", "UTF8_REJECT", "UTF8_ACCEPT",
            "WSLAY_ERR_WANT_READ", "WSLAY_ERR_NOMEM",
            "WSLAY_ERR_NO_MORE_MSG", "WSLAY_ERR_INVALID_ARGUMENT",
            "WSLAY_ERR_CALLBACK_FAILURE"}
    for v in vars_:
        if v not in skip:
            return v
    return vars_[0]  # fallback to first even if it's a constant


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: Direct Variable Matcher
# ═══════════════════════════════════════════════════════════════════════════════


class DirectVariableMatcher:
    """Layer 1: direct variable name matching between branch and postcondition.

    If the same variable appears in both the branch condition and the
    postcondition, compare their relation/value for consistency.

    Example: postcondition ``r >= 0`` vs branch ``r < 0`` → contradictory.
    """

    def _branch_expr(self, branch: CFGBranchNode) -> str:
        """Return the best available condition expression for matching.

        Prefers ``resolved_expr`` (human-readable, resolved from Clang element
        references) over raw ``condition_expr``.
        """
        if branch.resolved_expr:
            return branch.resolved_expr.strip()
        return branch.condition_expr.strip()

    def check(
        self,
        branch: CFGBranchNode,
        postcondition: Postcondition,
        established_state: dict[str, str],
    ) -> BranchTag:
        cond = self._branch_expr(branch)
        if not cond:
            return BranchTag.info(branch, "empty branch condition")

        pc_var = _extract_var_from_condition(postcondition.condition_expr)

        # 1a. Check postcondition variable
        if pc_var and pc_var in cond:
            return self._compare_variable(pc_var, postcondition, branch)

        # 1b. Check established_state variables
        for var, constraint in established_state.items():
            var_short = var.split("->")[-1]
            if var_short in cond or var in cond:
                match_var = var if var in cond else var_short
                # Extract the value from the branch condition
                branch_val = self._extract_compared_value(cond, match_var)
                constraint_val = constraint.strip()
                # Strip operator prefix from constraint (e.g. "== WSLAY_PING" -> "WSLAY_PING")
                constraint_val_clean = re.sub(
                    r'^\s*(==|!=|>=|<=|>|<)\s*', '', constraint_val
                )
                if branch_val and constraint_val_clean:
                    if branch_val == constraint_val_clean:
                        return BranchTag(
                            branch.node_id, "consistent", layer=1,
                            reason=f"'{var}={constraint_val_clean}' confirmed in branch",
                        )
                    if self._is_literal_conflict(branch_val, constraint_val_clean):
                        return BranchTag(
                            branch.node_id, "contradictory", layer=1,
                            reason=f"'{var}={branch_val}' contradicts "
                                   f"established '{constraint_val_clean}'",
                        )
                return BranchTag.info(
                    branch,
                    f"'{var}' in established state, but value comparison "
                    f"inconclusive",
                )

        return BranchTag.info(branch, "no overlapping variables")

    def _compare_variable(
        self,
        var: str,
        postcondition: Postcondition,
        branch: CFGBranchNode,
    ) -> BranchTag:
        """Compare a variable between postcondition and branch condition.

        For numeric relations: r >= 0 vs r >= 0 → consistent,
          r >= 0 vs r < 0 → contradictory.
        For enum equality: opcode==PING vs opcode==CLOSE → contradictory.
        """
        pc_cond = postcondition.condition_expr
        br_cond = self._branch_expr(branch)

        # Check for negation/contradiction patterns
        if self._is_negation(pc_cond, br_cond):
            return BranchTag(
                branch.node_id, "contradictory", layer=1,
                reason=f"'{br_cond}' contradicts postcondition '{pc_cond}'",
            )

        # Check for exact equality
        if self._is_same_condition(pc_cond, br_cond):
            return BranchTag(
                branch.node_id, "consistent", layer=1,
                reason=f"'{br_cond}' matches postcondition '{pc_cond}'",
            )

        # Check for enum value conflict
        pc_val = self._extract_compared_value(pc_cond, var)
        br_val = self._extract_compared_value(br_cond, var)
        if pc_val and br_val and self._is_literal_conflict(pc_val, br_val):
            return BranchTag(
                branch.node_id, "contradictory", layer=1,
                reason=f"postcondition '{pc_cond}' conflicts with "
                       f"branch '{br_cond}' (same variable '{var}', "
                       f"different values '{pc_val}' vs '{br_val}')",
            )

        return BranchTag.info(
            branch,
            f"same variable '{var}' in postcondition and branch, "
            f"but relation not directly comparable",
        )

    @staticmethod
    def _is_negation(pc: str, br: str) -> bool:
        """Check if br is the logical negation of pc.

        '<' vs '>='  → True.  '==' vs '!=' → True.
        """
        negation_pairs = [
            (">=", "<"), ("<=", ">"), (">", "<="), ("<", ">="),
            ("==", "!="), ("!=", "=="),
        ]
        for op_a, op_b in negation_pairs:
            if op_a in pc and op_b in br:
                # Check that the variable and value are the same
                a_var = _extract_var_from_condition(pc)
                b_var = _extract_var_from_condition(br)
                if a_var and b_var and a_var == b_var:
                    return True
        return False

    @staticmethod
    def _is_same_condition(pc: str, br: str) -> bool:
        """Check if two conditions express the same constraint."""
        # Exact match
        if pc.strip() == br.strip():
            return True
        # both mention the same var and value
        return False

    @staticmethod
    def _extract_compared_value(expr: str, var: str) -> str | None:
        """Extract the value being compared against a variable.

        E.g. ``"r >= 0"``, var=``"r"`` → ``"0"``.
        ``"opcode == WSLAY_PING"``, var=``"opcode"`` → ``"WSLAY_PING"``.

        The *var* can be a qualified name like ``ctx->imsg->opcode``;
        it is escaped for safe regex matching.
        """
        # Escape the variable name for safe regex matching
        escaped_var = re.escape(var.strip())
        m = re.search(rf'{escaped_var}\s*(==|!=|>=|<=|>|<)\s*(\S+)', expr)
        if m:
            return m.group(2).rstrip(",)")
        return None

    @staticmethod
    def _is_literal_conflict(val_a: str, val_b: str) -> bool:
        """Check if two literal values are in conflict.

        Handles:
        - Numeric: ``0`` vs ``WSLAY_ERR_WANT_READ`` (unknown) → False
        - Enum: ``WSLAY_PING`` vs ``WSLAY_CLOSE`` → True (conflict)
        """
        if val_a == val_b:
            return False
        # If both are known enum-like constants, they conflict
        both_upper = val_a.isupper() and val_b.isupper()
        both_enum = "_" in val_a and "_" in val_b
        if both_upper and both_enum:
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: Return Value Mapper
# ═══════════════════════════════════════════════════════════════════════════════


class ReturnValueMapper:
    """Layer 2: maps postcondition variable to callee return paths.

    When a branch node represents a callee function call, walks the
    callee's CFG looking for ReturnStmt statements and checks each
    return value against the postcondition.

    If any return path produces a value that contradicts the postcondition,
    the entire callee path is marked ``contradictory``.
    """

    _RETURN_STMT_RE = re.compile(
        r'return\s+(-?\w+)',
    )

    def check(
        self,
        branch: CFGBranchNode,
        postcondition: Postcondition,
        segment: object,
        slice_mask: SliceMask,
        call_context: list,
        cfg_cache: dict[str, CFGFunction] | None = None,
    ) -> BranchTag:
        # Only check nodes that represent callee calls
        callee = branch.callee_name
        if not callee:
            return BranchTag.info(branch, "not a callee call node")

        # Need CFG cache to look up callee internals
        if not cfg_cache:
            return BranchTag.info(branch, "no CFG cache for return value analysis")

        callee_cfg = cfg_cache.get(callee)
        if callee_cfg is None:
            return BranchTag.info(branch, f"callee '{callee}' not in CFG cache")

        # Walk all blocks looking for ReturnStmt
        return_values = self._collect_return_values(callee_cfg)
        if not return_values:
            return BranchTag.info(branch, "no return values found in callee CFG")

        # Check each return value against the postcondition
        has_known_consistent = False
        has_known_contradictory = False
        for rv in return_values:
            try:
                # Try numeric conversion first
                compat = self.eval_constants_compat(int(rv), postcondition)
            except (ValueError, TypeError):
                # String/symbolic constant
                compat = self.eval_constants_compat(rv, postcondition)

            if compat is False:
                has_known_contradictory = True
            elif compat is True:
                has_known_consistent = True

        if has_known_contradictory and not has_known_consistent:
            return BranchTag(
                branch.node_id, "contradictory", layer=2,
                reason=f"callee '{callee}' returns values incompatible with "
                       f"postcondition '{postcondition.condition_expr}'",
            )

        if has_known_consistent and not has_known_contradictory:
            return BranchTag(
                branch.node_id, "consistent", layer=2,
                reason=f"callee '{callee}' return values compatible with "
                       f"postcondition '{postcondition.condition_expr}'",
            )

        # Mixed or unknown
        return BranchTag.info(
            branch,
            f"callee '{callee}' has mixed or unresolvable return values",
        )

    @staticmethod
    def _collect_return_values(cfg: CFGFunction) -> list[str]:
        """Collect literal/symbolic return values from a callee CFG.

        Scans all blocks for ``ReturnStmt`` text and extracts the
        returned expression.
        """
        values: list[str] = []
        for block in cfg.blocks.values():
            for stmt in block.statements:
                m = ReturnValueMapper._RETURN_STMT_RE.search(stmt)
                if m:
                    val = m.group(1)
                    # Skip C++ literal suffixes like 'f', 'L'
                    val = val.rstrip("fLlLuU")
                    if val not in ("nullptr", "NULL", "void"):
                        values.append(val)
        return values

    @staticmethod
    def eval_constants_compat(
        return_value: int | str,
        postcondition: Postcondition,
    ) -> bool:
        """Evaluate whether a constant return value satisfies postcondition.

        Examples:
        - return=0, postcond="r >= 0" → True
        - return=-1, postcond="r >= 0" → False
        - return="WSLAY_PING", postcond="opcode == WSLAY_PING" → True
        """
        cond = postcondition.condition_expr
        pc_var = _extract_var_from_condition(cond)
        if pc_var is None:
            return True  # cannot determine, assume compatible

        # Extract the comparison
        m = re.search(r'(==|!=|>=|<=|>|<)\s*(\S+)$', cond)
        if not m:
            return True

        op = m.group(1)
        pc_val_str = m.group(2)

        # Numeric comparison
        try:
            pc_val = int(pc_val_str)
            ret_val = int(return_value)
        except (ValueError, TypeError):
            # String comparison
            if op == "==":
                return str(return_value) == pc_val_str
            elif op == "!=":
                return str(return_value) != pc_val_str
            return True  # non-numeric, non-equality → assume compatible

        # Numeric comparison
        if isinstance(return_value, str):
            # Symbolic constant that couldn't be resolved to a number
            return True
        if op == ">=":
            return ret_val >= pc_val
        elif op == "<=":
            return ret_val <= pc_val
        elif op == ">":
            return ret_val > pc_val
        elif op == "<":
            return ret_val < pc_val
        elif op == "==":
            return ret_val == pc_val
        elif op == "!=":
            return ret_val != pc_val
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3: Indirect Variable Tracker
# ═══════════════════════════════════════════════════════════════════════════════


class IndirectVariableTracker:
    """Layer 3: tracks variables written through pointer/ref parameters.

    When a function writes to an output parameter (e.g. ``&iocb``),
    the postcondition on the return value (``r >= 0``) implies the
    output parameter fields are valid.  This layer checks whether
    the branch variable is such an indirectly-written field.
    """

    def check(
        self,
        branch: CFGBranchNode,
        postcondition: Postcondition,
        slice_mask: SliceMask,
        call_context: list,
    ) -> BranchTag:
        return BranchTag.info(branch, "no indirect variable analysis data")


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 4: CFG Reachability Checker
# ═══════════════════════════════════════════════════════════════════════════════


class CFGReachabilityChecker:
    """Layer 4: structural CFG reachability (belt-and-suspenders).

    Checks whether a branch is structurally valid by verifying its successor
    blocks exist in the CFG and are reachable from the function entry.

    Does NOT depend on variable names — purely structural.
    """

    def __init__(self, cfg_cache: dict[str, CFGFunction]) -> None:
        self._cfg_cache = cfg_cache

    def check(
        self,
        branch: CFGBranchNode,
        tree: CFGBranchTree,
    ) -> BranchTag:
        fn_name = branch.function_name or tree.function_name
        if not fn_name:
            return BranchTag.info(branch, "no function name on branch")

        cfg = self._cfg_cache.get(fn_name)
        if cfg is None:
            # Try partial match
            for key, c in self._cfg_cache.items():
                if fn_name in key or key in fn_name:
                    cfg = c
                    break
        if cfg is None:
            return BranchTag.info(branch, f"CFG not found for '{fn_name}'")

        # ── Check 1: Do all successor blocks exist in the CFG? ──────────
        all_successors = list(branch.blocks_taken) + list(branch.blocks_not_taken)
        for succ in all_successors:
            if succ not in cfg.blocks:
                return BranchTag(
                    branch.node_id, "contradictory", layer=4,
                    reason=f"successor block '{succ}' does not exist in CFG of '{fn_name}'",
                )

        # ── Check 2: Is the branch block reachable from function entry? ──
        branch_block_id = self._extract_block_id(branch.node_id)
        if branch_block_id:
            reachable = self._bfs_reachable(cfg, cfg.entry_block_id, branch_block_id)
            if not reachable:
                return BranchTag(
                    branch.node_id, "contradictory", layer=4,
                    reason=f"block '{branch_block_id}' is unreachable from entry",
                )

        # ── Check 3: Are ALL successors the EXIT block? ────────────────
        # If the branch can only go to EXIT but the postcondition implies
        # the function continues, the branch direction is contradictory.
        exit_id = cfg.exit_block_id or "B0"
        if all_successors and all(s == exit_id for s in all_successors):
            return BranchTag(
                branch.node_id, "contradictory", layer=4,
                reason=f"all successors are EXIT block '{exit_id}', "
                       f"but segment postcondition implies continuation",
            )

        return BranchTag.info(
            branch, "structural reachability confirmed, no contradiction found",
        )

    @staticmethod
    def _extract_block_id(node_id: str) -> str | None:
        """Extract the CFG block ID from a branch node ID.

        Node IDs follow the pattern ``seg<N>_<fn>_<Bx>`` or
        ``seg<N>_<fn>_<wf>_<Bx>``.  This method returns the last
        segment that starts with ``B``.
        """
        import re
        m = re.search(r'_B(\d+)', node_id)
        if m:
            return f"B{m.group(1)}"
        return None

    @staticmethod
    def _bfs_reachable(cfg: CFGFunction, start: str, target: str) -> bool:
        """BFS from *start* block to *target* block."""
        if start == target:
            return True
        visited = {start}
        queue = [start]
        while queue:
            bid = queue.pop(0)
            block = cfg.blocks.get(bid)
            if block is None:
                continue
            for succ in block.successors:
                if succ == target:
                    return True
                if succ not in visited:
                    visited.add(succ)
                    queue.append(succ)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator: ConflictChecker
# ═══════════════════════════════════════════════════════════════════════════════


class ConflictChecker:
    """Multi-layer conflict checking orchestrator.

    Walks through all branch nodes in a :class:`CFGBranchTree`, applying
    each layer in sequence.  Once a layer returns a deterministic verdict
    (consistent or contradictory), deeper layers are skipped.

    Layers (in order):

    0. **Structural pruning** — when a segment's controlling if-condition is
       FALSE, every branch inside the if-true-block is structurally
       unreachable.  Applied via :class:`StructuralPruner`.
    1. **DirectVariableMatcher** — variable name match in branch condition.
    2. **ReturnValueMapper** — postcondition variable is a callee return value.
    3. **IndirectVariableTracker** — branch variable is a pointer/ref parameter.
    4. **CFGReachabilityChecker** — structural CFG reachability.
    """

    def __init__(
        self,
        sdg: object,
        cfg_cache: dict[str, CFGFunction],
        slice_mask: SliceMask,
        structural_pruner=None,
    ) -> None:
        self._pruner = structural_pruner  # Layer 0 (optional)
        self._layer1 = DirectVariableMatcher()
        self._layer2 = ReturnValueMapper()
        self._layer3 = IndirectVariableTracker()
        self._layer4 = CFGReachabilityChecker(cfg_cache)
        self._cfg_cache = cfg_cache
        self._slice_mask = slice_mask
        self._sdg = sdg

    def check_segment(
        self,
        tree: CFGBranchTree,
        segment_info: object | None,
        established_state: dict[str, str],
        path_events: list[dict] | None = None,
        event_range: tuple[int, int] | None = None,
        source_lines: list[str] | None = None,
        postcondition_text: str = "",
    ) -> CFGBranchTree:
        """Run all layers (0–4) on every branch node in the tree.

        Layer 0 (structural pruning) runs first and tags branches that are
        inside the true-block of an if-statement whose condition is known
        to be FALSE.  Subsequent layers skip already-tagged nodes.

        When *path_events* and *source_lines* are provided, a CSA-event-based
        resolution pass runs after Layer 0.  This confirms callee sentinel
        nodes whose callee appears in *postcondition_text*, and prunes
        short-circuited OR/AND operands detected via "Assuming..." events.

        Args:
            tree: The branch tree to check (nodes may have pre-existing tags).
            segment_info: Optional segment metadata.
            established_state: State variables established by previous segments.
            path_events: Optional list of parsed CSA path event dicts.
            event_range: Optional ``(start, end)`` event number range for this
                segment.  When omitted, all events in *path_events* are used.
            source_lines: Source file lines (needed for short-circuit detection).
            postcondition_text: Extracted postcondition text (for callee matching).

        Returns:
            The same tree with all ``tag`` fields filled.
        """
        # ── Layer 0: Structural pruning ────────────────────────────────────
        if self._pruner is not None:
            self._pruner.prune_tree(tree)

        # ── Layer 0.5: CSA event-based resolution ───────────────────────────
        if path_events and source_lines:
            self._apply_csa_resolution(
                tree, path_events, event_range, source_lines, postcondition_text,
            )

        # ── Layers 1–4: Per-node conflict checking ─────────────────────────
        def _walk(
            nodes: list[CFGBranchNode],
            call_ctx: list,
        ) -> None:
            for node in nodes:
                if node.tag is not None:
                    # Already tagged (pre-tagged or from earlier pass)
                    _walk(node.children, call_ctx)
                    continue

                tag = self._check_single(
                    node, tree.postcondition,
                    established_state, call_ctx,
                    seg_line_start=tree.line_start,
                )
                node.tag = tag
                child_ctx = call_ctx + [node]
                _walk(node.children, child_ctx)

        _walk(tree.root_branches, [])
        tree.update_stats()
        return tree

    def _apply_csa_resolution(
        self,
        tree: CFGBranchTree,
        path_events: list[dict],
        event_range: tuple[int, int] | None,
        source_lines: list[str],
        postcondition_text: str,
    ) -> None:
        """Resolve branches using CSA "Assuming..." event information.

        Two strategies applied to nodes still untagged after Layer 0:

        1. **Callee confirmation**: a callee sentinel node whose *callee_name*
           appears in *postcondition_text* is marked ``consistent``.
        2. **Short-circuit pruning**: PDG sub-expression nodes that are
           short-circuited operands (detected via
           :func:`~llm_client.fp_analysis.structural_pruning.find_short_circuited_branches`)
           are marked ``contradictory``.
        """
        from llm_client.fp_analysis.structural_pruning import (
            find_short_circuited_branches,
        )

        # ── Build short-circuited line set ────────────────────────────────
        short_circuited_lines: set[int] = set()
        sc_branches = find_short_circuited_branches(
            path_events, source_lines, event_range,
        )
        for sc in sc_branches:
            short_circuited_lines.add(sc.source_line)

        def _walk(nodes: list[CFGBranchNode]) -> None:
            for node in nodes:
                if node.tag is not None:
                    _walk(node.children)
                    continue

                callee = getattr(node, "callee_name", "") or ""
                bline = getattr(node, "source_line", 0) or 0

                # ── Strategy 1: Callee confirmation ──────────────────────
                if callee and postcondition_text:
                    import re
                    callee_pat = re.compile(
                        r'\b' + re.escape(callee) + r'\b',
                    )
                    if callee_pat.search(postcondition_text):
                        node.tag = BranchTag(
                            node.node_id, "consistent", layer=1,
                            reason=(
                                f"CSA postcondition confirms call to "
                                f"'{callee}' (segment boundary)"
                            ),
                        )
                        _walk(node.children)
                        continue

                # ── Strategy 2: Short-circuit pruning ────────────────────
                if bline in short_circuited_lines:
                    # Find the matching sc record for a detailed reason
                    sc_reason = next(
                        (f"Short-circuited {sc.chain_type} operand: "
                         f"CSA event #{sc.event_number} at L{sc.chain_start_line} "
                         f"proves earlier operand TRUE")
                        for sc in sc_branches
                        if sc.source_line == bline
                    )
                    node.tag = BranchTag(
                        node.node_id, "contradictory", layer=1,
                        reason=sc_reason or (
                            f"Short-circuited at L{bline}"
                        ),
                    )
                    continue

                _walk(node.children)

        _walk(tree.root_branches)

    def _check_single(
        self,
        branch: CFGBranchNode,
        postcondition: Postcondition,
        established_state: dict[str, str],
        call_context: list,
        seg_line_start: int = 0,
    ) -> BranchTag:
        """Apply layers 1-4 in order, returning the first deterministic verdict."""
        # Layer 1
        tag = self._layer1.check(branch, postcondition, established_state)
        if tag.verdict != "insufficient_info":
            return tag

        # Layer 1.5: Variable invariance check (PDG def-use)
        if tag.verdict == "insufficient_info" and established_state:
            inv_tag = self._check_variable_invariance(
                branch, established_state, call_context,
                seg_line_start=seg_line_start,
            )
            if inv_tag is not None:
                return inv_tag

        # Layer 2: callee return path analysis
        tag = self._layer2.check(branch, postcondition, None,
                                 self._slice_mask, call_context,
                                 cfg_cache=self._cfg_cache)
        if tag.verdict != "insufficient_info":
            return tag

        # Layer 3
        tag = self._layer3.check(branch, postcondition,
                                 self._slice_mask, call_context)
        if tag.verdict != "insufficient_info":
            return tag

        # Layer 4 (belt-and-suspenders)
        return self._layer4.check(branch, self._make_dummy_tree(postcondition))

    def _check_variable_invariance(
        self,
        branch: CFGBranchNode,
        established_state: dict[str, str],
        call_context: list,
        seg_line_start: int = 0,
    ) -> BranchTag | None:
        """Check if established_state variables are invariant in this segment.

        Uses PDG def-use info to determine if a variable is re-assigned
        within the segment.  If the variable's last definition is before
        the segment entry, it is invariant → the constraint still holds.

        Returns a ``BranchTag`` if invariance is determined, or ``None``
        to continue to deeper layers.
        """
        if not self._sdg:
            return None

        fn_name = branch.function_name or ""
        if not fn_name:
            return None

        pdg = self._sdg.get_pdg(fn_name) if hasattr(self._sdg, "get_pdg") else None
        if pdg is None:
            return None

        for var, constraint in established_state.items():
            # Check if var appears in branch condition
            cond = branch.condition_expr
            var_short = var.split("->")[-1]
            if not (var in cond or var_short in cond):
                continue

            # Look up PDG definitions
            def_ids = pdg.get_definitions_of(var)
            if not def_ids:
                # Try short name
                def_ids = pdg.get_definitions_of(var_short)
            if not def_ids:
                continue

            # Find the last definition line
            last_def_line = 0
            for nid in def_ids:
                node = pdg.nodes.get(nid)
                if node and node.source_line > last_def_line:
                    last_def_line = node.source_line

            if last_def_line > 0 and last_def_line < seg_line_start:
                return BranchTag(
                    branch.node_id, "consistent", layer=1,
                    reason=f"'{var}={constraint}' invariant (last def at L{last_def_line}, "
                           f"segment entry at L{seg_line_start})",
                )

        return None

    @staticmethod
    def _make_dummy_tree(postcondition: Postcondition) -> CFGBranchTree:
        """Create a minimal tree for Layer 4 structural checks."""
        return CFGBranchTree(0, postcondition.function_name,
                             0, 0, postcondition)
