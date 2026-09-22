"""Backward slicing engine for System Dependence Graphs.

Given a *slicing criterion* — a bug trigger point identified as
``(function_name, node_id)`` — this module computes the backward slice through
the SDG.  The slice includes all statements that influence the criterion through
data-dependence or control-dependence chains, both intra-procedurally and
across function calls.

Algorithm (Weiser-style):
    1. Start with the trigger node in the worklist.
    2. Pop a node from the worklist.  If not yet visited:
       a. Add it to the slice set.
       b. Add all its **data-dependence predecessors** (defs of variables it
          uses) to the worklist.
       c. Add all its **control-dependence predecessors** (predicates that
          control its execution) to the worklist.
       d. If the node is a **function call**, recurse into the callee's PDG.
    3. When the worklist is empty, compute the **slice frontier**: variables
       used inside the slice but defined outside it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from llm_client.fp_analysis.pdg_models import (
    PDGNode,
    SDG,
    SliceNode,
    SliceResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BTBS: control predicate node kinds for Phase 3 post-processing
# ---------------------------------------------------------------------------
# Duplicated from path_seed_extractor._CONTROL_PREDICATE_KINDS to avoid
# a cross-module dependency on a private constant.  These are node kinds
# produced by the C++ PDG builder for branch-condition expressions.
_BTBS_CONTROL_KINDS: frozenset[str] = frozenset({
    "if_cond", "while_cond", "for_cond", "binary_op", "unary_op",
})


# ---------------------------------------------------------------------------
# Slicer
# ---------------------------------------------------------------------------


@dataclass
class SlicerConfig:
    """Configuration for the backward slicer.

    Attributes:
        max_depth: Maximum recursion depth for cross-function calls — counted in
            **cross-function hops only**.  Intra-procedural edges (data_dep and
            control_dep) do not consume it: a function's control-dependence
            chain has no relation to how deep the call stack goes, and charging
            it there truncated the slice on ordinary if/else-if ladders.
            When this limit is reached, the call is conservatively treated as
            "all actual params are inputs."
        include_call_context: If True, also include the **call-site node**
            in the caller when slicing into a callee.  This provides better
            context for understanding why a particular callee was reached.
    """

    max_depth: int = 10
    include_call_context: bool = True


def compute_slice(
    sdg: SDG,
    function_name: str,
    node_id: str,
    config: SlicerConfig | None = None,
    annotations: dict[tuple[str, int], bool] | None = None,
    caller_map: dict[str, str] | None = None,
) -> SliceResult:
    """Compute the backward slice from *(function_name, node_id)*.

    This is the **single-seed** entry point.  For multi-seed slicing from
    all bug-path events, use :func:`compute_slice_from_seeds` instead.

    Args:
        sdg: The System Dependence Graph containing PDGs and call graph.
        function_name: Qualified name of the function containing the trigger.
        node_id: Node ID of the slicing criterion (the bug trigger point).
        config: Optional configuration (depth limit, flags).
        annotations: Optional BTBS path annotations for branch-pruning.
        caller_map: Optional BTBS callee→caller map for cross-function
            reverse propagation.  Restricts reverse traversal to only the
            caller on the bug path.

    Returns:
        A :class:`SliceResult` containing all slice nodes, the frontier of
        input variables, and metadata.

    Raises:
        ValueError: If *function_name* is not found in the SDG.
        ValueError: If *node_id* is not found in the trigger function's PDG.
    """
    return compute_slice_from_seeds(
        sdg,
        seeds=[(function_name, node_id, "trigger")],
        config=config,
        annotations=annotations,
        caller_map=caller_map,
    )


def compute_slice_from_seeds(
    sdg: SDG,
    seeds: list[tuple[str, str, str]],
    config: SlicerConfig | None = None,
    annotations: dict[tuple[str, int], bool] | None = None,
    caller_map: dict[str, str] | None = None,
) -> SliceResult:
    """Compute the backward slice from **multiple seed points**.

    Initialises the traversal worklist with **all** seeds, then runs the
    standard Weiser-style backward slice through data-dependence and
    control-dependence edges across the SDG.

    When *annotations* is provided (from :func:`~path_seed_extractor.build_path_annotations`),
    **Bug-path-Targeted Backward Slicing (BTBS)** is enabled:

    - **Phase 2 (control-dep pruning)**: When following control-dependence
      edges, the slicer checks whether the controlling predicate is on a
      NOT-taken branch (according to the bug path).  If so, the edge is
      skipped — the dependent statement was not executed on the actual bug
      path.
    - **Phase 3 (post-processing)**: After traversal, any control-predicate
      node on a NOT-taken branch line is removed from the slice.  Its
      data-flow children (variable definitions reached through def-use
      edges) are preserved.
    - **Phase 4 (reverse propagation)**: When the slicer reaches a synthetic
      parameter node in a callee, it propagates backward to the **single**
      caller on the bug path (via *caller_map*), instead of searching *all*
      callers as standard inter-procedural slicing would.

    Using multiple seeds (e.g. one per bug-path event) ensures that
    variables that appear *earlier* in the bug execution path — control
    predicates, intermediate assignments — are included in the slice even
    if they do not directly appear at the final bug trigger point.

    Args:
        sdg: The System Dependence Graph.
        seeds: List of ``(function_name, node_id, reason)`` triples.  The
            reason is one of ``"trigger"``, ``"path_event"``,
            ``"path_control"``, ``"path_call"``, ``"path_report"``, or
            any descriptive label.
        config: Optional configuration (depth limit, flags).
        annotations: Optional mapping from ``(function_name, source_line)``
            to ``bool`` (True = branch taken, False = NOT taken).  When
            provided, enables BTBS pruning.
        caller_map: Optional mapping from ``callee_qualified_name`` to
            ``caller_qualified_name`` built from bug path "Calling" events.
            Enables BTBS callee→caller reverse propagation.

    Returns:
        A :class:`SliceResult` containing the union of all backward slices
        from each seed.

    Raises:
        ValueError: If *seeds* is empty.
        ValueError: If any seed's function is not found in the SDG.
        ValueError: If any seed's node ID is not found in its function's PDG.
    """
    if config is None:
        config = SlicerConfig()

    if not seeds:
        raise ValueError("Seed set is empty — at least one seed is required.")

    # Validate all seeds
    for fn_name, nid, _reason in seeds:
        pdg = sdg.get_pdg(fn_name)
        if pdg is None:
            raise ValueError(
                f"Function '{fn_name}' (from seed) not found in SDG. "
                f"Available functions: {list(sdg.function_names)[:20]}..."
            )
        if nid not in pdg.nodes:
            raise ValueError(
                f"Node '{nid}' not found in PDG for '{fn_name}' (from seed). "
                f"Available nodes: {list(pdg.nodes.keys())[:20]}..."
            )

    trigger_fn = seeds[0][0]
    # Use the path_report seed (bug trigger) as the canonical trigger node,
    # falling back to the first seed.
    trigger_nid = seeds[0][1]
    for s in seeds:
        if len(s) >= 3 and s[2] == "path_report":
            trigger_nid = s[1]
            break

    # ---- Core traversal (shared with single-seed) ----
    visited: set[tuple[str, str]] = set()  # (function_name, node_id)
    slice_nodes: list[SliceNode] = []
    worklist: list[tuple[str, str, str, int]] = [
        (fn, nid, reason, 0) for fn, nid, reason in seeds
    ]
    truncated = False

    while worklist:
        cur_fn, cur_id, reason, depth = worklist.pop()
        key = (cur_fn, cur_id)

        if key in visited:
            continue
        visited.add(key)

        if depth > config.max_depth:
            truncated = True
            # The edge kind matters: ``control_dep`` is *intra*-procedural, so a
            # deep if/else-if chain inside one function can burn the budget that
            # ``SlicerConfig.max_depth`` documents as the cross-function limit.
            # Logging it is what tells the two apart in a real report.
            logger.debug(
                "Slice depth limit (%d) reached at (%s, %s) via %s edge",
                config.max_depth, cur_fn, cur_id, reason,
            )
            continue

        cur_pdg = sdg.get_pdg(cur_fn)
        if cur_pdg is None:
            logger.debug("PDG not found for '%s' during slicing; skipping.", cur_fn)
            continue

        cur_node = cur_pdg.nodes.get(cur_id)
        if cur_node is None:
            logger.debug("Node '%s' not found in '%s'; skipping.", cur_id, cur_fn)
            continue

        slice_nodes.append(SliceNode(
            function_name=cur_fn,
            source_file=cur_pdg.source_file,
            node=cur_node,
            reason=reason,
        ))

        # ---- 1. Backward data dependence ----
        for pred_id in cur_pdg.data_preds_by_node.get(cur_id, []):
            worklist.append((cur_fn, pred_id, "data_dep", depth))

        # ---- 2. Backward control dependence (★ BTBS-pruned) ----
        for pred_id in cur_pdg.ctrl_preds_by_node.get(cur_id, []):
            # BTBS Phase 2: if the controlling predicate is on a NOT-taken
            # branch (according to the bug path annotations), skip this
            # edge — the current node's branch was NOT executed on the bug
            # path, so its predicate shouldn't be pulled in via control-dep.
            if annotations is not None:
                pred_node = cur_pdg.nodes.get(pred_id)
                if pred_node and _is_btbs_not_taken(
                    annotations, cur_fn, pred_node.source_line,
                ):
                    logger.debug(
                        "BTBS: skip ctrl-dep edge %s -> %s "
                        "(predicate at L%d is NOT-taken)",
                        pred_id, cur_id, pred_node.source_line,
                    )
                    continue
            # Depth is unchanged: this edge stays inside ``cur_pdg``, and
            # ``max_depth`` bounds the *cross-function* walk (see SlicerConfig).
            # Charging it here let a long if/else-if chain inside ONE function
            # exhaust the budget and set ``truncated`` — which makes every
            # slice-driven branch "relevant" (branch_relevance.branch_relevant
            # is fully conservative under truncation), so nothing is ever fixed
            # and the path space is never compressed.  On read_index-484-2 five
            # purely intra-procedural control-dep hops in faiss::read_index did
            # exactly that: 473 relevant / 0 irrelevant, reduced bound 2.06e25;
            # not counting them gives 199 / 274 and 5.16e11.
            worklist.append((cur_fn, pred_id, "control_dep", depth))

        # ---- 3. Cross-function propagation (caller → callee) ----
        # ★ BTBS: when caller_map is provided, forward propagation into
        # callees is disabled.  Cross-function edges are handled by
        # Phase 4 reverse propagation (callee→caller) via synthetic
        # param nodes — see step 4 below.
        if cur_node.is_call and cur_node.callee:
            _propagate_through_call(
                sdg, cur_fn, cur_node, cur_pdg,
                worklist, visited, config, depth,
                caller_map=caller_map,
            )

        # ---- 4. Reverse cross-function propagation (callee → caller) ★ BTBS ----
        # When we visit a synthetic parameter node (source_line=0) in a callee
        # that's on the bug path, propagate to the caller on the bug path
        # (instead of searching ALL callers like standard slicing would).
        if caller_map is not None and cur_fn in caller_map:
            _propagate_to_caller_if_needed(
                sdg, cur_fn, cur_node, cur_pdg,
                caller_map, worklist, visited, depth,
            )

        # ---- 5. Cross-function [this].field propagation (BTBS only) ----
        # When a node uses a [this].field pseudo-variable that has NO local
        # definition, walk up the caller_map chain to find the definition in
        # a same-class caller or a constructor call in the calling context.
        if caller_map is not None:
            _propagate_this_field_along_callchain(
                sdg, caller_map, cur_fn, cur_id, cur_pdg,
                worklist, visited, depth,
            )

    # ---- ★ BTBS Phase 3: Post-processing — remove not-taken predicates ----
    if annotations:
        before = len(slice_nodes)
        slice_nodes = _btbs_prune_not_taken_predicates(slice_nodes, annotations)
        if len(slice_nodes) < before:
            logger.info(
                "BTBS Phase 3: removed %d not-taken predicate node(s) "
                "from slice (%d → %d)",
                before - len(slice_nodes), before, len(slice_nodes),
            )

    # ---- Build the result ----
    involved: set[str] = set()
    for sn in slice_nodes:
        involved.add(sn.function_name)

    # Input variables: used in slice but not defined by any slice node
    all_defined: set[str] = set()
    all_used: set[str] = set()
    for sn in slice_nodes:
        n = sn.node
        if n.variable:
            all_defined.add(n.variable)
        # Find used variables from def-use edges within the same function
        fn_pdg = sdg.get_pdg(sn.function_name)
        if fn_pdg:
            for due in fn_pdg.def_use_edges:
                if due.use_node_id == n.id:
                    all_used.add(due.variable)

    input_vars = all_used - all_defined

    # Detect if any global/static variables are involved
    involves_globals = any(
        sn.node.variable.startswith("static_") or "." in sn.node.variable
        for sn in slice_nodes
    )

    return SliceResult(
        trigger_function=trigger_fn,
        trigger_node_id=trigger_nid,
        nodes=slice_nodes,
        input_variables=input_vars,
        involved_functions=involved,
        involves_globals=involves_globals,
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# BTBS helpers
# ---------------------------------------------------------------------------


def _is_btbs_not_taken(
    annotations: dict[tuple[str, int], bool],
    fn_name: str,
    source_line: int,
) -> bool:
    """Check if *source_line* in *fn_name* is annotated as a NOT-taken branch.

    Returns ``True`` when the bug path's CONTROL event at this line has
    ``branch_condition=False`` (the branch was NOT taken on the bug path).
    """
    return annotations.get((fn_name, source_line)) is False


def _btbs_prune_not_taken_predicates(
    slice_nodes: list[SliceNode],
    annotations: dict[tuple[str, int], bool],
) -> list[SliceNode]:
    """Remove control-predicate nodes on NOT-taken branches from the slice.

    Keeps the data-flow chain (variable definitions reached through the
    predicate's def-use edges) intact while removing the predicate node
    itself.  Only nodes whose PDG ``kind`` matches :data:`_BTBS_CONTROL_KINDS`
    are candidates for removal.

    Example (parseArg)::

        S_167_9 (binary_op "c == -1") at L167, NOT-taken  →  REMOVED
        S_166_5 (decl       "c = getopt_long") at L166     →  KEPT
        S_171_9 (binary_op "c == 0") at L171, TAKEN        →  KEPT
    """
    return [
        sn for sn in slice_nodes
        if not (
            annotations.get((sn.function_name, sn.node.source_line)) is False
            and sn.node.kind in _BTBS_CONTROL_KINDS
        )
    ]


# ---------------------------------------------------------------------------
# BTBS Phase 4 — Reverse cross-function propagation (callee → caller)
# ---------------------------------------------------------------------------
# Standard inter-procedural backward slicing, when it exhausts local
# dependencies in a function F, searches ALL callers of F for backward
# propagation.  This can explode for widely-used functions.
#
# BTBS uses the bug path's call chain (built from "Calling"/"Returning
# from" events) to restrict reverse propagation to ONLY the caller that
# actually executed on the bug path.
#
# The key insight: PDG synthetic parameter nodes (source_line=0) mark
# function entry points.  When the backward slicer visits one, it knows
# the variable is a formal parameter — the definition is in the CALLER,
# not the callee.  BTBS then uses the caller_map to find that one caller.


def _propagate_to_caller_if_needed(
    sdg: SDG,
    cur_fn: str,
    cur_node: PDGNode,
    cur_pdg,
    caller_map: dict[str, str],
    worklist: list,
    visited: set,
    depth: int,
) -> None:
    """BTBS Phase 4: propagate from a callee's parameter entry to its caller.

    Trigger conditions (all must hold):
    1. The current node is a **synthetic parameter node** (``source_line=0``).
       These are created by ``PDGBuilder.cpp`` at function entry points.
    2. The node's variable is in the callee's ``summary.input_variables``
       (i.e., it is a formal parameter).
    3. The callee is in *caller_map* (means it is on the bug path).

    When triggered, finds the call site in the bug-path caller, maps the
    formal parameter to its actual argument, and adds the actual argument's
    definition in the caller to the worklist.
    """
    # ── Condition 1: synthetic parameter node ──
    if cur_node.source_line != 0:
        return

    var = cur_node.variable
    if not var:
        return

    # ── Condition 2: variable is a formal parameter ──
    if var not in cur_pdg.summary.input_variables:
        return

    # ── Condition 3: this function is a callee on the bug path ──
    caller_fn = caller_map.get(cur_fn)
    if not caller_fn:
        return

    caller_pdg = sdg.get_pdg(caller_fn)
    if not caller_pdg:
        logger.debug(
            "BTBS Phase 4: caller '%s' not in SDG (callee '%s', var '%s')",
            caller_fn, cur_fn, var,
        )
        return

    # ── Find the call site in the caller ──
    call_site = _find_call_site_in_caller(sdg, caller_fn, cur_fn)
    if not call_site:
        logger.debug(
            "BTBS Phase 4: no call site for '%s' in '%s'",
            cur_fn, caller_fn,
        )
        return

    call_nid, call_node = call_site

    # ── Map formal → actual parameter by position ──
    try:
        formal_idx = cur_pdg.summary.input_variables.index(var)
    except ValueError:
        return

    if formal_idx >= len(call_node.actual_params):
        logger.debug(
            "BTBS Phase 4: formal param idx %d out of range (actual_params=%s)",
            formal_idx, call_node.actual_params,
        )
        return

    actual_param = call_node.actual_params[formal_idx]
    if not actual_param or actual_param == "?":
        return

    logger.debug(
        "BTBS Phase 4: %s:%s → %s(%s)  (formal=%s, actual=%s)",
        caller_fn.split("::")[-1], call_nid,
        cur_fn.split("::")[-1], ",".join(call_node.actual_params),
        var, actual_param,
    )

    # ── Add the actual param's definition in the caller ──
    def_ids = caller_pdg.defs_by_variable.get(actual_param, [])
    for def_id in def_ids:
        key = (caller_fn, def_id)
        if key not in visited:
            worklist.append(
                (caller_fn, def_id, "btbs_caller_param", depth + 1)
            )

    # Also add the call site itself for context
    call_key = (caller_fn, call_nid)
    if call_key not in visited:
        worklist.append(
            (caller_fn, call_nid, "call_site", depth)
        )


def _find_call_site_in_caller(
    sdg: SDG,
    caller_fn: str,
    callee_fn: str,
) -> tuple[str, PDGNode] | None:
    """Find the ``(node_id, PDGNode)`` in *caller_fn* that calls *callee_fn*.

    Tries exact callee name match first, then short-name suffix match
    (``ends_with("::" + short_name)``), and finally raw token match
    (handles template instantiations like ``putOptions<...>``).
    """
    caller_pdg = sdg.get_pdg(caller_fn)
    if not caller_pdg:
        return None

    callee_short = callee_fn.split("::")[-1]

    for nid, node in caller_pdg.nodes.items():
        if not node.is_call:
            continue
        caller_callee = node.callee or ""
        # Exact match
        if caller_callee == callee_fn:
            return (nid, node)
        # Short-name suffix match
        if caller_callee.endswith("::" + callee_short):
            return (nid, node)
        # Raw token match (handles templates)
        if caller_callee.split("<")[0] == callee_short:
            return (nid, node)
        if callee_short in caller_callee:
            # Check if it's really the same function (not a substring)
            caller_parts = caller_callee.split("::")
            if caller_parts and caller_parts[-1] == callee_short:
                return (nid, node)

    return None


def _propagate_through_call(
    sdg: SDG,
    caller_fn: str,
    call_node: PDGNode,
    caller_pdg,
    worklist: list,
    visited: set,
    config: SlicerConfig,
    current_depth: int,
    caller_map: dict[str, str] | None = None,
) -> None:
    """Handle cross-function propagation when the slicer reaches a call node.

    In **BTBS mode** (when *caller_map* is provided, i.e. non-None and
    non-empty), forward propagation into callees is **disabled**.  Cross-
    function edges are handled by Phase 4 reverse propagation from the
    callee side — the callee's backward slice (from its own bug-path seeds)
    reaches a synthetic param node (``source_line=0``), which triggers
    :func:`_propagate_to_caller_if_needed` to bridge back to the caller.

    In **standard mode** (when *caller_map* is None or empty), three cases:
    1. **PDG available** for the callee → recurse into its PDG, mapping
       actual→formal parameters.
    2. **Summary available** (from the function's summary in the SDG) →
       use the summary's input-output dependencies.
    3. **No information** → conservatively add ALL actual parameters as inputs.
    """
    callee = call_node.callee
    if not callee:
        return

    # ★ BTBS: skip forward propagation entirely — Phase 4 handles it
    if caller_map:
        logger.debug(
            "BTBS: skip forward propagation into '%s' (handled by Phase 4 "
            "reverse propagation from callee-side param nodes)",
            callee,
        )
        return

    callee_pdg = sdg.get_callee_pdg(caller_fn, callee)

    if callee_pdg is not None:
        # Case 1: Full PDG available — recurse into callee from the
        # definitions of output variables that the caller depends on.
        #
        # Strategy: for each actual parameter that's in the *current slice
        # frontier* (i.e., the caller uses it as input and it's also a formal
        # param of the callee), find the callee's definitions of that param
        # and slice from there.
        for i, actual_param in enumerate(call_node.actual_params):
            if actual_param == "?" or not actual_param:
                continue

            # Map to formal parameter (if we can)
            if i < len(callee_pdg.summary.input_variables):
                formal_param = callee_pdg.summary.input_variables[i]
            else:
                formal_param = actual_param

            # Find where this formal parameter is *used* in the callee
            for node_id, node in callee_pdg.nodes.items():
                # Check if this node uses the parameter
                for due in callee_pdg.def_use_edges:
                    if due.use_node_id == node_id and due.variable == formal_param:
                        key = (callee_pdg.function_name, node_id)
                        if key not in visited:
                            worklist.append(
                                (callee_pdg.function_name, node_id,
                                 "call_context", current_depth + 1)
                            )

        # Also include the return value definition in the callee
        # (if the caller uses the return value)
        for node_id, node in callee_pdg.nodes.items():
            if node.kind == "return":
                # Find the return value expression's dependencies
                for due in callee_pdg.def_use_edges:
                    if due.use_node_id == node_id:
                        key = (callee_pdg.function_name, due.def_node_id)
                        if key not in visited:
                            worklist.append(
                                (callee_pdg.function_name, due.def_node_id,
                                 "call_context", current_depth + 1)
                            )

        # Include the callee PDG's entry context if config allows
        if config.include_call_context:
            # Also note the call node in the caller as context
            pass  # (already in slice_nodes)
    else:
        # Case 2 / 3: No PDG — use function summary or be conservative.
        #   If the summary says "output depends on input P[i]", only add
        #   the relevant params.  Otherwise add ALL actual params.
        logger.debug(
            "No PDG for callee '%s' (caller=%s), using conservative fallback.",
            callee, caller_fn,
        )

        # Try to find a summary in the caller's own summary set
        # (the callee may not have a PDG but the caller's fn summary
        #  might document what flows through the call).
        for i, actual_param in enumerate(call_node.actual_params):
            if actual_param == "?" or actual_param == "":
                continue
            # Conservatively: find the definition of this actual param
            # in the caller and add it to the worklist
            for due in caller_pdg.def_use_edges:
                if due.use_node_id == call_node.id and due.variable == actual_param:
                    worklist.append(
                        (caller_fn, due.def_node_id, "data_dep", current_depth)
                    )
                    break
            else:
                # The actual param itself was not tracked in def-use edges;
                # try a broader search for its definition
                for node_id, node in caller_pdg.nodes.items():
                    if node.variable == actual_param:
                        worklist.append(
                            (caller_fn, node_id, "data_dep", current_depth)
                        )
                        break


def find_trigger_node(
    trigger_function: str,
    trigger_line: int,
    sdg: SDG,
) -> Optional[str]:
    """Find the best node ID for the slicing criterion given a source line.

    Looks for the PDG node in *trigger_function* that is closest to
    *trigger_line* (usually the error-report event or null-dereference site).

    Returns ``None`` if no suitable node can be found.
    """
    pdg = sdg.get_pdg(trigger_function)
    if pdg is None:
        return None

    # Find all nodes at or near the trigger line, preferring nodes that
    # actually define or use a variable (not just syntactic markers).
    candidates: list[tuple[str, int]] = []  # (node_id, distance)
    for nid, node in pdg.nodes.items():
        dist = abs(int(node.source_line) - trigger_line)
        if dist <= 5:  # within 5 lines
            # Prefer 'call_expr', 'assign', 'return', 'decl' kinds
            priority = 0
            if node.is_call or node.kind in ("assign", "return", "decl"):
                priority = 1
            candidates.append((nid, dist - priority))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Step 5 — Cross-function [this].field propagation (BTBS only)
# ---------------------------------------------------------------------------
# When the backward slicer encounters a [this].field pseudo-variable that
# has no local definition, this module traces upward through the caller_map
# chain to find the definition.
#
# Two propagation rules:
#
# 1. Same-class caller: if the caller is a method of the same class AND the
#    call is on "this" (implicit or explicit), then [this].field in the
#    callee refers to the same object as in the caller — propagate the
#    field-definition search to the caller.
#
# 2. Constructor trace: if the caller constructs an object on which it then
#    calls the callee, find the constructor's [this].field definition and
#    connect it.
# ---------------------------------------------------------------------------


def _propagate_this_field_along_callchain(
    sdg: SDG,
    caller_map: dict[str, str],
    cur_fn: str,
    cur_id: str,
    cur_pdg: FunctionPDG,
    worklist: list,
    visited: set,
    depth: int,
) -> None:
    """Step 5: propagate unresolved ``[this].field`` uses up the call chain.

    Trigger condition: the current node uses a ``[this].field`` pseudo-
    variable that has no definition in the current function's PDG (i.e. it
    is an "imported" field value set by another function).

    Walks up the ``caller_map`` chain (BTBS bug-path call chain) and at
    each level applies two rules to find the field's definition.
    """
    # ── Step 5a: find [this].field variables used but not locally defined ──
    unresolved: set[str] = set()
    for due in cur_pdg.def_use_edges:
        if due.use_node_id == cur_id and due.variable.startswith("[this]."):
            if not cur_pdg.defs_by_variable.get(due.variable, []):
                unresolved.add(due.variable)

    if not unresolved:
        return

    logger.debug(
        "Step 5: %d unresolved [this].field var(s) at %s:%s",
        len(unresolved), cur_fn, cur_id,
    )

    added: set[tuple[str, str]] = set()

    # Walk up the caller_map chain.  Guard against cycles: a callee may
    # appear at multiple depths on the bug path, so caller_map (callee→caller,
    # last-write-wins) can form a cycle under recursion/mutual recursion.
    seen_callers: set[str] = set()
    fn = cur_fn
    while fn in caller_map:
        if fn in seen_callers:  # cycle guard (mutual recursion)
            break
        seen_callers.add(fn)
        caller_fn = caller_map[fn]
        if caller_fn == fn:  # safety: self-loop guard
            break
        caller_pdg = sdg.get_pdg(caller_fn)
        if caller_pdg is None:
            break

        # ── Rule 1: Same-class caller shares 'this' ────────────────────
        # If both caller and callee are methods of the same class, and the
        # call was on 'this' (or implicit this), the [this].field in the
        # callee refers to the SAME memory as in the caller.
        if _same_class(caller_fn, fn):
            # Verify the call is on 'this' (not on a different object)
            site = _find_call_site_in_caller(sdg, caller_fn, fn)
            obj = None
            if site:
                obj = _extract_object_from_expr(
                    site[1].expression or "", fn
                )
            is_this_call = (obj is None or obj.lower() == "this")

            if is_this_call:
                for field_var in unresolved:
                    for def_id in _get_field_def_ids(caller_pdg, field_var):
                        key = (caller_fn, def_id)
                        if key not in visited and key not in added:
                            worklist.append(
                                (caller_fn, def_id, "cross_fn_field", depth + 1)
                            )
                            added.add(key)
                if added:
                    logger.debug(
                        "  Rule 1 (same-class): %d def(s) from %s",
                        len([k for k in added if k[0] == caller_fn]),
                        caller_fn,
                    )

        # ── Rule 2: Constructor trace ─────────────────────────────────
        # Find the call site for 'fn' in 'caller_fn', extract the object
        # expression, trace its definition in the caller. If the definition
        # is a constructor call for the same class, connect its field defs.
        class_name = _get_class_name(fn)
        if class_name:
            site = _find_call_site_in_caller(sdg, caller_fn, fn)
            if site:
                call_nid, call_node = site
                obj = _extract_object_from_expr(
                    call_node.expression or "", fn
                )
                if obj and obj.lower() != "this":
                    for def_id in caller_pdg.defs_by_variable.get(obj, []):
                        def_node = caller_pdg.nodes.get(def_id)
                        if def_node and def_node.is_call:
                            if _is_constructor(def_node.callee, class_name):
                                ctor_fn = def_node.callee
                                ctor_pdg = sdg.get_pdg(ctor_fn)
                                if ctor_pdg:
                                    for field_var in unresolved:
                                        for cd_id in _get_field_def_ids(
                                                ctor_pdg, field_var):
                                            key = (ctor_fn, cd_id)
                                            if key not in visited and key not in added:
                                                worklist.append(
                                                    (ctor_fn, cd_id,
                                                     "cross_fn_field",
                                                     depth + 1)
                                                )
                                                added.add(key)
                                    # Also add the constructor call site
                                    # in the caller for context
                                    ctor_call_key = (caller_fn, def_id)
                                    if ctor_call_key not in visited and \
                                            ctor_call_key not in added:
                                        worklist.append(
                                            (caller_fn, def_id,
                                             "cross_fn_ctor", depth)
                                        )
                                        added.add(ctor_call_key)
                                if added:
                                    logger.debug(
                                        "  Rule 2 (constructor): %d def(s) "
                                        "from %s via %s",
                                        len([k for k in added
                                             if k[0] == ctor_fn]),
                                        ctor_fn, obj,
                                    )

        fn = caller_fn


# ---------------------------------------------------------------------------
# Helpers for cross-function [this].field propagation
# ---------------------------------------------------------------------------


def _get_field_def_ids(
    pdg: FunctionPDG, field_var: str,
) -> list[str]:
    """Return node IDs that define *field_var* in *pdg*.

    Checks both ``defs_by_variable`` (built from ``node.variable``) and
    ``def_use_edges`` directly.  The latter is needed for ``[this].field``
    pseudo-variables created by the augmentation pass — these are stored as
    edge variables rather than node variables.
    """
    ids: list[str] = []
    ids.extend(pdg.defs_by_variable.get(field_var, []))
    for due in pdg.def_use_edges:
        if due.variable == field_var and due.def_node_id:
            if due.def_node_id not in ids:
                ids.append(due.def_node_id)
    return ids


def _get_class_name(qualified_fn: str) -> str | None:
    """Extract the class name from a qualified function name.

    Returns ``None`` for free functions (no ``::`` qualifier).

    Examples::

        "DefaultPieceStorage::setBitfield"       → "DefaultPieceStorage"
        "aria2::BitfieldMan::setBitfield"         → "aria2::BitfieldMan"
        "aria2::BitfieldMan::BitfieldMan"         → "aria2::BitfieldMan"
        "freeFunction"                             → None
    """
    parts = qualified_fn.split("::")
    if len(parts) >= 2:
        return "::".join(parts[:-1])
    return None


def _short_name(qualified_fn: str) -> str:
    """Return the last component of a qualified function name.

    "DefaultPieceStorage::setBitfield"  → "setBitfield"
    "aria2::BitfieldMan::getBitfield"   → "getBitfield"
    """
    # Strip template parameters for comparison
    return qualified_fn.split("::")[-1].split("<")[0]


def _same_class(fn_a: str, fn_b: str) -> bool:
    """Check if *fn_a* and *fn_b* are methods of the same class."""
    cls_a = _get_class_name(fn_a)
    cls_b = _get_class_name(fn_b)
    return bool(cls_a and cls_b and cls_a == cls_b)


def _extract_object_from_expr(
    expr: str, callee_name: str,
) -> str | None:
    """Extract the object expression from a method-call expression.

    Examples::

        "storage->setBitfield(...)"   →  "storage"
        "this->setBitfield(...)"      →  "this"
        "tempBitfield.setBitfield()"  →  "tempBitfield"
        "setBitfield(...)"            →  None (implicit this)

    Uses a simple string search for the callee short name preceded by
    ``->`` or ``.``.
    """
    if not expr or not callee_name:
        return None
    short = _short_name(callee_name)
    if not short:
        return None

    # Search for "->" + short_name or "." + short_name
    for sep in (f"->{short}", f".{short}"):
        idx = expr.find(sep)
        if idx > 0:
            obj_text = expr[:idx].rstrip()
            # Extract the last word token
            import re
            words = re.findall(r"\b(\w[\w_]*)\b", obj_text)
            if words:
                return words[-1]

    return None


def _is_constructor(
    callee: str | None,
    class_name: str | None,
) -> bool:
    """Check if *callee* is a constructor of the class with *class_name*.

    The short name of a constructor equals the short name of the class.

    Examples::

        callee="BitfieldMan::BitfieldMan",  class_name="BitfieldMan"  → True
        callee="BitfieldMan::setBitfield",  class_name="BitfieldMan"  → False
    """
    if not callee or not class_name:
        return False
    cls_short = _short_name(class_name)
    callee_short = _short_name(callee)
    return bool(callee_short == cls_short)
