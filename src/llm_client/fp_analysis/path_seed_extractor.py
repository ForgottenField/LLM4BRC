"""Extract seed variable sets from CSA bug report path events for backward slicing.

Motivation
----------
The CSA bug report contains a sequence of path events that collectively describe
the symbolic execution path leading to the bug.  Each event may involve variables
(control conditions, data assignments, pointer operations) that are critical to
understanding *why* the path was taken.

Previously, the backward slicer started from only the **final bug trigger point**
(a single node).  This meant variables appearing earlier in the path — especially
control predicates that determined *which* branch was taken — could be missing
from the slice.

This module matches path events to PDG nodes using **only AST-derived metadata**
from the PDG (node ``kind``, ``variable``, ``is_call``, ``callee``, etc.).  No
source-code regex parsing is used — the PDG nodes themselves are the product of
Clang's AST, so querying them *is* AST-based analysis.

After the initial path-event→seed mapping, three enrichment passes run:

1. **Precise trigger extraction** — At report events describing "argument" or
   "parameter" issues, only the call expression (which carries actual parameter
   info) is included as a seed; incidental assignment nodes are excluded.

2. **Cross-function parameter tracing** — When a seed variable appears to be
   an actual parameter to a callee function, the corresponding formal parameter
   usage nodes in the callee's PDG are added to the seed set.

3. **Member variable dependency tracing** — When a seed variable iterates or
   derives from a class member (``handlers_.begin()``, ``this->field``),
   the member variable is recorded as a dependency.

Architecture
------------
::

    BugReport (HTML)  ──►  FPReportParser.parse()
                                  │
                           path_events: list[PathEvent]
                                  │
                                  ▼
                    extract_seeds_from_bug_path()
                          │
                          ├── Phase 1: Event→PDG matching
                          │    (via find_pdg_nodes_for_event)
                          │
                          ├── Phase 2: Cross-function param tracing
                          │    (via _enrich_with_cross_function_params)
                          │
                          └── Phase 3: Member dep tracing
                               (via _trace_member_dependencies)
                                  │
                                  ▼
                   seeds: list[(fn_name, node_id, reason)]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from llm_client.csa_analysis.models import BugPath, EventKind, PathEvent
from llm_client.fp_analysis.pdg_models import PDGNode, SDG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PDG node kinds that function as control predicates
# ---------------------------------------------------------------------------
_CONTROL_PREDICATE_KINDS: frozenset[str] = frozenset({
    "if_cond", "while_cond", "for_cond", "binary_op", "unary_op",
})

# PDG node kinds that are "trigger-like" (final report events)
_TRIGGER_LIKE_KINDS: frozenset[str] = frozenset({
    "deref", "call_expr", "assign", "return",
})


# ---------------------------------------------------------------------------
# File path matching
# ---------------------------------------------------------------------------


def _file_matches_event(pdg_file: str, event_file: str,
                        source_root: str | None) -> bool:
    """Check if a PDG's source file matches the event's file path.

    Tries multiple strategies (in order):
    1. Exact string equality.
    2. Filename (basename) equality.
    3. Common last-2-path-components suffix.
    4. Relative path resolution via *source_root*.
    """
    if not pdg_file or not event_file:
        return False
    if pdg_file == event_file:
        return True
    pdg_name = Path(pdg_file).name
    event_name = Path(event_file).name
    if pdg_name == event_name:
        return True
    # Common suffix (last 2+ components)
    pdg_parts = Path(pdg_file).parts
    event_parts = Path(event_file).parts
    min_len = min(len(pdg_parts), len(event_parts))
    if min_len >= 2:
        if (Path(*pdg_parts[-2:]).as_posix()
                == Path(*event_parts[-2:]).as_posix()):
            return True
    if source_root:
        root = Path(source_root).resolve()
        try:
            pdg_rel = Path(pdg_file).resolve().relative_to(root)
            event_rel = Path(event_file).resolve().relative_to(root)
            return str(pdg_rel) == str(event_rel)
        except (ValueError, OSError):
            pass
    return False


# ---------------------------------------------------------------------------
# PDG AST metadata matching
# ---------------------------------------------------------------------------


def _description_mentions(description: str, keyword: str) -> bool:
    """Case-insensitive substring check — no regex.

    CSA event descriptions are structured English like:
      - ``"'ptr' is null"``
      - ``"Assigning 'x' = 5"``
      - ``"Taking true branch"``
    """
    if not description or not keyword:
        return False
    return keyword.lower() in description.lower()


def _is_calling_event(description: str) -> bool:
    d = description.lower().strip()
    # Strip leading directional arrows (←, →)
    if d and not d[0].isalpha():
        d = d.lstrip("←→ \t")
    return (d.startswith("calling") or d.startswith("entering")
            or "called from" in d)


def _is_return_event(description: str) -> bool:
    d = description.lower().strip()
    if d and not d[0].isalpha():
        d = d.lstrip("←→ \t")
    return d.startswith("returning from")


def _node_matches_event(node: PDGNode, event: PathEvent) -> bool:
    """Check if a PDG node corresponds to a path event.

    Uses only PDG fields from Clang's AST: ``kind``, ``variable``,
    ``is_call``, ``callee``, ``expression``.

    Special handling for **report** events: when the bug description
    mentions "argument" or "parameter", only ``call_expr`` nodes (which
    carry ``actual_params``) are matched.  Assignment nodes that merely
    store the return value are excluded — the seed must name the *actual
    argument variable* that is problematic, not the return target.
    """
    desc = event.description

    # Control event
    if event.event_kind == EventKind.CONTROL:
        return node.kind in _CONTROL_PREDICATE_KINDS

    # Return event
    if _is_return_event(desc) and node.kind == "return":
        return True

    # Call event
    if _is_calling_event(desc):
        if node.is_call:
            if node.callee and _description_mentions(
                    desc, node.callee.split("::")[-1]):
                return True
            return True  # call node at the right line is good enough
        return False

    # ── Report event (the bug trigger) ──────────────────────────────
    if event.event_kind == EventKind.REPORT:
        # If the bug is about an argument/parameter, ONLY accept call_expr
        # nodes — they carry actual_params naming the problematic variable.
        # Skip assign/decl nodes that are just storage for the return value.
        if _description_mentions(desc, "argument") or _description_mentions(
                desc, "parameter"):
            return node.is_call

        # Otherwise accept trigger-like kinds
        if node.kind in _TRIGGER_LIKE_KINDS:
            return True
        if node.variable and _description_mentions(desc, node.variable):
            return True
        if node.expression and _description_mentions(desc,
                                                      node.expression[:40]):
            return True
        return False

    # ── Regular event ───────────────────────────────────────────────
    if node.variable and _description_mentions(desc, node.variable):
        return True
    if node.expression and _description_mentions(desc, node.expression[:40]):
        return True
    if node.is_call:
        return True

    return False


# ---------------------------------------------------------------------------
# Main matching entry point
# ---------------------------------------------------------------------------


def find_pdg_nodes_for_event(
    event: PathEvent,
    sdg: SDG,
    source_root: str | None = None,
    file_index: dict[str, list[FunctionPDG]] | None = None,
) -> list[tuple[str, str, str]]:
    """Find PDG node(s) in the SDG that correspond to a path event.

    Matching is entirely AST-based: queries the SDG for PDG nodes at the
    event's source line, then filters by node metadata.

    When *file_index* is provided (from :meth:`SDG.build_file_index`), the
    search is O(1) per file instead of O(N) over all 24K functions.
    """
    line = event.line_number
    if line <= 0:
        logger.debug("Event has no line number, skipping: %s",
                     event.description[:60])
        return []

    # Reason tag
    if event.event_kind == EventKind.CONTROL:
        reason = "path_control"
    elif event.event_kind == EventKind.REPORT:
        reason = "path_report"
    elif _is_calling_event(event.description):
        reason = "path_call"
    elif _is_return_event(event.description):
        reason = "path_return"
    else:
        reason = "path_event"

    # ---- Fast path: use file index (O(1) per file) ----
    if file_index is not None:
        return _find_nodes_via_index(
            event, sdg, source_root, file_index, line, reason,
        )

    # ---- Legacy path: scan all functions (O(N)) ----
    return _find_nodes_legacy(event, sdg, source_root, line, reason)


# ---------------------------------------------------------------------------
# Index-based fast path helpers
# ---------------------------------------------------------------------------


def _find_nodes_via_index(
    event: PathEvent,
    sdg: SDG,
    source_root: str | None,
    file_index: dict[str, list[FunctionPDG]],
    line: int,
    reason: str,
) -> list[tuple[str, str, str]]:
    """Fast path: match event using the file index.

    Instead of scanning all 24K functions, uses the file index to find only
    the functions whose ``source_file`` matches the event's ``file_path``,
    then checks nodes at the target line within those functions.

    Complexity: O(F_matched + N_matched) vs O(F_all + N_all).
    """
    # Find which file index key(s) match the event's file_path.
    # We only check unique file paths here (~760 vs 24K functions).
    matching_files: list[str] = []
    for sf in file_index:
        if _file_matches_event(sf, event.file_path, source_root):
            matching_files.append(sf)

    if not matching_files:
        return []

    candidates: list[tuple[str, str]] = []

    # ---- Step 1: Exact line match within matching files ----
    for sf in matching_files:
        for pdg in file_index[sf]:
            fn_name = pdg.function_name
            for node_id, node in pdg.nodes.items():
                if node.source_line == line:
                    candidates.append((fn_name, node_id))

    # ---- Step 1b: Line-range tolerance (±N lines) ----
    if not candidates:
        _LINE_TOLERANCE = 5
        for sf in matching_files:
            for pdg in file_index[sf]:
                fn_name = pdg.function_name
                for node_id, node in pdg.nodes.items():
                    if node.source_line <= 0:
                        continue
                    dist = abs(node.source_line - line)
                    if dist <= _LINE_TOLERANCE:
                        candidates.append((fn_name, node_id))
        if candidates:
            logger.debug(
                "Line-range match at %s:%d (±%d): %d candidates",
                event.file_path, line, _LINE_TOLERANCE, len(candidates),
            )

    if not candidates:
        logger.debug("No PDG nodes at %s:%d for: %s",
                     event.file_path, line, event.description[:60])
        return []

    # ---- Step 2: Filter via AST metadata ----
    results: list[tuple[str, str, str]] = []
    for fn_name, node_id in candidates:
        node = sdg.pdgs[fn_name].nodes[node_id]
        if _node_matches_event(node, event):
            results.append((fn_name, node_id, reason))

    # ---- Step 3: Fallback by event kind ----
    if not results:
        for fn_name, node_id in candidates:
            node = sdg.pdgs[fn_name].nodes[node_id]
            if (event.event_kind == EventKind.CONTROL
                    and node.kind in _CONTROL_PREDICATE_KINDS):
                results.append((fn_name, node_id, reason))
                break
            if (_is_calling_event(event.description) and node.is_call):
                results.append((fn_name, node_id, reason))
                break
            if (event.event_kind == EventKind.REPORT
                    and node.kind in _TRIGGER_LIKE_KINDS):
                results.append((fn_name, node_id, reason))
                break

    # ---- Step 4: Conservative — all candidates ----
    if not results:
        results.extend((fn, nid, reason) for fn, nid in candidates)

    return results


def _find_nodes_legacy(
    event: PathEvent,
    sdg: SDG,
    source_root: str | None,
    line: int,
    reason: str,
) -> list[tuple[str, str, str]]:
    """Legacy path: scan all functions O(N) — used when no index is available."""
    candidates: list[tuple[str, str]] = []

    # ---- Step 1: Candidates at this exact line ----
    for fn_name, pdg in sdg.pdgs.items():
        if not _file_matches_event(pdg.source_file, event.file_path,
                                   source_root):
            continue
        for node_id, node in pdg.nodes.items():
            if node.source_line == line:
                candidates.append((fn_name, node_id))

    # ---- Step 1b: Line-range tolerance (±N lines) ----
    if not candidates:
        _LINE_TOLERANCE = 5
        for fn_name, pdg in sdg.pdgs.items():
            if not _file_matches_event(pdg.source_file, event.file_path,
                                       source_root):
                continue
            for node_id, node in pdg.nodes.items():
                if node.source_line <= 0:
                    continue
                dist = abs(node.source_line - line)
                if dist <= _LINE_TOLERANCE:
                    candidates.append((fn_name, node_id))
        if candidates:
            logger.debug(
                "Line-range match at %s:%d (±%d): %d candidates",
                event.file_path, line, _LINE_TOLERANCE, len(candidates),
            )

    if not candidates:
        logger.debug("No PDG nodes at %s:%d for: %s",
                     event.file_path, line, event.description[:60])
        return []

    # ---- Step 2: Filter via AST metadata ----
    results: list[tuple[str, str, str]] = []
    for fn_name, node_id in candidates:
        node = sdg.pdgs[fn_name].nodes[node_id]
        if _node_matches_event(node, event):
            results.append((fn_name, node_id, reason))

    # ---- Step 3: Fallback by event kind ----
    if not results:
        for fn_name, node_id in candidates:
            node = sdg.pdgs[fn_name].nodes[node_id]
            if (event.event_kind == EventKind.CONTROL
                    and node.kind in _CONTROL_PREDICATE_KINDS):
                results.append((fn_name, node_id, reason))
                break
            if (_is_calling_event(event.description) and node.is_call):
                results.append((fn_name, node_id, reason))
                break
            if (event.event_kind == EventKind.REPORT
                    and node.kind in _TRIGGER_LIKE_KINDS):
                results.append((fn_name, node_id, reason))
                break

    # ---- Step 4: Conservative — all candidates ----
    if not results:
        results.extend((fn, nid, reason) for fn, nid in candidates)

    return results


# ---------------------------------------------------------------------------
# Fallback seed creation — trigger-only best-effort seed
# ---------------------------------------------------------------------------

# Line-number tolerance when matching fallback seeds (CSA report line
# numbers occasionally diverge from PDG source_line values).
_FALLBACK_LINE_TOLERANCE = 15


def _create_fallback_seed(
    trigger_fn: str,
    trigger_line: int,
    sdg: SDG,
    bug_path: BugPath | None = None,
) -> list[tuple[str, str, str]]:
    """Create best-effort seed(s) when event→PDG matching finds nothing.

    Strategy (tried in order):

    1. **Near-trigger-line** — PDG nodes within
       ``_FALLBACK_LINE_TOLERANCE`` lines of *trigger_line* in the
       trigger function.
    2. **Event-description match** — if the bug path has a report event
       with a description, search for PDG nodes whose ``expression``
       contains the description keyword.
    3. **First meaningful node** — the first non-recovery, non-parameter
       node in the trigger function's PDG.
    4. **Any first node** — the first node in the PDG (last resort).

    Returns:
        List of ``(fn_name, node_id, "fallback")`` triples, or empty if
        the trigger function itself cannot be located in the SDG.
    """
    pdg = sdg.get_pdg(trigger_fn)
    if pdg is None:
        logger.debug("Fallback: SDG has no function matching '%s'", trigger_fn)
        return []

    fn_name = pdg.function_name  # use the canonical qualified name

    # ── Strategy 1: near trigger line (closest within tolerance) ──
    best_near: tuple[int, str | None] = (999999, None)
    for node_id, node in pdg.nodes.items():
        if node.source_line <= 0:
            continue
        if node.kind == "param":
            continue  # skip formal-parameter bookkeeping nodes
        dist = abs(node.source_line - trigger_line)
        if dist < best_near[0]:
            best_near = (dist, node_id)

    if best_near[1] is not None and best_near[0] <= _FALLBACK_LINE_TOLERANCE:
        return [(fn_name, best_near[1], "fallback_near_line")]

    # ── Strategy 2: report-event description → expression match ──
    if bug_path:
        report_desc = ""
        for evt in bug_path.events:
            if evt.event_kind.value == "report":
                report_desc = evt.description.lower()
                break
        if report_desc:
            # Pick the first keyword from description (longest word ≥ 3 chars)
            words = sorted(
                [w for w in report_desc.split() if len(w) >= 3],
                key=len, reverse=True,
            )
            for word in words[:5]:
                for node_id, node in pdg.nodes.items():
                    if node.kind == "param":
                        continue
                    expr = (node.expression or "").lower()
                    if word in expr:
                        return [(fn_name, node_id, "fallback_desc")]
                # Also try against variable names
                for node_id, node in pdg.nodes.items():
                    if node.kind == "param":
                        continue
                    var = (node.variable or "").lower()
                    if word in var:
                        return [(fn_name, node_id, "fallback_desc_var")]

    # ── Strategy 3: first meaningful node ──
    meaningful_kinds = {"call_expr", "binary_op", "unary_op", "assign",
                        "decl", "return", "event", "deref",
                        "if_cond", "while_cond", "for_cond"}
    for node_id, node in pdg.nodes.items():
        if node.kind in meaningful_kinds and node.source_line > 0:
            return [(fn_name, node_id, "fallback_first_meaningful")]

    # ── Strategy 4: any first node ──
    for node_id, _node in pdg.nodes.items():
        return [(fn_name, node_id, "fallback_any")]

    return []


# ===================================================================
# Phase 2 — Cross-function parameter tracing
# ===================================================================


def _seed_variables(seeds: list[tuple[str, str, str]],
                    sdg: SDG) -> dict[str, set[str]]:
    """Extract the set of variable names per function from seeds."""
    result: dict[str, set[str]] = {}
    for fn_name, node_id, _reason in seeds:
        pdg = sdg.get_pdg(fn_name)
        if pdg and node_id in pdg.nodes:
            v = pdg.nodes[node_id].variable
            if v:
                result.setdefault(fn_name, set()).add(v)
    return result


def _related_param_name(source_var: str, candidate: str) -> bool:
    """Heuristic: does *candidate* relate to *source_var*?

    Used to detect cross-function parameter mappings where a variable
    name changes slightly across a call boundary::

        caller passes ``lopt``  →  callee names it ``plopt``
        caller passes ``handlers_.begin()`` →  callee names it ``first``
    """
    if not source_var or not candidate:
        return False
    if source_var == candidate:
        return True
    # Shared root (at least 3 chars in common)
    # e.g. "lopt" ↔ "plopt", "handler" ↔ "handlers_"
    shorter, longer = (source_var.lower(), candidate.lower())
    if len(shorter) > len(longer):
        shorter, longer = longer, shorter
    # Check if the shorter is a substring of the longer
    if shorter in longer:
        return True
    # Check common prefix of 3+ chars
    for i in range(min(len(shorter), len(longer)), 2, -1):
        if shorter[:i] == longer[:i]:
            return True
    return False


def _enrich_with_cross_function_params(
    seeds: list[tuple[str, str, str]],
    sdg: SDG,
    bug_path: BugPath | None = None,
) -> list[tuple[str, str, str]]:
    """Enrich seeds by tracing cross-function parameter mappings.

    For each seed variable V in function F, this pass looks for
    **call_expr nodes in F's PDG** where V appears as an actual
    parameter.  For each such call, the callee's PDG is located and
    nodes that define or use the corresponding formal parameter are
    added as seeds.

    **Call-site-free fallback** — When the PDG lacks an explicit
    call node (e.g. due to recovery expressions), the function
    searches the bug report's path events for "Calling" events and
    uses the function summary to infer the relationship.

    Example::

        ┌─ parseArg ──────────────────────┐
        │  int lopt;          ← seed      │
        │  putOptions(&lopt, ...)          │  ← call_expr with actual_params=["lopt"]
        └──────────────────────────────────┘
                      │  actual→formal mapping
                      ▼
        ┌─ putOptions ─────────────────────┐
        │  int* plopt;   ← new seed        │
        │  (*longOpts).flag = plopt        │
        └──────────────────────────────────┘
    """
    seed_vars = _seed_variables(seeds, sdg)
    if not seed_vars:
        return []

    new_seeds: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = {(s[0], s[1]) for s in seeds}

    # ── A. PDG call-site-based tracing ──────────────────────────
    for fn_name, vars_set in seed_vars.items():
        fn_pdg = sdg.get_pdg(fn_name)
        if not fn_pdg:
            continue

        # Find all call_expr nodes in this function
        for nid, call_node in fn_pdg.nodes.items():
            if not call_node.is_call or not call_node.callee:
                continue

            # Check if any actual param matches a seed variable
            for sv in vars_set:
                if sv not in call_node.actual_params:
                    continue

                callee_pdg = sdg.get_callee_pdg(fn_name,
                                                 call_node.callee)
                if not callee_pdg:
                    continue

                # Map actual→formal by position
                idx = call_node.actual_params.index(sv)
                formal_vars: list[str] = []
                if idx < len(callee_pdg.summary.input_variables):
                    formal_vars.append(
                        callee_pdg.summary.input_variables[idx])

                # Also try to find any formal param name related to sv
                for iv in callee_pdg.summary.input_variables:
                    if _related_param_name(sv, iv):
                        formal_vars.append(iv)

                # Add nodes in callee that involve the formal var
                for fv in set(formal_vars):
                    for cid, cnode in callee_pdg.nodes.items():
                        key = (callee_pdg.function_name, cid)
                        if key in seen:
                            continue
                        if cnode.variable == fv:
                            seen.add(key)
                            new_seeds.append(
                                (callee_pdg.function_name, cid,
                                 f"cross_fn_param({sv}->{fv})"))

    # ── B. Bug-report fallback (when PDG call nodes are missing) ─
    # If the PDG is incomplete (recovery-expr, missing call nodes),
    # use the bug report's "Calling" events to infer caller-callee
    # relationships and then map via function summaries.
    if bug_path and seed_vars:
        # Collect all called function names from "Calling" events
        # CSA formats: "Calling 'putOptions<...>'" → extract "putOptions"
        calling_tokens: set[str] = set()
        for evt in bug_path.events:
            desc = evt.description or ""
            if _is_calling_event(desc):
                # Extract function name before '('
                for token in desc.replace("'", "").split():
                    clean = token.split("(")[0].split("<")[0]
                    if clean and clean[0].isalpha():
                        calling_tokens.add(clean)

        for fn_name, vars_set in seed_vars.items():
            seed_pdg_local = sdg.get_pdg(fn_name)
            if not seed_pdg_local:
                continue
            seed_file = seed_pdg_local.source_file

            for callee_token in calling_tokens:
                # Try to match the token to a PDG function name
                callee_pdg = sdg.get_pdg(callee_token)
                if not callee_pdg:
                    for qname, cpdg in sdg.pdgs.items():
                        if cpdg.function_name.endswith("::" + callee_token):
                            callee_pdg = cpdg
                            break
                if not callee_pdg:
                    continue
                # Exclude self-match (same function)
                if callee_pdg.function_name == fn_name:
                    continue
                # Only match functions in the same file
                if Path(callee_pdg.source_file).name != Path(
                        seed_file).name:
                    continue

                for sv in vars_set:
                    for iv in callee_pdg.summary.input_variables:
                        if not _related_param_name(sv, iv):
                            continue
                        for cid, cnode in callee_pdg.nodes.items():
                            key = (callee_pdg.function_name, cid)
                            if key in seen:
                                continue

                            # Match by: variable field, expression text,
                            # or def-use edge reference
                            matched = False
                            if cnode.variable == iv:
                                matched = True
                            elif iv in (cnode.expression or ""):
                                matched = True
                            else:
                                for due in callee_pdg.def_use_edges:
                                    if due.variable == iv:
                                        if due.use_node_id == cid or due.def_node_id == cid:
                                            matched = True
                                            break

                            if matched:
                                seen.add(key)
                                new_seeds.append(
                                    (callee_pdg.function_name, cid,
                                     f"cross_fn_report({sv}->{iv})"))

    logger.debug("Cross-function param enrichment: +%d seeds",
                 len(new_seeds))
    return new_seeds


# ===================================================================
# Phase 3 — Member variable dependency tracing
# ===================================================================


def _trace_member_dependencies(
    seeds: list[tuple[str, str, str]],
    sdg: SDG,
) -> list[tuple[str, str, str]]:
    """Trace member variable dependencies from seed variables.

    When a seed variable V is defined via member-access expressions
    (``obj.method()``, ``obj.field``, ``*first`` → ``handlers_.begin()``),
    this pass finds the PDG node(s) in the *same function* that provide
    V's definition and checks whether the definition involves a member
    of the enclosing class.

    If a member reference is found (e.g. ``this->handlers_``), a seed is
    added for the definition node so the backward slicer reaches it.

    For iterators like ``first`` (whose value ``handlers_.begin()`` is
    set by the *caller*, not the current function), we trace the call
    site in the caller's PDG.
    """
    new_seeds: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = {(s[0], s[1]) for s in seeds}

    _MEMBER_INDICATORS = ("this->", "this.", "->begin()", ".begin()",
                          "->end()", ".end()", "handlers_", "members_",
                          "fields_", "children_", "items_")

    for fn_name, node_id, _reason in seeds:
        pdg = sdg.get_pdg(fn_name)
        if not pdg or node_id not in pdg.nodes:
            continue

        node = pdg.nodes[node_id]

        # ---- A. Check the seed node's expression for member access ----
        expr = node.expression or ""
        has_member_ref = any(marker in expr for marker in _MEMBER_INDICATORS)

        if has_member_ref:
            logger.debug("Member reference in seed %s :: %s: %s",
                         fn_name, node_id, expr[:60])

        # ---- B. Follow def-use edges backward from this seed ----
        # If the seed is a use (uses variable V), find the def of V
        # and check if the def's expression involves member access.
        if node.variable:
            # Find all defs of this variable
            def_ids = pdg.get_definitions_of(node.variable)
            for did in def_ids:
                if did == node_id:
                    continue
                def_node = pdg.nodes.get(did)
                if not def_node:
                    continue
                def_expr = def_node.expression or ""
                if any(marker in def_expr for marker in _MEMBER_INDICATORS):
                    key = (fn_name, did)
                    if key not in seen:
                        seen.add(key)
                        new_seeds.append(
                            (fn_name, did, "member_dep"))

        # ---- C. For call_expr seeds, check actual params ----
        if node.is_call:
            for ap in node.actual_params:
                if ap == "?" or not ap:
                    continue
                # Check if this actual param is defined via member access
                def_ids = pdg.get_definitions_of(ap)
                for did in def_ids:
                    def_node = pdg.nodes.get(did)
                    if not def_node:
                        continue
                    def_expr = def_node.expression or ""
                    if any(marker in def_expr
                           for marker in _MEMBER_INDICATORS):
                        key = (fn_name, did)
                        if key not in seen:
                            seen.add(key)
                            new_seeds.append(
                                (fn_name, did, f"member_dep({ap})"))

    logger.info("Member dependency tracing: +%d seeds", len(new_seeds))
    return new_seeds


# ===================================================================
# Path Annotations — BTBS foundation
# ===================================================================


def build_path_annotations(
    bug_path: BugPath,
    sdg: SDG,
    source_root: str | None = None,
    file_index: dict[str, list[FunctionPDG]] | None = None,
) -> dict[tuple[str, int], bool]:
    """Build a mapping of branch decisions from bug path CONTROL events.

    Each CONTROL event with an explicit ``branch_condition`` records whether
    a source-level predicate took the **true** or **false** branch on the
    actual bug execution path.

    The returned mapping is keyed by ``(qualified_function_name, source_line)``
    and is the foundation for **Bug-path-Targeted Backward Slicing (BTBS)**.

    BTBS uses these annotations in two ways:

    1. **Phase 2 (control-dep pruning)**: When the backward slicer follows
       a control-dependence edge, it checks whether the *controlling predicate*
       is on a NOT-taken branch.  If so, the edge is skipped — the dependent
       statement is on a branch of the predicate that was not executed on the
       bug path.

    2. **Phase 3 (post-processing removal)**: After traversal, any control
       predicate node that sits on a NOT-taken branch line is removed from
       the slice.  Its data-flow children (variable definitions reached via
       def-use edges) are *preserved* — they are on the actual bug path.

    Example (parseArg)::

        # CONTROL event: L167 "Taking false branch", branch_condition=False
        annotations[("aria2::OptionParser::parseArg", 167)] = False  # c == -1 NOT taken
        annotations[("aria2::OptionParser::parseArg", 171)] = True   # c == 0  taken

    Args:
        bug_path: Structured bug path.
        sdg: System Dependence Graph.
        source_root: Project root for file path resolution.
        file_index: Optional pre-built file index from
            :meth:`SDG.build_file_index`.  When provided, avoids scanning
            all 24K functions for each event.

    Returns:
        A dict ``{(function_name, line): branch_decision}``, where
        ``True`` = branch taken, ``False`` = branch NOT taken.
    """
    annotations: dict[tuple[str, int], bool] = {}

    # Use file_index if available (fast path), otherwise fall back to full scan
    if file_index is not None:
        ann_sources: list[tuple[str, FunctionPDG]] = [
            (fn_name, pdg)
            for file_pdgs in file_index.values()
            for pdg in file_pdgs
            for fn_name in [pdg.function_name]
        ]
        # Build a reverse mapping of file → list of (fn_name, pdg) for matching
        _iter_over = file_index
    else:
        _iter_over = None

    for event in bug_path.events:
        if event.event_kind != EventKind.CONTROL:
            continue
        if event.branch_condition is None:
            continue

        line = event.line_number
        if line <= 0:
            continue

        if file_index is not None:
            # Fast path: only scan PDGs from matching files
            matching_sfs = [
                sf for sf in file_index
                if _file_matches_event(sf, event.file_path, source_root)
            ]
            matched = False
            for sf in matching_sfs:
                if matched:
                    break
                for pdg in file_index[sf]:
                    if matched:
                        break
                    fn_name = pdg.function_name
                    for node_id, node in pdg.nodes.items():
                        if node.source_line != line:
                            continue
                        if node.kind in _CONTROL_PREDICATE_KINDS:
                            annotations[(fn_name, line)] = event.branch_condition
                            matched = True
                            break
        else:
            # Legacy path: scan all functions
            for fn_name, pdg in sdg.pdgs.items():
                if not _file_matches_event(pdg.source_file, event.file_path,
                                           source_root):
                    continue
                for node_id, node in pdg.nodes.items():
                    if node.source_line != line:
                        continue
                    if node.kind in _CONTROL_PREDICATE_KINDS:
                        key = (fn_name, line)
                        annotations[key] = event.branch_condition
                        break
                else:
                    continue
                break  # first matching function wins

    logger.info("BTBS path annotations: %d branch decisions",
                len(annotations))
    for (fn, ln), taken in sorted(annotations.items()):
        logger.debug("  %s:L%d → %s", fn.split("::")[-1], ln,
                     "TAKEN" if taken else "NOT TAKEN")

    return annotations


# "← Assuming the condition is false →" / "← Taking true branch →"
_CONDITION_DIRECTION_RE = re.compile(
    r"(?:Taking|Assuming the condition is)\s+(true|false)",
    re.IGNORECASE,
)


def build_dense_path_annotations(
    bug_path: BugPath,
    sdg: SDG,
    source_root: str | None = None,
    file_index: dict[str, list[FunctionPDG]] | None = None,
) -> dict[tuple[str, int], bool]:
    """Like :func:`build_path_annotations`, but reads *every* CSA event.

    ``build_path_annotations`` only accepts ``CONTROL`` events that already
    carry a parsed ``branch_condition``, which silently drops the
    ``← Assuming the condition is false →`` events Clang emits for each operand
    of a short-circuited ``a || b`` test.  On ``index_read.cpp``'s fourcc
    ``else if`` chain that loses ``L836``/``L837``/``L838`` — exactly the lines
    whose then-bodies we must not expand.

    This variant also mines the direction from the event *text*, and marks a
    ``(function, line)`` as undecided when the two sources disagree (an
    ``a || b`` test can report both directions for the same line, and there is
    no way to tell which operand each refers to).

    Returns the same shape as :func:`build_path_annotations`:
    ``{(qualified_function_name, line): branch_taken}``.
    """
    directions: dict[int, set[bool]] = {}

    def _note(line: int, taken: bool) -> None:
        if line > 0:
            directions.setdefault(line, set()).add(taken)

    for event in bug_path.events:
        taken: bool | None = None
        if event.event_kind == EventKind.CONTROL:
            taken = event.branch_condition
        elif event.branch_condition is not None:
            taken = event.branch_condition
        if taken is None:
            m = _CONDITION_DIRECTION_RE.search(event.description or "")
            if m:
                taken = m.group(1).lower() == "true"
        if taken is not None:
            _note(event.line_number, taken)

    annotations: dict[tuple[str, int], bool] = {}
    ambiguous = 0
    for event in bug_path.events:
        line = event.line_number
        if line <= 0 or line not in directions:
            continue
        line_dirs = directions[line]
        if len(line_dirs) != 1:
            ambiguous += 1
            continue
        taken = next(iter(line_dirs))
        key = _resolve_predicate_fn(sdg, line, event.file_path, source_root,
                                    file_index)
        if key is not None:
            annotations[(key, line)] = taken

    logger.info("Dense path annotations: %d branch decisions from %d event line(s)"
                " (%d ambiguous line(s) skipped)",
                len(annotations), len(directions), ambiguous)
    return annotations


def _resolve_predicate_fn(
    sdg: SDG,
    line: int,
    event_file: str,
    source_root: str | None,
    file_index: dict[str, list[FunctionPDG]] | None,
) -> str | None:
    """Qualified name of the function holding a control predicate at *line*.

    Same file-restricted PDG scan ``build_path_annotations`` performs, factored
    out so both callers share one implementation.
    """
    if file_index is not None:
        candidates = (
            (pdg.function_name, pdg)
            for sf, pdgs in file_index.items()
            if _file_matches_event(sf, event_file, source_root)
            for pdg in pdgs
        )
    else:
        candidates = (
            (fn_name, pdg)
            for fn_name, pdg in sdg.pdgs.items()
            if _file_matches_event(pdg.source_file, event_file, source_root)
        )
    for fn_name, pdg in candidates:
        for node in pdg.nodes.values():
            if node.source_line == line and node.kind in _CONTROL_PREDICATE_KINDS:
                return fn_name
    return None


# ===================================================================
# Call Chain — BTBS cross-function reverse propagation
# ===================================================================


def _extract_fn_name_from_event(description: str) -> str | None:
    """Extract the function name from a "Calling 'foo'" event description."""
    d = description.strip()
    # Strip leading directional arrows
    if d and not d[0].isalpha():
        d = d.lstrip("←→ \t")
    for prefix in ("Calling ", "Entering ", "called from "):
        if d.lower().startswith(prefix.lower()):
            name = d[len(prefix):].strip().split("(")[0].split("<")[0]
            name = name.strip("'\"")
            # Strip trailing arrows/direction markers that may remain
            # after "→" prevents .strip("'\"") from removing a trailing "'"
            name = name.rstrip("→← \t").strip("'\"").strip()
            # Handle "constructor for 'ClassName'" → "ClassName::ClassName"
            if name.lower().startswith("constructor for "):
                cls_name = name[len("constructor for "):].strip()
                cls_name = cls_name.strip("'\"").strip()
                if cls_name:
                    name = f"{cls_name}::{cls_name}"
            if name:
                return name
    # Token-based fallback (used by _enrich_with_cross_function_params)
    for token in d.replace("'", "").split():
        clean = token.split("(")[0].split("<")[0]
        if clean and clean[0].isalpha():
            return clean
    return None


def _resolve_fn_name(name: str, sdg: SDG) -> str | None:
    """Resolve an unqualified function name to a qualified SDG name.

    Tries exact match first, then ``ends_with("::" + name)``, then
    last-component match.  Returns ``None`` if no match is found.
    """
    pdg = sdg.get_pdg(name)
    if pdg:
        return pdg.function_name
    # Broader search (handles template args, anonymous namespaces)
    for qname, cpdg in sdg.pdgs.items():
        if cpdg.function_name.endswith("::" + name):
            return cpdg.function_name
        parts = qname.split("::")
        if parts and parts[-1] == name:
            return qname
    return None


def build_caller_map(
    bug_path: BugPath,
    sdg: SDG,
    entry_fn: str | None = None,
) -> dict[str, str]:
    """Build a **callee→caller** map from the bug path's calling events.

    In standard inter-procedural backward slicing, when the slicer reaches
    a function's parameter entry point, it must search ALL callers of that
    function.  This can explode for widely-used functions.

    **BTBS optimisation**: by recording the *single* call chain on the bug
    path, BTBS restricts reverse propagation to only the caller that
    actually executed on the bug path.

    How it works
    ------------
    The bug path contains ``"Calling 'foo'"`` and ``"Returning from 'foo'"``
    events.  This function walks the events maintaining a call stack::

        ┌─ Events                              ┌─ Call stack (push/pop)
        │  [2] Calling 'putOptions'      ──→   │  [parseArg, putOptions]
        │  [5] Returning from putOptions ──→   │  [parseArg]

    The result is a ``{callee: caller}`` mapping::

        {"aria2::(anonymous namespace)::putOptions": "aria2::OptionParser::parseArg"}

    Args:
        bug_path: The parsed bug path with execution events.
        sdg: System Dependence Graph (for name resolution).
        entry_fn: Qualified name of the entry function (from bug report
            metadata).  If ``None``, inferred from the first calling event.

    Returns:
        A dict ``{callee_qualified_name: caller_qualified_name}``.  Empty
        when the bug path has no calling events.
    """
    if not bug_path.events:
        return {}

    caller_map: dict[str, str] = {}
    # Resolve entry_fn to its fully-qualified SDG name so that any
    # caller_map values match the PDG's function_name format.
    resolved_entry = (
        _resolve_fn_name(entry_fn, sdg)
        if entry_fn is not None else None
    ) or entry_fn
    call_stack: list[str | None] = [resolved_entry]

    for event in bug_path.events:
        desc = event.description or ""

        if _is_calling_event(desc):
            callee_name = _extract_fn_name_from_event(desc)
            if not callee_name:
                continue
            callee_qname = _resolve_fn_name(callee_name, sdg)
            if not callee_qname:
                logger.debug("BTBS caller_map: could not resolve '%s'", callee_name)
                call_stack.append(callee_name)  # use raw name
                continue

            current_caller = call_stack[-1] if call_stack else None
            if current_caller is not None:
                caller_map[callee_qname] = current_caller
                logger.debug(
                    "BTBS caller_map: %s → %s", current_caller, callee_qname,
                )
            call_stack.append(callee_qname)

        elif _is_return_event(desc):
            if len(call_stack) > 1:
                call_stack.pop()

    logger.info("BTBS caller_map: %d callee→caller edge(s)", len(caller_map))
    return caller_map


def is_not_taken_branch(
    annotations: dict[tuple[str, int], bool] | None,
    fn_name: str,
    source_line: int,
) -> bool:
    """Check if *source_line* in *fn_name* is annotated as a NOT-taken branch.

    Convenience wrapper for :func:`build_path_annotations` consumers (the
    backward slicer) so they don't need to query the :class:`dict` directly.
    Returns ``False`` when *annotations* is ``None`` (BTBS disabled).
    """
    if annotations is None:
        return False
    return annotations.get((fn_name, source_line)) is False


# ===================================================================
# Public API — build complete seed set
# ===================================================================


def extract_seeds_from_bug_path(
    bug_path: BugPath,
    sdg: SDG,
    source_root: str | None = None,
    fallback_trigger_fn: str | None = None,
    fallback_trigger_line: int | None = None,
) -> list[tuple[str, str, str]]:
    """Extract the complete seed set from a bug report's path events.

    Three-phase extraction:

    1. **Event matching** — each path event is matched to PDG node(s) using
       AST-derived metadata (node kind, variable, call info).

    2. **Cross-function parameter tracing** — seed variables that appear at
       call boundaries are mapped to formal parameters in callee/caller PDGs.

    3. **Member dependency tracing** — seed variables that derive from class
       members (via ``.begin()``, ``this->``, etc.) are traced to their
       definition nodes.

    Args:
        bug_path: Structured bug path (from CSA report parsing).
        sdg: System Dependence Graph.
        source_root: Project root for file path resolution.
        fallback_trigger_fn: Function name if no seeds found.
        fallback_trigger_line: Line number if no seeds found.

    Returns:
        List of ``(function_name, node_id, reason)`` triples forming
        the seed set.
    """
    all_seeds: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    matched_event_count = 0

    # ── Pre-build file index (avoids O(N) scan over all 24K functions) ────
    file_index = sdg.build_file_index()

    # ── Phase 1: Event→PDG matching ────────────────────────────────
    for event in bug_path.events:
        event_seeds = find_pdg_nodes_for_event(
            event, sdg, source_root, file_index=file_index,
        )
        for fn_name, node_id, reason in event_seeds:
            key = (fn_name, node_id)
            if key not in seen:
                seen.add(key)
                all_seeds.append((fn_name, node_id, reason))
                matched_event_count += 1

    logger.info(
        "Phase 1 (event→PDG): %d unique seeds from %d path events "
        "(%d/%d = %.0f%%)",
        len(all_seeds), matched_event_count,
        matched_event_count, len(bug_path.events),
        matched_event_count / len(bug_path.events) * 100
        if bug_path.events else 0,
    )

    # ── Phase 2: Cross-function parameter tracing ──────────────────
    if all_seeds:
        xfn_seeds = _enrich_with_cross_function_params(all_seeds, sdg, bug_path)
        if xfn_seeds:
            before = len(all_seeds)
            for seed in xfn_seeds:
                key = (seed[0], seed[1])
                if key not in seen:
                    seen.add(key)
                    all_seeds.append(seed)
            logger.info("Phase 2 (cross-fn): +%d seeds → total %d",
                        len(all_seeds) - before, len(all_seeds))

    # ── Phase 3: Member dependency tracing ─────────────────────────
    if all_seeds:
        member_seeds = _trace_member_dependencies(all_seeds, sdg)
        if member_seeds:
            before = len(all_seeds)
            for seed in member_seeds:
                key = (seed[0], seed[1])
                if key not in seen:
                    seen.add(key)
                    all_seeds.append(seed)
            logger.info("Phase 3 (member dep): +%d seeds → total %d",
                        len(all_seeds) - before, len(all_seeds))

    if not all_seeds and fallback_trigger_fn and fallback_trigger_line is not None:
        logger.warning(
            "No seeds extracted; fall back to single trigger (%s:%d)",
            fallback_trigger_fn, fallback_trigger_line,
        )
        # ── Phase 0: Trigger-only fallback seed ─────────────────────
        # When event→PDG matching produces zero seeds (line-number
        # mismatch, sparse PDG, etc.), create a best-effort seed from
        # the trigger function+line.  This ensures the backward slicer
        # always has at least one entry point.
        fallback_seeds = _create_fallback_seed(
            fallback_trigger_fn, fallback_trigger_line,
            sdg, bug_path,
        )
        if fallback_seeds:
            all_seeds.extend(fallback_seeds)
            logger.info(
                "Fallback seed: %d node(s) in %s",
                len(fallback_seeds), fallback_trigger_fn,
            )

    return all_seeds
