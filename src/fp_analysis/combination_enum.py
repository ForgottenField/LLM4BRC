"""Systematic whole-path combination enumeration.

Replaces the per-segment blocked-candidate re-selection.  The old mechanism
excluded a chosen candidate *index* for a segment from every future path, which
over-prunes: infeasibility is combination-dependent — ``(segA=b1, segB=c1)``
being FP does not make ``b1`` infeasible, since ``(segA=b1, segB=c2)`` may be
feasible.  Instead we enumerate the cross-product of the free segments' feasible
candidates (from ``collect_path_space``), block *whole-path* combinations, and
conflict-check + simulate each.  A branch is only globally pruned when conflict
analysis *proves* it conflicts with every branch of another segment — never by
default.

Pure functions; no LLM calls.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Iterator, Optional, Sequence


def _ci_dims(path_space: dict) -> list[dict]:
    """The enumeration dimensions of *path_space*.

    With context-insensitive grouping (``ci_dims`` present), these are the
    collapsed groups + singletons; otherwise each segment is its own dimension.
    A dimension is ``{"members": [per_segment_idx...], "count": int}``.
    """
    dims = path_space.get("ci_dims")
    if dims is not None:
        return dims
    per_segment = path_space["per_segment"]
    return [
        {"members": [i], "count": ps.get("num_feasible_candidates", len(ps["candidates"]))}
        for i, ps in enumerate(per_segment)
    ]


def build_selections_from_combo(
    path_space: dict,
    combo: Sequence[int],
    segments: Sequence,
) -> list[dict]:
    """Build a v2 ``selections`` list from a whole-path candidate combination.

    ``combo`` is a tuple with one 0-based candidate index per *dimension*
    (a context-insensitive group is a single dimension; forced segments have
    only index 0).  Grouped segments share the SAME branch decisions — they are
    the same function's branches re-appearing, so their directions are bound
    together (全路径一致).  Each selection's ``branch_decisions`` carry BOTH the
    v1 keys (``line``, ``branch``) consumed by ``_check_global_path_consistency``
    and the v2 keys (``node_id``, ``branch_taken``, ``condition``,
    ``source_line``) consumed by the blueprint.  ``established_state`` is
    derived deterministically from segment metadata and the chosen branch
    decisions — it is advisory only (every downstream consumer has a fallback).
    """
    per_segment = path_space["per_segment"]
    dims = _ci_dims(path_space)
    selections: list[dict] = []
    for di, dim in enumerate(dims):
        rep = per_segment[dim["members"][0]]
        cand = rep["candidates"][combo[di]]
        bd_v2 = cand.get("branch_decisions") or []
        branch_decisions = []
        for d in bd_v2:
            taken = bool(d.get("branch_taken", True))
            line = d.get("source_line", 0)
            branch_decisions.append({
                # v1 keys (merge check)
                "line": line,
                "branch": "true" if taken else "false",
                # v2 keys (blueprint)
                "node_id": d.get("node_id", ""),
                "branch_taken": taken,
                "condition": d.get("condition_expr", ""),
                "source_line": line,
            })
        for mi in dim["members"]:
            seg = segments[mi]
            ps = per_segment[mi]
            selections.append({
                "segment_index": mi,
                "branch_decisions": branch_decisions,
                "established_state": _state_desc(
                    seg, mi, ps, branch_decisions,
                ),
                "selected_candidate": combo[di] + 1,
                "num_candidates": ps.get("num_feasible_candidates", len(ps["candidates"])),
                "is_feasible": True,
            })
    return selections


def _state_desc(seg, seg_idx: int, ps: dict, branch_decisions: list[dict]) -> str:
    """Deterministic advisory state description (mirrors the v2 linear pattern)."""
    fn = getattr(seg, "function_name", "?")
    label = getattr(seg, "label", f"S{seg_idx}")
    ls = getattr(seg, "line_start", 0)
    le = getattr(seg, "line_end", 0)
    if not branch_decisions:
        pc = ps.get("postcondition") or ""
        pc_text = f", postcondition: {pc}" if pc else ""
        return f"Linear segment {label} ({fn} L{ls}-L{le}){pc_text}"
    parts = []
    for d in branch_decisions:
        taken = "TRUE" if d["branch_taken"] else "FALSE"
        parts.append(f"L{d['source_line']}: {d.get('condition', '')} → {taken}")
    return f"Branch segment {label} ({fn} L{ls}-L{le}): " + "; ".join(parts)


def combo_signature(path_space: dict, combo: Sequence[int]) -> str:
    """Whole-path signature of a combination, matching ``path_signature`` format.

    Built from the per-segment candidate ``signature`` fields that
    ``collect_path_space`` already stores (``L{line}:T/F,...``), producing e.g.
    ``seg0[L10:T];seg1[L3:T,L5:F]`` — the same shape ``path_signature`` produces,
    so it dedups against ``blocked_paths``.  Grouped segments (context-insensitive
    dimensions) share the same candidate, so each member is emitted with the
    same signature — a grouped assignment therefore blocks consistently.
    """
    per_segment = path_space["per_segment"]
    dims = _ci_dims(path_space)
    parts = []
    for di, dim in enumerate(dims):
        cand = per_segment[dim["members"][0]]["candidates"][combo[di]]
        sig = cand.get("signature", "")
        for mi in dim["members"]:
            parts.append(f"seg{mi}[{sig}]")
    return ";".join(parts)


# Combos we are willing to *materialize* into the rank heap.  The full
# cross-product can be astronomically large (e.g. GlogFormatter ~2.4e24) and
# materializing + sorting it OOMs the server — so we only enumerate a bounded
# lexicographic prefix and keep the ``cap`` lowest-rank combos via a heap.
_ENUM_LIMIT = 200_000
# Absolute cap on combos ever yielded, whatever ``cap`` is requested.
_YIELD_CAP = 100_000


def iter_combinations(
    path_space: dict,
    segments: Sequence,
    cap: Optional[int] = None,
) -> Iterator[tuple[tuple, list[dict]]]:
    """Yield ``(combo, selections)`` for distinct whole-path combinations.

    Combos are produced in feasibility order (sum of per-dimension candidate
    ``rank`` ascending — most feasible first) by keeping the ``cap`` lowest-rank
    combos of a *bounded* enumeration of the cross-product.  A *dimension* is
    either a single segment or a context-insensitive group (identical-branch
    segments collapsed together, forced to share one candidate).  Forced
    segments (``num_feasible_candidates <= 1``) contribute only index 0 and are
    fixed.  ``cap`` bounds the number of combos yielded; ``_ENUM_LIMIT`` bounds
    how many combos are enumerated, so memory is O(cap) and time
    O(_ENUM_LIMIT·log cap) even for astronomically large spaces.  No per-segment
    pruning is done here — blocking happens on the whole-path signature, in the
    caller.
    """
    per_segment = path_space["per_segment"]
    dims = _ci_dims(path_space)
    index_ranges = [range(d["count"]) for d in dims]
    if not index_ranges:
        return

    total = 1
    for r in index_ranges:
        total *= len(r)

    limit = total if cap is None else min(cap, total)
    if limit > _YIELD_CAP:
        limit = _YIELD_CAP
    enum_n = min(total, _ENUM_LIMIT)

    def _rank(combo: tuple) -> int:
        return sum(
            per_segment[d["members"][0]]["candidates"][idx].get("rank", idx + 1)
            for d, idx in zip(dims, combo)
        )

    def _bounded_product():
        cnt = 0
        for combo in itertools.product(*index_ranges):
            if cnt >= enum_n:
                break
            cnt += 1
            yield combo

    top = heapq.nsmallest(limit, _bounded_product(), key=_rank)
    for combo in top:
        yield combo, build_selections_from_combo(path_space, combo, segments)
