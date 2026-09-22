"""Feasible path-space collection + terminal visualization.

The pipeline already *enumerates* the feasible path space implicitly:
``ShortestPathComputer.compute_topk`` DFS-enumerates complete paths through each
segment's CFG branch tree (pruning ``contradictory`` branches, auto-fixing
``consistent`` ones, branching both ways on ``insufficient_info``), ranks by
complexity ``(S↑, C↑, V↑)``, then keeps only the top-k.  The remaining candidates
are discarded.

This module makes that space explicit and persistent:

- :func:`collect_path_space` re-runs the (deterministic, LLM-free) enumeration per
  segment with an unbounded ``k`` and serializes every feasible candidate, plus
  aggregate stats and the whole-path-space upper bound (the product of per-segment
  candidate counts).  This is the structured dataset a future "efficient path-space
  search" algorithm can consume to find a feasible solution or prove infeasibility.
- :func:`render_path_space_ascii` prints a terminal ASCII branch tree per segment
  (verdict-colored, chosen path highlighted) with an efficiency comparison block,
  showing *how* the current method explores a bounded local candidate set instead
  of the full (exponential) whole-path product.

Only depends on the shortest-path/CFG-branch machinery — no LLM calls.
"""

from __future__ import annotations

import re

from llm_client.fp_analysis.shortest_path import (
    CalleePathCache,
    ShortestPathComputer,
)

# Sentinel for "no chosen path info provided".
_CHOSEN_NONE = object()

# Serialization cap for a candidate's branch-decision list.  A segment's tree can
# hold tens of thousands of branches (see ``cfg_branch_extractor``), and every
# enumerated candidate used to serialize *all* of them — a 7.3 GB result JSON for
# one faiss report.  The head of the list is kept (enough to identify a candidate
# by eye); the full count and signature are always preserved, so candidate
# equivalence is unaffected.
_MAX_SERIALIZED_DECISIONS = 40


def _chosen_by_seg(selections) -> dict[int, set[str]]:
    """Map each segment to the set of branch ``node_id``s on the chosen path.

    ``selections`` is the final path-selection result (a list of per-segment
    dicts with ``branch_decisions`` carrying ``node_id`` + ``branch_taken``).
    """
    chosen: dict[int, set[str]] = {}
    for sel in (selections or []):
        seg_idx = sel.get("segment_index")
        if seg_idx is None:
            continue
        node_ids = {
            str(d.get("node_id"))
            for d in (sel.get("branch_decisions") or [])
            if d.get("node_id")
        }
        chosen[seg_idx] = node_ids
    return chosen


def _segment_signature(branch_decisions) -> str:
    """Per-segment branch signature, e.g. ``seg0[L10:T,L12:F]``.

    Mirrors the whole-path ``path_signature`` in ``run_path_selection.py`` but
    scoped to one segment so it can serve as a stable candidate key.
    """
    parts = []
    for d in (branch_decisions or []):
        line = d.get("line", d.get("source_line", "?"))
        taken = d.get("branch_taken", True)
        parts.append(f"L{line}:{'T' if taken else 'F'}")
    return ",".join(parts)


def _chosen_pairs(selections) -> dict[int, frozenset]:
    """Per-segment set of ``(node_id, branch_taken)`` pairs on the chosen path.

    A path is uniquely identified by this pair-set (a ``node_id`` alone is not
    unique: candidates always include both children of a branch, differentiated
    only by their ``branch_taken`` flag).  Used to mark which enumerated
    candidate is the one actually selected.
    """
    pairs: dict[int, frozenset] = {}
    for sel in (selections or []):
        seg_idx = sel.get("segment_index")
        if seg_idx is None:
            continue
        pairs[seg_idx] = frozenset(
            (str(d.get("node_id")), bool(d.get("branch_taken", True)))
            for d in (sel.get("branch_decisions") or [])
            if d.get("node_id")
        )
    return pairs


def collect_path_space(analyzer, segments, selections=None) -> dict:
    """Enumerate and serialize the feasible path space of a report.

    Args:
        analyzer: The :class:`FPAnalyzer` running the pipeline (uses its SDG).
        segments: The report's ``SegmentInfo`` list.
        selections: The final path-selection result (a list of per-segment
            dicts).  When given, the exact chosen candidate per segment is
            marked ``chosen=True``.

    Returns a serializable dict:
        ``{"per_segment": [...], "total_feasible_candidates": int,
            "whole_path_bound": int, "capped": bool}``
    """
    chosen_pairs = _chosen_pairs(selections)
    sdg = getattr(analyzer, "_sdg", None)
    shortest_pc = ShortestPathComputer(sdg, CalleePathCache(sdg))
    cap = ShortestPathComputer._MAX_CANDIDATES

    per_segment: list[dict] = []
    total_candidates = 0
    capped = False

    for segment in segments or []:
        seg_idx = getattr(segment, "segment_index", None)
        tree = getattr(segment, "cfg_branch_tree", None)
        chosen_exact = chosen_pairs.get(seg_idx, frozenset())

        if tree is None or not getattr(tree, "root_branches", None):
            per_segment.append({
                "segment_index": seg_idx,
                "linear": True,
                "num_branches": 0,
                "num_consistent": 0,
                "num_contradictory": 0,
                "num_insufficient": 0,
                "num_feasible_candidates": 1,
                "num_exclusive_groups": 0,
                "grouped_branches": 0,
                "postcondition": _postcondition_str(tree),
                "ci_call_edges_only": False,
                "ci_branch_keys": [],
                "candidates": [{
                    "branch_decisions": [],
                    "num_branch_decisions": 0,
                    "metrics": {"S": 0, "C": 0, "V": 0},
                    "signature": _segment_signature([]),
                    "chosen": True,
                    "rank": 1,
                }],
            })
            total_candidates += 1
            continue

        pc = getattr(segment, "postcondition", None) or getattr(
            tree, "postcondition", None,
        )
        # Mutual-exclusion groups (same var == different consts).  Candidates
        # returned by compute_topk are already feasibility-filtered against
        # these, so num_feasible_candidates reflects the pruned set.
        groups = ShortestPathComputer._extract_equality_exclusive_groups(
            tree.root_branches,
        )
        num_exclusive_groups = len(groups)
        grouped_branches = sum(len(g) for g in groups)

        cands = shortest_pc.compute_topk(tree, k=cap, postcondition=pc)

        if len(cands) >= cap:
            capped = True

        cand_list: list[dict] = []
        ci_keys: set[str] = set()
        ci_call_edges_only = True
        for rank, (decisions, metrics) in enumerate(cands, start=1):
            bd = [
                {
                    "node_id": getattr(d, "node_id", ""),
                    "branch_taken": getattr(d, "branch_taken", True),
                    "condition_expr": getattr(d, "condition_expr", ""),
                    "source_line": getattr(d, "source_line", 0),
                }
                for d in decisions
            ]
            for d in bd:
                norm = _normalize_call_edge((d.get("node_id") or "").strip())
                if norm is None:
                    ci_call_edges_only = False
                else:
                    ci_keys.add(norm)
            cand_pairs = frozenset(
                (d.get("node_id"), d.get("branch_taken"))
                for d in bd if d.get("node_id")
            )
            cand_list.append({
                "branch_decisions": bd[:_MAX_SERIALIZED_DECISIONS],
                "num_branch_decisions": len(bd),
                "metrics": {
                    "S": getattr(metrics, "S", 0),
                    "C": getattr(metrics, "C", 0),
                    "V": getattr(metrics, "V", 0),
                },
                "signature": _segment_signature(bd),
                "chosen": bool(chosen_exact) and cand_pairs == chosen_exact,
                "rank": rank,
            })

        per_segment.append({
            "segment_index": seg_idx,
            "linear": False,
            "num_branches": getattr(tree, "num_branches", 0),
            "num_consistent": getattr(tree, "num_consistent", 0),
            "num_contradictory": getattr(tree, "num_contradictory", 0),
            "num_insufficient": getattr(tree, "num_insufficient", 0),
            "num_feasible_candidates": len(cand_list),
            "num_exclusive_groups": num_exclusive_groups,
            "grouped_branches": grouped_branches,
            "postcondition": _postcondition_str(pc),
            "ci_call_edges_only": ci_call_edges_only,
            "ci_branch_keys": sorted(ci_keys),
            "candidates": cand_list,
        })
        total_candidates += len(cand_list)

    whole_bound = 1
    for seg in per_segment:
        whole_bound *= seg["num_feasible_candidates"]

    return {
        "per_segment": per_segment,
        "total_feasible_candidates": total_candidates,
        "whole_path_bound": whole_bound,
        "capped": capped,
    }


def _postcondition_str(pc) -> str:
    if pc is None:
        return ""
    expr = getattr(pc, "condition_expr", "") or ""
    line = getattr(pc, "source_line", "") or ""
    return f"{expr} @ L{line}".strip()


# ── Context-insensitive grouping ─────────────────────────────────────────────
#
# The whole-path cross-product is inflated when the SAME function's branches
# re-appear across several segments.  Two distinct causes, with opposite
# context-sensitivity:
#
# 1. *Multiple calls to the same function* (the user's target use case).  The
#    path calls a callee at N distinct call sites; each call edge surfaces the
#    callee's branch conditions as ``callee_<caller>_<callee>_L<line>_B<id>``
#    nodes.  The only thing differing between call sites is the call line, so
#    the branch sets are *identical once the line is stripped* — and the callee
#    genuinely behaves the same regardless of which call site invoked it
#    (context-INSENSITIVE).  Collapsing these N segments to ONE dimension is
#    sound: their directions are bound together (全路径一致).
#
# 2. *Loop re-entry inside ONE call* (e.g. GlogFormatter: insertThousands
#    GroupingUnsafe is called once but its own loop body, ``seg_`` nodes,
#    re-appears across 4 path segments → 8⁴ = 4096).  These are CONTEXT-SENSITIVE
#    — each loop passage may legitimately take different branch directions as
#    indices/remaining_digits change.  Forcing them identical is UNSOUND and can
#    flip a genuine TP (the infinite-loop bug) to FP.  Such own-body segments
#    (``seg_``/``pdg_``) are therefore NEVER grouped.
#
# So only genuine call-edge segments qualify; ``branch_set_of`` (raw node-ids)
# would over-capture both, so grouping uses ``ci_branch_set_of``.


def _normalize_call_edge(nid: str) -> str | None:
    """Normalize a call-edge branch node_id, dropping the call line.

    ``callee_<caller>_<callee>_L<line>_B<id>`` → ``callee_<caller>_<callee>_B<id>``
    (call line stripped), so two calls to the same function at different lines
    collapse to the same branch-set key.  ``call_`` nodes carry no line and pass
    through.  Non-call-edge nodes (own-body ``seg_``/``pdg_``) return None.
    """
    if not (nid.startswith("callee_") or nid.startswith("call_")):
        return None
    return re.sub(r"_L\d+", "", nid)


def ci_branch_set_of(ps: dict) -> frozenset:
    """frozenset of normalized CALL-EDGE branch keys in a path-space entry.

    Only branches from call edges (``callee_``/``call_``) qualify for
    context-insensitive grouping — they represent genuine *calls* to a callee,
    whose repeated appearance across segments inflates the path space.  A
    segment with ANY own-body branch (``seg_``/``pdg_``, e.g. the bug function's
    own loop re-entering) yields an EMPTY set → never grouped (context-sensitive).
    """
    # ``collect_path_space`` records the FULL candidate set's keys alongside the
    # (now capped) serialized ``branch_decisions``, so grouping never depends on
    # how much of a candidate got serialized.
    if "ci_branch_keys" in ps:
        if not ps.get("ci_call_edges_only", False):
            return frozenset()
        return frozenset(ps["ci_branch_keys"])

    keys: set[str] = set()
    for cand in ps.get("candidates", []):
        for d in cand.get("branch_decisions", []):
            nid = (d.get("node_id") or "").strip()
            norm = _normalize_call_edge(nid)
            if norm is None:
                return frozenset()
            keys.add(norm)
    return frozenset(keys)


def annotate_ci_groups(path_space: dict, segments=None) -> dict:
    """Collapse *repeated-call-site* segments into context-insensitive dimensions.

    Groups segments that (a) carry the SAME normalized call-edge branch set —
    the same callee's branches re-appearing at different call sites — and
    (b) have equal candidate count.  Each group becomes ONE enumeration dimension
    (its members forced to identical branch directions — 全路径一致); ungrouped
    segments stay one dimension each.  Returns a shallow-copied ``path_space``
    annotated with:

    - ``ci_groups``   : list of per_segment index lists (len > 1)
    - ``ci_group_of`` : {per_segment_index: group_id}
    - ``ci_dims``     : list of {"members": [idx...], "count": int}
    - ``whole_path_bound`` : recomputed over the collapsed dimensions

    Segments whose branches are own-body (``seg_``/``pdg_``, e.g. a single
    call's loop re-entry) or linear yield no group, so the function is a no-op
    on reports without genuine repeated call sites.
    """
    per = path_space.get("per_segment", [])
    out = dict(path_space)
    if not per:
        out["ci_groups"] = []
        out["ci_group_of"] = {}
        out["ci_dims"] = []
        out["whole_path_bound"] = 1
        return out

    buckets: dict[tuple, list[int]] = {}
    for i, p in enumerate(per):
        key = (ci_branch_set_of(p), p.get("num_feasible_candidates", 1))
        buckets.setdefault(key, []).append(i)

    groups = [idx for idx in buckets.values()
              if len(idx) > 1 and ci_branch_set_of(per[idx[0]])]

    group_of: dict[int, int] = {}
    dims: list[dict] = []
    for gid, idx in enumerate(groups):
        for i in idx:
            group_of[i] = gid
        dims.append({
            "members": list(idx),
            "count": per[idx[0]].get("num_feasible_candidates", 1),
        })
    for i, p in enumerate(per):
        if i not in group_of:
            dims.append({"members": [i], "count": p.get("num_feasible_candidates", 1)})

    bound = 1
    for d in dims:
        bound *= d["count"]

    out["ci_groups"] = groups
    out["ci_group_of"] = group_of
    out["ci_dims"] = dims
    out["whole_path_bound"] = bound
    return out


# ── ASCII tree rendering ────────────────────────────────────────────────────


def render_path_space_ascii(segments, chosen_by_seg, path_space,
                            attempts=None, max_attempts=None) -> str:
    """Render per-segment ASCII branch trees + an efficiency summary block.

    Args:
        segments: The report's ``SegmentInfo`` list (with ``cfg_branch_tree``).
        chosen_by_seg: ``{seg_idx: set(node_id)}`` for the chosen path.
        path_space: The dict returned by :func:`collect_path_space`.
        attempts: How many whole-path attempts the search actually used.
        max_attempts: The path-traversal threshold (``--max-path-attempts``).

    Returns a single multi-line string ready to ``print``.
    """
    if chosen_by_seg is None:
        chosen_by_seg = {}
    lines: list[str] = []
    for idx, segment in enumerate(segments or []):
        seg_idx = getattr(segment, "segment_index", idx)
        tree = getattr(segment, "cfg_branch_tree", None)
        chosen = chosen_by_seg.get(seg_idx, set())
        fn = getattr(segment, "function_name", "?")
        label = getattr(segment, "label", f"S{seg_idx}")

        info = path_space.get("per_segment", [{}])[idx]
        nc = info.get("num_feasible_candidates", 0)
        lines.append(f"\n  ── Seg {seg_idx} ({fn} · {label}) ──")
        lines.append(
            f"     候选路径: {nc} 条 | "
            f"一致={info.get('num_consistent', 0)} "
            f"剪枝={info.get('num_contradictory', 0)} "
            f"待定={info.get('num_insufficient', 0)}"
        )
        ng = info.get("num_exclusive_groups", 0)
        if ng:
            gb = info.get("grouped_branches", 0)
            lines.append(
                f"     互斥裁剪: {ng} 组互斥分支({gb} 分支) → 候选数已滤除不可行组合"
            )
        if tree is None or not getattr(tree, "root_branches", None):
            lines.append("      (linear segment — 无分支，单一可行路径)")
            continue
        for node in tree.root_branches:
            lines += _render_node(node, chosen, "      ")

    lines.append("\n  ── 探索效率对比 (how we explore the feasible path space) ──")
    bound = path_space.get("whole_path_bound", 0)
    total = path_space.get("total_feasible_candidates", 0)
    lines.append(f"     整路径朴素空间 = ∏(各 segment 候选数) ≈ {bound:_}")
    lines.append(
        f"     实际探索代价   = ∑(各 segment 候选数) = {total}"
        f"  + 惰性重试 ≤ {attempts if attempts else 0} 次"
    )
    lines.append(
        "     → 并未枚举全部整路径组合：逐段本地枚举(有界) + 最短优先(S↑C↑V↑)"
        " + 仅冲突时屏蔽当前组合并重试"
    )
    if any(
        s.get("num_exclusive_groups", 0) for s in path_space.get("per_segment", [])
    ):
        lines.append(
            "     → 互斥分支(同变量 == 不同常量)组合在枚举后裁剪：候选数与整路径"
            "上界均为可行性筛选后的值"
        )
    if path_space.get("capped"):
        lines.append(
            f"     注: 某 segment 候选数达到上限({ShortestPathComputer._MAX_CANDIDATES})，"
            "枚举被截断"
        )
    if max_attempts:
        lines.append(
            f"     路径遍历阈值 = --max-path-attempts = {max_attempts}"
        )
    return "\n".join(lines)


def _render_node(node, chosen_node_ids, prefix, is_last=True) -> list[str]:
    """Render one CFG branch node + its children as an ASCII subtree."""
    tag = getattr(node, "tag", None)
    verdict = getattr(tag, "verdict", None) if tag is not None else None
    node_id = getattr(node, "node_id", "?")
    on_path = node_id in chosen_node_ids

    if verdict == "contradictory":
        mark = "[剪枝]"
    elif verdict == "consistent":
        mark = "[可走]"
    else:
        mark = "[待定]"
    if on_path:
        mark = "✅" + mark + " ← 选中"

    branch = getattr(node, "branch_label", "") or (
        "true" if getattr(node, "is_taken_branch", True) else "false"
    )
    expr = getattr(node, "condition_expr", "") or ""
    line = getattr(node, "source_line", "")
    conn = "└─ " if is_last else "├─ "
    child_prefix = prefix + ("   " if is_last else "│  ")

    lines = [
        f"{prefix}{conn}{mark} {expr} (L{line}, {branch}) "
        f"[{getattr(node, 'function_name', '')}]"
    ]
    children = getattr(node, "children", None) or []
    for i, child in enumerate(children):
        lines += _render_node(
            child, chosen_node_ids, child_prefix, is_last=(i == len(children) - 1),
        )
    return lines
