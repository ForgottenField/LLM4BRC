"""Slice-driven branch relevance — classify enumerated branches by whether they
have a data/control path to the bug's root cause, then compress the enumeration.

The pipeline enumerates a report's feasible path space (cross-product of
per-segment candidates from ``collect_path_space``).  Many of those branches are
*irrelevant*: their predicate has no data/control-dependence path to the bug
trigger (as established by the PDG backward slice).  Enumerating over them is
pure waste and, worse, hides the fact that the whole space is root-cause
invariant.  This module:

* ``branch_relevant``  — decide, for one branch decision, whether its predicate
  is retained in the slice (relevant) or removed (irrelevant).
* ``classify_branch_relevance`` — classify every distinct branch in a path space.
* ``reduce_path_space`` — fix irrelevant branches and keep one representative
  candidate per distinct *relevant-signature*, returning the compressed space
  plus ratio/compression statistics.

Routing uses the ``node_id`` prefix produced by ``cfg_branch_extractor``:

* ``pdg_<fn>_S_<line>_<col>[...]``  — a PDG control predicate; the tail ``S_...``
  is the exact PDG predicate id, so ``SliceMask.is_branch_retained(fn, pred)``
  answers relevance directly.
* ``callee_...`` / ``call_...``     — a call-edge sentinel (``condition_expr`` is
  ``"→ callee()"``); not a control predicate, so ``is_branch_retained`` does not
  apply.  Relevant iff the callee is involved in the slice.
* ``seg_<fn>_<block>``              — CFG fallback (Clang BasicBlock id); resolved
  via the SDG line index + control-predicate kind + condition disambiguation.

Pure functions; no LLM calls.
"""

from __future__ import annotations

from typing import Optional, Sequence

from llm_client.fp_analysis.slice_mask import SliceMask
from llm_client.fp_analysis.pdg_models import SDG

# Control-predicate node kinds (slice_mask._CONTROL_PREDICATE_KINDS + unary_op).
_CONTROL_KINDS: frozenset[str] = frozenset({
    "if_cond", "while_cond", "for_cond", "binary_op", "unary_op",
})

# Extra node kinds that should NEVER be mistaken for a control predicate when we
# resolve a CFG-fallback branch by line (a call is not a predicate).
_NON_PREDICATE_KINDS: frozenset[str] = frozenset({
    "call_expr", "assign", "decl", "return", "argument", "expr",
})


# ---------------------------------------------------------------------------
# Per-branch classification
# ---------------------------------------------------------------------------


def branch_relevant(
    sdg: SDG,
    slice_mask: SliceMask,
    decision: dict,
    *,
    bug_file: str | None = None,
    truncated: bool = False,
) -> bool:
    """True if *decision*'s branch predicate is retained in the slice (relevant).

    Conservative: any unparseable / unresolvable / truncated case returns True
    (do not compress), so we only ever FIX a branch when the slice confidently
    removed it.
    """
    node_id = (decision.get("node_id") or "").strip()
    if not node_id:
        return True

    relevant: bool
    if node_id.startswith("pdg_"):
        relevant = _pdg_branch_relevant(sdg, slice_mask, node_id)
    elif node_id.startswith("callee_") or node_id.startswith("call_"):
        relevant = _call_edge_relevant(sdg, slice_mask, decision)
    elif node_id.startswith("seg_"):
        relevant = _seg_branch_relevant(sdg, slice_mask, decision, bug_file)
    else:
        # Unknown node_id format → cannot prove removal → conservative.
        return True

    # Under truncation the slice may have missed a real path to the root cause,
    # so a "removed" verdict is not trustworthy.  Fully conservative.
    if truncated and not relevant:
        return True
    return relevant


def _pdg_branch_relevant(sdg: SDG, slice_mask: SliceMask, node_id: str) -> bool:
    """PDG predicate: parse fn + predicate id, ask the mask directly."""
    rest = node_id[len("pdg_"):]
    mark = rest.find("_S_")
    if mark == -1:
        return True  # not a control-predicate id → cannot judge
    fn = rest[:mark]
    pred = rest[mark + 1:]
    # Canonicalize fn (node_id may carry a short/unqualified name); get_pdg
    # tries exact → ::suffix → unqualified, so the mask key lines up.
    pdg = sdg.get_pdg(fn)
    canon = pdg.function_name if pdg is not None else fn
    return slice_mask.is_branch_retained(canon, pred)


def _call_edge_relevant(
    sdg: SDG, slice_mask: SliceMask, decision: dict,
) -> bool:
    """Call-edge sentinel: relevant iff the callee is involved in the slice.

    ``pdg_retained_nodes`` keys == the set of functions that have retained
    nodes (== ``SliceResult.involved_functions``), so membership means the callee
    participates in the root-cause slice.
    """
    callee = _parse_callee(decision)
    if not callee:
        return True  # cannot identify the call target → conservative
    pdg = sdg.get_pdg(callee)
    if pdg is None:
        return True  # callee not in the SDG → can't prove it is irrelevant
    return bool(slice_mask.pdg_retained_nodes.get(pdg.function_name))


def _parse_callee(decision: dict) -> str:
    """Extract a short callee name from a call-edge decision.

    Prefers ``condition_expr`` (``"→ callee()"``), falls back to the
    ``callee_<caller>_<callee>_L<line>`` node_id.
    """
    cond = (decision.get("condition_expr") or "").strip()
    if cond.startswith("→ ") and "()" in cond:
        name = cond[2:].split("()", 1)[0].strip()
        if name:
            return name
    node_id = (decision.get("node_id") or "").strip()
    if node_id.startswith("callee_"):
        parts = node_id.split("_")
        # node_id == callee_<caller>_<callee>_L<line>; callee is the longest
        # interior segment that ends before "_L<line>".  Find the _L marker.
        for i, p in enumerate(parts):
            if p.startswith("L") and p[1:].isdigit() and i >= 2:
                return "_".join(parts[2:i])
    return ""


def _seg_branch_relevant(
    sdg: SDG,
    slice_mask: SliceMask,
    decision: dict,
    bug_file: str | None,
) -> bool:
    """CFG-fallback branch: resolve the predicate by (file, line) + condition."""
    if not bug_file:
        return True
    line = decision.get("source_line", 0) or 0
    li = sdg.build_line_index()
    matches = _line_matches(li, bug_file, line)
    if not matches:
        return True  # no control predicate at that line → cannot judge
    cond = (decision.get("condition_expr") or "").strip()
    best = None
    for fn, nid in matches:
        pdg = sdg.get_pdg(fn)
        if pdg is None:
            continue
        node = pdg.nodes.get(nid)
        if node is None or node.kind not in _CONTROL_KINDS:
            continue
        if cond and (node.expression or "").strip() and \
                cond not in (node.expression or "") and \
                (node.expression or "") not in cond:
            continue  # condition text doesn't agree → different predicate
        best = (pdg.function_name, nid)
        if not cond:
            break
    if best is None:
        return True
    return slice_mask.is_branch_retained(*best)


def _line_matches(
    line_index: dict[tuple[str, int], list[tuple[str, str]]],
    bug_file: str,
    line: int,
) -> list[tuple[str, str]]:
    """Return ``[(fn, node_id)]`` at (bug_file, line) with suffix-tolerant file match."""
    key = (bug_file, line)
    if key in line_index:
        return line_index[key]
    base = bug_file.rsplit("/", 1)[-1]
    out: list[tuple[str, str]] = []
    for (sf, ln), refs in line_index.items():
        if ln != line:
            continue
        if sf == bug_file or sf.endswith("/" + bug_file) or \
                (sf and sf.rsplit("/", 1)[-1] == base):
            out.extend(refs)
    return out


# ---------------------------------------------------------------------------
# Whole-space classification + reduction
# ---------------------------------------------------------------------------


def classify_branch_relevance(
    sdg: SDG,
    slice_mask: SliceMask,
    path_space: dict,
    *,
    bug_file: str | None = None,
    truncated: bool = False,
) -> dict:
    """Classify every distinct branch node in *path_space*.

    Returns ``{"node_relevance": {node_id: bool}, "stats": {...}}``.  Two branch
    directions (``branch_taken`` T/F) share a node_id / predicate, so relevance
    is keyed by node_id and judged once.
    """
    node_relevance: dict[str, bool] = {}
    for seg in path_space.get("per_segment", []):
        for cand in seg.get("candidates", []):
            for d in cand.get("branch_decisions", []):
                nid = (d.get("node_id") or "").strip()
                if not nid or nid in node_relevance:
                    continue
                node_relevance[nid] = branch_relevant(
                    sdg, slice_mask, d,
                    bug_file=bug_file, truncated=truncated,
                )

    relevant = [n for n, r in node_relevance.items() if r]
    irrelevant = [n for n, r in node_relevance.items() if not r]
    stats = {
        "num_branch_nodes": len(node_relevance),
        "num_relevant": len(relevant),
        "num_irrelevant": len(irrelevant),
        "irrelevant_ratio": round(
            len(irrelevant) / len(node_relevance), 3,
        ) if node_relevance else 0.0,
        "relevant_nodes": sorted(relevant),
        "irrelevant_nodes": sorted(irrelevant),
        "truncated": bool(truncated),
    }
    return {"node_relevance": node_relevance, "stats": stats}


def reduce_path_space(
    path_space: dict,
    node_relevance: dict,
    segments: Sequence,
) -> tuple[dict, dict]:
    """Compress *path_space* by fixing irrelevant branches.

    Candidates that agree on every *relevant* branch decision are root-cause
    equivalent; one representative (lowest rank) is kept per such group.  The
    returned path space only varies the relevant branches.  Also returns
    per-segment + whole-path compression stats.
    """
    original_bound = int(path_space.get("whole_path_bound", 1))
    per_segment_in = path_space.get("per_segment", [])
    per_segment_out: list[dict] = []
    seg_stats: list[dict] = []

    for si, ps in enumerate(per_segment_in):
        cands = ps.get("candidates", [])
        groups: dict[frozenset, dict] = {}
        for cand in cands:
            key = _relevant_signature(cand, node_relevance)
            existing = groups.get(key)
            if existing is None or cand.get("rank", 1) < existing.get("rank", 1):
                groups[key] = cand
        reps = sorted(groups.values(), key=lambda c: c.get("rank", 1))

        new_ps = dict(ps)
        new_ps["candidates"] = reps
        new_ps["num_feasible_candidates"] = len(reps)

        seg_rel = seg_irrel = 0
        seen: set[str] = set()
        for cand in cands:
            for d in cand.get("branch_decisions", []):
                nid = (d.get("node_id") or "").strip()
                if not nid or nid in seen:
                    continue
                seen.add(nid)
                if node_relevance.get(nid, True):
                    seg_rel += 1
                else:
                    seg_irrel += 1

        seg_stats.append({
            "segment_index": ps.get("segment_index", si),
            "original_candidates": len(cands),
            "reduced_candidates": len(reps),
            "relevant_branches": seg_rel,
            "irrelevant_branches": seg_irrel,
        })
        per_segment_out.append(new_ps)

    reduced_bound = 1
    for ps in per_segment_out:
        reduced_bound *= ps["num_feasible_candidates"]

    reduced_ps = dict(path_space)
    reduced_ps["per_segment"] = per_segment_out
    reduced_ps["whole_path_bound"] = reduced_bound
    reduced_ps["total_feasible_candidates"] = sum(
        len(ps["candidates"]) for ps in per_segment_out
    )

    stats = {
        "segments": seg_stats,
        "original_bound": original_bound,
        "reduced_bound": reduced_bound,
        "compression_ratio": round(
            reduced_bound / original_bound, 6,
        ) if original_bound else 0.0,
    }
    return reduced_ps, stats


def _relevant_signature(cand: dict, node_relevance: dict) -> frozenset:
    """The relevant decisions of a candidate as an unordered frozenset.

    Missing node_ids default to relevant (conservative — keeps them distinct).
    """
    sig: set[tuple[str, bool]] = set()
    for d in cand.get("branch_decisions", []):
        nid = (d.get("node_id") or "").strip()
        if not nid:
            continue
        if node_relevance.get(nid, True):
            sig.add((nid, bool(d.get("branch_taken", True))))
    return frozenset(sig)
