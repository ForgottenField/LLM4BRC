#!/usr/bin/env python3
"""Deterministic A/B/C comparison over the regression reports (no LLM).

For each report the deterministic stages run **twice**: once with
``path_annotations=None`` and once with the dense CSA decision table.  That
only toggles **A** — B (slice retention) and C (recursion / environment /
serialization guards) have no switch, so they are on in both runs.  Prints, per
segment, the branch-tree node count, the conflict-checking tallies, the
feasible-candidate count and both the raw and CI-collapsed whole-path bounds,
then the delta.

Two bounds matter, and the bigger one is not the one that decides anything: the
pipeline's exploration budget reads the **CI-collapsed** bound
(``run_path_selection`` → ``_FULL_EXPLORE_THRESHOLD = 32``).  The raw product
barely moves when the tree shrinks, because any segment with more branch
combinations than ``ShortestPathComputer._MAX_CANDIDATES`` (200) contributes
exactly 200 regardless of how many nodes it has — so a tree-size win shows up
in ``nodes`` and in wall-clock, not in the bound.

The two runs must use separate ``CFGCacheSet`` instances, and each *label* run
must rebuild its segments from the parsed report: ``run_branch_filter`` attaches
``cfg_branch_tree`` to the segments it is given, so reusing them across runs
would compare a tree against itself.

Usage:
    python3.10 tools/abc_compare.py <report.html> [more.html ...]
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

from run_path_selection import (  # noqa: E402
    CFGCacheSet, _resolve_source_file_for_cfg, build_slice_mask,
    build_dense_path_annotations, load_sdg, parse_report, run_branch_filter,
)
from llm_client.fp_analysis.cfg_feasibility import segment_path  # noqa: E402
from llm_client.fp_analysis.path_space import (  # noqa: E402
    annotate_ci_groups, collect_path_space,
)

SOURCE_ROOT = Path.home() / "csa_reports" / "project" / "faiss"
_FULL_EXPLORE_THRESHOLD = 32  # run_path_selection's exploration budget
PDG = HERE / "pdg_faiss.json"

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


class _AnalyzerShim:
    """``collect_path_space`` only reads ``analyzer._sdg``."""

    def __init__(self, sdg):
        self._sdg = sdg


def _count_nodes(nodes) -> int:
    total = 0
    stack = list(nodes)
    while stack:
        node = stack.pop()
        total += 1
        stack.extend(node.children)
    return total


def _run(label, report_path, sdg, source_root, annotations, want_dense):
    cache_set = CFGCacheSet(project_name=source_root.name,
                            source_root=str(source_root))
    parsed_report, bug_path = parse_report(str(report_path), source_root)
    bug_source = _resolve_source_file_for_cfg(parsed_report, str(source_root))
    cache_set.ensure(bug_source)

    segments = segment_path(parsed_report, sdg=sdg, source_root=str(source_root))
    for s in segments:
        if s.function_file:
            cache_set.ensure(s.function_file)
    cfg_cache = cache_set.as_dict()

    slice_mask, _ = build_slice_mask(
        sdg, parsed_report, bug_path, source_root, with_slice_result=True)

    if want_dense:
        annotations = build_dense_path_annotations(
            bug_path, sdg, source_root=str(source_root),
            file_index=sdg.build_file_index())

    t0 = time.time()
    branch_results, _ = run_branch_filter(
        segments, parsed_report, cfg_cache, sdg, slice_mask, source_root,
        established_state={}, cache_set=cache_set,
        path_annotations=annotations)
    elapsed = time.time() - t0

    ps = collect_path_space(_AnalyzerShim(sdg), segments)
    # The pipeline's exploration budget is decided by the CI-collapsed bound,
    # not the raw product — a segment whose branches all stay "insufficient"
    # saturates at the per-segment candidate cap, so the raw bound barely moves.
    reduced_ps = annotate_ci_groups(ps, segments)
    by_idx = {r["segment_index"]: r for r in branch_results}
    rows = []
    for i, seg in enumerate(segments):
        r = by_idx.get(i) or {}
        tree = getattr(seg, "cfg_branch_tree", None)
        rows.append({
            "seg": i,
            "fn": seg.function_name or "<empty>",
            "lines": f"L{seg.line_start}-L{seg.line_end}",
            "nodes": _count_nodes(tree.root_branches) if tree else 0,
            "br": r.get("num_branches", 0),
            "con": r.get("num_consistent", 0),
            "ctr": r.get("num_contradictory", 0),
            "ins": r.get("num_insufficient", 0),
        })

    print(f"  [{label}] {len(segments)} segments, "
          f"{sum(r['nodes'] for r in rows)} nodes, "
          f"{sum(r['br'] for r in rows)} branches  [{elapsed:.1f}s]")
    for r in rows:
        print(f"    seg {r['seg']:>2d} {r['fn'][:26]:26s} {r['lines']:>14s} "
              f"nodes={r['nodes']:>7d} br={r['br']:>5d} "
              f"con={r['con']:>5d} ctr={r['ctr']:>5d} ins={r['ins']:>5d}")
    print(f"  [{label}] feasible candidates per segment: "
          f"{[s.get('num_feasible_candidates') for s in ps['per_segment']]}"
          f"  whole_path_bound={ps['whole_path_bound']:_}"
          f"{'  CAPPED' if ps.get('capped') else ''}")
    print(f"  [{label}] after CI collapse: bound="
          f"{reduced_ps['whole_path_bound']:_} "
          f"groups={len(reduced_ps.get('ci_groups', []))}"
          f"  (<{_FULL_EXPLORE_THRESHOLD} => full-space explore)"
          if reduced_ps['whole_path_bound'] <= _FULL_EXPLORE_THRESHOLD else
          f"  [{label}] after CI collapse: bound="
          f"{reduced_ps['whole_path_bound']:_} "
          f"groups={len(reduced_ps.get('ci_groups', []))}  (> threshold)")
    return rows, ps, reduced_ps


def main(argv: list[str]) -> int:
    reports = [Path(p) for p in argv[1:]]
    sdg = load_sdg(PDG)
    for report in reports:
        print("=" * 78)
        print(report.name)
        base_rows, base_ps, base_ci = _run("baseline", report, sdg, SOURCE_ROOT,
                                           None, want_dense=False)
        new_rows, new_ps, new_ci = _run("ABC", report, sdg, SOURCE_ROOT,
                                        None, want_dense=True)
        base_by = {r["seg"]: r for r in base_rows}
        print("  --- delta ---")
        for r in new_rows:
            b = base_by.get(r["seg"], {})
            if b.get("nodes", 0) == r["nodes"] and b.get("br", 0) == r["br"]:
                continue
            print(f"    seg {r['seg']:>2d} {r['fn'][:26]:26s} "
                  f"nodes {b.get('nodes', 0):>7d} -> {r['nodes']:>7d}   "
                  f"br {b.get('br', 0):>5d} -> {r['br']:>5d}")
        print(f"    bound {base_ps['whole_path_bound']:_} -> "
              f"{new_ps['whole_path_bound']:_}")
        print(f"    CI-collapsed bound {base_ci['whole_path_bound']:_} -> "
              f"{new_ci['whole_path_bound']:_}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
