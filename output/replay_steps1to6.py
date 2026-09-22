#!/usr/bin/env python3
"""Replay the deterministic stages (1-6) of the pipeline for a set of reports.

Steps 1-4 and 6 are pure static analysis; only Step 5.5 (assumption check)
needs the LLM, and it is skipped here — ``established_state={}`` makes the
conflict checker slightly more conservative but does NOT change which segments
resolve to a CFG function or how many branches are extracted from them.  That
is exactly the "extracted path count" question this script answers:

    per segment: function_name | function_file | CFG key found? | branches

Usage:
    python3.10 output/replay_steps1to6.py <report.html> [more.html ...]
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
    load_sdg, parse_report, run_branch_filter,
)
from llm_client.fp_analysis.cfg_feasibility import segment_path  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

SOURCE_ROOT = Path.home() / "csa_reports" / "project" / "faiss"
PDG = HERE / "pdg_faiss.json"


def replay(report_path: Path, sdg, cache_set) -> None:
    print("=" * 78)
    print(report_path.name)
    parsed_report, bug_path = parse_report(str(report_path), SOURCE_ROOT)

    bug_source = _resolve_source_file_for_cfg(parsed_report, str(SOURCE_ROOT))
    hit = cache_set.ensure(bug_source)
    print(f"  bug file: {bug_source}  (cache: {'yes' if hit else 'NO'})")

    segments = segment_path(parsed_report, sdg=sdg, source_root=str(SOURCE_ROOT))
    print(f"  segments: {len(segments)}")

    new_tus = [s.function_file for s in segments
               if s.function_file and cache_set.ensure(s.function_file)]
    cfg_cache = cache_set.as_dict()
    print(f"  step 3.5: +{len(new_tus)} cache(s) -> {len(cfg_cache)} fn total")

    slice_mask, _ = build_slice_mask(
        sdg, parsed_report, bug_path, SOURCE_ROOT, with_slice_result=True)

    branch_results, _ = run_branch_filter(
        segments, parsed_report, cfg_cache, sdg, slice_mask, SOURCE_ROOT,
        established_state={}, cache_set=cache_set)

    by_idx = {r["segment_index"]: r for r in branch_results}
    print(f"  {'seg':>3s} {'fn':28s} {'file':24s} {'cfgkey':>7s} {'br':>4s}  lines")
    unresolved = 0
    for i, seg in enumerate(segments):
        fn = seg.function_name or "<empty>"
        ffile = Path(seg.function_file).name if seg.function_file else "<none>"
        key = cache_set.find_key(seg.function_name,
                                 source_file=seg.function_file or "")
        r = by_idx.get(i)
        nbr = r["num_branches"] if r else 0
        mark = "ok" if key else "MISS"
        if not key:
            unresolved += 1
        print(f"  {i:>3d} {fn:28s} {ffile:24s} {mark:>7s} {nbr:>4d}  "
              f"L{seg.line_start}-L{seg.line_end}")
    total = sum(r["num_branches"] for r in branch_results)
    print(f"  -> {len(branch_results)}/{len(segments)} segments resolved, "
          f"{total} branches, {unresolved} unresolved")


def main(argv: list[str]) -> int:
    reports = [Path(p) for p in argv[1:]]
    t0 = time.time()
    sdg = load_sdg(PDG)
    print(f"SDG loaded: {len(sdg.pdgs)} fn [{time.time()-t0:.1f}s]")
    for r in reports:
        # A fresh cache set per report: the pipeline processes one report per
        # run, and the merged cache's key set decides which CFG key a bare
        # method name resolves to.  Sharing it across reports would let an
        # earlier report's translation units change a later report's answer.
        cache_set = CFGCacheSet(project_name=SOURCE_ROOT.name,
                                source_root=str(SOURCE_ROOT))
        replay(r, sdg, cache_set)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
