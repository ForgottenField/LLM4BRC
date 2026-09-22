#!/usr/bin/env python3
"""Probe the deterministic report self-contradiction check on real reports.

Runs the deterministic stages (1-6) — same harness as
``tools/replay_steps1to6.py``, no LLM — then reports, per report:

  * how many (condition, direction) claims the report path yields,
  * which source lines were skipped (ambiguous node/step counts),
  * every self-contradiction found.

Usage:
    python3.10 tools/probe_report_consistency.py <report.html> [more.html ...]
"""
from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))

from run_path_selection import (  # noqa: E402
    CFGCacheSet, _resolve_source_file_for_cfg, build_slice_mask,
    load_sdg, parse_report, run_branch_filter,
)
from llm_client.fp_analysis.cfg_feasibility import segment_path  # noqa: E402
from llm_client.fp_analysis.report_consistency import (  # noqa: E402
    condition_claims, detect_report_contradictions,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

SOURCE_ROOT = Path.home() / "csa_reports" / "project" / "faiss"
PDG = HERE / "pdg_faiss.json"


def probe(report_path: Path, sdg, cache_set) -> None:
    print("=" * 78)
    print(report_path.name)
    parsed_report, bug_path = parse_report(str(report_path), SOURCE_ROOT)

    bug_source = _resolve_source_file_for_cfg(parsed_report, str(SOURCE_ROOT))
    cache_set.ensure(bug_source)

    segments = segment_path(parsed_report, sdg=sdg, source_root=str(SOURCE_ROOT))
    for seg in segments:
        if seg.function_file:
            cache_set.ensure(seg.function_file)
    cfg_cache = cache_set.as_dict()

    slice_mask, _ = build_slice_mask(
        sdg, parsed_report, bug_path, SOURCE_ROOT, with_slice_result=True)

    run_branch_filter(
        segments, parsed_report, cfg_cache, sdg, slice_mask, SOURCE_ROOT,
        established_state={}, cache_set=cache_set)

    claims = condition_claims(segments, parsed_report.path_events)
    print(f"  segments: {len(segments)} | condition claims: {len(claims)}")
    per_fn = Counter(c.function_name for c in claims)
    for fn, n in per_fn.most_common(6):
        print(f"    {n:>4d}  {fn}")

    contradictions = detect_report_contradictions(
        segments, parsed_report.path_events, source_root=str(SOURCE_ROOT))
    print(f"  contradictions: {len(contradictions)}")
    for c in contradictions:
        print(f"    ✗ {c.condition}  vars={c.variables}")
        print(f"      L{c.first.line} ({'T' if c.first.direction else 'F'})"
              f" event#{c.first.event_number}  vs  "
              f"L{c.second.line} ({'T' if c.second.direction else 'F'})"
              f" event#{c.second.event_number}  [{c.first.function_name}]")
    if contradictions:
        print(f"  reason: {contradictions[0].reason()}")


def main(argv: list[str]) -> int:
    reports = [Path(p) for p in argv[1:]]
    t0 = time.time()
    sdg = load_sdg(PDG)
    print(f"SDG loaded: {len(sdg.pdgs)} fn [{time.time()-t0:.1f}s]")
    for r in reports:
        cache_set = CFGCacheSet(project_name=SOURCE_ROOT.name,
                               source_root=str(SOURCE_ROOT))
        probe(r, sdg, cache_set)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
