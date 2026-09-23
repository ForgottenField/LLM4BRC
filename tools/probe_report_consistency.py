#!/usr/bin/env python3
"""Probe the two deterministic stage-6 gates on real reports.

Runs the deterministic stages (1-6) — same harness as
``tools/replay_steps1to6.py``, no LLM — then reports, per report:

  * the fabricated-entry-state finding, if any (the report's entry null
    assumption refuted by the bug function's own always-compiled fatal check);
  * how many (condition, direction) claims the report path yields,
  * which source lines were skipped (ambiguous node/step counts),
  * every self-contradiction found.

Usage:
    python3.10 tools/probe_report_consistency.py [--project <name>] <report|dir> ...
    python3.10 tools/probe_report_consistency.py --project protobuf \
        ~/csa_reports/project/protobuf/reports

The project must be named explicitly (``--project``, default ``faiss``): the
corpus and the ``pdg_<project>.json`` are both derived from it, so a report from
one project probed against another project's graph yields plausible-looking
nonsense.  A directory argument expands to every ``*.html`` beneath it.
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
    CFGCacheSet, _check_pdg_artifact_version, _resolve_source_file_for_cfg,
    build_slice_mask, load_sdg, parse_report, run_branch_filter,
)
from llm_client.fp_analysis.cfg_feasibility import segment_path  # noqa: E402
from llm_client.fp_analysis.entry_state import (  # noqa: E402
    detect_fabricated_entry_state,
)
from llm_client.fp_analysis.report_consistency import (  # noqa: E402
    condition_claims, detect_report_contradictions,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

PROJECTS_DIR = Path.home() / "csa_reports" / "project"
DEFAULT_PROJECT = "faiss"

# Filled in by main() from --project; the probe is single-project by design.
SOURCE_ROOT = PROJECTS_DIR / DEFAULT_PROJECT
PDG = HERE / f"pdg_{DEFAULT_PROJECT}.json"


def probe(report_path: Path, sdg, cache_set) -> None:
    print("=" * 78)
    print(report_path.name)
    parsed_report, bug_path = parse_report(str(report_path), SOURCE_ROOT)

    # The other (and cheaper) halt-before-enumeration exit: the report's entry
    # state is refuted by the bug function's own always-compiled fatal check.
    # It needs only the report and the source tree — no SDG, no CFG cache — so
    # it is probed first and stays visible on reports whose stage-1/2 data is
    # incomplete.  Both gates share the stage-6 early return.
    entry = detect_fabricated_entry_state(parsed_report, str(SOURCE_ROOT))
    if entry is None:
        print("  entry-state fabrication: none")
    else:
        print(f"  entry-state fabrication: ✗ {entry.guard_macro} on "
              f"'{entry.parameter}' at L{entry.guard_line} (deref L{entry.deref_line})"
              f"  call_sites={entry.call_sites}")
        print(f"    reason: {entry.reason()}")

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
    global SOURCE_ROOT, PDG

    args = argv[1:]
    project = DEFAULT_PROJECT
    if args and args[0] in ("--project", "-p"):
        if len(args) < 2:
            print("--project needs a name", file=sys.stderr)
            return 2
        project = args[1]
        args = args[2:]

    SOURCE_ROOT = PROJECTS_DIR / project
    PDG = HERE / f"pdg_{project}.json"
    if not SOURCE_ROOT.is_dir():
        print(f"no such project corpus: {SOURCE_ROOT}", file=sys.stderr)
        return 2
    if not PDG.is_file():
        print(f"no such PDG: {PDG}", file=sys.stderr)
        return 2
    _check_pdg_artifact_version(PDG, project)

    reports: list[Path] = []
    for raw in args:
        p = Path(raw)
        if p.is_dir():
            reports.extend(sorted(p.rglob("*.html")))
        else:
            reports.append(p)
    if not reports:
        print("no reports given", file=sys.stderr)
        return 2

    print(f"project: {project}  source_root: {SOURCE_ROOT}  pdg: {PDG.name}")
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
