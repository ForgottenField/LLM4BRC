"""``process_report`` — the ten stages of one report, end to end.

This module is where a work item becomes a verdict.  The stage modules below it
(``stages``, ``selection``, ``exploration``) each produce *evidence*; the
report-level TP / FP / UNKNOWN is set here, in ``exploration``'s aggregation, and
nowhere else.

``process_report`` is now the ten stages as ten calls, one per line, in the order
``docs/pipeline-stages.md`` states them.  Each ``_stageN_*`` adapter holds that
stage's **verbatim** body from the single 1123-line function this replaced — the
prints stay inside the stage bodies, so call order is statement order is print
order, and a reader who knew the old function will recognise every line.

Two exits leave early and both are deterministic: the stage-6 self-contradiction
gate and the stage-7 semantic finite-domain gate.  They return the result JSON
they wrote, and ``process_report`` returns it unchanged.

The one thing the split had to say out loud is ``t0``: the old body used a single
local as the start timestamp for stage 3's elapsed print, then re-used it for the
analyzer's print, then re-set it for stage 5 and re-used it for the stage-6 gate's
prints.  It is now a parameter of the functions that print the elapsed time, set
by the caller immediately before the call that starts the clock — which is what
the old code did, and the reason it is a parameter rather than a fresh
``time.time()`` inside each callee.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

from llm_client.base import LLMConfig
from llm_client.config import provider_spec
from llm_client.credentials import load_api_key
from fp_analysis.branch_relevance import (
    classify_branch_relevance,
    reduce_path_space,
)
from fp_analysis.cfg_cache_set import CFGCacheSet
from fp_analysis.cfg_feasibility import resolve_source_file_for_cfg
from fp_analysis.combination_enum import iter_combinations
from fp_analysis.constraint_model import render_bundle
from fp_analysis.domain_facts import extract_domain_facts, format_domain_facts
from fp_analysis.entry_state import detect_fabricated_entry_state
from fp_analysis.models import code_model_of
from fp_analysis.path_analyzer import FPAnalyzer
from fp_analysis.path_seed_extractor import build_dense_path_annotations
from fp_analysis.path_space import (
    _chosen_by_seg,
    annotate_ci_groups,
    collect_path_space,
    render_path_space_ascii,
)
from fp_analysis.report_consistency import detect_report_contradictions

from pipeline.artifacts import (
    _write_output,
    code_model_report,
    report_fields,
    report_out_path,
)
from pipeline.exploration import (
    _aggregate_exploration,
    _run_path_attempts,
)
from pipeline.layout import DEFAULT_PROJECTS_DIR, get_logger
from pipeline.lite_graph import build_lite_graph
from pipeline.models import (
    AssembledProject,
    ExplorationState,
    SelectionContext,
    Stage1Parse,
    Stage3Slice,
    Stage4Assumptions,
    Stage5Branches,
    Stage8PathSpace,
    VerdictState,
)
from pipeline.outcomes import Verdict
from pipeline.project_paths import resolve_cfg_cache, load_sdg
from pipeline.project_paths import resolve_paths
from pipeline.schemas import ResultDocument
from pipeline.selection import _NO_PATH_STEP_KEYWORD
from pipeline.stages import (
    infer_entry_function,
    build_slice_mask,
    parse_report,
    run_branch_filter,
    segment_path,
    stage_label,
    stage_step,
)

logger = get_logger()

_DEFAULT_PATH_ATTEMPT_LIMIT = 10
_MAX_PATH_ATTEMPTS_HARD_CAP = 10

# Compatibility name used by tests and tools that patch the runner seam.
_resolve_paths = resolve_paths


def _path_exploration_budget(
    requested_max_attempts: int, reduced_bound: int,
) -> tuple[int, bool]:
    """Return the effective attempt count and whether the reduced space is exhausted.

    The configured value is clamped to a hard cap of ten.  A path space is marked
    exhausted only when every reduced-space combination is within that cap; a
    ten-attempt sample of a larger space must remain UNKNOWN when all attempts
    are unresolved or FP.
    """
    requested = max(1, int(requested_max_attempts))
    effective_cap = min(requested, _MAX_PATH_ATTEMPTS_HARD_CAP)
    normalized_bound = max(1, int(reduced_bound))
    attempts = min(normalized_bound, effective_cap)
    exhausted = normalized_bound > 1 and attempts == normalized_bound
    return attempts, exhausted

def _print_run_banner(args, report_ref, source_root, pdg_path, cfg_cache_path) -> None:
    """The run header: what is being analysed and against which project data.

    Printed before stage 1 so that a misinvocation is legible in the log even
    when stage 1 then fails — which is the same reason ``main`` resolves every
    work item up front.
    """
    print("=" * 70)
    print("END-TO-END PATH SELECTION PIPELINE")
    print("=" * 70)
    print(f"Report:      {report_ref.path}"
          + (f"\nFinding:     {report_ref.selector}" if report_ref.selector else ""))
    print(f"Project:     {source_root}      ← 项目源树（报告/源在此）")
    print(f"  └ projects-dir: {args.projects_dir}")
    print(f"PDG:         {pdg_path or 'not present'}")
    print("Graph mode:  "
          + ("exact (forced)" if getattr(args, "no_lite_graph", False)
             else "Tree-sitter fast path; exact PDG fallback"))
    print(f"CFG Cache:   {cfg_cache_path}")
    print("Algorithm:   conflict-driven path selection + POC verification")
    print(f"  (项目源+报告默认在 {DEFAULT_PROJECTS_DIR}/<name>/；"
          f"pdg/cfg 缓存在仓库根 pdg_<name>.json + cfg_cache/<name>/ 下)")
    print()


def _stage1_parse(args, report_ref, source_root) -> Stage1Parse:
    """Stage 1a: the report through the frontend, rendered into the legacy dicts."""
    # ── Stage 1: parse the report, then assemble the project data ──
    t0 = time.time()
    # Parse through the *ref*, not `report_path`: one finding is one unit of
    # work, and the selector is the half of the ref that names which one.  It
    # was dropped here (the line above rebinds `report_path` to `ref.path`), so
    # every run reached the frontend with a selector-less ref and a
    # multi-finding document — which `load` correctly refuses, since picking one
    # itself would be a silent choice.  `--report '<doc>::<finding>'` and the
    # `--scan-dir` expansion both died there, i.e. no SARIF finding could be
    # analysed at all.
    parsed_report, bug_path, loaded = parse_report(
        report_ref, source_root, report_format=getattr(args, "report_format", None),
    )
    entry_fn = parsed_report.metadata.function_name
    print(f"{stage_label(1)}: Parsed report — {entry_fn or '(unnamed)'} | "
          f"{parsed_report.metadata.bug_type} "
          f"({len(parsed_report.path_events)} events) [{time.time()-t0:.1f}s]")
    code_summary = loaded.summary()["code"]
    print(f"  └ {loaded.document.format.value} / {loaded.finding.analyzer.value}"
          + (f" · finding {loaded.ref.selector}" if loaded.ref.selector else "")
          + f" · 报告路径解析 {code_summary['resolved_files']}/"
            f"{code_summary['reported_files']}"
          + (f" · 弱匹配 {len(code_summary['weak'])}" if code_summary["weak"] else "")
          + (f" · 未解析 {len(code_summary['unresolved'])}"
             if code_summary["unresolved"] else ""))
    return Stage1Parse(parsed_report, bug_path, loaded, entry_fn)


def _stage1_assemble(
    source_root,
    cfg_cache_path,
    pdg_path,
    parsed_report,
    *,
    lite_graph=None,
    prefer_exact: bool = False,
) -> AssembledProject:
    """Stage 1b: select the graph backend and load the available CFG cache."""
    # ── Stage 1 (cont.): load the SDG + the CFG cache ──
    # The CFG cache is per translation unit, so start from the bug host's own
    # cache and pull in the rest on demand once segmentation reveals which
    # files the path actually visits (stage 2 below).
    t0 = time.time()
    cache_set = CFGCacheSet(
        project_name=source_root.name, source_root=str(source_root),
    )
    if cfg_cache_path is not None:
        cache_set.add_cache_file(cfg_cache_path)
    else:
        bug_source = resolve_source_file_for_cfg(parsed_report, str(source_root))
        if not cache_set.ensure(bug_source):
            logger.warning(
                "no CFG cache for bug file %s — path files will be loaded on "
                "demand; preheat cfg_cache/%s/ offline to cover the rest",
                bug_source, source_root.name,
            )
    cfg_cache = cache_set.as_dict()
    use_lite = (
        not prefer_exact
        and lite_graph is not None
        and lite_graph.available
        and (
            _lite_covers_report(lite_graph, parsed_report)
            or pdg_path is None
        )
    )
    if use_lite:
        sdg = lite_graph.sdg
        graph_backend = "tree-sitter"
    elif pdg_path is not None:
        sdg = load_sdg(pdg_path)
        graph_backend = "exact"
    else:
        detail = getattr(lite_graph, "error", "") if lite_graph is not None else ""
        raise RuntimeError(
            "No analysis graph is available: Tree-sitter produced no usable "
            "graph"
            + (f" ({detail})" if detail else "")
            + " and no exact PDG artifact exists. Pass --pdg to use the exact "
              "fallback."
        )
    if lite_graph is not None:
        lite_graph.summary["selected_backend"] = graph_backend
        lite_graph.summary["exact_fallback_available"] = pdg_path is not None
    print(
        f"{stage_label(1)}: Loaded CFG ({len(cfg_cache)} fn) + "
        f"{graph_backend} SDG ({len(sdg.pdgs)} fn) "
        f"[{time.time()-t0:.1f}s]"
    )
    return AssembledProject(cache_set, cfg_cache, sdg, graph_backend)


def _lite_covers_report(lite_graph, parsed_report) -> bool:
    """Whether the bounded syntax graph covers the report entry location."""
    if lite_graph is None or not lite_graph.available:
        return False
    function_name = getattr(
        getattr(parsed_report, "metadata", None), "function_name", "",
    )
    if function_name and lite_graph.sdg.get_pdg(function_name) is not None:
        return True
    bug_file = getattr(
        getattr(parsed_report, "metadata", None), "bug_file", "",
    )
    bug_line = int(getattr(
        getattr(parsed_report, "metadata", None), "bug_line", 0,
    ) or 0)
    bug_name = Path(bug_file).name if bug_file else ""
    if not bug_name:
        return False
    for pdg in lite_graph.sdg.pdgs.values():
        if Path(pdg.source_file or "").name != bug_name:
            continue
        if not bug_line or any(
            node.source_line == bug_line for node in pdg.nodes.values()
        ):
            return True
    return False


def _stage1_resolve_entry(entry_fn, sdg, parsed_report) -> str:
    """The entry function, inferred from the PDG when the report named none."""
    # A report that names no entry function (every SARIF one) gets it from the
    # PDG now that the PDG is loaded.  Before segmentation, because the
    # caller-map walk and the CFG-cache last resort both want it.
    if not entry_fn:
        entry_fn = infer_entry_function(sdg, parsed_report)
        parsed_report.metadata.function_name = entry_fn
        print(f"{stage_label(1)}: entry function from PDG — {entry_fn or '(none found)'}")
    return entry_fn


def _stage2_segment(parsed_report, sdg, source_root) -> list:
    """Stage 2: cut the bug path into segments at its control events."""
    # ── Stage 2: segment the bug path ──
    t0 = time.time()
    segments = segment_path(parsed_report, sdg=sdg, source_root=str(source_root))
    print(f"{stage_label(2)}: Segmented into {len(segments)} segment(s) "
          f"[{time.time()-t0:.1f}s]")
    return segments


def _stage2_extend_caches(cache_set, segments, cfg_cache, cfg_cache_path,
                          source_root, entry_fn) -> None:
    """Stage 2 (cont.): pull in the CFG of every file the segmented path crosses.

    ``cfg_cache`` is passed in only so the count printed is the merged view's, and
    so the last-resort lookup can tell "no cache anywhere" from "no cache for the
    bug file"; the mutation itself goes through ``cache_set``, which is what makes
    these visible to stage 5.
    """
    # ── Stage 2 (cont.): load the CFG of every file the path crosses ──
    # ``cfg_cache`` is the live merged view, so these show up for stage 5.
    t0 = time.time()
    new_tus = [seg.function_file for seg in segments
               if seg.function_file and cache_set.ensure(seg.function_file)]
    if new_tus:
        print(f"{stage_label(2)}: +{len(new_tus)} CFG cache(s) for path files "
              f"({len(cfg_cache)} fn total) [{time.time()-t0:.1f}s]")

    # Nothing anywhere: the bug file wasn't a TU and the path's own files are
    # unresolved (report without per-file sections).  Locate the TU that
    # defines the entry function as a last resort.
    if not cfg_cache and cfg_cache_path is None:
        fallback = resolve_cfg_cache(source_root, entry_fn)
        if fallback is not None:
            logger.warning(
                "no CFG cache for any path file — falling back to %s "
                "(defines '%s')", fallback.name, entry_fn,
            )
            cache_set.add_cache_file(fallback)
    cache_set.warn_missing()


def _stage3_slice(sdg, parsed_report, bug_path, source_root, t0) -> Stage3Slice:
    """Stage 3: the backward slice, plus the dense CSA branch decisions.

    ``t0`` is the caller's clock start (see the module docstring): it is *not*
    re-read here, because the analyzer's ready-print downstream measures from
    stage 3's start and the old body shared one local for both.
    """
    # ── Stage 3: build the SliceMask (PDG backward slice) ──
    file_index = sdg.build_file_index()
    slice_mask, slice_result = build_slice_mask(
        sdg, parsed_report, bug_path, source_root, with_slice_result=True,
        file_index=file_index,
    )

    # Dense CSA decisions: which branch each predicate line took on the reported
    # path.  Used by the branch filter to expand only calls the path can reach —
    # a segment's line range routinely spans the then-body of a branch the CSA
    # took as false (`_find_calls_in_pdg_range` alone would expand it).
    # Deliberately NOT fed to the slicer: the slice's own BTBS pass keeps its
    # narrower CONTROL-only view.
    path_annotations = build_dense_path_annotations(
        bug_path, sdg, source_root=str(source_root), file_index=file_index,
    )
    print(f"{stage_label(3)}: {len(path_annotations)} CSA branch decision line(s)")
    return Stage3Slice(file_index, slice_mask, slice_result, path_annotations)


def _build_analyzer(
    args, source_root, pdg_path, t0, sdg=None, lite_graph=None,
) -> FPAnalyzer:
    """The LLM provider and the analyzer — infrastructure for stages 4 and 9.

    Built here, *before* the branch filter, so the assumption-derived state can
    feed conflict checking (Layers 1b/1.5), which previously never received any
    ``established_state`` and therefore pruned nothing.
    """
    # ── LLM provider + FPAnalyzer (infrastructure for stages 4 and 9) ──
    # Created here (before the branch filter) so the assumption-derived state
    # can feed into conflict checking (Layers 1b/1.5), which previously never
    # received any established_state and therefore pruned nothing.
    spec = provider_spec(getattr(args, "provider", "deepseek"))
    api_key = getattr(args, "api_key", "") or load_api_key(
        spec.name, getattr(args, "api_key_file", None),
    )
    model = getattr(args, "model", None) or spec.default_model
    config = LLMConfig(
        provider_name=spec.name,
        api_key=api_key,
        default_model=model,
        default_timeout=getattr(args, "llm_timeout", 120.0),
    )
    # FPAnalyzer creates LLMProvider internally from config
    analyzer = FPAnalyzer(
        llm_config=config,
        model=model,
        source_root=str(source_root),
        pdg_path=str(pdg_path) if pdg_path is not None else None,
        report_format=getattr(args, "report_format", None),
        sdg=sdg,
    )
    if lite_graph is not None and lite_graph.prompt_text:
        analyzer.set_lite_evidence(lite_graph.prompt_text)
    print(f"  FPAnalyzer ready (model={analyzer.model_name}) [{time.time()-t0:.1f}s]")
    return analyzer


def _stage4_assumptions(analyzer, segments, parsed_report, source_root) -> Stage4Assumptions:
    """Stage 4: assumption feasibility — the pipeline's first LLM call."""
    # ── Stage 4: assumption feasibility (runs before the branch filter) ──
    t0 = time.time()
    global_constraints = analyzer.load_global_constraints(
        str(source_root),
        target_classes=analyzer.extract_class_names_from_path(parsed_report),
    )
    hard_constraints, cumulative_state = analyzer.check_assumptions_feasibility(
        segments=segments,
        parsed=parsed_report,
        global_constraints=global_constraints,
    )
    assumption_elapsed = time.time() - t0
    n_hc = len(hard_constraints)
    n_state = len(cumulative_state)
    print(f"{stage_label(4)}: Assumption check — {n_hc} hard constraint(s), "
          f"{n_state} state binding(s) [{assumption_elapsed:.1f}s]")
    for hc in hard_constraints:
        print(f"    🔒 {hc}")
    for var, val in cumulative_state.items():
        print(f"    📎 {var}: {val}")
    constraint_solution = getattr(
        analyzer, "_global_constraint_solution", None,
    )
    if constraint_solution:
        print(
            f"    🧮 Global data constraints: {constraint_solution.get('status')} "
            f"({len(constraint_solution.get('contradictions') or [])} contradiction(s))"
        )
    return Stage4Assumptions(hard_constraints, cumulative_state,
                             assumption_elapsed, n_hc, n_state,
                             constraint_solution)


def _stage5_branch_filter(segments, parsed_report, cfg_cache, sdg, slice_mask,
                          source_root, cumulative_state, cache_set,
                          path_annotations, t0) -> Stage5Branches:
    """Stage 5: the four-layer branch conflict check, seeded with stage 4's state.

    ``t0`` is shared with :func:`_stage6_gate` (see the module docstring).
    """
    # ── Stage 5: branch filter (seeded with assumption-derived state) ──
    branch_results, insufficient_counts = run_branch_filter(
        segments, parsed_report, cfg_cache, sdg, slice_mask, source_root,
        established_state=cumulative_state, cache_set=cache_set,
        path_annotations=path_annotations,
    )
    print(f"{stage_label(5)}: Branch filter ({len(branch_results)} segments) "
          f"[{time.time()-t0:.1f}s]")

    # ── Print branch filter summary ──
    total_b = sum(r["num_branches"] for r in branch_results)
    total_con = sum(r["num_consistent"] for r in branch_results)
    total_ctr = sum(r["num_contradictory"] for r in branch_results)
    total_ins = sum(r["num_insufficient"] for r in branch_results)
    print(f"  Branches: {total_b} total | {total_con} consistent | {total_ctr} pruned | {total_ins} insufficient")
    total_b = sum(r["num_branches"] for r in branch_results)
    total_con = sum(r["num_consistent"] for r in branch_results)
    total_ctr = sum(r["num_contradictory"] for r in branch_results)
    total_ins = sum(r["num_insufficient"] for r in branch_results)
    return Stage5Branches(branch_results, insufficient_counts,
                         total_b, total_con, total_ctr, total_ins)


def _stage6_gate(args, report_ref, segments, parsed_report, source_root, entry_fn,
                 t0, total_start, lite_graph=None,
                 constraint_solution=None) -> ResultDocument | None:
    """Stage 6: the two deterministic FP gates, sharing one exit.

    Returns the result JSON when a gate fired, ``None`` otherwise.  ``t0`` is
    stage 5's clock start (the console shows the whole branch-filter-to-gate span
    as one elapsed figure) and ``total_start`` the run's, for the JSON.
    """
    # ── Stage 6: report self-contradiction gate (deterministic, no LLM) ──
    # The report's OWN path states a direction for every branch condition it
    # walks.  When one condition is asserted BOTH ways on that path, with no
    # write to its variables in between, no input can realize the path: the
    # report is an FP, provable with no path enumeration, no POC and no model
    # call — settled here, ahead of path selection, POC generation and the
    # simulated execution (the stage-4 assumption check is the only LLM call
    # that has run by this point).  CSA cannot refute such a pair itself when an
    # operand comes from an out-of-line function (faiss' `fourcc`): its two call
    # results are independent free symbols to the analyzer's solver, which is
    # why the report was emitted at all.
    # `fp_analysis/report_consistency.py` documents the soundness guards (same
    # function+file, different lines, no intervening write, no call step
    # between the claims, unambiguous line pairing) — every one fails closed, so
    # a missed contradiction only degrades to the pipeline's usual behaviour.
    #
    # A second, independent detector shares this exit: `entry_state.py` reports
    # the case where the report's own path *manufactures the entry state* — CSA
    # assumes a pointer parameter is null, walks the FAILURE arm of an
    # always-compiled check on that parameter (ABSL_RAW_CHECK/CHECK/
    # FAISS_THROW_IF_NOT, whose failure arm aborts or throws) and then claims the
    # deref is reached.  The failure arm does not return, so the path is not
    # realizable.  Same character as the pair detector (the report contradicts
    # itself, deterministically, without any input), same exit, different
    # evidence — hence one shared block below.
    report_contradictions: list = []
    try:
        report_contradictions = detect_report_contradictions(
            segments, parsed_report.path_events, source_root=str(source_root),
            model=code_model_of(parsed_report))
    except Exception as e:  # noqa: BLE001 — gate must never be fatal
        logger.warning("Report self-contradiction gate failed: %s", e)
        report_contradictions = []
    entry_finding = None
    try:
        entry_finding = detect_fabricated_entry_state(
            parsed_report, source_root)
    except Exception as e:  # noqa: BLE001 — gate must never be fatal
        logger.warning("Entry-state fabrication gate failed: %s", e)
        entry_finding = None
    constraint_conflicts = []
    if constraint_solution and constraint_solution.get("status") == "unsat":
        constraint_conflicts = list(
            constraint_solution.get("contradictions") or []
        )
    if report_contradictions or entry_finding or constraint_conflicts:
        _fp_reasons: list[str] = []
        if report_contradictions:
            _c0 = report_contradictions[0]
            print(f"\n{stage_label(6)}: report path self-contradiction — "
                  f"{len(report_contradictions)} pair(s) "
                  f"[{time.time()-t0:.1f}s]")
            for _c in report_contradictions[:3]:
                print(f"    ✗ {_c.condition} — L{_c.first.line}"
                      f"({'TRUE' if _c.first.direction else 'FALSE'}, "
                      f"event #{_c.first.event_number}) vs L{_c.second.line}"
                      f"({'TRUE' if _c.second.direction else 'FALSE'}, "
                      f"event #{_c.second.event_number})")
            _fp_reasons.append(_c0.reason())
        if entry_finding:
            print(f"\n{stage_label(6)}: fabricated entry state — "
                  f"{entry_finding.guard_macro} at L{entry_finding.guard_line} "
                  f"[{time.time()-t0:.1f}s]")
            print(f"    ✗ {entry_finding.guard_source}")
            print(f"    ✗ CSA 报告在 L{entry_finding.guard_line} 取失败分支"
                  f"（事件 #{entry_finding.branch_event_number}），"
                  f"却声明 L{entry_finding.deref_line} 处解引用 "
                  f"'{entry_finding.parameter}'")
            _fp_reasons.append(entry_finding.reason())
        if constraint_conflicts:
            _c = constraint_conflicts[0]
            print(f"\n{stage_label(6)}: global data constraints unsatisfiable — "
                  f"{len(constraint_conflicts)} contradiction(s) "
                  f"[{time.time()-t0:.1f}s]")
            print(f"    ✗ {_c.get('reason') or _c.get('subject', '')}")
            _fp_reasons.append(
                str(_c.get("reason")
                    or f"Global data constraints for {_c.get('subject', '')} are unsatisfiable")
            )
        _fp_reason = " ".join(_fp_reasons)
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print("  Verdict:    ❌ FP (no feasible path — the report contradicts itself)")
        print(f"  Reason:     {_fp_reason}")
        _out = report_out_path(args, report_ref)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _output = {
            **report_fields(report_ref),
            "algorithm": "conflict-driven",
            "entry_function": entry_fn,
            "bug_type": parsed_report.metadata.bug_type,
            "verdict": Verdict.FP.value,
            "verdict_reason": _fp_reason,
            "verdict_steps": (
                [f"报告自相矛盾判定: {c.reason()}" for c in report_contradictions]
                + ([f"入口状态不可实现判定: {entry_finding.reason()}"]
                   if entry_finding else [])
                + [f"全局数据约束求解: {c.get('reason', '')}"
                   for c in constraint_conflicts]
            ),
            "total_elapsed_sec": round(time.time() - total_start, 1),
        }
        if report_contradictions:
            _c0 = report_contradictions[0]
            _output["report_self_contradiction"] = {
                "decisive_fp": True,
                "reason": _c0.reason(),
                "condition": _c0.condition,
                "variables": list(_c0.variables),
                "first": {
                    "line": _c0.first.line,
                    "direction": _c0.first.direction,
                    "event_number": _c0.first.event_number,
                },
                "second": {
                    "line": _c0.second.line,
                    "direction": _c0.second.direction,
                    "event_number": _c0.second.event_number,
                },
                "function": _c0.first.function_name,
                "file": _c0.first.source_file,
                "pairs": [
                    {
                        "condition": c.condition,
                        "line": c.first.line,
                        "event_number": c.first.event_number,
                        "other_line": c.second.line,
                        "other_event_number": c.second.event_number,
                    }
                    for c in report_contradictions
                ],
            }
        if entry_finding:
            _output["entry_state_fabrication"] = {
                "decisive_fp": True,
                "kind": entry_finding.kind,
                "reason": entry_finding.reason(),
                "function": entry_finding.function,
                "file": str(entry_finding.file),
                "parameter": entry_finding.parameter,
                "guard_macro": entry_finding.guard_macro,
                "guard_condition": entry_finding.guard_condition,
                "guard_line": entry_finding.guard_line,
                "guard_source": entry_finding.guard_source,
                "guard_semantics": entry_finding.guard_semantics,
                "assumption_event_number": entry_finding.assumption_event_number,
                "branch_event_number": entry_finding.branch_event_number,
                "deref_line": entry_finding.deref_line,
                "call_sites_in_project": entry_finding.call_sites,
                "call_site_samples": list(entry_finding.call_site_samples),
            }
        if constraint_conflicts:
            _output["global_constraint_solve"] = {
                "decisive_fp": True,
                **constraint_solution,
            }
        # The verdict step names the detector that fired; when both did, the
        # condition-pair detector keeps the canonical suffix and the entry-state
        # finding is listed in `verdict_steps` alongside it.
        _output["verdict_step"] = (
            stage_step(6, "（同一条件同路径反向）")
            if report_contradictions
            else (
                stage_step(6, "（入口状态制造：穿过必然终止的检查）")
                if entry_finding
                else stage_step(6, "（全局数据约束不可满足）")
            )
        )
        _output["code_model"] = code_model_report(parsed_report, source_root)
        _output["lite_graph"] = (
            lite_graph.to_dict() if lite_graph is not None else None
        )
        _write_output(_out, _output, total_start)
        return _output

    return None


def _stage7_preamble(cumulative_state, constraint_solution=None,
                     constraint_bundle=None) -> list[str]:
    """Stage 7 preamble: stage 4's state becomes a block of the selection prompt."""
    # ── Stage 7 preamble: inject assumption-derived state as preconditions ──
    # (the stage-4 state becomes part of the selection prompt in stage 9)
    extra_preconditions: list[str] = []
    if cumulative_state:
        state_lines = "\n".join(
            f"  - {var}: {val}" for var, val in cumulative_state.items()
        )
        extra_preconditions.append(
            f"[Assumption-derived state from Error Start / CSA assumptions]\n"
            f"{state_lines}"
        )
    if constraint_solution or constraint_bundle is not None:
        parts = ["[Global structured data constraints]"]
        if constraint_solution:
            parts.append(f"Solver status: {constraint_solution.get('status')}")
            if constraint_solution.get("derived_facts"):
                parts.append(
                    "Derived facts: "
                    + "; ".join(
                        f"{item.get('subject')} {item.get('operator')} "
                        f"{item.get('value')}"
                        for item in constraint_solution["derived_facts"][:20]
                    )
                )
            if constraint_solution.get("missing_information"):
                parts.append(
                    "Missing information: "
                    + "; ".join(constraint_solution["missing_information"][:10])
                )
        if constraint_bundle is not None:
            parts.append(render_bundle(constraint_bundle))
        extra_preconditions.append("\n".join(parts))
    return extra_preconditions


def _stage7_supplement(analyzer, segments, parsed_report, sdg, slice_mask,
                       extra_preconditions, lite_graph=None) -> None:
    """Stage 7 (advisory): guard-pinned / finite-domain facts, prose only.

    Mutates ``extra_preconditions`` in place and stashes the same text on the
    analyzer for the simulation auditor.  ``_domain_facts_text`` is reset inside
    the ``try`` and *after* the call, and the ``except`` path deliberately does not
    reset it — the attribute is a side channel the verifier reads, so where the
    reset sits is behaviour, not style.
    """
    # ── Stage 7 (supplement, advisory): guard-pinned / finite-domain variables ──
    # CSA (and the tool's own constraint model) treat some variables — e.g. a file-read
    # header discriminator compared against an out-of-line constant factory such as
    # `fourcc("rrot")` in another TU — as able to take *any* value, when source actually
    # pins them to a small, fixed constant set.  Surfacing that to the path-selection and
    # simulation LLM lets it conclude reachability is impossible where CSA saw none.
    # This half is advisory prose only; the decisive gate is immediately below.
    if lite_graph is not None and lite_graph.prompt_text:
        extra_preconditions.append(lite_graph.prompt_text)
    try:
        target_fns = {
            s.function_name for s in segments if getattr(s, "function_name", "")
        }
        target_fns.add(parsed_report.metadata.function_name)
        domain_facts = extract_domain_facts(sdg, slice_mask, target_fns)
        # Default: no facts (nothing appended to selection prompt nor the simulator).
        analyzer.set_domain_facts_text("")
        if domain_facts:
            fact_text = format_domain_facts(domain_facts)
            extra_preconditions.append(
                "[Guard-pinned / finite-domain variables (semantic completion)]\n"
                f"{fact_text}"
            )
            # Also stash on the analyzer so the simulation auditor (reuses this same
            # FPAnalyzer) can append the facts to its own prompt (see simulate_execution).
            analyzer.set_domain_facts_text(fact_text)
            print(f"{stage_label(7)}: domain supplement — {len(domain_facts)} guard-pinned "
                  f"variable fact(s) surfaced to LLM")
            for df in domain_facts:
                ffn = df.function_name.split("::")[-1] or df.function_name
                print(f"    🎯 {df.variable} ({ffn}): {len(df.folded)} distinct constants")
    except Exception as e:  # noqa: BLE001 — best-effort supplement, never fatal
        print(f"{stage_label(7)}: domain supplement skipped ({e})")


def _stage7_gate(analyzer, parsed_report, args, report_ref, entry_fn, source_root,
                 total_start, lite_graph=None) -> ResultDocument | None:
    """Stage 7: the decisive semantic finite-domain conflict gate.

    Returns the result JSON on a decisive answer, ``None`` when the gate declines
    (no answer, a failed call, or a non-decisive one) — in which case selection
    proceeds normally.
    """
    # ── Stage 7: decisive semantic finite-domain conflict gate (pre-enumeration) ──
    # Before enumerating the (possibly huge) path space, ask the LLM whether the
    # reported path forces a guard-pinned / finite-domain variable (e.g. a fourcc
    # tag `h`, which CSA treats as free because `fourcc` is an opaque/out-of-line
    # constant factory) into a self-contradiction — admitted into the deref region
    # only when `h ∈ S`, yet every pointer-assigning dispatch arm taken FALSE so the
    # pointer stays null.  If so the reported bug state is UNREACHABLE => decisive
    # FP, concluded BEFORE path selection.  Best-effort + conservative: a single
    # LLM call; on any failure or non-decisive answer we proceed to normal selection.
    semantic_fp_reason: str | None = None
    try:
        semantic_fp_reason = analyzer.semantic_fp_reason(parsed_report)
    except Exception as e:  # noqa: BLE001 — gate must never be fatal
        logger.warning("Semantic finite-domain FP gate failed: %s", e)
        semantic_fp_reason = None
    if semantic_fp_reason:
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print("  Verdict:    ❌ FP (no feasible path — semantic finite-domain conflict)")
        print(f"  Reason:     {semantic_fp_reason}")
        _out = report_out_path(args, report_ref)
        _out.parent.mkdir(parents=True, exist_ok=True)
        _output = {
            **report_fields(report_ref),
            "algorithm": "conflict-driven",
            "entry_function": entry_fn,
            "bug_type": parsed_report.metadata.bug_type,
            "semantic_feasibility": {
                "decisive_fp": True,
                "reason": semantic_fp_reason,
            },
            "verdict": Verdict.FP.value,
            "verdict_step": stage_step(7, "（约束补全，枚举前判定）"),
            "verdict_reason": semantic_fp_reason,
            "verdict_steps": [f"约束补全判定: {semantic_fp_reason}"],
            "total_elapsed_sec": round(time.time() - total_start, 1),
            "code_model": code_model_report(parsed_report, source_root),
            "lite_graph": lite_graph.to_dict() if lite_graph is not None else None,
        }
        _write_output(_out, _output, total_start)
        return _output

    return None


def _stage8_path_space(args, report_ref, analyzer, segments, parsed_report,
                      slice_mask, slice_result, sdg, verdict_state,
                      explore) -> Stage8PathSpace:
    """Stage 8: collect and compress the feasible path space; set the budget.

    Also the stage-9 boundary: this is where the verdict and exploration state are
    reset (``final_step``/``final_reason`` go from ``""`` to ``None``) and where
    the stage-9 banner prints.  Both are here rather than in the caller because
    nothing between them prints, so the console order is unchanged either way —
    and keeping them adjacent to the state they belong to is what makes the
    ``""``→``None`` reset legible.
    """
    report_stem = report_ref.key
    out_path = report_out_path(args, report_ref)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(stage_label(8))
    print(f"{'='*70}")

    # ── Path traversal with a third outcome (UNKNOWN).  When a selected path's
    # simulation comes back conflicted/unresolved, we abandon that path, mark it
    # blocked, and re-select a DIFFERENT feasible path, up to ``max_path_attempts``.
    # Exceeding the threshold with no decisive TP/FP → UNKNOWN (path space too
    # large / undecided). ──
    #
    # Diversity mechanism = SYSTEMATIC WHOLE-PATH COMBINATION ENUMERATION.  We
    # collect the per-segment feasible candidates once (deterministic, no LLM)
    # and walk their cross-product on re-selection.  We block WHOLE-PATH
    # combinations (never a per-segment branch): infeasibility is combination-
    # dependent — (segA=b1, segB=c1) being FP does not make b1 infeasible, since
    # (segA=b1, segB=c2) may be feasible.  So a path's FP blocks that exact
    # combination and the next enumeration step tries a different one.
    requested_max_attempts = (
        getattr(args, "max_path_attempts", _DEFAULT_PATH_ATTEMPT_LIMIT)
        or _DEFAULT_PATH_ATTEMPT_LIMIT
    )
    max_path_attempts = min(
        max(1, int(requested_max_attempts)),
        _MAX_PATH_ATTEMPTS_HARD_CAP,
    )
    path_space_enum = collect_path_space(analyzer, segments)

    # ── Slice-driven branch relevance: fix root-cause-irrelevant branches and
    # enumerate only the relevant ones.  Compressing the space both avoids wasted
    # enumeration over branches that cannot change the verdict and, when the
    # reduced bound collapses to 1, lets a single decisive FP generalize to the
    # whole space (graph evidence, not text matching). ──
    bug_file = parsed_report.metadata.bug_file
    slice_truncated = bool(getattr(slice_result, "truncated", False))
    slice_relevance = classify_branch_relevance(
        sdg, slice_mask, path_space_enum,
        bug_file=bug_file, truncated=slice_truncated,
    )
    reduced_ps, slice_reduce_stats = reduce_path_space(
        path_space_enum, slice_relevance["node_relevance"], segments,
    )
    # Context-insensitive collapse: segments carrying the IDENTICAL branch set
    # (the same function's branches re-appearing across multiple segments, e.g.
    # insertThousandsGroupingUnsafe's loop B5/B8/B10 in 4 segments) become ONE
    # enumeration dimension — their directions are bound together (全路径一致),
    # collapsing the whole-path cross-product from ∏counts to the distinct
    # branch-set product (GlogFormatter: 4096 → 8).
    reduced_ps = annotate_ci_groups(reduced_ps, segments)
    ci_stats = {
        "collapsed_bound": int(reduced_ps.get("whole_path_bound", 1)),
        "num_groups": len(reduced_ps.get("ci_groups", [])),
        "grouped_segments": reduced_ps.get("ci_groups", []),
    }
    reduced_bound_one = reduced_ps.get("whole_path_bound", 1) == 1

    # Exploration budget.  At most ten distinct path attempts are executed.  A
    # reduced space is exhausted only when it fits inside that hard cap; larger
    # spaces are sampled, so per-path FP observations remain insufficient for a
    # report-level FP.
    # (There used to be a `_STRUCTURAL_GP_MAX = 1000` cap here, gating the
    # structural-FP cause-invariance generalization on the reduced bound.  It was
    # a *soundness* cap, not a cost one: `cause_invariant` enumerated the
    # cross-product, which combination_enum bounds (200k enumerated / 100k
    # yielded), so above the cap the "invariant" answer would have come from a
    # partial scan.  The scan is now exhaustive without enumerating the
    # cross-product, so a decisive structural FP generalizes inside a huge space
    # too — that is what makes it possible to conclude FP instead of UNKNOWN
    # there.  Do not reintroduce a bound-based cap without re-checking that.)
    reduced_bound = int(reduced_ps.get("whole_path_bound", 1))
    loop_attempts, full_space_explore = _path_exploration_budget(
        max_path_attempts, reduced_bound,
    )
    combo_iter = iter_combinations(
        reduced_ps, segments,
        cap=max(loop_attempts, 16),
    )
    verdict_state.verdict = None
    verdict_state.final_step = None
    verdict_state.final_reason = None
    explore.selections = None
    explore.ps_stats = {}
    explore.poc_result = None
    explore.poc_code = None
    explore.poc_path = None
    explore.poc_attempts: list[dict] = []
    explore.path_consistency_violations: list[dict] = []
    explore.attempts_run = 0

    # Exploration state: a decisive FP on ONE path never concludes the report is
    # FP (that path may just be the wrong one).  We collect evidence across the
    # distinct feasible paths we explore (blocking each FP signature and
    # re-selecting another), then aggregate at the end.
    explore.saw_unresolved = False
    explore.fp_observations: list[dict] = []  # {signature, final_reason}
    # Leak-layer (memory-leak reports) verdicts per attempt.  The leak layer settles a
    # leak from the object's ownership chain — a code-level claim, not a per-path
    # execution result — so the report-level loop requires the attempts to AGREE
    # before either direction is accepted (see the post-loop consensus).  Keeping them
    # here also keeps the verdict independent of the attempt order.
    explore.leak_fp_reasons: list[str] = []
    explore.leak_tp_reasons: list[str] = []

    explore.combo = None
    explore.combo_sig = None
    print(f"\n{'='*70}")
    print(stage_label(9))
    print(f"{'='*70}")
    return Stage8PathSpace(
        report_stem=report_stem,
        out_path=out_path,
        max_path_attempts=max_path_attempts,
        path_space_enum=path_space_enum,
        slice_truncated=slice_truncated,
        slice_relevance=slice_relevance,
        reduced_ps=reduced_ps,
        slice_reduce_stats=slice_reduce_stats,
        ci_stats=ci_stats,
        reduced_bound_one=reduced_bound_one,
        reduced_bound=reduced_bound,
        loop_attempts=loop_attempts,
        full_space_explore=full_space_explore,
        combo_iter=combo_iter,
    )


def _print_results(ctx, verdict_state, explore) -> dict:
    """Print the RESULTS block and the path-space summaries; return the latter.

    The returned ``path_space`` is the *collected* space (candidates, bound,
    attempts used) — a different object from the ``path_space_enum`` stage 8
    enumerated, and the one the result JSON serializes.
    """
    args = ctx.args
    analyzer = ctx.analyzer
    segments = ctx.segments
    reduced_ps = ctx.reduced_ps
    loop_attempts = ctx.loop_attempts
    max_path_attempts = ctx.max_path_attempts
    full_space_explore = ctx.full_space_explore
    slice_relevance = ctx.slice_relevance
    slice_reduce_stats = ctx.slice_reduce_stats
    fp_observations = explore.fp_observations
    selections = explore.selections
    ps_stats = explore.ps_stats
    verdict = verdict_state.verdict
    final_step = verdict_state.final_step

    # ── Print results ──
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print("  Algorithm:  conflict-driven path selection + POC verification")
    print(f"  Duration:   {ps_stats.get('elapsed_sec', 0):.1f}s")
    print(f"  Segments:   {ps_stats.get('num_segments', 0)}")
    if verdict == "TP":
        print("  Verdict:    ✅ TP (feasible path found)")
    elif verdict == "FP":
        print("  Verdict:    ❌ FP (no feasible path)")
    else:
        print("  Verdict:    ❓ UNKNOWN ("
              + (f"{_NO_PATH_STEP_KEYWORD}（报告无路径，不作 TP 验证）"
                 if _NO_PATH_STEP_KEYWORD in (final_step or "")
                 else "path space too large / undecided") + ")")

    if fp_observations:
        print("\n  🧭 可行路径空间探索 (探索式 FP 判定):")
        print(f"    已探索不同可行路径: {len(fp_observations)} 条")
        for obs in fp_observations:
            print(f"      - {obs.get('signature', '')[:80]}")
            r = obs.get('final_reason', '')
            if r:
                print(f"         理由: {r[:120]}")

    # ── Feasible path-space collection + display ──
    # Deterministic, no LLM: re-enumerates the per-segment feasible candidates
    # (via compute_topk with an unbounded k) and serializes them, so future
    # efficient path-space-search algorithms can consume the structured data.
    chosen_by_seg = _chosen_by_seg(selections)
    path_space = collect_path_space(analyzer, segments, selections)
    path_space["attempts_used"] = explore.attempts_run
    path_space["max_path_attempts"] = loop_attempts
    path_space["configured_max_path_attempts"] = max_path_attempts
    path_space["full_space_explore"] = full_space_explore
    path_space["chosen_by_seg"] = {
        str(k): sorted(v) for k, v in chosen_by_seg.items()
    }
    if getattr(args, "path_space", False):
        print(f"\n{'='*70}")
        print("FEASIBLE PATH SPACE (per-segment candidates + whole-path bound)")
        print(f"{'='*70}")
        print(render_path_space_ascii(
            segments, chosen_by_seg, path_space,
            attempts=path_space["attempts_used"], max_attempts=loop_attempts,
        ))
    else:
        # Compact always-on summary in the RESULTS block.
        seg_summary = ", ".join(
            f"S{info.get('segment_index')}:{info.get('num_feasible_candidates')}"
            for info in path_space.get("per_segment", [])
        )
        print("\n  Path space:")
        print(f"    per-segment candidates: {seg_summary}")
        print(f"    total enumerated      : {path_space.get('total_feasible_candidates')}")
        print(f"    whole-path bound ∏N_i : {path_space.get('whole_path_bound'):_}"
              + (f" → {reduced_ps.get('whole_path_bound'):_}"
                 f" (上下文不敏感折叠 {len(reduced_ps.get('ci_groups', []))} 组)"
                 if reduced_ps.get('ci_groups') else ""))
        print(f"    attempts used         : {path_space.get('attempts_used')}/{loop_attempts}"
              + ("  [全空间探索]" if full_space_explore else ""))

    # ── Slice-driven branch-relevance summary (compression) ──
    rel_stats = slice_relevance["stats"]
    print("\n  Slice relevance (root-cause branches):")
    print(f"    branch nodes: {rel_stats['num_branch_nodes']} "
          f"| relevant: {rel_stats['num_relevant']} "
          f"| irrelevant: {rel_stats['num_irrelevant']} "
          f"(irrelevant ratio {rel_stats['irrelevant_ratio']})"
          + (" | ⚠️ slice truncated → 保守不压缩" if rel_stats["truncated"] else ""))
    print(f"    path space: {slice_reduce_stats['original_bound']:_} "
          f"→ {slice_reduce_stats['reduced_bound']:_}"
          f"  (compression {slice_reduce_stats['compression_ratio']})")
    return path_space


def _build_output_dict(
    ctx, verdict_state, explore, stage4, stage5, path_space,
) -> ResultDocument:
    """Assemble the result JSON.  Does not write it — :func:`_write_output` does.

    Every key here has a reader: ``tools/summarize_faiss_eval.py`` and the eval
    tooling match ``verdict_step``'s Chinese keywords and read
    ``path_selection.{generalized_fp,structural_generalized_fp,leak_consensus_fp,
    leak_layer,fp_exploration}`` by name, which is why those are emitted as
    explicit booleans rather than being left to be inferred from prose.
    """
    report_ref = ctx.report_ref
    entry_fn = ctx.entry_fn
    parsed_report = ctx.parsed_report
    source_root = ctx.source_root
    total_start = ctx.total_start
    hard_constraints = stage4.hard_constraints
    cumulative_state = stage4.cumulative_state
    n_hc = stage4.num_hard_constraints
    n_state = stage4.num_state_bindings
    assumption_elapsed = stage4.elapsed_sec
    branch_results = stage5.results
    total_b = stage5.total_branches
    total_con = stage5.total_consistent
    total_ctr = stage5.total_contradictory
    total_ins = stage5.total_insufficient
    ci_stats = ctx.ci_stats
    slice_relevance = ctx.slice_relevance
    slice_reduce_stats = ctx.slice_reduce_stats
    blocked_paths = explore.blocked_paths
    fp_observations = explore.fp_observations
    leak_fp_reasons = explore.leak_fp_reasons
    leak_tp_reasons = explore.leak_tp_reasons
    saw_unresolved = explore.saw_unresolved
    selections = explore.selections
    ps_stats = explore.ps_stats
    poc_result = explore.poc_result
    verdict = verdict_state.verdict
    final_step = verdict_state.final_step
    final_reason = verdict_state.final_reason
    verdict_steps = verdict_state.steps
    loop_attempts = ctx.loop_attempts
    max_path_attempts = ctx.max_path_attempts
    full_space_explore = ctx.full_space_explore

    output = {
        **report_fields(report_ref),
        "algorithm": "conflict-driven",
        "entry_function": entry_fn,
        "bug_type": parsed_report.metadata.bug_type,
        "assumption_check": {
            "num_hard_constraints": n_hc,
            "hard_constraints": hard_constraints,
            "num_state_bindings": n_state,
            "state_bindings": cumulative_state,
            "global_constraint_solve": stage4.constraint_solution,
            "elapsed_sec": round(assumption_elapsed, 1),
        },
        "branch_filter": {
            "num_segments": len(branch_results),
            "total_branches": total_b,
            "consistent": total_con,
            "contradictory_pruned": total_ctr,
            "insufficient_info": total_ins,
            "per_segment": branch_results,
        },
        "path_selection": {
            "verdict": verdict,
            "elapsed_sec": ps_stats.get("elapsed_sec", 0),
            "num_selections": len(selections) if selections else 0,
            "selections": selections if selections else None,
            "blocked_paths": blocked_paths,
            "max_path_attempts": loop_attempts,
            "configured_max_path_attempts": max_path_attempts,
            "full_space_explore": full_space_explore,
            "generalized_fp": bool(ps_stats.get("generalized_fp", False)),
            "structural_generalized_fp": bool(ps_stats.get("structural_generalized_fp", False)),
            "leak_consensus_fp": bool(ps_stats.get("leak_consensus_fp", False)),
            "csa_contradiction_fp": bool(ps_stats.get("csa_contradiction_fp", False)),
            "report_control_contradiction": ps_stats.get(
                "report_control_contradiction", []
            ),
            "path_consistency": {
                "violation_count": sum(
                    len(entry.get("violations") or [])
                    for entry in explore.path_consistency_violations
                ),
                "attempts": explore.path_consistency_violations,
            },
            "leak_layer": {
                "fp_verdicts": len(leak_fp_reasons),
                "tp_verdicts": len(leak_tp_reasons),
                "reasons": (leak_fp_reasons or leak_tp_reasons)[:3],
            },
            "fp_exploration": {
                "num_distinct_paths_fp": len(fp_observations),
                "saw_unresolved": saw_unresolved,
                "observations": fp_observations,
            },
        },
        "slice_relevance": {
            "branch_stats": slice_relevance["stats"],
            "reduction": slice_reduce_stats,
        },
        "context_insensitive": ci_stats,
        "total_elapsed_sec": round(time.time() - total_start, 1),
        "poc_attempts": explore.poc_attempts,
    }
    if poc_result is not None:
        output["poc_verification"] = poc_result
        sim = poc_result.get("simulation") or {}
        output["verification_status"] = sim.get("verified_status", "NONE")
    else:
        output["verification_status"] = "NONE"
    # Record whether the verification feedback loop flipped an initial TP to FP
    # (only a conclusive FP judgment flips; unresolved stays TP/UNVERIFIED).
    output["verification_flipped_to_fp"] = (
        (poc_result or {}).get("simulation", {}).get("conclusion") == "fp"
    )
    output["project"] = source_root.name
    output["verdict"] = verdict
    output["verdict_step"] = final_step
    output["verdict_reason"] = final_reason
    output["verdict_steps"] = verdict_steps
    output["path_space"] = path_space
    # ── Code model: the report's own copy vs the local checkout, plus what the
    #    report's index made the model read.  See `code_model_report`. ──
    output["code_model"] = code_model_report(parsed_report, source_root)
    output["lite_graph"] = (
        ctx.lite_graph.to_dict() if getattr(ctx, "lite_graph", None) is not None else None
    )
    return output


def _print_final_verdict(verdict_state, explore) -> None:
    """The consolidated verdict block: decisive step, evidence, and the POC."""
    verdict = verdict_state.verdict
    final_step = verdict_state.final_step
    final_reason = verdict_state.final_reason
    verdict_steps = verdict_state.steps
    poc_code = explore.poc_code
    poc_path = explore.poc_path
    poc_result = explore.poc_result

    # ── Final consolidated verdict: decisive step + evidence (TP test case &
    #    execution-passed result; FP cause; UNKNOWN reason) ──
    print("\n" + "=" * 70)
    print("最终判定 VERDICT — 判定步骤与依据")
    print("=" * 70)
    if verdict == "TP":
        print("  Verdict:  ✅ TP (真阳性，路径可达且可复现)")
    elif verdict == "FP":
        print("  Verdict:  ❌ FP (假阳性，路径不可达/不成立)")
    else:
        print("  Verdict:  ❓ UNKNOWN ("
              + (f"{_NO_PATH_STEP_KEYWORD}：报告只给出位置，POC 逐步对照无对象"
                 if _NO_PATH_STEP_KEYWORD in (final_step or "")
                 else "路径空间过大 / 经多轮探索仍无法确定") + ")")
    print(f"  判定步骤: {final_step or '（未记录）'}")
    if verdict == "TP":
        if poc_code:
            print(f"  测试用例: {poc_path if poc_path else '（未保存到文件）'}  "
                  f"({len(poc_code.splitlines())} 行)")
            print("      -----------------------------------------------------")
            for ln in poc_code.splitlines():
                print(f"      {ln}")
            print("      -----------------------------------------------------")
            print("  反馈执行通过结果（模拟执行判定触发 bug 的轮次）：")
            rounds = (poc_result or {}).get("simulation", {}).get("rounds", [])
            if rounds:
                for r in rounds:
                    tag = "✅触发" if r.get("triggered") else "❌未触发"
                    print(f"      Round {r['round']} [{tag}] "
                          f"fp_likelihood={r.get('fp_likelihood')}")
                    for tr in (r.get("trace") or [])[:6]:
                        print(f"          {tr}")
                    print(f"          依据: {(r.get('reason') or '')[:220]}")
            else:
                print("      （模拟执行未产生轮次记录）")
        else:
            print("  测试用例: （未生成 —— 未获得 POC）")
    else:
        print(f"  原因: {final_reason or '（未提供具体原因）'}")
    if verdict_steps:
        print("  判定过程:")
        for s in verdict_steps:
            print(f"      · {s}")


def process_report(args, report_path) -> ResultDocument:
    """Run the full pipeline for a single report and write its result JSON.

    Returns the output dict (with a ``verdict`` field) for batch aggregation.

    The body is the ten stages in order.  The two deterministic gates return
    early with the JSON they wrote; everything else records into ``verdict_state``
    and ``explore``, which the aggregation then reasons over.
    """
    # Resolve paths
    require_pdg = bool(getattr(args, "no_lite_graph", False))
    if "require_pdg" in inspect.signature(_resolve_paths).parameters:
        report_ref, cfg_cache_path, pdg_path, source_root = _resolve_paths(
            args, report_path, require_pdg=require_pdg,
        )
    else:
        # Compatibility with test/evaluation seams that still expose the
        # historical two-argument resolver.
        report_ref, cfg_cache_path, pdg_path, source_root = _resolve_paths(
            args, report_path,
        )
    report_path = report_ref.path
    total_start = time.time()

    verdict_state = VerdictState()
    explore = ExplorationState()

    _print_run_banner(args, report_ref, source_root, pdg_path, cfg_cache_path)

    stage1 = _stage1_parse(args, report_ref, source_root)
    lite_graph = build_lite_graph(
        stage1.loaded,
        source_root,
        enabled=not (
            getattr(args, "no_lite_graph", False)
            or getattr(args, "pdg", None)
        ),
        timeout=getattr(args, "lite_graph_timeout", 30.0),
        compile_commands=getattr(args, "compile_commands", None),
    )
    project = _stage1_assemble(
        source_root,
        cfg_cache_path,
        pdg_path,
        stage1.parsed_report,
        lite_graph=lite_graph,
        prefer_exact=bool(getattr(args, "pdg", None)),
    )
    entry_fn = _stage1_resolve_entry(stage1.entry_fn, project.sdg, stage1.parsed_report)
    segments = _stage2_segment(stage1.parsed_report, project.sdg, source_root)
    _stage2_extend_caches(project.cache_set, segments, project.cfg_cache,
                          cfg_cache_path, source_root, entry_fn)

    t0 = time.time()          # stage 3's clock start (shared with _build_analyzer)
    stage3 = _stage3_slice(project.sdg, stage1.parsed_report, stage1.bug_path,
                           source_root, t0)
    analyzer = _build_analyzer(
        args, source_root, pdg_path, t0, sdg=project.sdg, lite_graph=lite_graph,
    )
    stage4 = _stage4_assumptions(analyzer, segments, stage1.parsed_report,
                                  source_root)

    t0 = time.time()          # stage 5's clock start (shared with _stage6_gate)
    stage5 = _stage5_branch_filter(
        segments, stage1.parsed_report, project.cfg_cache, project.sdg,
        stage3.slice_mask, source_root, stage4.cumulative_state,
        project.cache_set, stage3.path_annotations, t0,
    )

    early = _stage6_gate(args, report_ref, segments, stage1.parsed_report,
                         source_root, entry_fn, t0, total_start, lite_graph,
                         constraint_solution=stage4.constraint_solution)
    if early is not None:
        return early

    extra_preconditions = _stage7_preamble(
        stage4.cumulative_state,
        constraint_solution=stage4.constraint_solution,
        constraint_bundle=getattr(analyzer, "_global_constraint_bundle", None),
    )
    _stage7_supplement(analyzer, segments, stage1.parsed_report, project.sdg,
                       stage3.slice_mask, extra_preconditions, lite_graph)
    early = _stage7_gate(analyzer, stage1.parsed_report, args, report_ref,
                         entry_fn, source_root, total_start, lite_graph)
    if early is not None:
        return early

    stage8 = _stage8_path_space(
        args, report_ref, analyzer, segments, stage1.parsed_report,
        stage3.slice_mask, stage3.slice_result, project.sdg,
        verdict_state, explore,
    )
    ctx = SelectionContext(
        args=args,
        report_ref=report_ref,
        analyzer=analyzer,
        parsed_report=stage1.parsed_report,
        loaded=stage1.loaded,
        segments=segments,
        slice_mask=stage3.slice_mask,
        source_root=source_root,
        cfg_cache=project.cfg_cache,
        sdg=project.sdg,
        extra_preconditions=extra_preconditions,
        hard_constraints=stage4.hard_constraints,
        reduced_ps=stage8.reduced_ps,
        reduced_bound_one=stage8.reduced_bound_one,
        slice_truncated=stage8.slice_truncated,
        slice_relevance=stage8.slice_relevance,
        combo_iter=stage8.combo_iter,
        loop_attempts=stage8.loop_attempts,
        max_path_attempts=stage8.max_path_attempts,
        full_space_explore=stage8.full_space_explore,
        slice_reduce_stats=stage8.slice_reduce_stats,
        ci_stats=stage8.ci_stats,
        out_path=stage8.out_path,
        report_stem=stage8.report_stem,
        entry_fn=entry_fn,
        lite_graph=lite_graph,
        total_start=total_start,
    )

    _run_path_attempts(ctx, verdict_state, explore)
    _aggregate_exploration(ctx, verdict_state, explore)
    path_space = _print_results(ctx, verdict_state, explore)
    output = _build_output_dict(ctx, verdict_state, explore, stage4, stage5, path_space)
    _write_output(stage8.out_path, output, total_start)
    _print_final_verdict(verdict_state, explore)

    return output
