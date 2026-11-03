"""Stage 9's mechanism and stage 10's verification — plus the verdict prose.

``run_path_selection`` is the selection step itself (LLM-driven seed or
deterministic whole-path combination); ``_verify_path`` is stage 10 (generate a
POC, audit it, simulate execution, feed the failures back).  Neither owns the
report-level verdict: ``_verify_path`` returns a ``conclusion`` and the caller's
exploration loop (see :mod:`~pipeline.exploration`)
decides TP / FP / UNKNOWN from it.

The verdict *prose* lives here too because it is a property of the stage that
reaches it: ``_no_path_verdict`` (a location-only report can never be TP),
``_tp_verdict_reason`` (a TP must name its own evidence) and the
``verdict_step`` keyword constants the eval tooling greps for.
"""

from __future__ import annotations

import time
from pathlib import Path

from fp_analysis.models import CompletedPath, Confidence, PathAnalysis
from fp_analysis.path_analyzer import FPAnalyzer, PathSpaceExhausted
from fp_analysis.prompt_templates import parse_poc_unreachable_note
from fp_analysis.simulation_verifier import SimulationVerifier
from fp_analysis.slice_mask import SliceMask

from pipeline.layout import get_logger
from pipeline.stages import stage_step

logger = get_logger()

def run_path_selection(
    analyzer: FPAnalyzer,
    parsed_report,
    segments: list,
    slice_mask: SliceMask,
    source_root: Path,
    cfg_cache: dict | None = None,
    extra_preconditions: list[str] | None = None,
    hard_constraints: list[str] | None = None,
    blocked_paths: list[str] | None = None,
    blocked_candidates: list[dict] | None = None,
    combo: tuple | None = None,
    path_space: dict | None = None,
) -> tuple[list[dict] | None, dict]:
    """Run segment-by-segment path selection via FPAnalyzer (conflict-driven).

    Two selection modes:

    - ``combo is None``: the LLM-driven selection
      (``_select_feasible_path_segments``).  Best-feasible-first path that
      uses the global constraints/state context.  This is the attempt-1 seed.
    - ``combo is not None``: deterministic whole-path combination enumeration.
      ``combo`` is a 0-based candidate index per segment into ``path_space``'s
      per-segment candidates; selections are built without an LLM pick and
      conflict-checked.  ``selections is None`` here means THAT combination is
      globally infeasible (a whole-path FP) — the caller blocks it and enumerates
      the next, NOT a report-level FP.

    Args:
        extra_preconditions: Optional state facts to inject into the
            CompletedPath.preconditions (e.g. from assumption checking).
        hard_constraints: Hard constraints from assumption feasibility check
            (e.g. ``["blockLength_ > 0"]``).  Passed to path selection so
            constraint conflicts are discovered **by path selection**.
        blocked_paths: Whole-path signatures already tried and abandoned
            (simulation conflicted/unresolved/FP).  Passed to path selection so
            it avoids re-deriving them and picks a different feasible path.
        blocked_candidates: Kept for call-compat only — no longer used for
            re-selection (whole-path combination enumeration replaces the
            per-segment candidate blocking that over-pruned).  Passed empty.

    Returns (selections, stats_dict).  When ``combo is not None`` and the
    combination is globally infeasible, ``stats["combo_infeasible"]`` is True.
    """
    # Build a minimal CompletedPath — the path selection algorithm uses
    # completed.completed_steps / preconditions / concretized_values for
    # context prompts.  Other fields are irrelevant for path selection.
    preconditions = []
    if extra_preconditions:
        preconditions.extend(extra_preconditions)
    completed = CompletedPath(
        original_path_summary="",
        analysis=PathAnalysis(
            path_id="bug_path",
            is_likely_fp=False,
            gaps=[],
            confidence=Confidence.LOW,
            reasoning="placeholder for path selection",
        ),
        completed_steps=[],
        final_path_summary="",
        preconditions=preconditions,
        concretized_values={},
        error_start_is_real=True,
    )

    # Build a minimal AnalysisResult for the segment-by-segment selector.
    # _select_single_segment_path accesses: parsed_report, reduced_code,
    # source_root, annotated_path, call_chain_context, bug_path.
    from types import SimpleNamespace
    analysis_result = SimpleNamespace()
    analysis_result.parsed_report = parsed_report
    analysis_result.source_root = str(source_root)
    analysis_result.reduced_code = ""
    analysis_result.annotated_path = ""
    analysis_result.call_chain_context = ""
    analysis_result.bug_path = None

    t0 = time.time()

    logger.info("Conflict-driven path selection with shortest-path candidates ...")
    exhausted = False
    combo_infeasible = False
    try:
        if combo is not None and path_space is not None:
            # Deterministic whole-path combination: build + conflict-check only.
            selections = analyzer.select_combination_segments(
                analysis_result, completed, segments, path_space, combo,
            )
            if selections is None:
                combo_infeasible = True
        else:
            # LLM-driven selection (attempt-1 seed).  No blocked_candidates:
            # whole-path combination enumeration (in the caller) replaces the
            # per-segment candidate blocking that over-pruned.
            selections = analyzer.select_feasible_path_segments(
                analysis_result, completed, slice_mask,
                max_rounds=3,
                segments=segments,
                cfg_cache=cfg_cache,
                hard_constraints=hard_constraints,
                blocked_paths=blocked_paths,
                blocked_candidates=[],
            )
    except PathSpaceExhausted as e:
        # Exhaustion = no NEW distinct top-k whole-path remains (all candidates
        # of some segment are blocked).  This is NOT a conflict-based
        # infeasibility — it does not prove the report is FP, so we must NOT
        # fall through to the caller's hard-FP branch.  Signal it distinctly;
        # the caller routes it to bounded-exploration aggregation.
        logger.info("Path space exhausted: %s", e)
        exhausted = True
        selections = None

    elapsed = time.time() - t0

    stats = {
        "elapsed_sec": round(elapsed, 1),
        "num_segments": len(segments),
        "selected": selections is not None,
        "exhausted": exhausted,
        "combo_infeasible": combo_infeasible,
    }

    if selections:
        stats["num_selections"] = len(selections)
    elif combo_infeasible:
        stats["result"] = "Combination globally infeasible (whole-path FP) — not a report-level FP"
    elif exhausted:
        stats["result"] = "Exhausted top-k path space (no NEW distinct path) — not a conflict-based FP"
    else:
        stats["result"] = "FP — no globally consistent path"

    return selections, stats


def path_signature(selections):
    """Stable, human-readable signature of a whole-path branch combination.

    Used to mark a path as *blocked* after its simulation came back
    conflicted/unresolved, so a later path-selection attempt (with
    ``blocked_paths``) avoids re-deriving the same combination.
    """
    parts = []
    for sel in (selections or []):
        seg_idx = sel.get("segment_index", "?")
        bds = sel.get("branch_decisions", [])
        b = ",".join(
            f"L{d.get('line', d.get('source_line', '?'))}:"
            f"{'T' if d.get('branch_taken', True) else 'F'}"
            for d in bds
        )
        parts.append(f"seg{seg_idx}[{b}]")
    return ";".join(parts)


def _blocked_candidate_map(selections):
    """Per-segment ``{segment_index: selected_candidate}`` for a selection.

    Used alongside ``path_signature`` so a blocked path can be re-avoided
    DETERMINISTICALLY (by excluding the chosen candidate indices from the next
    selection), not just via the advisory prompt hint.

    FORCED segments are excluded: a segment with ``num_candidates <= 1`` (or
    linear, carrying no ``selected_candidate``) has exactly one path that every
    whole-path must pass through.  Blocking it would eliminate *every*
    whole-path and yield a premature FP, without ever checking whether the
    other (free) segments can combine to reach the bug.  Forced segments are
    fixed constraints of the search, never blocked — FP for them must come from
    cross-segment conflict (or bounded exploration of the free segments), not
    from "blocking" the only path.
    """
    return {
        sel.get("segment_index"): sel.get("selected_candidate")
        for sel in (selections or [])
        if sel.get("selected_candidate")
        and (sel.get("num_candidates", 0) or 0) > 1
    }


def _last_round_reason(sim_res) -> str:
    """The last simulation round's own reason text (``""`` if there is none)."""
    rounds = (sim_res or {}).get("rounds") or []
    if not rounds:
        return ""
    sim = rounds[-1].get("sim")
    return str(getattr(sim, "reason", "") or "")


def _tp_verdict_reason(leak_tp: bool = False, round_reason: str = "",
                       leak_reason: str = "") -> str:
    """Why the report is a TP — the reason a TP verdict carries (item C).

    The TP exit used to leave ``final_reason`` untouched, while the FP exits all
    set it.  Since ``final_reason`` lives outside the attempt loop, a report whose
    attempt 1 concluded FP (e.g. the leak layer's "stored/owned on well-formed
    paths" argument) and whose attempt 2 concluded TP printed attempt 1's FP
    reasoning next to a TP verdict, in both the summary and the result JSON.
    Every verdict now owns its reason.
    """
    base = str(round_reason or "").strip()
    if leak_tp:
        head = (
            "leak 层判定 TP：对象在 well-formed 调用路径上被 abandon 且无人接管"
            "（未释放、未移交 owner/return）→ 真实可达泄漏"
        )
        base = str(leak_reason or base).strip()
    else:
        head = "模拟执行判定触发 bug（测试用例复现 CSA 报告路径）"
    return f"{head}。模拟依据：{base[:300]}" if base else head


# The `verdict_step` keyword of the no-path UNKNOWN, kept in one place: the
# summary tooling greps these phrases (see `stage_step`), and the console must
# not label this UNKNOWN "path space too large" — the space was never the
# reason.  Deliberately distinct from `遍历超限`, which must keep meaning
# "exploration budget exhausted".
_NO_PATH_STEP_KEYWORD = "报告未给出路径"


def _no_path_verdict(path_evidence: str) -> tuple[str, str] | None:
    """The UNKNOWN that a report stating only a *location* must end in.

    ``path_evidence`` is the analyzer's own statement — ``report_frontend``
    records it per finding (``PathEvidence``) and ``report_bridge`` carries it
    to here as ``ReportMetadata.path_evidence``; it is **not** re-derived from
    the number of events (a location-only finding still renders one, the report
    step).  ``"trace"`` means the analyzer walked a path and printed it, so this
    returns ``None`` and the pipeline verifies as usual.  Otherwise the report
    never described a path, and POC verification cannot certify the claim:

    * Stage 9 builds a feasible path *itself*.  Handing that path's POC to the
      step-by-step conformance audit gives the audit nothing to check against —
      only the report step is left, so it passes vacuously and ``triggered``
      then rests on a path the pipeline chose, not on the one the analyzer
      claimed.  That is the "POC manufactures its own precondition" failure
      mode, at report scale instead of call scale, and it is exactly the
      instrument that would have caught it.
    * Nothing sound is lost by stopping here.  Every stage that can conclude
      without a reported path has already run by this point (stages 6/7 and
      stage 9's conflict-based FP): FP says reachability is *refuted* and needs
      no path of its own, whereas TP is a claim *about* a path.

    Returns ``(verdict_step, verdict_reason)``, or ``None`` to verify normally.
    """
    if path_evidence == "trace":
        return None
    return (
        stage_step(10, f"{_NO_PATH_STEP_KEYWORD}（path_evidence={path_evidence}）"),
        (
            f"报告只给出被报位置（path_evidence={path_evidence}），"
            f"没有它所走过的路径；阶段 9 构造出的可行路径不是报告路径，"
            f"其 POC 的逐步一致性对照无对象、必然空过 → 无法证明报告所述路径可达，"
            f"不作 TP 判定，返回 UNKNOWN"
        ),
    )


def _verify_path(analyzer, report_path, source_root, out_path, report_stem,
                 selections=None):
    """7a–7e: generate a POC (LLM-only, no real compile), LLM-audit its
    correctness, and LLM-simulate execution with a feedback loop.

    Returns the outcome WITHOUT mutating the verdict — the caller's path-traversal
    loop decides TP / FP / UNKNOWN based on ``conclusion``:

        {"poc_result", "conclusion": "tp"|"fp"|"unresolved"|None,
         "final_reason", "poc_code", "poc_path", "sim_res"}

    ``conclusion is None`` means verification errored (environmental), not a
    classification verdict.
    """
    poc_result: dict | None = None
    sim_res = None
    poc_code = None
    poc_path = None
    final_reason = None

    analyzer.ensure_project_name(source_root.name)

    try:
        # 7a: Parse report + extract source context via analyzer
        t0 = time.time()
        analysis_poc = analyzer.analyze(report_path, str(source_root))
        print(f"  7a. Source context extracted [{time.time()-t0:.1f}s]")

        # 7b: Constraint completion → CompletedPath.  Anchored to the selected
        # feasible path so a different selection yields a different POC (P1).
        t0 = time.time()
        print("  7b. Running constraint completion (LLM) ...")
        completed_path = analyzer.complete_constraints(
            analysis_poc, selected_branches=selections,
        )
        print(f"  7b. Constraint completion done: "
              f"{len(completed_path.preconditions)} preconditions, "
              f"{len(completed_path.concretized_values)} concretized values "
              f"[{time.time()-t0:.1f}s]")
        if completed_path.preconditions:
            for p in completed_path.preconditions[:5]:
                print(f"      - {p}")
        if completed_path.concretized_values:
            for k, v in list(completed_path.concretized_values.items())[:5]:
                print(f"      - {k} = {v}")

        # 7c: Generate POC — LLM only, NO real compilation. Correctness is
        # audited by the LLM during the simulated execution.
        t0 = time.time()
        print("  7c. Generating POC (LLM, no compilation) ...")
        poc_code = analyzer.generate_poc(
            completed_path, analysis_poc, selected_branches=selections,
        )
        poc_elapsed = time.time() - t0

        poc_result = {
            "compiled": False,
            "compile_attempts": 0,
            "compile_mode": "llm_only",
            "elapsed_sec": round(poc_elapsed, 1),
            "poc_code_lines": len(poc_code.splitlines()) if poc_code else 0,
            "compile_log": [],
        }

        if poc_code:
            print(f"      POC code: {len(poc_code.splitlines())} lines — "
                  f"LLM-generated, correctness audited during simulation")
            poc_path = out_path.parent / f"poc_{report_stem}.cpp"
            poc_path.write_text(poc_code)
            print(f"      POC saved to: {poc_path}")
        else:
            print("      POC generation returned no code.")

        # 7d: LLM simulated execution + feedback loop.
        if not poc_code:
            poc_result["simulation"] = {
                "performed": False,
                "triggered": False,
                "verified_status": "NONE",
                "reason": "No POC generated — simulation skipped",
            }
            conclusion = "unresolved"
        else:
            t0 = time.time()
            print("  7d. LLM simulated execution + feedback loop ...")
            sim_verifier = SimulationVerifier(analyzer)
            # The generator's own `POC_UNREACHABLE:` claim (see
            # prompts.POC_GENERATION_PROMPT rule 9), if it made one.  Handed to
            # the auditor as a claim to adjudicate; it can support an FP only
            # if the audit independently reproduces it.
            poc_note = parse_poc_unreachable_note(poc_code)
            if poc_note:
                print(f"      Generator reported unreachable: {poc_note[:160]}")
            sim_res = sim_verifier.verify_with_feedback(
                poc_code, completed_path, analysis_poc, max_rounds=3,
                selected_branches=selections, generator_note=poc_note,
            )
            sim_elapsed = time.time() - t0
            conclusion = sim_res.get("conclusion", "unresolved")
            verified_status = (
                "VERIFIED" if conclusion == "tp" else "UNVERIFIED"
            )
            poc_result["simulation"] = {
                "performed": True,
                "compiled": False,
                "compile_mode": "llm_only",
                "triggered": sim_res["triggered"],
                "conclusion": conclusion,
                "verified_status": verified_status,
                "num_rounds": len(sim_res["rounds"]),
                "elapsed_sec": round(sim_elapsed, 1),
                "test_case_correct": sim_res["rounds"][-1]["sim"].test_case_correct,
                "test_case_issues": sim_res["rounds"][-1]["sim"].test_case_issues,
                "path_conformance_ok": sim_res["rounds"][-1]["sim"].path_conformance_ok,
                "parse_degraded": sim_res["rounds"][-1]["sim"].parse_degraded,
                "rounds": [
                    {
                        "round": r["round"],
                        "triggered": r["sim"].triggered,
                        "reason": r["sim"].reason,
                        "trace": r["sim"].trace,
                        "blocking_reason": r["sim"].blocking_reason,
                        "suggested_input_change": r["sim"].suggested_input_change,
                        "csa_contradiction": r["sim"].csa_contradiction,
                        "fp_likelihood": r["sim"].fp_likelihood,
                        "test_case_correct": r["sim"].test_case_correct,
                        "test_case_issues": r["sim"].test_case_issues,
                        "path_conformance": r["sim"].path_conformance,
                        "path_conformance_ok": r["sim"].path_conformance_ok,
                        "parse_degraded": r["sim"].parse_degraded,
                        "bug_value_source": r["sim"].bug_value_source,
                        # the leak verdict's raw evidence (memory-leak reports only), so a
                        # report-level verdict can be audited after the fact
                        "leak_chain": r["sim"].leak_chain,
                    }
                    for r in sim_res["rounds"]
                ],
                "leak_verdict": sim_res.get("leak_verdict"),
                "final_reason": sim_res.get("final_reason"),
                # The POC generator's own POC_UNREACHABLE claim, if any (the
                # auditor's adjudication of it lands in final_reason).
                "generator_unreachable_note": poc_note or None,
            }
            # Step-by-step path conformance (hallucination guard): print, per
            # round, whether the POC reproduced each CSA report path step.
            for r in sim_res["rounds"]:
                rsim = r["sim"]
                conf = getattr(rsim, "path_conformance", None) or []
                ok = getattr(rsim, "path_conformance_ok", True)
                if not conf:
                    if getattr(rsim, "parse_degraded", False):
                        print(f"      路径一致性 [第{r['round']}轮]: ⚠️ 响应被截断/"
                              f"审计不可用 — 逐步审计缺失，本轮不足以判定 TP")
                    continue
                n_bad = sum(1 for c in conf if not bool(c.get("satisfied", True)))
                flag = "✅ 一致" if ok else f"❌ 不一致({n_bad} 步)"
                print(f"      路径一致性 [第{r['round']}轮]: {flag} "
                      f"— {len(conf) - n_bad}/{len(conf)} 步与 CSA 报告一致")
                for c in conf:
                    if bool(c.get("satisfied", True)):
                        continue
                    print(f"        ✗ 步骤 {c.get('step', '?')}: "
                          f"报告预期={str(c.get('report_expectation', '?'))[:80]} | "
                          f"POC 实际={str(c.get('poc_result', '?'))[:80]}")

            if sim_res["triggered"]:
                print(f"      TP 模拟验证: ✅ 触发 bug [{sim_elapsed:.1f}s]")
            else:
                if conclusion == "fp":
                    print(f"      TP 模拟验证: ⚠️ 模拟执行判定 FP → 停止细化，翻转判定为 FP "
                          f"[{sim_elapsed:.1f}s]")
                else:
                    print(f"      TP 模拟验证: ⚠️ 未触发且无明确 FP 结论 → UNVERIFIED "
                          f"[{sim_elapsed:.1f}s]")
                if sim_res.get("final_reason"):
                    print(f"          依据: {sim_res['final_reason'][:200]}")

        if conclusion == "fp":
            final_reason = (
                (sim_res.get("final_reason") if sim_res and sim_res.get("final_reason")
                 else poc_result.get("simulation", {}).get("final_reason", ""))
            )
        elif conclusion == "tp":
            # The TP exit must own its reason too (see `_tp_verdict_reason`): leaving it
            # unset let a report whose earlier attempt had concluded FP print/carry that
            # attempt's FP (or leak-FP) reason next to a TP verdict.
            final_reason = (
                (sim_res or {}).get("final_reason")
                or _tp_verdict_reason(
                    leak_tp=(sim_res or {}).get("leak_verdict") == "tp",
                    round_reason=_last_round_reason(sim_res),
                )
            )

        return {
            "poc_result": poc_result,
            "conclusion": conclusion,
            "final_reason": final_reason,
            "poc_code": poc_code,
            "poc_path": poc_path,
            "sim_res": sim_res,
            "leak_verdict": (sim_res or {}).get("leak_verdict"),
        }

    except Exception as e:
        logger.error("POC verification failed: %s", e, exc_info=True)
        poc_result = {"compiled": False, "error": str(e)}
        print(f"  ❌ POC verification error: {e}")
        return {
            "poc_result": poc_result,
            "conclusion": None,
            "final_reason": str(e),
            "poc_code": None,
            "poc_path": None,
            "sim_res": None,
        }
