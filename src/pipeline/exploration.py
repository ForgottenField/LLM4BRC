"""The attempt loop and the post-loop aggregation — where a verdict is decided.

These two blocks are separated from the rest of ``runner`` because they are the
only code in the pipeline that can conclude a **report-level FP without a
per-attempt conclusion**.  Everything above them produces evidence about *one*
path; a decisive FP on one path is not a statement about the report, since a
different feasible combination may reach the bug.  Only the aggregation reasons
across paths, and it does so with two rules that took real defects to get right:

* a single-path FP **never** concludes — the loop blocks that whole-path
  combination and enumerates the next;
* FP from a *sample* of a still-huge space is unsound, so the aggregate FP needs
  either ``full_space_explore`` (the slice-compressed relevant space was
  exhausted) or the leak layer's cross-path agreement, which is a claim about
  the callee's code rather than about the enumerated branch directions.

Both rules are stated in ``docs/pipeline-stages.md`` §8 and both are load-bearing;
the tests in ``tests/test_fp_analysis/test_process_report_characterization.py``
pin the exits.

**The loop's body is verbatim.**  It rebinds ``verdict``, ``selections``,
``ps_stats`` and the ``poc_*`` values on various branches and ``break``s out at
six different points, so its inputs are aliased to the state carriers at the top
and written back once at the bottom.  The aliasing is deliberate and must not be
"simplified" into direct ``ctx.x`` references: a rebind through an attribute read
and an attribute write is the same thing only while every exit is accounted for,
and the ``continue``s are what make that hard to see.
"""

from __future__ import annotations

from fp_analysis.cause_invariance import cause_invariant, extract_deref_pointer
from fp_analysis.combination_enum import combo_signature

from pipeline.models import (
    ExplorationState,
    SelectionContext,
    VerdictState,
)
from pipeline.outcomes import Verdict
from pipeline.selection import (
    _last_round_reason,
    _no_path_verdict,
    _tp_verdict_reason,
    _verify_path,
    path_signature,
    run_path_selection,
)
from pipeline.stages import stage_label, stage_step


def _run_path_attempts(
    ctx: SelectionContext, verdict_state: VerdictState, explore: ExplorationState,
) -> None:
    """Walk the feasible path space until a decisive verdict or the budget runs out.

    An *attempt* is one whole-path combination: the first attempt asks the LLM to
    select the best-feasible path, every later one pulls the next untried
    combination from ``ctx.combo_iter``.  The loop is the only place a **TP** is
    set from a single path (and it is a claim about *that* path), and the six
    ``break``/``continue`` edges below are all load-bearing: a ``break`` hands the
    verdict to the aggregation, a ``continue`` blocks this combination and
    re-selects.

    The aliases are the read-only context and the two state carriers; the
    writeback at the end is what makes a rebind inside the loop visible to the
    aggregation.  A plain trailing writeback (no ``try``/``finally``) is exact
    here: the body has no ``return``, so every exit — including each ``break`` —
    reaches it, and an exception escaping the loop aborted the whole report
    before this refactor too.
    """
    analyzer = ctx.analyzer
    parsed_report = ctx.parsed_report
    segments = ctx.segments
    slice_mask = ctx.slice_mask
    source_root = ctx.source_root
    cfg_cache = ctx.cfg_cache
    extra_preconditions = ctx.extra_preconditions
    hard_constraints = ctx.hard_constraints
    sdg = ctx.sdg
    reduced_ps = ctx.reduced_ps
    reduced_bound_one = ctx.reduced_bound_one
    slice_relevance = ctx.slice_relevance
    slice_truncated = ctx.slice_truncated
    bug_file = ctx.parsed_report.metadata.bug_file
    combo_iter = ctx.combo_iter
    loop_attempts = ctx.loop_attempts
    full_space_explore = ctx.full_space_explore
    out_path = ctx.out_path
    report_stem = ctx.report_stem
    report_ref = ctx.report_ref

    verdict_steps = verdict_state.steps
    verdict = verdict_state.verdict
    final_step = verdict_state.final_step
    final_reason = verdict_state.final_reason

    blocked_paths = explore.blocked_paths
    fp_observations = explore.fp_observations
    leak_fp_reasons = explore.leak_fp_reasons
    leak_tp_reasons = explore.leak_tp_reasons
    saw_unresolved = explore.saw_unresolved
    selections = explore.selections
    ps_stats = explore.ps_stats
    poc_result = explore.poc_result
    poc_code = explore.poc_code
    poc_path = explore.poc_path
    combo = explore.combo
    combo_sig = explore.combo_sig

    for attempt in range(1, loop_attempts + 1):
        print(f"\n── Path-selection attempt {attempt}/{loop_attempts} "
              f"(blocked paths: {len(blocked_paths)})"
              + (" — 全空间探索(压缩相关空间小)" if full_space_explore else "")
              + " ──")
        if attempt == 1:
            # Best-feasible-first LLM path (uses global constraints/state context).
            combo = None
            combo_sig = None
            selections, ps_stats = run_path_selection(
                analyzer, parsed_report, segments, slice_mask,
                source_root=source_root, cfg_cache=cfg_cache,
                extra_preconditions=extra_preconditions,
                hard_constraints=hard_constraints,
                blocked_paths=blocked_paths,
            )
        else:
            # Systematic whole-path combination enumeration: pull the next untried
            # combination (skip ones already blocked as whole-path), build +
            # conflict-check it deterministically.
            combo = None
            combo_sig = None
            while True:
                nxt = next(combo_iter, None)
                if nxt is None:
                    break
                c, _sel = nxt
                csig = combo_signature(reduced_ps, c)
                if csig in blocked_paths:
                    continue
                combo, combo_sig = c, csig
                break
            if combo is None:
                # Every whole-path combination tried → not a conflict-based FP;
                # route to the bounded-exploration aggregation.
                print(f"  attempt {attempt}: ⚠️ 全部整路径组合已探索 → 交由探索聚合判定")
                verdict_steps.append(
                    f"{stage_label(9)} attempt {attempt} → 全部整路径组合已探索，交由路径探索聚合判定"
                )
                break  # verdict stays None → post-loop aggregation
            selections, ps_stats = run_path_selection(
                analyzer, parsed_report, segments, slice_mask,
                source_root=source_root, cfg_cache=cfg_cache,
                extra_preconditions=extra_preconditions,
                hard_constraints=hard_constraints,
                blocked_paths=blocked_paths,
                combo=combo,
                path_space=reduced_ps,
            )

        if selections is None:
            if ps_stats.get("combo_infeasible"):
                # THIS whole-path combination is globally infeasible → block that
                # combination and enumerate the next.  NOT a report-level FP: a
                # different combination may be feasible (or reach the bug).
                sig = combo_sig or path_signature([])
                blocked_paths.append(sig)
                fp_observations.append({
                    "signature": sig,
                    "final_reason": "组合级全局冲突（该整路径组合不可行）",
                })
                verdict_steps.append(
                    f"{stage_label(9)} attempt {attempt} → 组合 {combo} 全局不可行，"
                    f"屏蔽该整路径并继续枚举其它组合"
                )
                print(f"  attempt {attempt}: ⚠️ 组合 {combo} 全局不可行 → 屏蔽并继续枚举 "
                      f"(blocked={len(blocked_paths)})")
                poc_result = None
                continue
            if ps_stats.get("exhausted"):
                # No NEW distinct whole-path remains.  This is NOT a conflict-based
                # infeasibility — it does not prove the report is FP: there may be
                # untried combinations that reach the bug.  Route to the post-loop
                # bounded-exploration aggregation (FP only if every explored
                # distinct path was decisive-FP with no unresolved).
                print(f"  attempt {attempt}: ⚠️ 路径空间已穷尽（无新的不同组合）"
                      f" → 交由探索聚合判定")
                verdict_steps.append(
                    f"{stage_label(9)} attempt {attempt} → 路径空间已穷尽，交由路径探索聚合判定"
                )
                break  # verdict stays None → post-loop aggregation
            # Genuine conflict-based infeasibility (conflict checker / stuck):
            # no globally-consistent feasible path exists → FP (sound).
            final_step = stage_label(9)
            final_reason = ps_stats.get("result", "未找到全局一致可行路径")
            verdict_steps.append(f"{stage_label(9)} → {final_reason}（FP）")
            verdict = Verdict.FP
            break

        print(f"  attempt {attempt}: ✅ 找到全局一致可行路径（初始 TP）")
        print(f"\n  Segment Selections ({len(selections)} segments):")
        for sel in selections:
            seg_idx = sel.get("segment_index", "?")
            state = sel.get("established_state", "")
            if isinstance(state, str) and len(state) > 120:
                state = state[:120] + "..."
            branches = sel.get("branch_decisions", [])
            if branches:
                print(f"    Seg {seg_idx}: {len(branches)} branch decision(s)")
                for b in branches[:3]:  # show first 3
                    if isinstance(b, dict):
                        print(f"      - {b.get('node_id', '?')}: {b.get('branch_taken', '?')}")
            else:
                print(f"    Seg {seg_idx}: linear (no branch decisions)")

        # ── A report that states only a location cannot be certified TP ──
        # Stage 9 just built a feasible path *itself*; whether the report ever
        # described one is the analyzer's own statement, carried in the model
        # (see `_no_path_verdict` and docs/report-frontend.md).  When it did
        # not, generate nothing: the verification would have no reported path
        # to check the POC against, and a TP would rest on the path we chose.
        _no_path = _no_path_verdict(parsed_report.metadata.path_evidence)
        if _no_path is not None:
            final_step, final_reason = _no_path
            verdict_steps.append(f"{final_step} → 不作 TP 判定 → UNKNOWN")
            verdict = Verdict.UNKNOWN
            print(f"\n  ⚠️ 报告只给出位置、没有路径（path_evidence="
                  f"{parsed_report.metadata.path_evidence}）→ 不作 POC 验证"
                  f"（逐步对照无报告路径可依）→ UNKNOWN")
            break

        # ── Generate POC + LLM simulate + feedback (see _verify_path) ──
        print(f"\n{'='*70}")
        print("POC VERIFICATION (generate + LLM correctness audit + simulate; no compile)")
        print(f"{'='*70}")
        vres = _verify_path(
            analyzer, report_ref, source_root, out_path, report_stem,
            selections=selections,
        )
        poc_result = vres["poc_result"]
        poc_code = vres["poc_code"]
        poc_path = vres["poc_path"]
        concl = vres["conclusion"]
        leak_v = vres.get("leak_verdict")

        if concl == "tp":
            if leak_v == "tp" and not leak_fp_reasons and attempt < loop_attempts:
                # A leak-layer TP is a CODE-level ownership JUDGMENT, not
                # execution evidence (`triggered`), and the leak chain is read once
                # per path attempt — on read_index-484-1 the same function came back
                # leak-FP on one attempt and leak-TP on another, so accepting the
                # first decisive attempt made the verdict depend on attempt order
                # alone.  Block this path and keep exploring: the leak layer is
                # accepted once the conforming attempts agree (post-loop below).
                sig = path_signature(selections)
                blocked_paths.append(sig)
                leak_tp_reasons.append(vres.get("final_reason") or "")
                verdict_steps.append(
                    f"{stage_label(9)} attempt {attempt} → leak 层判定 TP，屏蔽该路径并继续探索"
                    f"以求一致（leak 判定为整函数级结论，不由单次尝试定案）"
                )
                print(f"  attempt {attempt}: ✅ leak 层判定 TP → 屏蔽该路径并继续探索 "
                      f"以验证跨路径一致 (blocked={len(blocked_paths)})")
                poc_result = None
                continue
            final_step = stage_step(10, "模拟执行反馈")
            # Every verdict owns its reason (see `_tp_verdict_reason`): a TP row must
            # never inherit an earlier attempt's FP/leak-FP reasoning.
            final_reason = vres.get("final_reason") or _tp_verdict_reason(
                leak_tp=leak_v == "tp",
                round_reason=_last_round_reason(vres.get("sim_res")),
            )
            verdict_steps.append(
                stage_step(10, "模拟执行反馈 → leak 层一致性判定 TP（多路径一致）")
                if leak_v == "tp" else
                stage_step(10, "模拟执行反馈 → 测试用例模拟执行判定触发 bug（TP）")
            )
            verdict = Verdict.TP
            break
        if concl == "fp":
            final_reason = vres["final_reason"] or (
                (poc_result or {}).get("simulation", {}).get("final_reason", "")
            )
            if leak_v == "fp":
                leak_fp_reasons.append(final_reason)
            if reduced_bound_one:
                # The compressed path space collapsed to a single root-cause
                # equivalence class: every enumerated branch is slice-confirmed
                # root-cause-irrelevant (its predicate has no data/control path to
                # the bug trigger).  No other feasible combination can therefore
                # change the verdict, so a decisive FP on THIS path generalizes to
                # the whole enumerated space — graph evidence, not a text match.
                sig = path_signature(selections)
                blocked_paths.append(sig)
                fp_observations.append({
                    "signature": sig, "final_reason": final_reason,
                })
                final_step = stage_step(9, "路径空间压缩推广")
                verdict_steps.append(
                    stage_step(9, "路径空间压缩推广 → 枚举分支全部根因无关"
                                  "（切片压缩 whole-path bound→1），"
                                  "该路径判定推广到整个路径空间 → FP")
                )
                ps_stats["generalized_fp"] = True
                verdict = Verdict.FP
                print(f"  attempt {attempt}: ❌ 决定性 FP + 枚举分支全部根因无关"
                      f" → 压缩推广到整个路径空间 → FP (generalized)")
                break

            # A decisive STRUCTURAL FP — the CSA path is internally
            # self-contradictory (csa_contradiction) or the LLM marks it
            # definite_fp — is a report-level claim about the ROOT-CAUSE pointer
            # (e.g. "the deref'd pointer cannot be null under the real API
            # contract").  If no *relevant* branch that writes that pointer takes
            # a different direction anywhere in the reduced space, the
            # contradiction holds on every combination → the single decisive FP
            # generalizes to the whole (relevant) space (16→1 simulation).  This
            # is a deterministic PDG check, not a text match.  Soundness: it only
            # fires on a structural FP + a successfully-recovered pointer + a
            # non-truncated slice, and never re-opens a TP.
            #
            # The generalization is a property of the code and the reduced space, not
            # of a particular attempt, so it is attempted on EVERY decisive-FP attempt
            # (the reference path is the one that produced the decisive verdict).  The
            # old `attempt == 1` restriction silently dropped the generalization exactly
            # when re-selection was what revealed the root cause, and the old
            # `reduced_bound <= _STRUCTURAL_GP_MAX` cap existed only because the scan
            # used to be a bounded prefix of the cross-product; `cause_invariant` is now
            # exhaustive without enumerating it, so a decisive structural FP in an
            # astronomically large space can conclude FP instead of always UNKNOWN.
            structural_fp = any(
                r.get("csa_contradiction") or r.get("fp_likelihood") == "definite_fp"
                for r in ((poc_result or {}).get("simulation", {}).get("rounds") or [])
            )
            if structural_fp:
                var = extract_deref_pointer(parsed_report, sdg)
                if var and cause_invariant(
                    sdg, selections, reduced_ps, segments,
                    slice_relevance["node_relevance"], var,
                    truncated=slice_truncated, bug_file=bug_file,
                ):
                    final_step = stage_step(9, "结构性FP根因不变性推广")
                    verdict_steps.append(
                        stage_step(9, "结构性FP根因不变性推广 → 决定性结构性 FP"
                                      "（矛盾在 report 级 CSA 路径），"
                                      f"相关分支不写根因指针 {var} 且不随组合变化 → "
                                      "推广到整个相关路径空间 → FP")
                    )
                    ps_stats["structural_generalized_fp"] = True
                    verdict = Verdict.FP
                    print(f"  attempt {attempt}: ❌ 决定性结构性 FP + 根因指针 {var} 不变"
                          f" → 推广到整个相关路径空间 → FP (structural generalized)")
                    break

            # A decisive FP on ONE path only shows THAT path is infeasible — it
            # does not prove the report is a false positive: other feasible
            # combinations might genuinely reach the bug.  So a single-path FP
            # never concludes the verdict by itself.  Instead we EXPLORE the
            # feasible path space: block this WHOLE-PATH combination (never a
            # per-segment branch — infeasibility is combination-dependent) and
            # enumerate the next distinct combination.  We only conclude FP after
            # the exploration loop has examined distinct combinations and found
            # none that is a genuine TP (see the post-loop aggregation).
            sig = path_signature(selections)
            blocked_paths.append(sig)
            fp_observations.append({
                "signature": sig,
                "final_reason": final_reason,
            })
            verdict_steps.append(
                f"{stage_label(9)} attempt {attempt} → 模拟执行判定单条路径 FP，"
                f"屏蔽该路径({sig})并重选其它可行路径继续探索"
            )
            print(f"  attempt {attempt}: ❌ 单路径判定 FP → 屏蔽该路径并探索其它路径 "
                  f"(blocked={len(blocked_paths)})")
            poc_result = None  # next attempt will rebuild
            if attempt < loop_attempts:
                continue
            # Fall through to post-loop aggregation (budget exhausted).
        if concl is None:
            # Verification errored (environmental, not a classification verdict):
            # keep the found path as TP (unverified), matching prior behavior.
            final_step = stage_label(9)
            verdict_steps.append(f"{stage_label(9)} → 找到全局一致可行路径（TP；验证出错未确认）")
            verdict = Verdict.TP
            break

        # ── unresolved (conflict): abandon path, block it, retry a different one ──
        # Guarded to `concl == "unresolved"` — the FP branch already fell through
        # to post-loop aggregation when the budget is exhausted and must NOT be
        # re-classified as unresolved here.
        if concl == "unresolved":
            sig = path_signature(selections)
            blocked_paths.append(sig)
            saw_unresolved = True
            verdict_steps.append(
                f"{stage_label(9)} attempt {attempt} → 模拟执行冲突/无法触发，放弃并屏蔽该路径，重试"
            )
            print(f"  attempt {attempt}: ⚠️ 模拟执行冲突(unresolved) → 放弃该路径并重试 "
                  f"(blocked={len(blocked_paths)})")
        poc_result = None  # next attempt will rebuild

    verdict_state.verdict = verdict
    verdict_state.final_step = final_step
    verdict_state.final_reason = final_reason
    explore.selections = selections
    explore.ps_stats = ps_stats
    explore.poc_result = poc_result
    explore.poc_code = poc_code
    explore.poc_path = poc_path
    explore.combo = combo
    explore.combo_sig = combo_sig
    explore.saw_unresolved = saw_unresolved


def _aggregate_exploration(
    ctx: SelectionContext, verdict_state: VerdictState, explore: ExplorationState,
) -> None:
    """Reason across the explored paths — the only verdict set without a per-attempt one.

    Reaching here means the loop ended with no decisive single-path TP/FP and no
    path-invariance proof.  Three pieces of evidence can decide:

    * the **leak layer agreeing** across the conforming attempts (a code-level
      ownership claim, universal over the space, so it needs no exhaustion);
    * an **exhausted** slice-compressed relevant space in which every distinct
      path was a decisive FP;
    * otherwise UNKNOWN — and the two ways of being undecided (too large to
      enumerate / an unresolved attempt) are reported separately, because they
      call for different follow-up work.

    The leak layer is accepted only on agreement, and with at least two witnesses
    when the space allowed more than one attempt: the chain is re-read per
    attempt, so a single decisive attempt made the verdict depend on attempt
    order alone.
    """
    verdict_steps = verdict_state.steps
    verdict = verdict_state.verdict
    final_step = verdict_state.final_step
    final_reason = verdict_state.final_reason

    leak_fp_reasons = explore.leak_fp_reasons
    leak_tp_reasons = explore.leak_tp_reasons
    fp_observations = explore.fp_observations
    saw_unresolved = explore.saw_unresolved
    ps_stats = explore.ps_stats

    loop_attempts = ctx.loop_attempts
    full_space_explore = ctx.full_space_explore

    # ── Post-loop aggregation (Tier 2) / Threshold exceeded → UNKNOWN ──
    # Reaching here means the loop ended with no decisive single-path TP/FP and
    # no path-invariance proof.  We aggregate the evidence gathered across the
    # distinct feasible paths we explored:
    #   - any path that triggered the bug        → TP (handled by immediate break)
    #   - FULL space exhausted AND all were FP  → FP (sound: no combo left untried)
    #   - bounded sample of a still-huge space  → UNKNOWN (unexplored combos may
    #     reach the bug, so a bounded all-FP sample does NOT prove FP)
    #   - any unresolved                        → UNKNOWN
    #
    # Note: a single-path FP NEVER concludes the verdict.  FP is only provable
    # when the slice-compressed relevant space was exhaustively enumerated
    # (`full_space_explore`: reduced_bound was small, ≤ _FULL_EXPLORE_THRESHOLD).
    # A BOUNDED exploration of a huge uncompressible space (reduced_bound >
    # threshold) samples only `loop_attempts` combos of an astronomically larger
    # set — every one being FP does not rule out a feasible path in the untried
    # combinations, so the honest verdict is UNKNOWN, not FP.
    if verdict is None:
        # ── Leak-layer consensus (memory-leak reports) ──────────────────────
        # The leak layer decides a leak from the object's ownership chain: WHO ends up
        # owning the object on each continuation, and whether the abandon needs
        # out-of-contract input.  That is a claim about the CALLEE's code, not about the
        # enumerated branch directions — the enumerated branches select which index
        # format is read, i.e. *which* object exists, not who owns it.  So unlike a
        # per-path simulation FP (which stays a sample of the space and therefore needs
        # `full_space_explore`), a leak verdict is already universal over the space and
        # does not need the space exhausted: on a combo where the object is never
        # allocated nothing leaks, and where it is allocated the chain the leak layer
        # classified is the chain that runs.
        #
        # It is accepted only on AGREEMENT: every conforming attempt that produced a leak
        # verdict must say the same thing (`leak_tp_reasons` empty for FP), with at least
        # two witnesses when the space allowed more than one attempt.  A single decisive
        # attempt was previously enough — and since the leak chain is re-read per attempt,
        # read_index-484-1 came back FP,FP,TP across attempts, so attempt ORDER alone
        # decided the verdict.  Disagreement voids the leak layer (the report then falls
        # through to the deterministic aggregation, i.e. UNKNOWN on a huge space) rather
        # than picking whichever attempt concluded first.
        min_witnesses = min(2, loop_attempts)
        leak_contradicted = bool(leak_tp_reasons and leak_fp_reasons)
        if leak_contradicted:
            leak_note = (
                f"leak 层跨路径判定自相矛盾（TP {len(leak_tp_reasons)} 次 / "
                f"FP {len(leak_fp_reasons)} 次）"
            )
            verdict_steps.append(
                stage_step(9, f"leak 层一致性 → {leak_note}，该层证据作废"))
            print(f"  ⚠️ {leak_note} → leak 层证据作废，交由路径空间聚合判定")
        if (
            not leak_tp_reasons
            and len(leak_fp_reasons) >= min_witnesses
        ):
            verdict = Verdict.FP
            final_step = stage_step(9, "leak 层一致性推广")
            final_reason = (
                f"leak 层在 {len(leak_fp_reasons)} 条不同可行路径上一致判定 FP："
                f"对象在其 ownership chain 上被释放/移交 owner（或仅在损坏输入下 abandon），"
                f"枚举分支只改变读取的索引类型、不改变该对象的归属 → 整个路径空间 → FP。"
                f"依据：{leak_fp_reasons[0][:300]}"
            )
            verdict_steps.append(
                stage_step(9, f"leak 层一致性推广 → "
                              f"{len(leak_fp_reasons)} 条路径一致判定 FP → FP")
            )
            ps_stats["leak_consensus_fp"] = True
            print(f"  ❌ leak 层一致性推广：{len(leak_fp_reasons)} 条路径一致判定 FP → FP")
        elif fp_observations and not saw_unresolved and full_space_explore:
            verdict = Verdict.FP
            final_step = stage_step(9, "路径探索聚合")
            final_reason = (
                f"全空间探索（压缩相关空间穷尽）: 已探索 {len(fp_observations)} 条不同可行路径，"
                f"均模拟判定为决定性 FP，整个压缩路径空间无可行解 → FP"
            )
            verdict_steps.append(
                stage_step(9, f"路径探索聚合 → 压缩相关路径空间已穷尽，"
                              f"已探索 {len(fp_observations)} 条路径均判定 FP → FP")
            )
        else:
            if not full_space_explore:
                undecided = f"路径空间过大未穷举（仅作有界探索 {len(fp_observations)} 条）"
            elif saw_unresolved:
                undecided = f"探索中存在 unresolved（{len(fp_observations)} 条 FP 观察）"
            else:
                undecided = f"探索后仍无确定结果（{len(fp_observations)} 条 FP 观察）"
            if leak_contradicted:
                undecided += f"；{leak_note}"
            verdict = Verdict.UNKNOWN
            final_step = stage_step(9, "路径遍历超限")
            final_reason = (
                f"{undecided}；未穷举的可行空间中仍可能存在可行解，"
                f"无法排除路径空间存在 TP，返回 UNKNOWN"
            )
            verdict_steps.append(
                stage_step(9, f"路径遍历超限 → UNKNOWN"
                              f"（{undecided}，无法排除未探索空间的可行解）")
            )

    verdict_state.verdict = verdict
    verdict_state.final_step = final_step
    verdict_state.final_reason = final_reason
    explore.ps_stats = ps_stats
