"""TP verification via LLM simulated execution of a generated test case.

After the pipeline judges a report TP and the verification stage produces a
POC test case, this module asks the LLM to *simulate executing* the test case
with its concrete inputs and decide whether it actually triggers the flagged
bug.  When the simulated run reports "not triggered", a feedback loop refines
the POC (correcting inputs/preconditions based on the blocking reason) and
re-simulates, up to ``max_rounds``.

The LLM must ALSO verify **path conformance** step by step: the ordered CSA
report path events are fed in, and the LLM pairs each step's expected result
with what the POC's execution actually produces there.  A POC that reaches a
crash through a state the report's own path never assumes (a *hallucinated*
test case — wrong path, right crash) is non-conforming; it is rejected as a TP
and regenerated with the mismatched steps as feedback, never accepted and never
flipped to FP.

The result does not flip the TP verdict (by design — see the ``UNVERIFIED``
status) but records per-round simulation traces as evidence of how confident
the TP classification is.

Reuses the host :class:`FPAnalyzer` for LLM plumbing (``_call_llm``,
``_parse_json_response``, ``_extract_code``, ``_make_compile_checker``,
``_fix_code_with_llm``) so behaviour stays consistent with the rest of the
pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from llm_client.fp_analysis.prompt_templates import (
    MEMORY_LEAK_SIMULATION_RULES,
    SIMULATION_PROMPT,
    SIMULATION_REFINE_PROMPT,
    SIMULATION_REFINE_SYSTEM_PROMPT,
    SIMULATION_SYSTEM_PROMPT,
)

# Completion budget for the simulation / refinement calls.  The simulation prompt asks for a
# per-step path-conformance record for EVERY report path step plus a critical-node audit, so its
# JSON reply is the longest in the pipeline: on read_index-484 it was cut off mid-string at
# ~12.2k chars (the 4096-token default) in 5 of 8 rounds.  The strict parse then failed, the regex
# fallback dropped `path_conformance` entirely, and its default `path_conformance_ok=True` let a
# truncated `"triggered": true` be accepted as a TP.  Pinned explicitly (rather than inheriting
# the FPAnalyzer default) so a future change to that default cannot silently re-truncate the
# pipeline's longest reply.
_SIMULATION_MAX_TOKENS = 16384

# Bug types that describe a memory / resource leak.  Unlike CSA's exact ``"Memory leak"`` string
# (matched by path_analyzer._is_memory_leak), this set covers the leak aliases CSA emits, so those
# reports get leak-specific simulation semantics instead of the null-deref rules.
_LEAK_BUG_TYPES = {
    "memory leak",
    "potential memory leak",
    "potential leak of memory",
    "memory never released",
    "leak of memory",
    "leak",
    "resource leak",
    "cwe-401",
}


def _is_leak_type(bug_type) -> bool:
    """Whether *bug_type* denotes a memory/resource leak (fuzzy match)."""
    if not bug_type:
        return False
    t = str(bug_type).strip().lower()
    return t in _LEAK_BUG_TYPES or any(a in t for a in ("leak", "never released", "cwe-401"))


def _chain_bool(lc: dict, key: str, default: bool = False) -> bool:
    v = lc.get(key)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return default


def _normalize_leak_chain(raw) -> dict | None:
    """Coerce an LLM-emitted ``leak_chain`` dict into a normalized form with sane bool defaults.

    Returns ``None`` for non-dicts (non-leak reports, or the LLM left it null).  Unknown/missing
    keys keep their semantic default (conservative: no fire, no free, no owner) rather than being
    treated as decisive evidence.
    """
    if not isinstance(raw, dict):
        return None
    store = raw.get("wellformed_continuation_stores_to")
    if isinstance(store, str):
        store = store.strip().lower()
        if store not in ("return", "owner"):
            store = None
    abandon = raw.get("abandon_path")
    if isinstance(abandon, str):
        abandon = abandon.strip().lower()
        if abandon not in ("exception_after_alloc", "manual_no_free_early_return"):
            abandon = None
    return {
        "alloc_line": raw.get("alloc_line"),
        "alloc_expr": str(raw.get("alloc_expr") or ""),
        "wellformed_continuation_stores_to": store,
        "freed_on_any_path": _chain_bool(raw, "freed_on_any_path"),
        "abandon_path": abandon,
        "requires_broken_input": _chain_bool(raw, "requires_broken_input"),
        "leak_fires_on_wellformed": _chain_bool(raw, "leak_fires_on_wellformed"),
    }

def _expand_sparse_audit(data: dict, steps: list[dict]):
    """Expand the SPARSE step audit into the full per-step ``path_conformance``.

    The simulation prompt used to ask for one 5-key record per report path step
    (``{step, report_expectation, poc_result, satisfied, note}``).  On a 92-step
    report that reply was 12,482 chars, of which 61% was JSON scaffolding (five
    long key names + indentation, 92 times over) and another 17% echoed back the
    step text the prompt itself had just supplied — 78% of the step audit, and
    the step audit was 78% of the whole reply, carrying *no* information in a
    conforming round (92/92 satisfied).  The model now reports sparsely:

        "steps_checked": 92,      # how many numbered steps it walked
        "path_mismatches": [{"step": 47, "poc_result": …, "why": …}]  # only the rest

    and this rebuilds the full record list, so every downstream consumer
    (``_format_conformance_mismatches``, the console audit, the result JSON)
    keeps reading the same shape.  Everything the model does not list is taken
    as satisfied, which is what makes the conforming round nearly free.

    Two earlier aggregates were tried and dropped, both for the same reason —
    they turned a formatting slip into a lost round:

    * a per-step BIT STRING (``"1111…"``): deepseek emitted 90 bits for 92 steps
      while reporting ``steps_checked=92`` (rightly discarding a decisive round
      and flipping read_index-484-1 from TP to UNKNOWN);
    * a satisfied COUNT: on read_index-484-2 the model wrote
      ``steps_satisfied: 0`` next to only 11 mismatch rows for 110 steps —
      it meant "the path as a whole is not reproduced", not "11 of 110 steps
      hold", and the arithmetic check rejected the round.

    ``steps_checked`` is the one claim left (it must equal the step count, so a
    model that audited only part of the list is caught), and the mismatch LIST
    is the audit of record.  A ``steps_satisfied`` field is tolerated and
    ignored if the model volunteers one.

    Returns ``(records, ok, degraded)``:

    * ``records is None``  — the reply carried no sparse audit at all (a legacy
      reply uses the explicit ``path_conformance`` array instead).
    * ``ok``               — no mismatches, i.e. every step conforms.
    * ``degraded``         — a sparse audit was present but unusable
      (``steps_checked`` ≠ the step count, a mismatch index out of range or
      duplicated).  Such a round carries no evidence about the POC's path and
      must not certify a TP, exactly like a truncated reply: it regenerates
      instead.
    """
    if "path_mismatches" not in data:
        return None, None, False

    def _unusable(reason: str):
        # Logged loudly: an unusable audit costs a regeneration round exactly
        # like a truncated reply, so its cause must be visible in the run log.
        logger.warning(
            "Simulation: sparse step audit unusable (%s) — steps_checked=%r "
            "path_mismatches=%r for %d step(s)",
            reason, data.get("steps_checked"), data.get("path_mismatches"),
            len(steps),
        )
        return None, None, True

    def _as_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # The one claim the reply must get right: it walked every rendered step.
    checked_n = _as_int(data.get("steps_checked"))
    if checked_n != len(steps):
        return _unusable("steps_checked ≠ the number of steps rendered")
    mismatches: dict[int, dict] = {}
    for m in data.get("path_mismatches") or []:
        if not isinstance(m, dict):
            continue
        idx = _as_int(m.get("step"))
        if idx is None or not (1 <= idx <= len(steps)):
            return _unusable(f"mismatch row with an out-of-range step {m.get('step')!r}")
        if idx in mismatches:
            return _unusable(f"duplicate mismatch row for step {idx}")
        mismatches[idx] = m
    if not mismatches and data.get("path_conformance_ok") is False:
        # "The POC contradicts the report" with nothing to point at: the round
        # has no actionable evidence to regenerate from, so treat it as an
        # unusable audit rather than an empty mismatch list.
        return _unusable("non-conforming verdict with no mismatch row")
    records = []
    for i in range(1, len(steps) + 1):
        rec: dict = {"step": i, "satisfied": i not in mismatches}
        text = steps[i - 1].get("text")
        if text:
            rec["report_expectation"] = text
        if i in mismatches:
            m = mismatches[i]
            rec["poc_result"] = str(m.get("poc_result") or "")
            why = m.get("why") or m.get("note")
            if why:
                rec["note"] = str(why)
        records.append(rec)
    return records, not mismatches, False


logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Structured outcome of one LLM simulated execution of a test case.

    The audit of the CSA report path lives in two fields: ``path_conformance``
    (one record per ordered path step, the step audit of record) and
    ``csa_contradiction`` (the report's own claims being mutually impossible),
    with ``fp_likelihood``/``reason`` carrying the overall judgement — so an FP
    misclassified as TP can be explained precisely.
    """

    triggered: bool
    reason: str
    trace: list[str] = field(default_factory=list)
    blocking_reason: str | None = None
    suggested_input_change: str | None = None
    csa_contradiction: str | None = None
    fp_likelihood: str | None = None
    # Test-case correctness audit (the POC is NOT compiled by a real toolchain;
    # the LLM verifies it before/while simulating).
    test_case_correct: bool = True
    test_case_issues: list[str] = field(default_factory=list)
    # Step-by-step PATH-CONFORMANCE audit: one record per ordered CSA report
    # path step, pairing the report's expected result with what the POC's
    # simulated execution actually produced there.  ``path_conformance_ok`` is
    # false when ANY step does not match — i.e. the POC reaches its state via a
    # path the report does not describe (a hallucinated test case), so its
    # "trigger" cannot be trusted as reproducing the reported bug.
    #
    # On the wire this list is SPARSE: the prompt asks for a `steps_checked`
    # count plus a `path_mismatches` row per unsatisfied step (see
    # `_expand_sparse_audit`), which fills this list in `from_dict`.  Keep
    # the expanded shape — it is what the console audit, `_format_conformance_
    # mismatches` and the result JSON consume.
    path_conformance: list[dict] = field(default_factory=list)
    path_conformance_ok: bool = True
    # True when this round's reply could not be parsed as JSON and was salvaged
    # (truncation repair) or read by the regex fallback — or when its sparse
    # step audit was unusable (a `steps_checked` that disagrees with the
    # rendered step count, self-contradicting mismatch rows, or no step
    # evidence at all).  Either way the ``path_conformance`` audit is
    # missing/partial, so ``path_conformance_ok`` carries no evidence and the
    # round cannot certify a TP.
    parse_degraded: bool = False
    # Memory-leak evidence chain, filled ONLY for memory-leak reports (see
    # MEMORY_LEAK_SIMULATION_RULES).  Lets the verifier reach a decisive leak verdict instead of
    # getting stuck in a "reachable the allocation but never fires" refine limbo.
    leak_chain: dict | None = None

    @classmethod
    def from_dict(cls, data: dict, steps: list[dict] | None = None) -> "SimulationResult":
        """*steps*: the numbered CSA path steps the prompt rendered (see
        ``SimulationVerifier._csa_step_rows``) — needed to expand the sparse
        audit and to fill each record's ``report_expectation``."""
        steps = steps or []
        conform = data.get("path_conformance") or []
        conform = [c for c in conform if isinstance(c, dict)]
        degraded = False
        sparse = None
        if not conform:
            # Sparse audit (the prompt's format).  A legacy reply that still
            # carries the explicit array above takes precedence — it is the
            # richer evidence and needs no expansion.
            sparse, sparse_ok, degraded = _expand_sparse_audit(data, steps)
            if sparse is not None:
                conform = sparse
        if sparse is None and not conform and not degraded:
            # Neither the sparse audit nor the array is present: the reply
            # carries no step evidence at all, so the conformance gate below
            # must not read the defaulted `path_conformance_ok=True` as proof.
            # (This is the hole a truncated-and-regex-parsed reply fell into.)
            degraded = True
            logger.warning(
                "Simulation: reply carries NO step audit for %d step(s) "
                "(steps_checked=%r steps_satisfied=%r path_mismatches=%r "
                "path_conformance=%r); treating the round as non-conforming.",
                len(steps), data.get("steps_checked"), data.get("steps_satisfied"),
                data.get("path_mismatches"), data.get("path_conformance"),
            )
        # Robustness: if the LLM listed per-step mismatches but forgot the
        # aggregate bool, infer it from the list (any unsatisfied step ⇒ not ok).
        ok_raw = data.get("path_conformance_ok", None)
        if ok_raw is None:
            conform_ok = all(bool(c.get("satisfied", True)) for c in conform) if conform else True
        else:
            conform_ok = bool(ok_raw)
        if any(not bool(c.get("satisfied", True)) for c in conform):
            conform_ok = False
        if sparse is not None and sparse_ok is False:
            # The mismatch LIST is the audit of record: a non-empty list means
            # non-conforming even if the model also wrote
            # `path_conformance_ok: true`.
            conform_ok = False
        if degraded:
            conform_ok = False
        return cls(
            triggered=bool(data.get("triggered", False)),
            reason=str(data.get("reason", "")),
            trace=[str(t) for t in (data.get("trace") or [])],
            blocking_reason=data.get("blocking_reason"),
            suggested_input_change=data.get("suggested_input_change"),
            csa_contradiction=data.get("csa_contradiction"),
            fp_likelihood=data.get("fp_likelihood"),
            test_case_correct=bool(data.get("test_case_correct", True)),
            test_case_issues=[
                str(t) for t in (data.get("test_case_issues") or [])
            ],
            path_conformance=conform,
            path_conformance_ok=conform_ok,
            parse_degraded=degraded,
            leak_chain=_normalize_leak_chain(data.get("leak_chain")),
        )


class SimulationVerifier:
    """Generate-free TP validation: LLM simulates a POC, with a refine loop."""

    def __init__(self, analyzer) -> None:
        """*analyzer*: the :class:`FPAnalyzer` instance running the pipeline."""
        self._analyzer = analyzer

    # ── prompt helpers ──────────────────────────────────────────────────────

    def _bug_info(self, analysis) -> dict:
        md = analysis.parsed_report.metadata
        return {
            "bug_type": md.bug_type,
            "bug_file": md.bug_file,
            "bug_line": md.bug_line,
            "bug_function": md.function_name,
            "project": getattr(self._analyzer, "_project_name", "") or "target project",
        }

    def _source_context(self, analysis) -> str:
        if getattr(analysis, "reduced_code", ""):
            return analysis.reduced_code
        try:
            source_ctx = self._analyzer._source_extractor.extract(
                analysis.parsed_report
            )
            return source_ctx.format_for_llm(max_lines=60)
        except Exception as e:  # noqa: BLE001 — best-effort source context
            logger.warning("Simulation: source-context extraction failed: %s", e)
            return "(source context unavailable)"

    def _csa_step_rows(self, analysis) -> list[dict]:
        """The ordered CSA report path steps as numbered rows.

        Single source of truth for step numbering: the rendered prompt block
        numbers these 1..N, and the reply's ``path_mismatches`` rows are indexed
        by *that* numbering (``row["step"] == n`` ⇔ ``rows[n-1]``), as is
        ``steps_checked``.  Rendering and audit expansion must never number the
        steps independently — an off-by-one here would silently attribute a
        mismatch to the wrong step.

        Source of truth: ``parsed_report.path_events`` (ordered by path_number,
        already populated by the HTML parser).  Each row carries the report's
        claimed event kind / branch decision / description, plus the source line
        snippet when available.
        """
        pr = getattr(analysis, "parsed_report", None)
        events = getattr(pr, "path_events", None) if pr else None
        if not events:
            return []
        snippets = getattr(pr, "source_snippets", {}) or {}
        rows: list[dict] = []
        for i, ev in enumerate(events, 1):
            kind = (ev.get("event_kind") or "event")
            desc = " ".join((ev.get("description") or "").split())
            ln = ev.get("line_number")
            bc = ev.get("branch_condition")
            tag = ""
            if kind == "control":
                if bc is True:
                    tag = " [branch: TRUE]"
                elif bc is False:
                    tag = " [branch: FALSE]"
                else:
                    tag = " [branch]"
            elif kind == "report":
                tag = " [BUG REPORTED HERE]"
            loc = f"L{ln} " if ln else ""
            snip = ""
            try:
                code = snippets.get(int(ln)) if ln else None
            except (TypeError, ValueError):
                code = None
            if code:
                snip = f"  `{' '.join(code.split())[:120]}`"
            rows.append({"step": i, "text": f"[{kind}]{tag} {loc}{desc}{snip}"})
        return rows

    def _csa_path_steps(self, analysis) -> str:
        """Render the numbered step list the LLM walks (see ``_csa_step_rows``)."""
        rows = self._csa_step_rows(analysis)
        if not rows:
            return "(no ordered CSA path events available in the report)"
        return "\n".join(f"{r['step']}. {r['text']}" for r in rows)

    @staticmethod
    def _format_conformance_mismatches(sim: "SimulationResult") -> str:
        """Render the steps where the POC's execution did NOT reproduce the
        report's expected result, for feedback into the refinement prompt."""
        rows = [
            c for c in (getattr(sim, "path_conformance", None) or [])
            if not bool(c.get("satisfied", True))
        ]
        if not rows:
            if getattr(sim, "parse_degraded", False):
                return (
                    "(no step audit available — the simulation reply was "
                    "truncated at the token limit or unparsable, or its "
                    "`steps_checked` did not match the number of steps, so no "
                    "step evidence survives; re-emit the COMPLETE JSON object "
                    "with `steps_checked` equal to the last step number and a "
                    "`path_mismatches` row for every step that does not conform)"
                )
            return "(none reported)"
        out = []
        for c in rows:
            out.append(
                f"- step {c.get('step', '?')}: report expects → "
                f"{c.get('report_expectation', '?')} | POC produces → "
                f"{c.get('poc_result', '?')}"
                + (f" | note: {c.get('note')}" if c.get("note") else "")
            )
        return "\n".join(out)

    def _preconditions_str(self, completed) -> str:
        if completed is None:
            return "(none)"
        lines = []
        if getattr(completed, "preconditions", None):
            lines += [f"  - {p}" for p in completed.preconditions]
        if getattr(completed, "concretized_values", None):
            lines += [
                f"  - {k} = {v}"
                for k, v in completed.concretized_values.items()
            ]
        summary = getattr(completed, "final_path_summary", "")
        if summary:
            lines.insert(0, summary)
        return "\n".join(lines) if lines else "(none)"

    # ── simulation ──────────────────────────────────────────────────────────

    def simulate_execution(self, poc_code, completed, analysis,
                           selected_branches=None) -> SimulationResult:
        """Ask the LLM to simulate executing *poc_code* and report whether the
        flagged bug triggers.  Returns a :class:`SimulationResult`."""
        info = self._bug_info(analysis)
        system_prompt = SIMULATION_SYSTEM_PROMPT.format(**info)
        step_rows = self._csa_step_rows(analysis)
        csa_path_steps = (
            "\n".join(f"{r['step']}. {r['text']}" for r in step_rows)
            if step_rows
            else "(no ordered CSA path events available in the report)"
        )
        user_prompt = SIMULATION_PROMPT.format(
            bug_type=info["bug_type"],
            bug_file=info["bug_file"],
            bug_line=info["bug_line"],
            bug_function=info["bug_function"],
            completed_path_summary=self._preconditions_str(completed),
            preconditions=self._preconditions_str(completed),
            csa_path_steps=csa_path_steps,
            poc_code=poc_code,
            source_context=self._source_context(analysis),
        )
        # Inject the selected feasible-path branch blueprint so the simulation
        # judges THIS specific branch path (not an arbitrary one).  This is what
        # makes Tier-2 "is this other path infeasible?" a real per-path check.
        if selected_branches:
            blueprint = self._analyzer._format_selected_branch_blueprint(
                selected_branches
            )
            if blueprint:
                user_prompt += (
                    "\n\n## Selected Feasible Path Being Validated\n"
                    "The path selector forced the test case down the following "
                    "branch decisions. Judge whether THIS exact path is feasible "
                    "and whether it reaches the bug; if these branch decisions "
                    "are mutually impossible or cannot reach the flagged state "
                    "in real execution, conclude FP:\n"
                    f"{blueprint}\n"
                )

        # Memory-leak reports get leak-specific semantics; without these the generic (crash-centric)
        # `triggered` rules treat "reaches the allocation" as evidence and the verifier can spin in a
        # "tp but never fires" refine loop → unresolved → UNKNOWN for an actually-decisive leak FP.
        if _is_leak_type(info["bug_type"]):
            user_prompt += "\n\n" + MEMORY_LEAK_SIMULATION_RULES

        # Guard-pinned / finite-domain supplement (semantic completion).  CSA treats some
        # variables (e.g. a file header discriminator compared against an opaque constant
        # factory) as able to take any value, when source pins them to a small fixed set.
        # Surface that advisory domain fact to the auditor so it can judge reachability where
        # CSA saw none.  Set by run_path_selection Step 5.5 on this same FPAnalyzer instance.
        domain_text = getattr(self._analyzer, "_domain_facts_text", "") if self._analyzer else ""
        if domain_text:
            user_prompt += "\n\n" + domain_text
            logger.info("Simulation: appended %d guard-pinned domain-fact lines",
                        domain_text.count("\n") + 1)

        raw = self._analyzer._call_llm(
            system_prompt, user_prompt, max_tokens=_SIMULATION_MAX_TOKENS
        )
        parse_degraded = False
        try:
            data = self._analyzer._parse_json_response(raw, "simulated execution")
            result = SimulationResult.from_dict(data, step_rows)
        except Exception as strict_err:  # noqa: BLE001 — try the repairing parse next
            try:
                # A reply cut off at the token limit is a *truncated* document,
                # not a malformed one: repair the JSON prefix instead of dropping
                # to the regex fallback, which loses the entire per-step
                # path-conformance audit.
                data = self._analyzer._parse_json_response_tolerant(
                    raw, "simulated execution"
                )
                result = SimulationResult.from_dict(data, step_rows)
                parse_degraded = True
                logger.warning(
                    "Simulation: strict JSON parse failed (%s) — salvaged a "
                    "repaired JSON prefix; the path-conformance audit may be "
                    "incomplete, so this round cannot certify a TP.", strict_err,
                )
            except Exception as e:  # noqa: BLE001 — fall back to heuristic parse
                logger.warning(
                    "Simulation: LLM JSON parse failed (%s); using heuristic parse.",
                    e,
                )
                result = self._heuristic_parse(raw)
                parse_degraded = True
        # A degraded round is one whose reply was proven incomplete (parse
        # failure / regex fallback) OR whose step audit came back unusable
        # (`from_dict` set the flag: a `steps_checked` that disagrees with the
        # rendered step count, self-contradicting mismatch rows, or no step
        # evidence at all).  In both cases an absent
        # (hence defaulted) `path_conformance_ok=True` is an artifact, not
        # evidence that the POC reproduced the report's path — so force the
        # conformance gate below to treat the round as non-conforming: never a
        # TP, and never flipped to FP.
        if parse_degraded or result.parse_degraded:
            result.parse_degraded = True
            result.path_conformance_ok = False
        logger.info(
            "Simulation: triggered=%s | fp_likelihood=%s | path_conformance_ok=%s "
            "| parse_degraded=%s | reason=%s",
            result.triggered, result.fp_likelihood, result.path_conformance_ok,
            result.parse_degraded, result.reason[:160],
        )
        # Print the step-by-step path-conformance audit so a hallucinated test
        # case (right crash, wrong path) is visible, not just summarized.
        if result.path_conformance:
            n_bad = sum(
                1 for c in result.path_conformance
                if not bool(c.get("satisfied", True))
            )
            logger.info(
                "Simulation: step-by-step path conformance — %d/%d step(s) matched the CSA report path.",
                len(result.path_conformance) - n_bad, len(result.path_conformance),
            )
            for c in result.path_conformance:
                mark = "✓" if bool(c.get("satisfied", True)) else "✗"
                logger.info(
                    "  %s step %s: report=%s | poc=%s",
                    mark, c.get("step", "?"),
                    str(c.get("report_expectation", "?"))[:120],
                    str(c.get("poc_result", "?"))[:120],
                )
        if result.csa_contradiction:
            logger.info(
                "Simulation: CSA path contradiction → %s",
                result.csa_contradiction[:240],
            )
        return result

    def _heuristic_parse(self, raw: str) -> SimulationResult:
        triggered = bool(re.search(r'"triggered"\s*:\s*true', raw))
        reason = ""
        m = re.search(r'"reason"\s*:\s*"([^"]+)"', raw, re.DOTALL)
        if m:
            reason = m.group(1)
        blocking = None
        m = re.search(r'"blocking_reason"\s*:\s*"([^"]+)"', raw, re.DOTALL)
        if m and m.group(1) not in ("null", ""):
            blocking = m.group(1)
        if not reason:
            reason = raw[:400]
        return SimulationResult(
            triggered=triggered, reason=reason, blocking_reason=blocking,
        )

    # ── refinement ──────────────────────────────────────────────────────────

    def refine_poc(self, poc_code, sim: SimulationResult, completed, analysis,
                   selected_branches=None, path_conformance=None) -> str:
        """Ask the LLM to refine *poc_code* so it actually reaches the bug,
        using the simulation's blocking reason as feedback."""
        info = self._bug_info(analysis)
        system_prompt = SIMULATION_REFINE_SYSTEM_PROMPT
        conformance_text = path_conformance if path_conformance is not None \
            else self._format_conformance_mismatches(sim)
        user_prompt = SIMULATION_REFINE_PROMPT.format(
            bug_type=info["bug_type"],
            bug_file=info["bug_file"],
            bug_line=info["bug_line"],
            bug_function=info["bug_function"],
            poc_code=poc_code,
            triggered=sim.triggered,
            reason=sim.reason,
            blocking_reason=sim.blocking_reason or "unknown",
            suggested_input_change=sim.suggested_input_change or "unknown",
            trace="\n".join(sim.trace) if sim.trace else "(no trace)",
            path_conformance=conformance_text,
            source_context=self._source_context(analysis),
        )
        if selected_branches:
            blueprint = self._analyzer._format_selected_branch_blueprint(
                selected_branches
            )
            if blueprint:
                user_prompt += (
                    "\n\n## Keep the POC on the selected feasible path\n"
                    "Refine the POC while still forcing these branch decisions "
                    "(do not silently switch to a different path):\n"
                    f"{blueprint}\n"
                )
        # Same budget as the simulation: the refinement prompt carries the
        # full original POC + the mismatch report, and its reply is a whole
        # C++ file — a truncated file would be compiled/audited as if complete.
        raw = self._analyzer._call_llm(
            system_prompt, user_prompt, max_tokens=_SIMULATION_MAX_TOKENS
        )
        refined = self._analyzer._extract_code(raw)
        if not refined.strip():
            logger.warning("Simulation: LLM returned empty refined POC; keeping original.")
            return poc_code
        return refined

    def _compile_fix(self, code: str) -> str:
        """No-op in LLM-only mode: the POC is deliberately NOT compiled by a real
        toolchain.  The test case's correctness is audited by the LLM during the
        simulated execution instead, so we return *code* unchanged (avoids the
        real-toolchain include-path failures that are unrelated to the bug's
        reachability)."""
        return code

    # ── deterministic fabrication guard ─────────────────────────────────────
    # A deterministic complement to the (nondeterministic) LLM audit rule for
    # "DIRECT CONSTRUCTION of a dependency-produced struct".  The prompt rule
    # catches this in some runs but not others; this code-level check catches
    # the known impossible-state patterns regardless of what the LLM said.
    #
    # Each entry: (compiled regex over the POC, human reason).  The regex MUST
    # only match a state real production code can NEVER produce, so firing it
    # is safe (no false-positive flip of a genuine TP).
    #
    # addrinfo.ai_addr: real folly callers obtain addrinfo only from the OS
    # getaddrinfo, whose success postcondition guarantees a non-null ai_addr
    # (on failure it returns an error instead).  A POC that sets ai_addr to
    # null — whether by hand-building the struct or by mocking getaddrinfo —
    # fabricates a state no real caller can produce, so the reported null
    # dereference is unreachable.
    _FABRICATION_PATTERNS = [
        (
            re.compile(r"\b(?:->|\.)?\s*ai_addr\s*=\s*(?:nullptr|NULL|0)\b"),
            "POC fabricates `addrinfo.ai_addr = null` (directly, or via a "
            "mocked getaddrinfo). Real callers obtain addrinfo only from the OS "
            "getaddrinfo, whose success postcondition guarantees a non-null "
            "ai_addr (on failure it returns an error instead). This state cannot "
            "occur in real usage → the reported dereference is unreachable, so "
            "the report is an FP.",
        ),
        (
            re.compile(
                r"\b[A-Za-z_]*[Aa]ddr(?:info|Info)[A-Za-z_]*\s*\("
                r"[^)]*\b(?:nullptr|NULL|0)\b[^)]*\)"
            ),
            "POC hand-builds an addrinfo via a helper that receives a null "
            "sockaddr (e.g. `createMockAddrInfo(AF_INET, nullptr, ...)`). Real "
            "callers never construct addrinfo manually — it comes only from the "
            "OS getaddrinfo, whose success postcondition guarantees a non-null "
            "ai_addr. A hand-built addrinfo with a null address is an impossible "
            "state that cannot occur in real usage → the reported dereference is "
            "unreachable, so the report is an FP.",
        ),
    ]

    def _fabrication_evidence(self, poc_code: str) -> str | None:
        """Return a reason string if *poc_code* fabricates an impossible
        dependency-produced state, else None.  Deterministic (no LLM)."""
        for pat, reason in self._FABRICATION_PATTERNS:
            if pat.search(poc_code or ""):
                return reason
        return None

    # ── feedback loop ───────────────────────────────────────────────────────

    def _is_fp_conclusion(self, sim: SimulationResult) -> bool:
        """Whether a simulation round has *concluded* the report is an FP.

        A round is a definite-FP conclusion when the LLM says the path is
        ``definite_fp`` (no test case can reach it), or it found an internal
        contradiction in the CSA path (self-inconsistent ⇒ cannot hold ⇒ FP).
        A ``likely_fp`` / ``uncertain`` / ``tp``-but-not-triggered round is
        NOT an FP conclusion: the POC setup may simply be wrong (fixable), so
        it warrants refinement.
        """
        if sim.fp_likelihood == "definite_fp":
            return True
        if sim.csa_contradiction:
            return True
        return False

    # ── memory-leak decisive verdict ────────────────────────────────────────
    # A deterministic complement to the LLM audit for memory-leak reports.  The generic rules decide
    # a leak only via `triggered` (crash semantics) or `definite_fp`; a leak that is "reachable but
    # the object is never abandoned on a well-formed path" was getting stuck as `tp` + `not
    # triggered` and refining forever → unresolved → UNKNOWN.  Here the LLM's `leak_chain` fields let
    # us reach a decisive TP/FP per round, mirroring how the deterministic fabrication guard works.
    def _leak_decisive_conclusion(self, sim: SimulationResult):
        """Return ``"tp"`` / ``"fp"`` when *sim*'s ``leak_chain`` decisively settles a memory leak,
        else ``None`` (no leak_chain, or it is not conclusive → fall back to the generic loop).

        A chain that contradicts ITSELF is never decisive.  The fields are read as one ownership
        story, so `leak_fires_on_wellformed` (the object is abandoned-unowned on a real
        well-formed caller path) cannot hold together with `requires_broken_input` (that very
        abandon needs corrupt/truncated input): the leak would then have to happen on an
        in-contract path *and* require an out-of-contract one.  DeepSeek emits that pair when it
        is unsure — on read_index-484-1 the SAME code and the SAME conforming path were read as
        ``leak_fires_on_wellformed=true`` in one run and as
        ``..._stores_to=owner`` + ``requires_broken_input=true`` in another, so which verdict the
        report got was decided by which attempt happened to conclude first.  Voiding the
        self-contradictory chain removes that flip without picking a side (the report then falls
        through to the deterministic path-space machinery).
        """
        lc = getattr(sim, "leak_chain", None)
        if not lc:
            return None
        if lc.get("leak_fires_on_wellformed"):
            if lc.get("requires_broken_input"):
                logger.warning(
                    "Simulation: leak_chain self-contradictory — leak_fires_on_wellformed=true "
                    "together with requires_broken_input=true (a well-formed leak cannot need "
                    "broken input); not decisive. chain=%s", lc,
                )
                return None
            # A genuine leak on an in-contract caller path → TP.
            return "tp"
        freed = lc.get("freed_on_any_path")
        store = lc.get("wellformed_continuation_stores_to")
        abandon = lc.get("abandon_path")
        broken = lc.get("requires_broken_input")
        if freed:
            # The object is freed/RAII'd on some path before exit → not leaked there.
            return "fp"
        if store in ("return", "owner") and broken:
            # Every well-formed continuation hands the object to a return/owner; the only way it is
            # abandoned is a mid-stream throw on corrupt/truncated input (no cleanup on that
            # exception path).  No well-formed real caller leaks it → FP.
            return "fp"
        if abandon is None and store in ("return", "owner"):
            # No abandon path exists at all — the object always ends up owned → not a leak.
            return "fp"
        return None

    @staticmethod
    def _compose_leak_reason(lc: dict, sim: SimulationResult) -> str:
        """Build a human reason string from a decisive leak_chain + the LLM's own reason."""
        alloc = lc.get("alloc_expr") or f"line {lc.get('alloc_line')}"
        sim_reason = (sim.reason or "").strip()
        if lc.get("leak_fires_on_wellformed"):
            head = (
                f"Memory-leak 判 TP：对象 {alloc} 在 well-formed 的真实调用路径上被 abandon"
                "且无人接管（既未释放、也未移交 owner/return）→ 真实可达的泄漏"
            )
        elif lc.get("freed_on_any_path"):
            head = (
                f"Memory-leak 判 FP：对象 {alloc} 在函数退出前于某条路径上被释放"
                "（delete/RAII/unique_ptr/清理 helper），并未在该路径上泄漏"
            )
        elif lc.get("wellformed_continuation_stores_to") in ("return", "owner") and lc.get(
            "requires_broken_input"
        ):
            head = (
                f"Memory-leak 判 FP：对象 {alloc} 在每条 well-formed 续行上都会存入 "
                f"{lc.get('wellformed_continuation_stores_to')}（如实参被赋给返回值/owner 并随之退出），"
                "绝不在 well-formed 输入下泄漏；唯一 abandon 是中途读/解析抛异常，需 corrupt/truncated"
                "（损坏输入前置条件）。无 well-formed 真实 caller 能到达该泄漏"
            )
        else:
            head = (
                f"Memory-leak 判 FP：对象 {alloc} 无 abandon 路径，在每条路径上都转入 owner/return，"
                "不会被泄漏"
            )
        if sim_reason:
            return f"{head}。模拟依据：{sim_reason[:300]}"
        return head

    def verify_with_feedback(
        self, poc_code, completed, analysis, max_rounds: int = 3,
        selected_branches=None,
    ) -> dict:
        """Run the TP-validation feedback loop.

        Each round simulates the current POC.  The loop stops at the first
        *decisive* judgment:

        - the round *concludes* the report is an **FP** (``definite_fp`` or a
          CSA-path internal contradiction) → stop immediately and return the FP
          reason (conclusion=fp).  We do NOT refine a definite-FP result —
          refining would only give the LLM another chance to rationalise a TP
          out of a genuinely-impossible path.
        - **PATH-CONFORMANCE GATE**: the POC must reproduce the CSA report's
          ordered path steps.  If ``path_conformance_ok`` is false — the POC
          reaches its state through a path the report does not describe (e.g. a
          manufactured field value) — the test case is WRONG: it is NOT accepted
          as a TP (even when ``triggered`` is true), and it is NOT an FP either.
          The mismatched steps are fed back and the POC is regenerated.
        - the memory-leak decisive verdict (``leak_verdict``) is judged AFTER
          the conformance gate, for the same reason: it is the one decisive
          layer that can conclude FP without exhausting the path space, so a
          non-conforming POC (or an unusable step audit) gives it no evidence.
        - ``triggered`` (and conforming) → the test case fires the bug → **TP**
          (conclusion=tp).
        - otherwise the non-trigger is a *fixable* POC (uncertain / POC setup
          wrong) → refine the POC and repeat.

        After ``max_rounds`` with neither a trigger nor an FP conclusion, the
        report is ``UNVERIFIED`` (conclusion=unresolved; the verdict is left to
        the caller, which does not flip an unresolved case to FP).  A POC that
        never conforms therefore degrades to UNRESOLVED — never to FP.

        Returns:
            {"triggered": bool, "conclusion": "tp"|"fp"|"unresolved",
             "leak_verdict": "tp"|"fp"|None,   # this round's leak-layer verdict
             "rounds": [{"round", "sim", "poc_code"}, ...],
             "final_reason": str | None}
        """
        rounds: list[dict] = []
        code = poc_code
        final_reason_override: str | None = None
        # Set when the leak layer (memory-leak reports) reached a decisive verdict in
        # this round.  The caller accumulates it across path attempts: it is a CODE-level
        # ownership judgment, not per-path execution evidence, so the report-level loop
        # requires the attempts' leak verdicts to agree before accepting one.
        leak_verdict: str | None = None
        is_leak = _is_leak_type(self._bug_info(analysis)["bug_type"])
        for rnd in range(1, max_rounds + 1):
            logger.info("Simulation round %d/%d ...", rnd, max_rounds)
            sim = self.simulate_execution(
                code, completed, analysis, selected_branches=selected_branches
            )
            rounds.append({"round": rnd, "sim": sim, "poc_code": code})

            # A definite-FP conclusion (or a CSA-path internal contradiction) is
            # decisive and is checked BEFORE any trigger: a round that reports
            # ``triggered=true`` alongside ``fp_likelihood=definite_fp`` or a
            # ``csa_contradiction`` is self-contradictory (no single test case can
            # both fire the bug and prove the bug unreachable), so the trigger
            # cannot be trusted.  This also deterministically catches the
            # "reached the buggy line but the dereference was safe" failure mode
            # (the LLM marks ``triggered`` while its own reason says the deref
            # operated on a valid/non-null pointer) — overriding such a trigger to
            # FP regardless of the ``triggered`` bit.
            if self._is_fp_conclusion(sim):
                reason = sim.reason or sim.blocking_reason or \
                    "模拟执行判定报告为 FP（测试用例无法触发 bug）"
                logger.info(
                    "Simulation round %d: definite FP (triggered=%s overridden) "
                    "→ stopping (no refine).", rnd, sim.triggered,
                )
                return {
                    "triggered": False, "conclusion": "fp",
                    "rounds": rounds, "final_reason": reason,
                    "leak_verdict": None,
                }

            # ── Path-conformance gate (hallucination guard) ──────────────────
            # The POC must reproduce the CSA report's path step by step.  When
            # any reported step's expected result is NOT what the POC's execution
            # produces (e.g. the POC reaches a crash through a state the report's
            # path never assumes), the test case is WRONG — its "trigger" is an
            # artifact of a path the report does not describe, so it must NOT be
            # accepted as a TP.  This is checked BEFORE the trigger shortcut, so
            # a right-crash/wrong-path POC cannot sneak through.
            #
            # Safety: a conformance failure NEVER flips the report to FP (the
            # report may still be a genuine TP that a better test case would
            # reach) — it only forces regeneration; if it never conforms the
            # outcome degrades to UNRESOLVED (UNVERIFIED), which does not flip
            # the verdict.
            if not getattr(sim, "path_conformance_ok", True):
                if getattr(sim, "parse_degraded", False):
                    # The reply was incomplete (truncated / regex-parsed) or its
                    # sparse step audit was unusable (steps_checked ≠ step count,
                    # mismatch row contradicting its own bit, no step evidence at
                    # all), so there is no audit to judge.  Regenerate rather than
                    # treat the missing audit as a pass — that is exactly how a
                    # truncated `"triggered": true` was accepted as a TP on
                    # read_index-484.
                    conformance_reason = (
                        "模拟执行的响应被截断/未解析成功，或逐步路径一致性审计不可用"
                        "（位数与步骤数不符/自相矛盾）→ "
                        "重新生成测试用例并要求输出完整 JSON：\n"
                        + self._format_conformance_mismatches(sim)
                    )
                    logger.warning(
                        "Simulation round %d: reply parse degraded (%s) — no "
                        "path-conformance evidence; regenerating.\n%s",
                        rnd, sim.reason[:200],
                        self._format_conformance_mismatches(sim),
                    )
                else:
                    conformance_reason = (
                        "测试用例未复现 CSA 报告的路径：以下步骤的执行结果与报告预期不一致 → "
                        "重新生成测试用例：\n"
                        + self._format_conformance_mismatches(sim)
                    )
                    logger.warning(
                        "Simulation round %d: POC does NOT conform to the CSA report "
                        "path (triggered=%s) — test case is wrong; regenerating.\n%s",
                        rnd, sim.triggered, self._format_conformance_mismatches(sim),
                    )
                if rnd < max_rounds:
                    code = self.refine_poc(
                        code, sim, completed, analysis,
                        selected_branches=selected_branches,
                        path_conformance=self._format_conformance_mismatches(sim),
                    )
                    code = self._compile_fix(code)
                    continue
                # Final round still non-conforming: do not accept TP — unresolved.
                final_reason_override = conformance_reason
                break

            # ── Memory-leak decisive verdict (AFTER the conformance gate) ────
            # The LLM's leak_chain settles a leak round the generic (crash-centric)
            # rules never reach — a leak whose object is freed/owned on well-formed
            # paths, or is abandoned only on corrupt-input, was previously stuck at
            # `fp_likelihood=tp` + `triggered=false` and refining forever.
            #
            # It is judged only on a round that reproduced the report's path: the leak
            # verdict is the ONE decisive layer that can conclude FP without exhausting
            # the path space, so it must clear the same bar as a `triggered` TP.  A POC
            # that does not follow the reported path (or whose step audit is unusable)
            # says nothing about the reported path — on read_index-484-1 the leak layer
            # returned "decisive FP" off POCs whose audit was 0/92 conforming, and off
            # another whose audit was unusable.  Do not move this back above the gate.
            if is_leak:
                leak_v = self._leak_decisive_conclusion(sim)
                leak_verdict = leak_v
                if leak_v == "tp":
                    reason = self._compose_leak_reason(sim.leak_chain, sim)
                    logger.info(
                        "Simulation round %d: leak_chain → object genuinely leaked on a "
                        "well-formed path → TP.", rnd,
                    )
                    return {
                        "triggered": True, "conclusion": "tp",
                        "rounds": rounds, "final_reason": reason,
                        "leak_verdict": "tp",
                    }
                if leak_v == "fp":
                    reason = self._compose_leak_reason(sim.leak_chain, sim)
                    logger.info(
                        "Simulation round %d: leak_chain decisive FP (stored/owned on "
                        "well-formed paths, or freed) → stopping (no refine).", rnd,
                    )
                    return {
                        "triggered": False, "conclusion": "fp",
                        "rounds": rounds, "final_reason": reason,
                        "leak_verdict": "fp",
                    }

            if sim.triggered:
                # Deterministic fabrication guard: a "trigger" that comes from a
                # POC fabricating a dependency state real code can never produce
                # is an artifact of the fabrication, NOT a reachable real-world
                # state.  Override to FP even though the LLM reported triggered.
                fabric = self._fabrication_evidence(code)
                if fabric:
                    logger.info(
                        "Simulation round %d: POC fabricates an impossible "
                        "dependency state → overriding TP to FP.", rnd,
                    )
                    return {
                        "triggered": False, "conclusion": "fp",
                        "rounds": rounds, "final_reason": fabric,
                        "leak_verdict": None,
                    }
                logger.info("Simulation round %d: bug TRIGGERED → TP.", rnd)
                return {
                    "triggered": True, "conclusion": "tp",
                    "rounds": rounds, "final_reason": None,
                    "leak_verdict": None,
                }

            # Fixable / uncertain non-trigger → refine the POC and continue.
            if rnd < max_rounds:
                logger.info(
                    "Simulation round %d: not triggered, not conclusive — "
                    "refining POC.", rnd,
                )
                code = self.refine_poc(
                    code, sim, completed, analysis,
                    selected_branches=selected_branches,
                )
                code = self._compile_fix(code)

        final_reason = final_reason_override or rounds[-1]["sim"].blocking_reason or (
            rounds[-1]["sim"].reason if rounds else None
        )
        logger.warning(
            "Simulation: neither triggered nor conclusive FP after %d round(s) "
            "→ UNVERIFIED.", max_rounds,
        )
        return {
            "triggered": False,
            "conclusion": "unresolved",
            "rounds": rounds,
            "final_reason": final_reason,
            "leak_verdict": leak_verdict,
        }
