"""Leak-layer evidence gating (task A) and its TP reason (task C).

The leak layer (``leak_chain``) is the one decisive layer that can conclude FP
*without* exhausting the path space, which is why it is the layer whose evidence
has to clear the highest bar:

1. a ``leak_chain`` that contradicts itself is never decisive — on
   read_index-484-1 the same code came back ``leak_fires_on_wellformed=true`` in
   one run and ``..._stores_to=owner`` + ``requires_broken_input=true`` in
   another, so the report's verdict was decided by which attempt concluded first;
2. the leak verdict is judged only on a round that reproduced the report's path
   (it used to be checked *before* the path-conformance gate: on 484-1 it
   returned "decisive FP" off POCs whose step audit was 0/92 conforming, and off
   another whose audit was unusable);
3. a leak TP carries a grounded reason (it used to return ``final_reason=None``).
"""

import json
from unittest.mock import MagicMock

from llm_client.fp_analysis.path_analyzer import FPAnalyzer
from llm_client.fp_analysis.simulation_verifier import (
    SimulationResult,
    SimulationVerifier,
    _normalize_leak_chain,
)

# 484-2 shape: stored into the return on every well-formed continuation; the only
# abandon is a mid-stream throw needing corrupt/truncated input → FP.
_LEAK_FP_CHAIN = {
    "alloc_line": 909,
    "alloc_expr": "new IndexHNSWSQ()",
    "wellformed_continuation_stores_to": "return",
    "requires_broken_input": True,
    "leak_fires_on_wellformed": False,
    "abandon_path": "exception_after_alloc",
}


def _leak_analysis():
    """A 2-step memory-leak report; the step list comes from the real renderer."""
    md = MagicMock()
    md.bug_type = "Memory leak"
    md.bug_file = "index_read.cpp"
    md.bug_line = 987
    md.function_name = "read_index"
    pr = MagicMock()
    pr.metadata = md
    pr.path_events = [
        {"event_kind": "control", "description": "Taking false branch",
         "line_number": 916, "branch_condition": False},
        {"event_kind": "report",
         "description": "Potential leak of memory pointed to by 'idxhnsw'",
         "line_number": 987},
    ]
    pr.source_snippets = {}
    analysis = MagicMock()
    analysis.parsed_report = pr
    analysis.reduced_code = "Index* read_index(IOReader* f) { return nullptr; }"
    return analysis


class _StubAnalyzer:
    """``FPAnalyzer`` surface used by ``SimulationVerifier``, real parsers."""

    _project_name = ""
    _domain_facts_text = ""

    def __init__(self, raw: str) -> None:
        self._raw = raw
        self.budgets: list[int | None] = []

    _parse_json_response = staticmethod(FPAnalyzer._parse_json_response)
    _parse_json_response_tolerant = staticmethod(
        FPAnalyzer._parse_json_response_tolerant
    )
    _extract_code = staticmethod(FPAnalyzer._extract_code)

    def _call_llm(self, system_prompt, user_prompt, max_tokens=None):
        self.budgets.append(max_tokens)
        return self._raw

    def _format_selected_branch_blueprint(self, selections):
        return ""


def _reply(conform: bool, chain: dict | None = None, triggered: bool = False) -> str:
    """A simulation reply whose step audit is conforming or not (2 steps rendered)."""
    return json.dumps({
        "triggered": triggered,
        "reason": "read_index_header and read_HNSW succeed, then the final else throws",
        "fp_likelihood": "tp",
        "steps_checked": 2,
        "path_conformance_ok": conform,
        "path_mismatches": (
            [] if conform else [{"step": 1, "poc_result": "took the IHNp branch",
                                 "why": "the report takes the IHNs branch"}]
        ),
        "leak_chain": dict(chain or _LEAK_FP_CHAIN),
    })


def _verify(raw: str, max_rounds: int = 2) -> dict:
    verifier = SimulationVerifier(_StubAnalyzer(raw))
    return verifier.verify_with_feedback(
        "int main(){}", None, _leak_analysis(), max_rounds=max_rounds
    )


# ── A2: a self-contradictory chain is never decisive ─────────────────────────

class TestLeakChainContradiction:
    def test_wellformed_leak_needing_broken_input_is_not_decisive(self):
        lc = _normalize_leak_chain({**_LEAK_FP_CHAIN, "leak_fires_on_wellformed": True})

        # the normalized chain really does carry the contradiction
        assert lc["leak_fires_on_wellformed"] is True
        assert lc["requires_broken_input"] is True
        assert SimulationVerifier(None)._leak_decisive_conclusion(
            SimulationResult(triggered=False, reason="sim", leak_chain=lc)
        ) is None

    def test_consistent_wellformed_leak_still_decides_tp(self):
        lc = _normalize_leak_chain({
            "leak_fires_on_wellformed": True,
            "wellformed_continuation_stores_to": None,
            "requires_broken_input": False,
        })

        assert SimulationVerifier(None)._leak_decisive_conclusion(
            SimulationResult(triggered=False, reason="sim", leak_chain=lc)
        ) == "tp"


# ── A1: the leak verdict needs a round that reproduced the report's path ─────

class TestLeakVerdictNeedsAConformingRound:
    def test_non_conforming_round_is_not_a_leak_fp(self):
        out = _verify(_reply(conform=False))

        # before the gate was moved ahead of the leak block this was "fp" — decided
        # by a POC that took a different branch than the report at step 1
        assert out["conclusion"] == "unresolved"
        assert out["leak_verdict"] is None

    def test_conforming_round_is_a_leak_fp(self):
        out = _verify(_reply(conform=True))

        assert out["conclusion"] == "fp"
        assert out["leak_verdict"] == "fp"
        assert "Memory-leak 判 FP" in out["final_reason"]

    def test_round_without_step_evidence_is_not_a_leak_fp(self):
        raw = json.dumps({"triggered": False, "reason": "sim",
                          "leak_chain": dict(_LEAK_FP_CHAIN)})

        out = _verify(raw)

        assert out["conclusion"] == "unresolved"
        assert out["leak_verdict"] is None

    def test_triggered_round_with_an_undecided_chain_is_a_plain_tp(self):
        # an UNDECIDED chain (abandoned, but neither via broken input nor stored to
        # an owner) must not interfere with execution evidence: the trigger stands.
        out = _verify(_reply(conform=True, triggered=True, chain={
            "abandon_path": "manual_no_free_early_return",
            "requires_broken_input": False,
            "wellformed_continuation_stores_to": None,
        }))

        assert out["conclusion"] == "tp"
        # execution evidence, not a leak judgment: the report-level loop must not
        # treat it as a leak verdict that needs cross-path agreement
        assert out["leak_verdict"] is None

    def test_triggered_round_with_a_leak_fp_chain_contradicts_itself(self):
        # `triggered=true` beside a chain whose fields say the object is NOT leaked
        # on well-formed paths is self-contradictory; like `definite_fp`, the
        # code-level leak evidence wins over the trigger it contradicts.
        out = _verify(_reply(conform=True, triggered=True))

        assert out["conclusion"] == "fp"
        assert out["leak_verdict"] == "fp"


# ── C: a leak TP carries a grounded reason ──────────────────────────────────

class TestLeakTpReason:
    def test_conforming_wellformed_leak_is_a_tp_with_a_reason(self):
        chain = {"leak_fires_on_wellformed": True, "requires_broken_input": False,
                 "alloc_expr": "new IndexHNSWSQ()"}

        out = _verify(_reply(conform=True, chain=chain))

        assert out["conclusion"] == "tp"
        assert out["leak_verdict"] == "tp"
        assert out["final_reason"]
        assert "判 TP" in out["final_reason"]
        assert "new IndexHNSWSQ()" in out["final_reason"]
