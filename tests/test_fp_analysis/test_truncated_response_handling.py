"""Regression tests for the token-limit truncation failure mode.

On read_index-484 five of eight simulated-execution replies were cut off at the
then-default 4096-token budget (all ending mid-string at ~12.2k chars).  The
strict JSON parse failed, the regex fallback dropped ``path_conformance``
entirely, and the dataclass default ``path_conformance_ok=True`` let a truncated
``"triggered": true`` be accepted as a TP — the conformance gate was silently
disarmed.  These tests pin the three fixes:

1. ``LLMResponse.finish_reason`` survives the provider layer, and ``_call_llm``
   both warns on a truncated reply and accepts a per-call budget override.
2. The pipeline's default budget is no longer at the truncation point.
3. A reply that had to be repaired/unparsable marks the round ``parse_degraded``
   and can never certify a TP.
"""

import json
from unittest.mock import MagicMock, patch

from llm_client.base import LLMConfig, LLMRequest, LLMResponse
from llm_client.fp_analysis.path_analyzer import FPAnalyzer
from llm_client.fp_analysis.simulation_verifier import (
    SimulationResult,
    SimulationVerifier,
    _SIMULATION_MAX_TOKENS,
)
from tests.conftest import mock_openai_response


class _FakeProvider:
    """Records the requests it is sent and replays a canned reply."""

    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self._content = content
        self._finish_reason = finish_reason
        self.requests: list[LLMRequest] = []

    def send(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            content=self._content,
            model=request.model,
            provider="fake",
            finish_reason=self._finish_reason,
        )


def _bare_analyzer(provider, max_tokens: int) -> FPAnalyzer:
    """An FPAnalyzer with only the state ``_call_llm`` touches."""
    analyzer = FPAnalyzer.__new__(FPAnalyzer)
    analyzer._provider = provider
    analyzer._max_tokens = max_tokens
    analyzer._model = "deepseek-chat"
    analyzer._temperature = 0.0
    return analyzer


def _fake_analysis():
    """Minimal stand-in for the PathAnalysis the verifier prompts from."""
    md = MagicMock()
    md.bug_type = "Dereference of null pointer"
    md.bug_file = "index_read.cpp"
    md.bug_line = 484
    md.function_name = "read_index"
    pr = MagicMock()
    pr.metadata = md
    pr.path_events = [
        {"event_kind": "control", "description": "Taking false branch",
         "line_number": 507, "branch_condition": False},
        {"event_kind": "report", "description": "Null pointer value stored",
         "line_number": 484},
    ]
    pr.source_snippets = {}
    analysis = MagicMock()
    analysis.parsed_report = pr
    analysis.reduced_code = "int read_index(int f) { return 0; }"
    return analysis


class _StubAnalyzer:
    """FPAnalyzer surface used by ``SimulationVerifier``; parsing is the real one."""

    _project_name = ""
    _domain_facts_text = ""

    def __init__(self, raw: str) -> None:
        self._raw = raw
        self.budgets: list[int | None] = []

    # real static methods, so the tests exercise the production parsers
    _parse_json_response = staticmethod(FPAnalyzer._parse_json_response)
    _parse_json_response_tolerant = staticmethod(
        FPAnalyzer._parse_json_response_tolerant
    )
    # the refine loop extracts a C++ file out of the refined reply
    _extract_code = staticmethod(FPAnalyzer._extract_code)

    def _call_llm(self, system_prompt, user_prompt, max_tokens=None):
        self.budgets.append(max_tokens)
        return self._raw

    def _format_selected_branch_blueprint(self, selections):
        return ""


# ── 1. finish_reason reaches the caller ──────────────────────────────────────

class TestFinishReasonPropagation:
    @patch("llm_client.deepseek_provider.OpenAI")
    def test_provider_reports_truncation(self, MockOpenAI):
        from llm_client.factory import ProviderFactory

        resp = mock_openai_response(content='{"triggered": true, "reason": "cut', model="deepseek-chat")
        resp.choices[0].finish_reason = "length"
        MockOpenAI.return_value.chat.completions.create.return_value = resp

        provider = ProviderFactory.create(
            LLMConfig(provider_name="deepseek", api_key="k", default_model="deepseek-chat")
        )
        out = provider.send(LLMRequest(prompt="hi", max_tokens=16))

        assert out.finish_reason == "length"

    @patch("llm_client.deepseek_provider.OpenAI")
    def test_provider_reports_normal_stop(self, MockOpenAI):
        from llm_client.factory import ProviderFactory

        resp = mock_openai_response(content="ok", model="deepseek-chat")
        resp.choices[0].finish_reason = "stop"
        MockOpenAI.return_value.chat.completions.create.return_value = resp

        provider = ProviderFactory.create(
            LLMConfig(provider_name="deepseek", api_key="k", default_model="deepseek-chat")
        )
        assert provider.send(LLMRequest(prompt="hi")).finish_reason == "stop"


# ── 2. the budget: per-call override + a warning on truncation ───────────────

class TestCallLLMBudget:
    def test_default_budget_is_above_the_old_truncation_point(self):
        # 4096 cut the simulation replies at ~12.2k chars; the default must not
        # sit exactly there any more.
        assert FPAnalyzer.__init__.__defaults__ is not None
        import inspect

        sig = inspect.signature(FPAnalyzer.__init__)
        assert sig.parameters["max_tokens"].default >= 8192

    def test_per_call_override_wins(self):
        provider = _FakeProvider("ok")
        analyzer = _bare_analyzer(provider, max_tokens=8192)

        analyzer._call_llm("sys", "user", max_tokens=1234)

        assert provider.requests[-1].max_tokens == 1234

    def test_default_budget_used_without_override(self):
        provider = _FakeProvider("ok")
        analyzer = _bare_analyzer(provider, max_tokens=777)

        analyzer._call_llm("sys", "user")

        assert provider.requests[-1].max_tokens == 777

    def test_truncated_reply_is_warned_about(self, caplog):
        provider = _FakeProvider('{"triggered": true, "reason": "cut', finish_reason="length")
        analyzer = _bare_analyzer(provider, max_tokens=4096)

        with caplog.at_level("WARNING"):
            analyzer._call_llm("sys", "user")

        assert any("TRUNCATED" in r.message for r in caplog.records), caplog.text

    def test_complete_reply_is_not_warned_about(self, caplog):
        provider = _FakeProvider("{}", finish_reason="stop")
        analyzer = _bare_analyzer(provider, max_tokens=4096)

        with caplog.at_level("WARNING"):
            analyzer._call_llm("sys", "user")

        assert not any("TRUNCATED" in r.message for r in caplog.records)


# ── 3. a degraded round cannot certify a TP ─────────────────────────────────

_CUT_REPLY = (
    '{"triggered": true, "reason": "the POC produces the null deref",\n'
    ' "path_conformance": [{"step": 1, "satisfied": true},\n'
    '                      {"step": 2, "satisfied": true},\n'
    ' "path_audit": ["L507 false branch taken", "L484 deref of a value the report says is null'
)


def _simulate(raw: str):
    analyzer = _StubAnalyzer(raw)
    verifier = SimulationVerifier(analyzer)
    result = verifier.simulate_execution("int main(){}", None, _fake_analysis())
    return analyzer, result


class TestDegradedRound:
    def test_truncated_reply_marks_degraded_and_blocks_tp(self):
        analyzer, result = _simulate(_CUT_REPLY)

        # the regex fallback can still read a `triggered: true` out of the raw
        # text — which is exactly why the round must not be trusted
        assert result.triggered is True
        assert result.parse_degraded is True
        assert result.path_conformance_ok is False
        # the simulation gets the longest budget in the pipeline
        assert analyzer.budgets == [_SIMULATION_MAX_TOKENS]

    def test_complete_reply_is_not_degraded(self):
        raw = ('{"triggered": true, "reason": "fires", "path_conformance_ok": true,'
               ' "path_conformance": [{"step": 1, "satisfied": true}]}')
        _, result = _simulate(raw)

        assert result.parse_degraded is False
        assert result.path_conformance_ok is True

    def test_genuine_mismatch_is_not_degraded(self):
        raw = ('{"triggered": true, "reason": "fires",'
               ' "path_conformance": [{"step": 1, "satisfied": false}]}')
        _, result = _simulate(raw)

        assert result.parse_degraded is False
        assert result.path_conformance_ok is False

    def test_degraded_reason_is_explained_to_the_refiner(self):
        sim = SimulationResult(triggered=True, reason="cut", parse_degraded=True)
        text = SimulationVerifier._format_conformance_mismatches(sim)

        assert "truncated" in text or "unparsable" in text
        assert "(none reported)" not in text


class TestSparseStepAudit:
    """The step audit is reported sparsely (two counts + mismatch rows).

    Rationale (measured on read_index-484-1, 92 steps): the old per-step record
    list was 12,482 chars — 61% JSON scaffolding, 17% echoing back the step text
    the prompt had just supplied — i.e. 78% of the step audit, which was 78% of
    the reply, carrying no information in a conforming round.  The gate must be
    *stronger*, not weaker: the counts are the audit of record and every way of
    getting them wrong degrades the round to "no evidence" (never a TP).

    Two aggregates were tried and dropped, each time because a formatting slip
    in an otherwise decisive round cost the round: a 92-char BIT STRING (deepseek
    emitted 90 bits for 92 steps, flipping read_index-484-1 TP→UNKNOWN) and a
    satisfied COUNT (on read_index-484-2 the model wrote steps_satisfied=0 beside
    11 mismatch rows for 110 steps — it meant "the path as a whole is not
    reproduced", and the arithmetic check rejected the round). The mismatch LIST
    is the audit of record; everything unlisted is satisfied, which is also what
    makes a conforming round cost nothing.
    """

    def _result(self, data: dict):
        from llm_client.fp_analysis.simulation_verifier import SimulationResult

        # the real renderer: `_fake_analysis()`'s report has exactly 2 steps
        steps = SimulationVerifier(_StubAnalyzer("{}"))._csa_step_rows(_fake_analysis())
        assert len(steps) == 2
        return SimulationResult.from_dict(data, steps)

    def test_mismatch_list_expands_to_full_records(self):
        res = self._result({
            "triggered": True, "reason": "fires",
            "steps_checked": 2,
            "path_mismatches": [{"step": 2, "poc_result": "writes 4 bytes",
                                 "why": "report expects a null store"}],
        })

        assert [c["satisfied"] for c in res.path_conformance] == [True, False]
        # the report text is not echoed by the model — it is filled from the
        # numbered step list the prompt rendered, so the console/dump keep working
        assert res.path_conformance[0]["report_expectation"] == (
            "[control] [branch: FALSE] L507 Taking false branch"
        )
        assert res.path_conformance[1]["poc_result"] == "writes 4 bytes"
        assert res.path_conformance[1]["note"] == "report expects a null store"
        assert res.path_conformance_ok is False
        assert res.parse_degraded is False

    def test_empty_mismatch_list_is_conforming(self):
        res = self._result({"triggered": True, "steps_checked": 2,
                            "path_mismatches": [], "path_conformance_ok": True})

        assert res.path_conformance_ok is True
        assert res.parse_degraded is False
        assert all(c["satisfied"] for c in res.path_conformance)

    def test_mismatch_list_overrides_a_self_reported_ok(self):
        res = self._result({"triggered": True, "path_conformance_ok": True,
                            "steps_checked": 2,
                            "path_mismatches": [{"step": 1, "why": "differs"}]})

        assert res.path_conformance_ok is False

    def test_volunteered_satisfied_count_is_ignored(self):
        # read_index-484-2's model wrote steps_satisfied=0 beside 11 mismatch
        # rows for 110 steps; the list is the audit, so the round survives.
        res = self._result({"triggered": True, "path_conformance_ok": False,
                            "steps_checked": 2, "steps_satisfied": 0,
                            "path_mismatches": [{"step": 1, "why": "differs"}]})

        assert res.parse_degraded is False
        assert res.path_conformance_ok is False
        assert [c["satisfied"] for c in res.path_conformance] == [False, True]

    def test_partial_audit_is_degraded(self):
        # 2 steps, only 1 walked: the model audited part of the path.  That is
        # the failure mode the old missing-array default used to swallow.
        res = self._result({"triggered": True, "path_conformance_ok": True,
                            "steps_checked": 1, "path_mismatches": []})

        assert res.parse_degraded is True
        assert res.path_conformance_ok is False

    def test_not_ok_without_a_mismatch_row_is_degraded(self):
        # "the POC contradicts the report" with nothing to point at leaves the
        # regeneration prompt empty-handed.
        res = self._result({"triggered": True, "steps_checked": 2,
                            "path_mismatches": [], "path_conformance_ok": False})

        assert res.parse_degraded is True
        assert res.path_conformance_ok is False

    def test_out_of_range_mismatch_step_is_degraded(self):
        res = self._result({
            "triggered": True, "steps_checked": 2,
            "path_mismatches": [{"step": 9, "why": "?"}],
        })

        assert res.parse_degraded is True
        assert res.path_conformance_ok is False

    def test_no_step_evidence_at_all_is_degraded(self):
        # No bits, no explicit array — the reply answers the trigger question but
        # carries nothing about the path, so `path_conformance_ok` must not
        # default to True.
        res = self._result({"triggered": True, "reason": "fires"})

        assert res.parse_degraded is True
        assert res.path_conformance_ok is False

    def test_duplicate_mismatch_rows_are_degraded(self):
        res = self._result({
            "triggered": True, "steps_checked": 2,
            "path_mismatches": [{"step": 1, "why": "a"}, {"step": 1, "why": "b"}],
        })

        assert res.parse_degraded is True
        assert res.path_conformance_ok is False

    def test_legacy_explicit_array_still_wins(self):
        res = self._result({
            "triggered": True, "path_conformance_ok": True,
            "path_conformance": [{"step": 1, "satisfied": True, "poc_result": "x"},
                                 {"step": 2, "satisfied": True, "poc_result": "y"}],
        })

        assert res.path_conformance_ok is True
        assert res.parse_degraded is False
        assert len(res.path_conformance) == 2

    def test_partial_audit_round_is_not_accepted_as_tp(self):
        raw = json.dumps({"triggered": True, "reason": "fires",
                          "path_conformance_ok": True, "steps_checked": 1,
                          "path_mismatches": []})
        analyzer = _StubAnalyzer(raw)
        out = SimulationVerifier(analyzer).verify_with_feedback(
            "int main(){}", None, _fake_analysis(), max_rounds=2
        )

        assert out["conclusion"] == "unresolved"
        assert all(r["sim"].parse_degraded for r in out["rounds"])

    def test_sparse_conforming_round_is_tp(self):
        raw = json.dumps({"triggered": True, "reason": "fires",
                          "steps_checked": 2, "path_mismatches": [],
                          "path_conformance_ok": True})
        out = SimulationVerifier(_StubAnalyzer(raw)).verify_with_feedback(
            "int main(){}", None, _fake_analysis(), max_rounds=2
        )

        assert out["conclusion"] == "tp"


class TestDegradedRoundNeverAcceptedAsTP:
    """``verify_with_feedback`` must not return ``tp`` for a degraded round."""

    def _verifier(self):
        verifier = SimulationVerifier(_StubAnalyzer("{}"))
        verifier.simulate_execution = lambda *a, **k: SimulationResult(
            triggered=True, reason="truncated reply", parse_degraded=True,
            path_conformance_ok=False,
        )
        verifier.refine_poc = lambda code, sim, *a, **k: code
        verifier._compile_fix = lambda code: code
        return verifier

    def test_triggered_truncated_round_degrades_to_unresolved(self):
        out = self._verifier().verify_with_feedback(
            "int main(){}", None, _fake_analysis(), max_rounds=2
        )

        assert out["conclusion"] == "unresolved"
        assert out["triggered"] is False
        assert all(r["sim"].parse_degraded for r in out["rounds"])

    def test_conforming_triggered_round_still_is_tp(self):
        verifier = SimulationVerifier(_StubAnalyzer("{}"))
        verifier.simulate_execution = lambda *a, **k: SimulationResult(
            triggered=True, reason="fires", path_conformance_ok=True,
            path_conformance=[{"step": 1, "satisfied": True}],
        )

        out = verifier.verify_with_feedback(
            "int main(){}", None, _fake_analysis(), max_rounds=2
        )

        assert out["conclusion"] == "tp"
