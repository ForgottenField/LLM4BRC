"""Tests for State Lifecycle Analysis (Step 2.7).

Tests cover:
- Data models: CallChainEntry, LifecycleAssessment
- Static analysis: CallChainExtractor.extract()
- Static analysis: CallChainExtractor.identify_state_variables()
- Static analysis: CallChainExtractor.detect_known_patterns()
- Prompt rendering
"""

from __future__ import annotations

from typing import Optional

import pytest

from llm_client.fp_analysis.models import CallChainEntry, LifecycleAssessment

# =========================================================================
# Model Tests
# =========================================================================


class TestCallChainEntry:
    """CallChainEntry — a single function call entry from CSA path."""

    def test_create_minimal(self):
        """A CallChainEntry can be created with only required fields."""
        entry = CallChainEntry(
            event_index=5,
            function_name="wslay_event_recv",
            is_entry=True,
        )
        assert entry.event_index == 5
        assert entry.function_name == "wslay_event_recv"
        assert entry.is_entry is True
        assert entry.callee is None
        assert entry.source_line == 0
        assert entry.state_params == []

    def test_create_all_fields(self):
        """A CallChainEntry can be created with all fields."""
        entry = CallChainEntry(
            event_index=35,
            function_name="wslay_event_recv",
            callee="decode",
            source_line=650,
            is_entry=True,
            state_params=["ctx->imsg->utf8state"],
        )
        assert entry.event_index == 35
        assert entry.function_name == "wslay_event_recv"
        assert entry.callee == "decode"
        assert entry.source_line == 650
        assert entry.is_entry is True
        assert entry.state_params == ["ctx->imsg->utf8state"]

    def test_return_entry(self):
        """A returning-from entry has is_entry=False."""
        entry = CallChainEntry(
            event_index=36,
            function_name="wslay_event_recv",
            callee="decode",
            is_entry=False,
        )
        assert entry.is_entry is False


class TestLifecycleAssessment:
    """LifecycleAssessment — result of Step 2.7 analysis."""

    def test_create_minimal(self):
        """Default values are safe (conservative: no lifecycle init)."""
        assessment = LifecycleAssessment()
        assert assessment.call_chain == []
        assert assessment.state_variables == []
        assert assessment.reset_functions == []
        assert assessment.non_reset_functions == []
        assert assessment.error_start_initialized_by_lifecycle is False
        assert assessment.lifecycle_type is None
        assert assessment.explanation == ""

    def test_create_all_fields(self):
        """A LifecycleAssessment can be created with all fields."""
        assessment = LifecycleAssessment(
            call_chain=[
                CallChainEntry(
                    event_index=35,
                    function_name="wslay_event_recv",
                    callee="decode",
                    source_line=650,
                    is_entry=True,
                    state_params=["ctx->imsg->utf8state"],
                ),
            ],
            state_variables=["utf8state", "ctx->imsg->utf8state"],
            reset_functions=["wslay_event_imsg_reset"],
            non_reset_functions=["wslay_event_imsg_set"],
            error_start_initialized_by_lifecycle=True,
            lifecycle_type="state_reset",
            explanation=(
                "utf8state is set to UTF8_ACCEPT by imsg_reset before each message. "
                "imsg_set does not touch utf8state, so on decode's first call "
                "*state == UTF8_ACCEPT -> FALSE branch -> codep IS initialized."
            ),
        )
        assert assessment.error_start_initialized_by_lifecycle is True
        assert assessment.lifecycle_type == "state_reset"
        assert len(assessment.state_variables) == 2
        assert len(assessment.call_chain) == 1
        assert len(assessment.reset_functions) == 1
        assert len(assessment.non_reset_functions) == 1

    def test_lifecycle_type_enum_values(self):
        """lifecycle_type accepts all expected values."""
        for lt in ("state_reset", "loop_carried_init", "union_type_punning", None):
            assessment = LifecycleAssessment(lifecycle_type=lt)
            assert assessment.lifecycle_type == lt

    def test_conservative_default(self):
        """Default should not claim initialization (safe for FP detection)."""
        assessment = LifecycleAssessment()
        assert assessment.error_start_initialized_by_lifecycle is False
        assert assessment.explanation == ""


# =========================================================================
# Static Analysis Tests — CallChainExtractor
# =========================================================================


# ── Fixtures: sample path_events ──


@pytest.fixture
def empty_events() -> list[dict]:
    """No path events."""
    return []


@pytest.fixture
def no_call_events() -> list[dict]:
    """Path events with only branch/assignment events, no function calls."""
    return [
        {
            "event_number": 1,
            "event_kind": "control",
            "description": "Taking false branch",
            "line_number": 50,
            "function_name": "test_func",
        },
        {
            "event_number": 2,
            "event_kind": "event",
            "description": "int x = 0",
            "line_number": 51,
            "function_name": "test_func",
        },
    ]


@pytest.fixture
def single_call_events() -> list[dict]:
    """Simple path with one Calling → Returning pair."""
    return [
        {
            "event_number": 10,
            "event_kind": "Calling",
            "description": "Calling decode",
            "line_number": 650,
            "function_name": "wslay_event_recv",
        },
        {
            "event_number": 11,
            "event_kind": "Return",
            "description": "← Returning from decode",
            "line_number": 651,
            "function_name": "decode",
        },
    ]


@pytest.fixture
def nested_call_events() -> list[dict]:
    """Nested call: outer → inner → return inner → return outer."""
    return [
        {
            "event_number": 5,
            "event_kind": "Calling",
            "description": "Calling wslay_event_imsg_reset",
            "line_number": 452,
            "function_name": "wslay_event_recv",
        },
        {
            "event_number": 6,
            "event_kind": "Return",
            "description": "← Returning from wslay_event_imsg_reset",
            "line_number": 453,
            "function_name": "wslay_event_imsg_reset",
        },
        {
            "event_number": 10,
            "event_kind": "Calling",
            "description": "Calling decode",
            "line_number": 650,
            "function_name": "wslay_event_recv",
        },
        {
            "event_number": 11,
            "event_kind": "Return",
            "description": "← Returning from decode",
            "line_number": 651,
            "function_name": "decode",
        },
    ]


@pytest.fixture
def wslay_decode_events() -> list[dict]:
    """Full decode path with context init, imsg_set, and decode call events."""
    return [
        {
            "event_number": 1,
            "event_kind": "Calling",
            "description": "Calling wslay_event_imsg_reset",
            "line_number": 452,
            "function_name": "wslay_event_recv",
        },
        {
            "event_number": 2,
            "event_kind": "Return",
            "description": "← Returning from wslay_event_imsg_reset",
            "line_number": 453,
            "function_name": "wslay_event_imsg_reset",
        },
        {
            "event_number": 30,
            "event_kind": "Calling",
            "description": "Calling decode",
            "line_number": 650,
            "function_name": "wslay_event_recv",
        },
        {
            "event_number": 31,
            "event_kind": "Return",
            "description": "← Returning from decode",
            "line_number": 651,
            "function_name": "decode",
        },
    ]


# ── extract() tests ──


class TestCallChainExtraction:
    """CallChainExtractor.extract() — parsing CSA path events."""

    def test_empty_events(self, empty_events):
        """Empty events list produces empty call chain."""
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        result = CallChainExtractor.extract(empty_events)
        assert result == []

    def test_no_call_events(self, no_call_events):
        """No call events produces empty call chain."""
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        result = CallChainExtractor.extract(no_call_events)
        assert result == []

    def test_single_call(self, single_call_events):
        """Single Calling → Returning pair is extracted."""
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        result = CallChainExtractor.extract(single_call_events)
        assert len(result) == 2
        assert result[0].function_name == "wslay_event_recv"
        assert result[0].callee == "decode"
        assert result[0].is_entry is True
        assert result[0].source_line == 650
        assert result[0].event_index == 10
        assert result[1].is_entry is False
        assert result[1].callee == "decode"

    def test_nested_calls(self, nested_call_events):
        """Multiple sequential calls are extracted in order."""
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        result = CallChainExtractor.extract(nested_call_events)
        assert len(result) == 4
        # Order: imsg_reset entry, imsg_reset return, decode entry, decode return
        assert result[0].callee == "wslay_event_imsg_reset"
        assert result[0].is_entry is True
        assert result[2].callee == "decode"
        assert result[2].is_entry is True
        assert result[3].callee == "decode"
        assert result[3].is_entry is False

    def test_wslay_decode_path(self, wslay_decode_events):
        """The full wslay decode path produces correct call chain."""
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        result = CallChainExtractor.extract(wslay_decode_events)
        assert len(result) == 4
        assert result[0].callee == "wslay_event_imsg_reset"
        assert result[2].callee == "decode"

    def test_no_callee_when_not_a_call(self):
        """Events without 'Calling' in their kind produce no entries."""
        events = [
            {
                "event_number": 1,
                "event_kind": "Calling",
                "description": "Calling foo",
                "line_number": 10,
                "function_name": "main",
            },
        ]
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        result = CallChainExtractor.extract(events)
        assert len(result) == 1
        assert result[0].callee == "foo"


# ── detect_known_patterns() tests ──


class TestDetectKnownPatterns:
    """CallChainExtractor.detect_known_patterns() — static analysis reuse."""

    def test_loop_carried_init_pattern(self):
        """detect_known_patterns detects loop-carried initialization."""
        code = """
        void func() {
            for (int i = 0; i < 10; i++) {
                uint32_t codep;
                decode(&state, &codep, byte);
            }
        }
        """
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        patterns = CallChainExtractor.detect_known_patterns(code)
        assert "loop_carried_init" in patterns

    def test_union_type_punning_pattern(self):
        """detect_known_patterns detects union type-punning."""
        code = """
        union { struct sockaddr_in in; struct sockaddr_storage storage; } ad;
        memcpy(&ad.storage, ifa->ifa_addr, addrlen);
        if (ad.in.sin_addr.s_addr != htonl(INADDR_LOOPBACK)) {}
        """
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        patterns = CallChainExtractor.detect_known_patterns(code)
        assert "union_type_punning" in patterns

    def test_no_patterns(self):
        """Clean code with no known patterns returns empty list."""
        code = """
        int add(int a, int b) {
            return a + b;
        }
        """
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        patterns = CallChainExtractor.detect_known_patterns(code)
        assert patterns == []

    def test_empty_code(self):
        """Empty code returns empty list."""
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        assert CallChainExtractor.detect_known_patterns("") == []
        assert CallChainExtractor.detect_known_patterns("   ") == []


# ── identify_state_variables() tests ──


class TestIdentifyStateVariables:
    """CallChainExtractor.identify_state_variables() — PDG-based analysis."""

    def test_no_sdg_returns_empty(self):
        """When SDG is None, returns empty list (no PDG info available)."""
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        result = CallChainExtractor.identify_state_variables(
            call_chain=[],
            error_start_var="codep",
            sdg=None,
        )
        assert result == []

    def test_empty_call_chain_returns_empty(self):
        """Empty call chain returns empty state variables list."""
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor

        result = CallChainExtractor.identify_state_variables(
            call_chain=[],
            error_start_var="codep",
            sdg="not_none_but_no_chain",
        )
        assert result == []


# =========================================================================
# Prompt Rendering Tests
# =========================================================================


class TestStateLifecyclePrompt:
    """STATE_LIFECYCLE_PROMPT template renders correctly."""

    def test_prompt_imports(self):
        """STATE_LIFECYCLE_SYSTEM_PROMPT and STATE_LIFECYCLE_PROMPT are importable."""
        from llm_client.fp_analysis.prompt_templates import (
            STATE_LIFECYCLE_PROMPT,
            STATE_LIFECYCLE_SYSTEM_PROMPT,
        )

        assert STATE_LIFECYCLE_SYSTEM_PROMPT is not None
        assert STATE_LIFECYCLE_PROMPT is not None

    def test_prompt_renders_with_minimal_args(self):
        """STATE_LIFECYCLE_PROMPT renders without error with all required fields."""
        from llm_client.fp_analysis.prompt_templates import STATE_LIFECYCLE_PROMPT

        rendered = STATE_LIFECYCLE_PROMPT.format(
            bug_type="Result of operation is garbage or undefined",
            bug_location="wslay_event.c",
            bug_line=93,
            bug_function="decode",
            error_start_var="codep",
            call_sequence="1: Calling decode (L650)",
            source_context="// simplified source",
            known_patterns="loop_carried_init",
            function_behavior_table="| Function | State params | Writes state? |",
        )
        assert "codep" in rendered
        assert "wslay_event.c" in rendered
        assert "Calling decode" in rendered
        assert "loop_carried_init" in rendered
        assert "Step 3" in rendered


# =========================================================================
# Integration Test: Step 2.7 End-to-End
# =========================================================================
# Requires PDG data (pdg_aria2.json) and mocks the LLM call to avoid
# API dependency.  Tests the glue logic: when Phase 2.5 returns
# error_start_is_real=True but Step 2.7 detects lifecycle initialization,
# does the pipeline correctly reclassify as FP?

import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest

# Mark this module-level: skip if PDG file not present
_HAS_PDG = Path(__file__).resolve().parent.parent.parent / "pdg_aria2.json"
_HAS_PDG = _HAS_PDG.exists()

pytestmark = pytest.mark.skipif(
    not _HAS_PDG,
    reason="pdg_aria2.json not found — requires PDG build artifacts",
)


def _make_constraint_completion_json() -> str:
    """Return a realistic constraint-completion LLM response with
    ``error_start_is_real=True`` (so Phase 2.5 does NOT filter it out,
    forcing the path to Step 2.7)."""
    return json.dumps({
        "original_path_summary": "Path: ...",
        "final_path_summary": "With constraints applied: ...",
        "preconditions": [],
        "concretized_values": {},
        "error_start_is_real": True,
        "completed_steps": [
            {
                "original_event_index": None,
                "new_event_kind": "event",
                "description": "Test step",
                "file_path": "test.c",
                "line_number": 1,
            },
        ],
    })


def _make_lifecycle_json() -> str:
    """Return a lifecycle-analysis LLM response that correctly identifies
    the state-reset pattern (``error_start_initialized_by_lifecycle=True``)."""
    return json.dumps({
        "state_variables": ["utf8state", "ctx->imsg->utf8state"],
        "reset_functions": ["wslay_event_imsg_reset"],
        "non_reset_functions": ["wslay_event_imsg_set"],
        "error_start_initialized_by_lifecycle": True,
        "lifecycle_type": "state_reset",
        "explanation": (
            "imsg_reset sets utf8state=UTF8_ACCEPT before each message. "
            "Since decode's first call has *state==UTF8_ACCEPT, the FALSE "
            "branch initializes *codep."
        ),
        "calling_sequence_summary": "imsg_reset -> imsg_set -> decode",
    })


class TestStep27Integration:
    """Full integration: does Step 2.7 correctly catch FP when
    Phase 2.5 misses it (error_start_is_real=True scenario)?"""

    @pytest.fixture(scope="class")
    def analyzer_with_pdg(self):
        """Create an FPAnalyzer with real PDG loaded (no LLM config needed
        — we mock all LLM calls)."""
        from llm_client.fp_analysis import FPAnalyzer

        pdg_path = str(
            Path(__file__).resolve().parent.parent.parent / "pdg_aria2.json"
        )
        return FPAnalyzer(
            llm_config=None,
            source_root="project/aria2",
            pdg_path=pdg_path,
            max_tokens=16384,
        )

    @pytest.fixture
    def prepared_analysis(self, analyzer_with_pdg):
        """Parse the wslay decode report and run all pre-LLM steps
        (HTML parsing + PDG slicing).  No LLM calls yet."""
        rp = Path(
            __file__).resolve().parent.parent.parent / "project" / "aria2" \
            / "reports" / "FP" / "report-wslay_event.c-decode-4-1.html"
        if not rp.exists():
            pytest.skip(f"Report not found: {rp}")

        analyzer = analyzer_with_pdg
        analysis = analyzer.analyze(str(rp), source_root="project/aria2")
        slice_result = analyzer._run_pdg_slicing(analysis)
        if slice_result:
            from llm_client.fp_analysis.code_reducer import reduce_code
            analysis.reduced_code = reduce_code(
                slice_result, source_root="project/aria2",
            )
        # Build slice mask
        if slice_result:
            from llm_client.fp_analysis.slice_mask import SliceMask
            analysis.slice_mask = SliceMask.from_slice_result(
                slice_result, analyzer._sdg,
            ).to_dict()
        return analysis, analyzer

    def test_step27_catches_fp_when_phase25_misses(
        self, prepared_analysis,
    ):
        """When Phase 2.5 returns error_start_is_real=True (missed the FP
        pattern) but Step 2.7 detects the lifecycle, the pipeline should
        still classify as FP and record the state_lifecycle step."""
        analysis, analyzer = prepared_analysis

        # Mock _call_llm to return controlled responses:
        # Call 1 (constraint completion): error_start_is_real=True
        # Call 2 (state lifecycle): detects lifecycle → override to FP
        responses = [
            _make_constraint_completion_json(),
            _make_lifecycle_json(),
        ]
        response_iter = iter(responses)

        def mock_call_llm(system_prompt, user_prompt):
            return next(response_iter)

        with patch.object(analyzer, "_call_llm", side_effect=mock_call_llm):
            result, completed = analyzer.complete_and_verify(
                analysis, max_iterations=1,
            )

        # ── Assertions ──
        # 1. Pipeline correctly classifies as FP
        from llm_client.fp_analysis.models import Classification
        assert result.classification == Classification.FALSE_POSITIVE, (
            f"Expected FP, got {result.classification}"
        )

        # 2. Verification history contains state_lifecycle step
        vh = result.verification_history
        lifecycle_entries = [
            e for e in vh if e.get("step") == "state_lifecycle"
        ]
        assert len(lifecycle_entries) == 1, (
            f"Expected 1 state_lifecycle entry, got {len(lifecycle_entries)}: "
            f"{[e.get('step') for e in vh]}"
        )

        # 3. Lifecycle details are recorded
        entry = lifecycle_entries[0]
        assert entry.get("is_feasible") is False
        assert entry.get("lifecycle_type") == "state_reset"
        assert "imsg_reset" in str(entry.get("reset_functions", []))
        assert "utf8state" in str(entry.get("state_variables", []))

        # 4. No further steps ran (Step 3 should be skipped)
        # Note: when path is classified as FP, complete_and_verify
        # returns (AnalysisResult, None) — completed is None by design.
        step3_entries = [
            e for e in vh if e.get("step") == "single_path_feasibility"
        ]
        assert len(step3_entries) == 0, (
            "Step 3 should not run when Step 2.7 overrides to FP"
        )

    def test_step27_skipped_when_no_state_vars(self, prepared_analysis):
        """When static analysis finds no state variables, _analyze_state_lifecycle
        skips the LLM call and returns a conservative default."""
        analysis, analyzer = prepared_analysis

        # Patch identify_state_variables to return empty
        from llm_client.fp_analysis.state_lifecycle import CallChainExtractor
        from llm_client.fp_analysis.models import CompletedPath, PathAnalysis, Confidence

        with patch.object(CallChainExtractor, "identify_state_variables",
                          return_value=[]):
            dummy_completed = CompletedPath(
                original_path_summary="test",
                analysis=PathAnalysis(
                    path_id="test", is_likely_fp=False, gaps=[],
                    confidence=Confidence.HIGH, reasoning="",
                ),
                completed_steps=[],
                final_path_summary="test",
                preconditions=[],
                concretized_values={},
                error_start_is_real=True,
            )

            # Patch _call_llm to prove it is NOT called
            with patch.object(analyzer, "_call_llm") as mock_llm:
                lifecycle = analyzer._analyze_state_lifecycle(
                    analysis, dummy_completed,
                )

        assert lifecycle.error_start_initialized_by_lifecycle is False, (
            "Should return conservative default when no state vars"
        )
        assert lifecycle.state_variables == [], (
            f"Should have empty state_vars, got {lifecycle.state_variables}"
        )
        mock_llm.assert_not_called()
