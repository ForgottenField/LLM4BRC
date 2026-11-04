"""Simulation gates for selected internal paths and trigger provenance."""

from types import SimpleNamespace

from fp_analysis.report_provenance import report_path_literal_trigger
from fp_analysis.simulation_verifier import SimulationResult, SimulationVerifier


def _analysis():
    metadata = SimpleNamespace(
        bug_type="Argument with 'nonnull' attribute passed null",
        bug_file="FollyMemcpy.cpp",
        bug_line=23,
        function_name="__folly_memcpy",
    )
    parsed = SimpleNamespace(
        metadata=metadata,
        path_events=[],
        source_snippets={},
    )
    return SimpleNamespace(parsed_report=parsed, source_root=None)


def _verifier_with_simulation(sim: SimulationResult) -> SimulationVerifier:
    analyzer = SimpleNamespace(
        project_name="folly",
        domain_facts_text="",
        lite_evidence="",
    )
    verifier = SimulationVerifier(analyzer)
    verifier.simulate_execution = lambda *args, **kwargs: sim
    verifier.refine_poc = lambda code, *args, **kwargs: code
    verifier._compile_fix = lambda code: code
    return verifier


def test_selected_internal_branch_mismatch_cannot_certify_tp():
    sim = SimulationResult(
        triggered=True,
        reason="CSA path conforms but an internal branch does not",
        path_conformance_ok=True,
        path_conformance=[{"step": 1, "satisfied": True}],
        selected_path_conformance_ok=False,
        selected_path_conformance=[{
            "branch": 1,
            "segment": 5,
            "line": 640,
            "condition": "nextBuf == buffer_",
            "expected_branch_taken": False,
            "satisfied": False,
            "poc_result": "TRUE",
        }],
    )

    result = _verifier_with_simulation(sim).verify_with_feedback(
        "int main(){}",
        None,
        _analysis(),
        max_rounds=1,
        selected_branches=[{"segment_index": 5, "branch_decisions": [{
            "kind": "control",
            "source_line": 640,
            "branch_taken": False,
        }]}],
    )

    assert result["conclusion"] == "unresolved"
    assert "selected internal feasible path" in result["final_reason"]


def test_report_path_literal_overrides_poc_argument_without_flipping_tp(
    monkeypatch,
):
    import fp_analysis.simulation_verifier as verifier_module

    monkeypatch.setattr(
        verifier_module,
        "report_path_literal_trigger",
        lambda *args, **kwargs: {
            "kind": "report_path_literal",
            "value": "null",
            "initialization_events": [{
                "event_number": 2,
                "line": 52,
                "description": "'dstNull' initialized to a null pointer value",
                "source": "char* dstNull = nullptr;",
            }],
            "passing_events": [3],
            "reason": "the report path contains the null initialization",
        },
    )
    sim = SimulationResult(
        triggered=True,
        reason="the report's own test passes null",
        path_conformance_ok=True,
        path_conformance=[{"step": 1, "satisfied": True}],
        bug_value_source="poc_argument",
    )

    result = _verifier_with_simulation(sim).verify_with_feedback(
        "int main(){}", None, _analysis(), max_rounds=1,
    )

    assert result["conclusion"] == "tp"
    assert sim.bug_value_source == "report_path_argument"
    assert sim.bug_value_provenance["initialization_events"][0]["event_number"] == 2


def test_poc_argument_absent_from_report_path_still_flips_to_fp(monkeypatch):
    import fp_analysis.simulation_verifier as verifier_module

    monkeypatch.setattr(
        verifier_module,
        "report_path_literal_trigger",
        lambda *args, **kwargs: None,
    )
    sim = SimulationResult(
        triggered=True,
        reason="the POC invented null",
        path_conformance_ok=True,
        path_conformance=[{"step": 1, "satisfied": True}],
        bug_value_source="poc_argument",
    )

    result = _verifier_with_simulation(sim).verify_with_feedback(
        "int main(){}", None, _analysis(), max_rounds=1,
    )

    assert result["conclusion"] == "fp"


def test_selected_path_sparse_audit_is_expanded_and_validated():
    steps = [{"text": "Taking false branch"}]
    selected = [
        {"segment": 5, "line": 722, "condition": "outer", "branch_taken": True},
        {"segment": 5, "line": 640, "condition": "inner", "branch_taken": False},
    ]
    result = SimulationResult.from_dict(
        {
            "triggered": True,
            "reason": "fires",
            "steps_checked": 1,
            "path_mismatches": [],
            "selected_branches_checked": 2,
            "selected_path_mismatches": [{
                "branch": 2,
                "poc_result": "TRUE",
                "why": "selected path requires FALSE",
            }],
            "selected_path_conformance_ok": True,
        },
        steps,
        selected,
    )

    assert result.selected_path_conformance_ok is False
    assert result.selected_path_conformance[1]["satisfied"] is False
    assert result.selected_path_conformance[1]["poc_result"] == "TRUE"


def test_partial_selected_path_audit_is_degraded():
    result = SimulationResult.from_dict(
        {
            "triggered": True,
            "reason": "fires",
            "steps_checked": 1,
            "path_mismatches": [],
            "selected_branches_checked": 1,
            "selected_path_mismatches": [],
            "selected_path_conformance_ok": True,
        },
        [{"text": "Taking false branch"}],
        [
            {"segment": 5, "line": 722, "condition": "outer", "branch_taken": True},
            {"segment": 5, "line": 640, "condition": "inner", "branch_taken": False},
        ],
    )

    assert result.selected_path_audit_degraded is True
    assert result.selected_path_conformance_ok is False


def test_missing_selected_path_mismatch_list_is_degraded():
    result = SimulationResult.from_dict(
        {
            "triggered": True,
            "reason": "fires",
            "steps_checked": 1,
            "path_mismatches": [],
            "selected_branches_checked": 1,
            "selected_path_conformance_ok": True,
        },
        [{"text": "Taking false branch"}],
        [{"segment": 5, "line": 640, "condition": "inner", "branch_taken": False}],
    )

    assert result.selected_path_audit_degraded is True
    assert result.selected_path_conformance_ok is False


def test_report_path_literal_trigger_is_read_from_the_owning_event_file(tmp_path):
    source = tmp_path / "MemcpyTest.cpp"
    source.write_text(
        "void test() {\n"
        "  char* dstNull = nullptr;\n"
        "  __folly_memcpy(dstNull, nullptr, 0);\n"
        "}\n",
        encoding="utf-8",
    )
    parsed = SimpleNamespace(
        metadata=SimpleNamespace(
            bug_type="Argument with 'nonnull' attribute passed null",
            bug_file=str(tmp_path / "FollyMemcpy.cpp"),
        ),
        path_events=[
            {
                "path_number": 2,
                "description": "'dstNull' initialized to a null pointer value",
                "line_number": 2,
                "file_path": str(source),
            },
            {
                "path_number": 3,
                "description": "Passing null pointer value via 1st parameter",
                "line_number": 3,
                "file_path": str(source),
            },
        ],
        source_snippets={},
    )

    evidence = report_path_literal_trigger(parsed, str(tmp_path))

    assert evidence["kind"] == "report_path_literal"
    assert evidence["initialization_events"][0]["event_number"] == 2
    assert evidence["passing_events"] == [3]
