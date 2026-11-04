"""POC attempt persistence and final-artifact consistency."""

from types import SimpleNamespace

from fp_analysis.simulation_verifier import SimulationResult
from pipeline.selection import _verify_path


class _Analyzer:
    def ensure_project_name(self, _name):
        return None

    def analyze(self, _report_path, _source_root):
        return object()

    def complete_constraints(self, _analysis, selected_branches=None):
        return SimpleNamespace(preconditions=[], concretized_values={})

    def generate_poc(self, _completed, _analysis, selected_branches=None):
        return "int initial_poc;"


class _Verifier:
    def __init__(self, _analyzer):
        pass

    def verify_with_feedback(self, *_args, **_kwargs):
        first = SimulationResult(triggered=False, reason="refine once",
                                 path_conformance_ok=True)
        final = SimulationResult(triggered=True, reason="final round",
                                 path_conformance_ok=True)
        return {
            "triggered": True,
            "conclusion": "tp",
            "leak_verdict": None,
            "final_reason": None,
            "rounds": [
                {"round": 1, "sim": first, "poc_code": "int round_one;"},
                {"round": 2, "sim": final, "poc_code": "int round_two;"},
            ],
        }


def test_every_simulation_round_and_final_poc_are_saved(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.selection.SimulationVerifier", _Verifier)
    out_path = tmp_path / "reports" / "path_selection_report.json"
    poc_dir = tmp_path / "pocs"

    result = _verify_path(
        _Analyzer(),
        report_path=tmp_path / "report.html",
        source_root=tmp_path,
        out_path=out_path,
        report_stem="report",
        poc_output_dir=poc_dir,
        attempt=2,
    )

    round_one = poc_dir / "poc_report_attempt_2_round_1.cpp"
    round_two = poc_dir / "poc_report_attempt_2_round_2.cpp"
    canonical = poc_dir / "poc_report.cpp"
    assert round_one.read_text() == "int round_one;"
    assert round_two.read_text() == "int round_two;"
    assert canonical.read_text() == "int round_two;"
    assert result["poc_code"] == "int round_two;"
    assert result["poc_path"] == canonical
    assert result["poc_result"]["attempt"] == 2
    assert result["poc_result"]["poc_path"] == str(canonical)
    rounds = result["poc_result"]["simulation"]["rounds"]
    assert [r["poc_path"] for r in rounds] == [str(round_one), str(round_two)]
