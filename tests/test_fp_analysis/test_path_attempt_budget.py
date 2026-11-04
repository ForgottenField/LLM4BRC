"""Path-attempt budget tests for bounded exploration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pipeline import exploration
from pipeline.models import ExplorationState, VerdictState
from pipeline.outcomes import Verdict
from pipeline.runner import _path_exploration_budget


def _context(loop_attempts: int, full_space_explore: bool) -> SimpleNamespace:
    return SimpleNamespace(
        args=SimpleNamespace(poc_output_dir=None),
        analyzer=object(),
        parsed_report=SimpleNamespace(
            metadata=SimpleNamespace(
                bug_file="sample.cpp",
                path_evidence="trace",
            ),
        ),
        loaded=object(),
        segments=[],
        slice_mask=object(),
        source_root=Path("."),
        cfg_cache={},
        extra_preconditions=[],
        hard_constraints=[],
        sdg=object(),
        reduced_ps={"whole_path_bound": 24},
        reduced_bound_one=False,
        slice_relevance={"node_relevance": {}},
        slice_truncated=False,
        combo_iter=iter(
            ({"id": index}, None)
            for index in range(1, loop_attempts)
        ),
        loop_attempts=loop_attempts,
        full_space_explore=full_space_explore,
        out_path=Path("result.json"),
        report_stem="report-x",
        report_ref=object(),
    )


def test_reduced_space_above_the_cap_runs_ten_attempts_and_is_not_exhausted() -> None:
    assert _path_exploration_budget(50, 24) == (10, False)


def test_reduced_space_inside_the_cap_is_fully_exhausted() -> None:
    assert _path_exploration_budget(10, 8) == (8, True)
    assert _path_exploration_budget(4, 8) == (4, False)


def test_single_combination_is_left_to_generalization_not_sampling() -> None:
    assert _path_exploration_budget(10, 1) == (1, False)


def test_ten_all_fp_attempts_of_a_capped_sample_return_unknown(monkeypatch) -> None:
    attempts: list[int] = []

    monkeypatch.setattr(
        exploration, "evaluate_control_direction_conflicts",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        exploration, "run_path_selection",
        lambda *args, **kwargs: ([], {"elapsed_sec": 0.0}),
    )
    monkeypatch.setattr(
        exploration, "selected_path_violations",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(exploration, "_no_path_verdict", lambda *args: None)
    monkeypatch.setattr(
        exploration, "path_signature",
        lambda selections: f"path-{len(attempts)}",
    )
    monkeypatch.setattr(
        exploration, "combo_signature",
        lambda reduced_ps, combo: f"combo-{combo['id']}",
    )

    def verify(*args, **kwargs):
        attempts.append(kwargs["attempt"])
        return {
            "poc_result": {"simulation": {"rounds": []}},
            "poc_code": None,
            "poc_path": None,
            "conclusion": "fp",
            "final_reason": "this path cannot trigger the reported bug",
            "leak_verdict": None,
            "sim_res": None,
        }

    monkeypatch.setattr(exploration, "_verify_path", verify)

    verdict_state = VerdictState()
    explore_state = ExplorationState()
    context = _context(loop_attempts=10, full_space_explore=False)

    exploration._run_path_attempts(context, verdict_state, explore_state)
    exploration._aggregate_exploration(context, verdict_state, explore_state)

    assert attempts == list(range(1, 11))
    assert explore_state.attempts_run == 10
    assert len(explore_state.fp_observations) == 10
    assert verdict_state.verdict is Verdict.UNKNOWN
    assert "路径空间过大未穷举" in verdict_state.final_reason
