"""``process_report`` end to end, with every stage boundary stubbed.

This is the characterization harness for the one function that owns the
report-level verdict.  It exists because of a gap, not a bug: before it, the
whole suite contained **exactly one** test that entered ``process_report``, and
that one raises ``_Stop`` at the end of stage 1
(``test_cli_integration.TestTheSelectorReachesStageOne``).  Stages 2 through the
final ``return`` — some 1100 lines, including all four verdict-setting blocks —
had no test at all.  Nothing in the suite could tell a correct reordering of that
body from an incorrect one.

So every seam the function crosses is replaced by a double: path resolution, the
frontend, the CFG-cache set, the SDG, segmentation, slicing, the analyzer, both
deterministic gates, the domain supplement, the path-space reduction, the
selection call and the POC verification.  What remains under test is the
*orchestration* — which stage runs after which, which exit fires on which
evidence, and what the result JSON says when it does.

Two kinds of assertion are deliberately **not** made:

* no golden byte-for-byte JSON, because every result carries a wall-clock
  duration and the console a per-stage elapsed time;
* no golden *whole console*, because the same reason applies.

What is pinned instead is what the methods doc calls stable: the **key set** of
the result JSON, the **exact** ``verdict_step`` suffix (those Chinese phrases are
matched by ``tools/summarize_faiss_eval.py``, so they are an interface, not
prose), the relative **order** of the console's section markers, and the
side-channel that stage 7 writes on the analyzer.

These tests are pure-deterministic on purpose: they make no LLM call and need no
API key, so they run in milliseconds and can guard every step of a refactor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_path_selection as rps  # noqa: E402
from llm_client.fp_analysis.cfg_models import SegmentInfo  # noqa: E402
from llm_client.fp_analysis.models import ParsedReport, ReportMetadata  # noqa: E402
from llm_client.fp_analysis.pdg_models import SDG  # noqa: E402
from llm_client.fp_analysis.pipeline import exploration as pipeline_exploration  # noqa: E402
from llm_client.fp_analysis.pipeline import runner as pipeline_runner  # noqa: E402
from llm_client.report_frontend import ReportRef  # noqa: E402


# ─── doubles ─────────────────────────────────────────────────────────────────────


class _FakeCacheSet:
    """Stands in for ``CFGCacheSet``; ``as_dict()`` is a live alias, not a copy.

    The real class returns ``self._merged``, and stage 2 relies on the writes it
    makes being visible to stage 5 (H2).  Keeping one dict here is what makes
    that visible if the refactor ever starts copying it.
    """

    def __init__(self, project_name=None, source_root=None):
        self._merged: dict = {}

    def add_cache_file(self, path):
        self._merged["fn_from_cache"] = {}
        return True

    def ensure(self, path):
        return True

    def as_dict(self):
        return self._merged

    def find_key(self, name, source_file=""):
        return None

    def warn_missing(self):
        pass


class _FakeAnalyzer:
    """The three things the orchestration asks of ``FPAnalyzer`` before stage 9.

    ``_domain_facts_text`` is *assigned*, never read, by ``process_report`` — it
    is the side channel stage 7 writes and ``SimulationVerifier`` later reads
    (H4), so the last test asserts on it rather than letting a ``try/finally``
    refactor silently change when it is reset.
    """

    last: "_FakeAnalyzer | None" = None

    def __init__(self, llm_config=None, source_root="", pdg_path="", report_format=None):
        self._model = "fake-model"
        self.source_root = source_root
        self._domain_facts_text = "<unset>"
        self.semantic_reason: str | None = None
        _FakeAnalyzer.last = self

    def _extract_class_names_from_path(self, parsed):
        return set()

    def _load_global_constraints(self, root, target_classes=None):
        return []

    def check_assumptions_feasibility(self, segments=None, parsed=None,
                                      global_constraints=None):
        return ["assumed x != 0"], {"x": "1"}

    def semantic_fp_reason(self, parsed):
        return self.semantic_reason


class _Contradiction:
    """One ``ReportContradiction`` (stage 6a)."""

    condition = "h == fourcc('rrot')"
    variables = ("h",)

    def __init__(self):
        self.first = SimpleNamespace(
            line=507, direction=True, event_number=3,
            function_name="read_index", source_file="faiss/impl/index_read.cpp",
        )
        self.second = SimpleNamespace(
            line=530, direction=False, event_number=9,
            function_name="read_index", source_file="faiss/impl/index_read.cpp",
        )

    def reason(self) -> str:
        return "same condition both ways on the report's own path: L507 vs L530"


class _EntryFinding:
    """One ``EntryStateFinding`` (stage 6b)."""

    kind = "fabricated_entry_state"
    function = "SimpleAtob"
    file = Path("src/numbers.cc")
    parameter = "out"
    guard_macro = "ABSL_RAW_CHECK"
    guard_condition = "out != nullptr"
    guard_line = 109
    guard_source = "ABSL_RAW_CHECK(out != nullptr, ...)"
    guard_semantics = "abort"
    assumption_event_number = 1
    branch_event_number = 2
    deref_line = 200
    call_sites = 0
    call_site_samples = ()

    def reason(self) -> str:
        return "report manufactures a null entry state, then walks the guard's failure arm"


# ─── the harness ─────────────────────────────────────────────────────────────────


@pytest.fixture
def run(monkeypatch, tmp_path: Path):
    """Drive ``process_report`` once, with every stage boundary stubbed.

    Returns a callable taking keyword overrides; every override is either an
    object the corresponding stage returns or a replacement callable.
    """
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-used")
    ref = ReportRef(tmp_path / "report-x.html")
    (tmp_path / "report-x.html").write_text("<html></html>")
    out_dir = tmp_path / "out"

    def _run(
        *,
        contradictions=None,
        entry_finding=None,
        semantic_reason=None,
        domain_facts=None,
        selections_result=None,
        verdict_result=None,
        path_evidence="trace",
        segments=None,
    ):
        parsed = ParsedReport(
            metadata=ReportMetadata(
                bug_type="cpp/NullDereference", bug_file="faiss/impl/index_read.cpp",
                bug_line=507, function_name="read_index", path_length=3,
                path_evidence=path_evidence,
            ),
            path_events=[], source_snippets={}, raw_html_path=str(ref.path),
        )
        loaded = SimpleNamespace(
            document=SimpleNamespace(format=SimpleNamespace(value="csa-html")),
            finding=SimpleNamespace(analyzer=SimpleNamespace(value="clang-sa")),
            ref=SimpleNamespace(selector=""),
            summary=lambda: {"code": {
                "resolved_files": 1, "reported_files": 1, "weak": [], "unresolved": [],
            }},
        )
        segs = segments if segments is not None else [
            SegmentInfo(segment_index=0, label="", event_range=(1, 2),
                        line_start=1, line_end=2, function_name="read_index",
                        function_file="faiss/impl/index_read.cpp"),
        ]
        _FakeAnalyzer.last = None

        monkeypatch.setattr(pipeline_runner, "_resolve_paths", lambda args, rp: (
            ref, tmp_path / "cfg.json", tmp_path / "pdg.json", tmp_path,
        ))
        monkeypatch.setattr(pipeline_runner, "parse_report", lambda r, sr, report_format=None: (
            parsed, None, loaded,
        ))
        monkeypatch.setattr(pipeline_runner, "CFGCacheSet", _FakeCacheSet)
        monkeypatch.setattr(pipeline_runner, "load_sdg", lambda p: SDG(pdgs={}))
        monkeypatch.setattr(pipeline_runner, "segment_path", lambda p, sdg=None, source_root=None: segs)
        monkeypatch.setattr(pipeline_runner, "build_slice_mask", lambda *a, **k: (
            SimpleNamespace(pdg_retained_nodes=set()),
            SimpleNamespace(truncated=False),
        ))
        monkeypatch.setattr(pipeline_runner, "build_dense_path_annotations", lambda *a, **k: {})
        monkeypatch.setattr(pipeline_runner, "run_branch_filter", lambda *a, **k: ([{
            "num_branches": 0, "num_consistent": 0,
            "num_contradictory": 0, "num_insufficient": 0,
        }], {}))
        monkeypatch.setattr(pipeline_runner, "FPAnalyzer", _FakeAnalyzer)
        monkeypatch.setattr(pipeline_runner, "detect_report_contradictions",
                            lambda *a, **k: list(contradictions or []))
        monkeypatch.setattr(pipeline_runner, "detect_fabricated_entry_state",
                            lambda *a, **k: entry_finding)
        monkeypatch.setattr(pipeline_runner, "extract_domain_facts",
                            lambda *a, **k: list(domain_facts or []))
        monkeypatch.setattr(pipeline_runner, "classify_branch_relevance", lambda *a, **k: {
            "node_relevance": {},
            "stats": {"num_branch_nodes": 0, "num_relevant": 0, "num_irrelevant": 0,
                      "irrelevant_ratio": "0.0", "truncated": False},
        })
        monkeypatch.setattr(pipeline_runner, "reduce_path_space", lambda *a, **k: (
            {"whole_path_bound": 1, "ci_groups": []},
            {"original_bound": 1, "reduced_bound": 1, "compression_ratio": "1.0"},
        ))
        monkeypatch.setattr(pipeline_runner, "annotate_ci_groups", lambda ps, segs: ps)
        monkeypatch.setattr(pipeline_runner, "iter_combinations", lambda *a, **k: iter(()))
        monkeypatch.setattr(pipeline_runner, "collect_path_space", lambda *a, **k: {
            "per_segment": [], "total_feasible_candidates": 0, "whole_path_bound": 1,
        })
        monkeypatch.setattr(pipeline_runner, "_chosen_by_seg", lambda s: {})
        # The attempt loop lives in `exploration`, so its two seams are looked
        # up in *that* module's globals.  A re-export would not do: a bare name
        # resolves in the caller's own module, so patching `runner` would bind a
        # second, unread name and the real LLM call would go out.
        monkeypatch.setattr(pipeline_exploration, "run_path_selection",
                            selections_result or (lambda *a, **k: (None, {"exhausted": True})))
        if verdict_result is not None:
            monkeypatch.setattr(pipeline_exploration, "_verify_path", verdict_result)

        args = argparse.Namespace(
            report=str(ref), project="faiss", projects_dir=str(tmp_path),
            source_root=None, pdg=None, cfg_cache=None, scan_dir=None,
            report_format=None, output=str(out_dir / "path_selection_report-x.json"),
            path_space=False,
            max_path_attempts=3, max_reports=None, batch_summary=None,
        )
        if semantic_reason is not None:
            analyzer_ctor = _FakeAnalyzer.__init__

            def _ctor(self, *a, **k):
                analyzer_ctor(self, *a, **k)
                self.semantic_reason = semantic_reason
            monkeypatch.setattr(_FakeAnalyzer, "__init__", _ctor)

        return pipeline_runner.process_report(args, ref)

    return _run, out_dir


def _written(out_dir: Path) -> dict:
    files = list(out_dir.glob("path_selection_*.json"))
    assert len(files) == 1, f"expected exactly one result JSON, got {files}"
    return json.loads(files[0].read_text())



# ─── the four exits ──────────────────────────────────────────────────────────────


class TestStage6Exit:
    """The stage-6 gate returns before stage 7; both detectors share the exit."""

    def test_report_self_contradiction(self, run) -> None:
        drive, out_dir = run
        output = drive(contradictions=[_Contradiction()])

        assert output["verdict"] == "FP"
        assert output["verdict_step"] == "Step 6 确定性一致性闸门（同一条件同路径反向）"
        assert output["report_self_contradiction"]["decisive_fp"] is True
        assert output["report_self_contradiction"]["first"]["line"] == 507
        assert output["report_self_contradiction"]["second"]["direction"] is False
        assert "entry_state_fabrication" not in output
        # The two paths that must NOT run: no stage-8 path space, no POC.
        assert "path_space" not in output
        assert "poc_verification" not in output
        # Every exit emits the code model (it is what the report's index cost).
        assert set(output["code_model"]) == {
            "drift", "drift_truncated", "code_set", "lazy_reads", "lazy_reads_truncated",
        }
        assert json.loads((out_dir / "path_selection_report-x.json").read_text()) == output

    def test_fabricated_entry_state_keeps_its_own_suffix(self, run) -> None:
        drive, _ = run
        output = drive(entry_finding=_EntryFinding())

        assert output["verdict"] == "FP"
        assert output["verdict_step"] == "Step 6 确定性一致性闸门（入口状态制造：穿过必然终止的检查）"
        assert output["entry_state_fabrication"]["parameter"] == "out"
        assert output["entry_state_fabrication"]["guard_line"] == 109
        assert output["entry_state_fabrication"]["deref_line"] == 200
        assert "report_self_contradiction" not in output

    def test_both_detectors_list_both_reasons_but_keep_the_canonical_suffix(self, run) -> None:
        # Documented rule: when both fire, the condition-pair detector keeps the
        # canonical suffix and the entry-state finding is listed alongside it.
        drive, _ = run
        output = drive(contradictions=[_Contradiction()], entry_finding=_EntryFinding())

        assert output["verdict_step"] == "Step 6 确定性一致性闸门（同一条件同路径反向）"
        steps = output["verdict_steps"]
        assert any(s.startswith("报告自相矛盾判定:") for s in steps)
        assert any(s.startswith("入口状态不可实现判定:") for s in steps)


class TestStage7Exit:
    def test_semantic_gate_suffix_and_shape(self, run) -> None:
        drive, _ = run
        output = drive(semantic_reason="h ∈ S but every dispatch arm taken FALSE")

        assert output["verdict"] == "FP"
        assert output["verdict_step"] == "Step 7 语义有限域冲突（约束补全，枚举前判定）"
        assert output["semantic_feasibility"] == {
            "decisive_fp": True,
            "reason": "h ∈ S but every dispatch arm taken FALSE",
        }
        assert output["verdict_reason"] == "h ∈ S but every dispatch arm taken FALSE"
        assert "path_space" not in output  # stage 8 never ran

    def test_no_semantic_reason_falls_through_to_enumeration(self, run) -> None:
        drive, _ = run
        output = drive(semantic_reason=None)
        assert "path_space" in output
        assert output["verdict"] == "UNKNOWN"


class TestFullRun:
    """The pre-enumeration gates declined, so the report reaches stage 9."""

    def test_exhausted_space_aggregates_to_unknown(self, run) -> None:
        drive, _ = run
        output = drive()

        assert output["verdict"] == "UNKNOWN"
        assert output["verdict_step"] == "Step 9 路径选择 路径遍历超限"

    def test_the_result_json_key_set_is_stable(self, run) -> None:
        drive, _ = run
        output = drive()

        assert set(output) == {
            "report", "report_file", "finding",
            "algorithm", "entry_function", "bug_type",
            "assumption_check", "branch_filter", "path_selection",
            "slice_relevance", "context_insensitive",
            "total_elapsed_sec", "verification_status", "verification_flipped_to_fp",
            "project", "verdict", "verdict_step", "verdict_reason", "verdict_steps",
            "path_space", "code_model",
        }
        # The five keys the eval tooling reads by name.
        ps = output["path_selection"]
        assert ps["generalized_fp"] is False
        assert ps["structural_generalized_fp"] is False
        assert ps["leak_consensus_fp"] is False
        assert ps["leak_layer"] == {"fp_verdicts": 0, "tp_verdicts": 0, "reasons": []}
        assert ps["fp_exploration"] == {
            "num_distinct_paths_fp": 0, "saw_unresolved": False, "observations": [],
        }
        assert output["verification_status"] == "NONE"
        assert output["verification_flipped_to_fp"] is False
        assert output["assumption_check"]["num_hard_constraints"] == 1
        assert output["branch_filter"]["num_segments"] == 1
        assert output["algorithm"] == "conflict-driven"
        assert output["project"] == "report-x" or output["project"]

    def test_the_pre_enumeration_gates_leave_the_side_channel_empty(self, run) -> None:
        # H4: stage 7 assigns `_domain_facts_text` *after* calling
        # `extract_domain_facts`, inside the try.  With no facts it must be the
        # empty string, not the `<unset>` the double starts with — a refactor
        # that moved the assignment into a `finally`, or dropped it, shows here.
        drive, _ = run
        drive(domain_facts=[])

        analyzer = _FakeAnalyzer.last
        assert analyzer is not None, "stage 4/7 never constructed the analyzer"
        assert analyzer._domain_facts_text == ""

    def test_console_sections_print_in_order(self, run, capsys) -> None:
        drive, _ = run
        drive(contradictions=[_Contradiction()])
        out = capsys.readouterr().out

        markers = [
            "END-TO-END PATH SELECTION PIPELINE",
            "✗ h == fourcc",                  # the gate's evidence line
            "RESULTS",
            "Verdict:    ❌ FP",
            "Full results saved to:",
            "Total pipeline time:",
        ]
        positions = [out.index(m) for m in markers]
        assert positions == sorted(positions), f"console order broke: {positions}"
        # …and the exit is an early one: nothing below stage 6 printed.
        assert "POC VERIFICATION" not in out
        assert "FEASIBLE PATH SPACE" not in out


class TestDecisiveVerdicts:
    """The two ways stage 9 can set a verdict without the post-loop aggregation."""

    @staticmethod
    def _selections():
        return [{"segment_index": 0, "established_state": "", "branch_decisions": []}]

    def test_tp_from_the_simulation_feedback(self, run) -> None:
        drive, _ = run
        output = drive(
            selections_result=lambda *a, **k: (self._selections(), {"elapsed_sec": 1.0}),
            verdict_result=lambda *a, **k: {
                "poc_result": {"simulation": {"range": [], "rounds": [],
                                              "verified_status": "VERIFIED"}},
                "poc_code": "int main() { return 0; }",
                "poc_path": None,
                "conclusion": "tp",
                "final_reason": "POC reproduces the deref",
                "leak_verdict": None,
                "sim_res": None,
            },
        )

        assert output["verdict"] == "TP"
        assert output["verdict_step"] == "Step 10 POC 验证与结论 模拟执行反馈"
        assert output["verdict_reason"] == "POC reproduces the deref"
        assert output["verification_status"] == "VERIFIED"
        assert output["path_selection"]["num_selections"] == 1

    def test_a_location_only_report_cannot_be_certified_tp(self, run) -> None:
        # The stage-10 verdict rule: with no reported path the conformance audit
        # passes vacuously, so a TP would rest on a path the pipeline chose.
        drive, _ = run
        output = drive(
            path_evidence="location",
            selections_result=lambda *a, **k: (self._selections(), {"elapsed_sec": 1.0}),
            verdict_result=lambda *a, **k: pytest.fail(
                "POC verification must not run for a location-only report"
            ),
        )

        assert output["verdict"] == "UNKNOWN"
        assert "报告未给出路径" in output["verdict_step"]

    def test_generalized_fp_names_the_compression(self, run) -> None:
        drive, _ = run
        output = drive(
            selections_result=lambda *a, **k: (self._selections(), {"elapsed_sec": 1.0}),
            verdict_result=lambda *a, **k: {
                "poc_result": {"simulation": {"rounds": [],
                                              "conclusion": "fp"}},
                "poc_code": None,
                "poc_path": None,
                "conclusion": "fp",
                "final_reason": "the deref'd pointer is a const global",
                "leak_verdict": None,
                "sim_res": None,
            },
        )

        # reduce_path_space returns whole_path_bound == 1 in this harness, so the
        # slice-compression generalization is what fires.
        assert output["verdict"] == "FP"
        assert output["verdict_step"] == "Step 9 路径选择 路径空间压缩推广"
        assert output["path_selection"]["generalized_fp"] is True
        assert output["path_selection"]["structural_generalized_fp"] is False
