"""Batch output contract tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from pipeline import batch
from pipeline.artifacts import report_out_path
from pipeline.results import configure_results_args, create_results_layout
from report_frontend import ReportRef


def test_batch_summary_persists_unknown_count(tmp_path, monkeypatch) -> None:
    ref = ReportRef(tmp_path / "report-x.html")

    def fake_process_report(args, report_ref):
        return {
            "report": report_ref.path.name,
            "project": "demo",
            "bug_type": "demo-rule",
            "verdict": "UNKNOWN",
            "verdict_step": "Step 9",
            "verdict_reason": "bounded exploration",
            "total_elapsed_sec": 0.1,
        }

    monkeypatch.setattr("pipeline.runner.process_report", fake_process_report)
    summary_path = tmp_path / "batch_summary.json"
    batch.run_batch(SimpleNamespace(batch_summary=str(summary_path)), [ref])

    summary = json.loads(summary_path.read_text())
    assert summary["unknown"] == 1
    assert summary["tp"] == 0
    assert summary["fp"] == 0


def test_results_layout_separates_reports_pocs_and_metadata(tmp_path) -> None:
    layout = create_results_layout(tmp_path / "results", "aria2")
    args = SimpleNamespace(output=None, batch_summary=None)

    configure_results_args(args, layout)

    assert layout.project_dir == layout.run_dir / "aria2"
    assert layout.reports_dir.is_dir()
    assert layout.pocs_dir.is_dir()
    assert args.output == str(layout.reports_dir)
    assert args.batch_summary == str(layout.summary_path)
    assert args.poc_output_dir == str(layout.pocs_dir)
    assert args.output_is_dir is True


def test_report_output_respects_implicit_results_directory(tmp_path) -> None:
    ref = ReportRef(tmp_path / "report-x.html")
    args = SimpleNamespace(
        output=str(tmp_path / "reports"),
        output_is_dir=True,
        report=str(ref),
    )

    assert report_out_path(args, ref) == tmp_path / "reports" / "path_selection_report-x.json"
