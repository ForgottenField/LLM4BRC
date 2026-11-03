"""Batch output contract tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

from pipeline import batch
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
