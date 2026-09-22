"""Tests for result persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from llm_client.bug_analysis.schemas import (
    AnalysisMetadata,
    AnalysisResult,
    BatchAnalysisResult,
    BatchSummary,
    BugFinding,
    InputFormat,
    Severity,
    SingleReportResult,
)
from llm_client.bug_analysis.storage import load_analysis, load_batch, save_analysis, save_batch


@pytest.fixture
def sample_result() -> AnalysisResult:
    finding = BugFinding(
        bug_type="CWE-120",
        severity=Severity.HIGH,
        description="Buffer overflow",
        root_cause="No bounds check",
    )
    return AnalysisResult(
        metadata=AnalysisMetadata(
            input_format=InputFormat.SARIF,
            model_used="test",
            total_findings=1,
        ),
        original_report="test report",
        findings=[finding],
    )


@pytest.fixture
def sample_batch(sample_result) -> BatchAnalysisResult:
    srr = SingleReportResult(report_index=0, status="success", result=sample_result)
    return BatchAnalysisResult(
        batch_id="test-batch",
        results=[srr],
        summary=BatchSummary(total_reports=1, successful=1, failed=0, total_findings=1),
    )


class TestSaveAnalysis:
    def test_save_json(self, sample_result):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_analysis(sample_result, tmp, fmt="json")
            assert Path(path).exists()
            data = json.loads(Path(path).read_text())
            assert data["schema_version"] == "1.0.0"
            assert len(data["findings"]) == 1

    def test_save_yaml(self, sample_result):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_analysis(sample_result, tmp, fmt="yaml")
            assert Path(path).exists()
            assert path.endswith(".yaml")

    def test_save_unsupported_format(self, sample_result):
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="Unsupported format"):
                save_analysis(sample_result, tmp, fmt="xml")


class TestLoadAnalysis:
    def test_round_trip_json(self, sample_result):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_analysis(sample_result, tmp)
            loaded = load_analysis(path)
            assert loaded.schema_version == sample_result.schema_version
            assert loaded.findings[0].bug_type == "CWE-120"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_analysis("/nonexistent/file.json")


class TestSaveBatch:
    def test_save_batch_json(self, sample_batch):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_batch(sample_batch, tmp)
            assert Path(path).exists()
            data = json.loads(Path(path).read_text())
            assert data["batch_id"] == "test-batch"

    def test_save_batch_creates_individual_files(self, sample_batch):
        with tempfile.TemporaryDirectory() as tmp:
            save_batch(sample_batch, tmp)
            # Individual file should exist
            individual = Path(tmp) / "batch_test-batch" / "report_0000.json"
            assert individual.exists()


class TestLoadBatch:
    def test_round_trip(self, sample_batch):
        with tempfile.TemporaryDirectory() as tmp:
            path = save_batch(sample_batch, tmp)
            loaded = load_batch(path)
            assert loaded.batch_id == "test-batch"
            assert loaded.summary.total_reports == 1
