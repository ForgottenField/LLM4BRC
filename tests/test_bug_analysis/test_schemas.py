"""Tests for bug analysis data schemas."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from llm_client.bug_analysis.schemas import (
    AnalysisMetadata,
    AnalysisResult,
    BatchAnalysisResult,
    BatchSummary,
    BugFinding,
    BugReport,
    InputFormat,
    Location,
    POCViability,
    Severity,
    SingleReportResult,
    ViabilityRating,
)


class TestBugReport:
    def test_minimal(self):
        r = BugReport(content="test warning")
        assert r.content == "test warning"
        assert r.source_name is None
        assert r.input_format is None

    def test_empty_content_raises(self):
        with pytest.raises(ValidationError):
            BugReport(content="")

    def test_with_all_fields(self):
        r = BugReport(
            content="test",
            source_name="scan-1",
            input_format=InputFormat.SARIF,
        )
        assert r.source_name == "scan-1"
        assert r.input_format == InputFormat.SARIF


class TestLocation:
    def test_minimal(self):
        loc = Location(file="src/main.c", line=42)
        assert loc.file == "src/main.c"
        assert loc.line == 42
        assert loc.end_line is None

    def test_full(self):
        loc = Location(file="f.c", line=1, end_line=10, column=3, end_column=15, snippet="code")
        assert loc.snippet == "code"


class TestPOCViability:
    def test_default_preconditions(self):
        pv = POCViability(rating=ViabilityRating.HIGH)
        assert pv.exploit_preconditions == []
        assert pv.impact is None

    def test_full(self):
        pv = POCViability(
            rating=ViabilityRating.MEDIUM,
            exploit_preconditions=["requires auth"],
            impact="data leak",
            viability_reasoning="needs authenticated session",
        )
        assert pv.impact == "data leak"


class TestBugFinding:
    def test_minimal(self):
        bf = BugFinding(
            bug_type="CWE-79: XSS",
            severity=Severity.HIGH,
            description="XSS in search field",
            root_cause="Unsanitized input",
        )
        assert bf.bug_type == "CWE-79: XSS"
        assert bf.affected_location is None
        assert bf.poc_viability is None

    def test_invalid_severity_raises(self):
        with pytest.raises(ValidationError):
            BugFinding(
                bug_type="X",
                severity="invalid",
                description="d",
                root_cause="r",
            )


class TestAnalysisMetadata:
    def test_timestamp_defaults_to_now(self):
        meta = AnalysisMetadata(
            input_format=InputFormat.SARIF,
            model_used="deepseek-v4-pro",
            total_findings=2,
        )
        assert isinstance(meta.analysis_timestamp, datetime)
        assert meta.total_findings == 2


class TestAnalysisResult:
    def test_round_trip(self, sarif_report):
        finding = BugFinding(
            bug_type="CWE-120",
            severity=Severity.HIGH,
            description="Buffer overflow",
            root_cause="No bounds check",
        )
        result = AnalysisResult(
            metadata=AnalysisMetadata(
                input_format=InputFormat.SARIF,
                model_used="test-model",
                total_findings=1,
            ),
            original_report=sarif_report,
            findings=[finding],
        )
        assert result.schema_version == "1.0.0"
        assert len(result.findings) == 1

        # Round-trip through JSON
        dumped = result.model_dump_json()
        loaded = AnalysisResult.model_validate_json(dumped)
        assert loaded.metadata.total_findings == 1
        assert loaded.findings[0].bug_type == "CWE-120"


class TestBatchEntities:
    def test_single_report_result_success(self):
        srr = SingleReportResult(
            report_index=0,
            status="success",
        )
        assert srr.status.value == "success"
        assert srr.error is None

    def test_single_report_result_error(self):
        srr = SingleReportResult(
            report_index=0,
            status="error",
            error="LLM timeout",
        )
        assert srr.status.value == "error"
        assert srr.error == "LLM timeout"

    def test_batch_summary(self):
        bs = BatchSummary(total_reports=3, successful=2, failed=1, total_findings=5)
        assert bs.severity_breakdown is None

    def test_batch_analysis_result(self):
        srr = SingleReportResult(report_index=0, status="error", error="fail")
        bar = BatchAnalysisResult(
            batch_id="test-batch",
            results=[srr],
            summary=BatchSummary(total_reports=1, successful=0, failed=1, total_findings=0),
        )
        assert bar.batch_id == "test-batch"
        assert bar.summary.failed == 1
