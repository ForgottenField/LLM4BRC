"""Main orchestrator for bug report analysis."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Optional

from llm_client.base import LLMConfig, LLMRequest, LLMProvider
from llm_client.bug_analysis.format_detector import detect_format
from llm_client.bug_analysis.prompts import build_messages
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
    ReportStatus,
    Severity,
    SingleReportResult,
    ViabilityRating,
)
from llm_client.factory import ProviderFactory

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "deepseek-v4-pro"


class BugAnalyzer:
    """Analyze static-analysis bug reports via an LLM.

    Parameters
    ----------
    llm_config:
        Provider configuration (use ``provider_name="deepseek"`` for DeepSeek).
    model:
        Model identifier (default ``deepseek-v4-pro``).
    temperature:
        LLM temperature.  Keep at ``0.0`` for deterministic analysis.
    max_tokens:
        Maximum tokens in the LLM response.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._provider: LLMProvider = ProviderFactory.create(llm_config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        report: str | BugReport,
        source_name: str | None = None,
    ) -> AnalysisResult:
        """Analyze a single bug report.

        Parameters
        ----------
        report:
            Raw report content (string) or a :class:`BugReport` instance.
        source_name:
            Optional label (ignored when *report* is a ``BugReport``).

        Returns
        -------
        AnalysisResult
            Structured analysis with extracted findings.
        """
        bug_report = self._normalize_input(report, source_name)
        content = bug_report.content.strip()
        if not content:
            raise ValueError("Bug report content is empty")

        input_format = (
            bug_report.input_format
            if bug_report.input_format is not None
            else detect_format(content)
        )

        messages = build_messages(content)
        llm_req = LLMRequest(
            messages=messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        start = time.monotonic()
        llm_resp = self._provider.send(llm_req)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        raw_json = llm_resp.content
        findings = self._parse_findings(raw_json)

        return AnalysisResult(
            schema_version="1.0.0",
            metadata=AnalysisMetadata(
                input_format=input_format,
                model_used=llm_resp.model,
                analysis_timestamp=datetime.now(),
                total_findings=len(findings),
                processing_time_ms=elapsed_ms,
            ),
            original_report=content,
            findings=findings,
        )

    def analyze_from_file(self, file_path: str, source_name: str | None = None) -> AnalysisResult:
        """Read a bug report from a file and analyze it.

        Parameters
        ----------
        file_path:
            Path to a file containing the bug report.
        source_name:
            Optional label.

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        ValueError
            If the file is empty.
        """
        import pathlib

        path = pathlib.Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Bug report file not found: {file_path}")
        content = path.read_text(encoding="utf-8")
        return self.analyze(content, source_name or path.name)

    def analyze_batch(
        self,
        reports: list[str | BugReport],
        source_names: list[str] | None = None,
    ) -> BatchAnalysisResult:
        """Analyze multiple bug reports independently.

        Each report is processed independently; a failure in one does not
        block the others.

        Parameters
        ----------
        reports:
            List of report contents or ``BugReport`` instances.
        source_names:
            Optional list of labels (ignored for ``BugReport`` entries).

        Returns
        -------
        BatchAnalysisResult
            Individual results and aggregate summary.
        """
        if not reports:
            raise ValueError("Report list is empty")

        single_results: list[SingleReportResult] = []
        for i, report in enumerate(reports):
            try:
                src = source_names[i] if source_names and i < len(source_names) else None
                result = self.analyze(report, source_name=src)
                single_results.append(
                    SingleReportResult(
                        report_index=i,
                        status=ReportStatus.SUCCESS,
                        result=result,
                    )
                )
            except Exception as exc:
                logger.warning("Report %d failed: %s", i, exc)
                single_results.append(
                    SingleReportResult(
                        report_index=i,
                        status=ReportStatus.ERROR,
                        error=str(exc),
                    )
                )

        successful = [r for r in single_results if r.status == ReportStatus.SUCCESS]

        # Build severity breakdown
        severity_breakdown: dict[Severity, int] = {}
        total_findings = 0
        for entry in successful:
            if entry.result:
                for finding in entry.result.findings:
                    total_findings += 1
                    sev = finding.severity
                    severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1

        summary = BatchSummary(
            total_reports=len(reports),
            successful=len(successful),
            failed=len(reports) - len(successful),
            total_findings=total_findings,
            severity_breakdown=severity_breakdown or None,
        )

        return BatchAnalysisResult(
            batch_id=uuid.uuid4().hex[:12],
            schema_version="1.0.0",
            created_at=datetime.now(),
            results=single_results,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_input(report: str | BugReport, source_name: str | None) -> BugReport:
        if isinstance(report, BugReport):
            return report
        return BugReport(content=report, source_name=source_name)

    @staticmethod
    def _parse_findings(raw_json: str) -> list[BugFinding]:
        """Parse and validate the LLM's JSON output into BugFinding objects."""
        cleaned = _extract_json(raw_json)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON output: %s\nRaw: %s", exc, raw_json)
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

        raw_findings = data.get("findings", [])
        if not isinstance(raw_findings, list):
            raise ValueError("LLM output 'findings' is not a list")

        findings: list[BugFinding] = []
        for item in raw_findings:
            findings.append(_parse_single_finding(item))

        return findings


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """Extract JSON from *text*, stripping markdown fences if present."""
    # Strip markdown code fences (```json ... ```)
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _parse_single_finding(item: dict) -> BugFinding:
    """Convert a raw dict from the LLM into a validated BugFinding."""
    loc_raw = item.get("affected_location") or {}
    poc_raw = item.get("poc_viability") or {}

    location = None
    if loc_raw and isinstance(loc_raw, dict):
        # Only create Location if at least one field has a value
        if loc_raw.get("file") or loc_raw.get("line"):
            location = Location(
                file=loc_raw.get("file"),
                line=_int_or_none(loc_raw.get("line")),
                end_line=_int_or_none(loc_raw.get("end_line")),
                column=_int_or_none(loc_raw.get("column")),
                end_column=_int_or_none(loc_raw.get("end_column")),
                snippet=loc_raw.get("snippet"),
            )

    poc_viability = None
    if poc_raw and isinstance(poc_raw, dict) and poc_raw.get("rating"):
        poc_viability = POCViability(
            rating=ViabilityRating(poc_raw["rating"]),
            exploit_preconditions=poc_raw.get("exploit_preconditions", []),
            impact=poc_raw.get("impact"),
            viability_reasoning=poc_raw.get("viability_reasoning"),
        )

    return BugFinding(
        bug_type=str(item.get("bug_type", "unknown")),
        severity=Severity(item.get("severity", "informational")),
        affected_location=location,
        description=str(item.get("description", "")),
        root_cause=str(item.get("root_cause", "")),
        suggested_fix=item.get("suggested_fix"),
        poc_viability=poc_viability,
    )


def _int_or_none(val: object) -> int | None:
    if val is None:
        return None
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
