"""Pydantic data models for bug report analysis."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class InputFormat(str, Enum):
    """Detected or user-specified input format."""

    SARIF = "sarif"
    JSON = "json"
    PLAIN_TEXT = "plain_text"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Severity level of a bug finding."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class ViabilityRating(str, Enum):
    """How viable POC generation is for a finding."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class ReportStatus(str, Enum):
    """Status of an individual report in a batch."""

    SUCCESS = "success"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


class BugReport(BaseModel):
    """Raw bug report submitted for analysis."""

    content: str = Field(
        ..., min_length=1, description="Raw text/content of the bug report"
    )
    source_name: Optional[str] = Field(None, description="User-provided label")
    input_format: Optional[InputFormat] = Field(
        None, description="User-specified format override; auto-detected if omitted"
    )


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class Location(BaseModel):
    """Code location reference within a finding."""

    file: Optional[str] = Field(None, description="File path (relative or absolute)")
    line: Optional[int] = Field(None, description="Starting line number")
    end_line: Optional[int] = Field(None, description="Ending line number (if range)")
    column: Optional[int] = Field(None, description="Starting column number")
    end_column: Optional[int] = Field(None, description="Ending column number")
    snippet: Optional[str] = Field(None, description="Relevant code snippet")


class POCViability(BaseModel):
    """Exploitability and POC-generation assessment for a single finding."""

    rating: ViabilityRating = Field(..., description="How viable POC generation is")
    exploit_preconditions: list[str] = Field(
        default_factory=list, description="Conditions required for exploitation"
    )
    impact: Optional[str] = Field(None, description="Potential impact if exploited")
    viability_reasoning: Optional[str] = Field(
        None, description="Justification for the rating"
    )


# ---------------------------------------------------------------------------
# Core entities
# ---------------------------------------------------------------------------


class BugFinding(BaseModel):
    """A single vulnerability identified in the report."""

    bug_type: str = Field(
        ..., description="Short name or CWE identifier (e.g. 'CWE-79: XSS')"
    )
    severity: Severity = Field(..., description="Severity level")
    affected_location: Optional[Location] = Field(
        None, description="Where the bug occurs"
    )
    description: str = Field(
        ..., description="Concise natural-language summary (2-3 sentences)"
    )
    root_cause: str = Field(
        ..., description="Brief explanation of the underlying programming error"
    )
    suggested_fix: Optional[str] = Field(
        None, description="High-level remediation guidance"
    )
    poc_viability: Optional[POCViability] = Field(
        None, description="POC generation assessment"
    )


class AnalysisMetadata(BaseModel):
    """Processing metadata attached to each analysis result."""

    input_format: InputFormat = Field(..., description="Detected or specified format")
    model_used: str = Field(..., description="LLM model identifier")
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now, description="ISO-8601 timestamp of analysis"
    )
    total_findings: int = Field(..., description="Number of bug findings extracted")
    processing_time_ms: Optional[int] = Field(
        None, description="Wall-clock time for LLM processing"
    )
    llm_confidence: Optional[str] = Field(
        None, description="LLM self-reported confidence (high/medium/low)"
    )


class AnalysisResult(BaseModel):
    """Top-level output container returned to the caller."""

    schema_version: str = Field(
        "1.0.0", description="Schema version for forward compatibility"
    )
    metadata: AnalysisMetadata = Field(..., description="Processing metadata")
    original_report: str = Field(
        ..., description="Preserved original input for cross-reference"
    )
    findings: list[BugFinding] = Field(
        ..., description="All bugs extracted from the report"
    )


# ---------------------------------------------------------------------------
# Batch entities
# ---------------------------------------------------------------------------


class SingleReportResult(BaseModel):
    """Result of analyzing a single report within a batch."""

    report_index: int = Field(..., description="Index in the original batch input")
    status: ReportStatus = Field(..., description="success or error")
    result: Optional[AnalysisResult] = Field(None, description="Analysis output")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchSummary(BaseModel):
    """Aggregate summary across all reports in a batch."""

    total_reports: int = Field(..., description="Total reports submitted")
    successful: int = Field(..., description="Number of successful analyses")
    failed: int = Field(..., description="Number of failed analyses")
    total_findings: int = Field(..., description="Aggregate count of all findings")
    severity_breakdown: Optional[dict[Severity, int]] = Field(
        None, description="Count of findings per severity level"
    )


class BatchAnalysisResult(BaseModel):
    """Container for batch analysis results."""

    batch_id: str = Field(..., description="Unique batch identifier")
    schema_version: str = Field(
        "1.0.0", description="Schema version for forward compatibility"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Batch creation timestamp"
    )
    results: list[SingleReportResult] = Field(
        ..., description="Individual analysis results"
    )
    summary: Optional[BatchSummary] = Field(None, description="Aggregate summary")
