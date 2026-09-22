"""Data models for CSA path extraction."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Confidence(str, Enum):
    """Correlation confidence level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class AnalyzerMode(str, Enum):
    """CSA analyzer execution mode."""

    SINGLE_FILE = "single-file"
    SCAN_BUILD = "scan-build"


class EventKind(str, Enum):
    """Type of a path event in a CSA bug path."""

    EVENT = "event"
    CONTROL = "control"
    REPORT = "report"


class CSAStatus(str, Enum):
    """Status of a CSA invocation."""

    SUCCESS = "success"
    ERROR = "error"


class BugFindingRef(BaseModel):
    """A reference to a bug finding from feature 002 (bug report analysis).

    Used as input to CSA analysis to guide which file/line to analyze.
    """

    bug_type: str = Field(description="Bug type/checker name")
    file_path: str = Field(description="Path to the affected source file")
    line_number: int = Field(ge=1, description="Line number where the bug occurs")
    end_line: Optional[int] = Field(None, ge=1, description="End line number for multi-line bugs")
    severity: str = Field(description="Severity level from bug analysis")
    description: str = Field(description="Human-readable bug description")

    @field_validator("end_line")
    @classmethod
    def end_line_must_be_ge_line(cls, v: Optional[int], info) -> Optional[int]:
        if v is not None and info.data.get("line_number") and v < info.data["line_number"]:
            raise ValueError("end_line must be >= line_number")
        return v


class CSAConfig(BaseModel):
    """Configuration for invoking Clang Static Analyzer."""

    compiler_flags: list[str] = Field(default_factory=list, description="Additional compiler flags")
    analyzer_mode: AnalyzerMode = Field(default=AnalyzerMode.SINGLE_FILE, description="Analysis mode")
    build_command: Optional[str] = Field(None, description="Build command for scan-build mode")
    output_dir: Optional[str] = Field(None, description="Directory for CSA output files")
    timeout_seconds: int = Field(default=60, ge=1, description="Timeout for CSA invocation")
    checkers: Optional[list[str]] = Field(None, description="Specific checkers to enable")
    language_standard: Optional[str] = Field(None, description="C/C++ language standard")


class CSAResult(BaseModel):
    """The raw result from a CSA invocation."""

    status: CSAStatus = Field(description="Success or error status")
    plist_path: Optional[str] = Field(None, description="Path to the generated plist file")
    plist_paths: Optional[list[str]] = Field(None, description="Paths to multiple plist files (scan-build)")
    command: str = Field(description="The exact CSA command executed")
    stdout: str = Field(default="", description="Stdout from the CSA process")
    stderr: str = Field(default="", description="Stderr from the CSA process")
    duration_ms: int = Field(default=0, ge=0, description="Execution time in milliseconds")
    exit_code: int = Field(default=-1, description="Process exit code")


class PathEvent(BaseModel):
    """A single step in a bug execution path."""

    event_kind: EventKind = Field(description="Type of event: event, control, or report")
    file_path: str = Field(description="Source file path")
    line_number: int = Field(default=0, ge=0, description="Line number (0 = synthetic event)")
    column_number: Optional[int] = Field(None, ge=1, description="Column number")
    description: str = Field(default="", description="Human-readable event description")
    code_snippet: Optional[str] = Field(None, description="Relevant source code at this point")
    branch_condition: Optional[bool] = Field(None, description="For control events: True=taken, False=not-taken")
    depth: Optional[int] = Field(None, description="Call stack depth")


class BugPath(BaseModel):
    """A structured representation of a single CSA bug path (parsed from plist)."""

    path_id: str = Field(description="Unique identifier for the bug path")
    bug_type: str = Field(description="CSA bug type/checker name")
    category: str = Field(description="CSA category (Logic error, Security, etc.)")
    check_name: Optional[str] = Field(None, description="Specific checker that found the bug")
    issue_context: Optional[str] = Field(None, description="Function name where the bug occurs")
    events: list[PathEvent] = Field(description="Ordered list of events in the execution path")
    issue_hash: Optional[str] = Field(None, description="CSA issue hash for deduplication")

    @field_validator("events")
    @classmethod
    def events_must_have_report_at_end(cls, v: list[PathEvent]) -> list[PathEvent]:
        if v and v[-1].event_kind != EventKind.REPORT:
            raise ValueError("The last event must have event_kind == 'report'")
        return v


class CorrelatedResult(BaseModel):
    """A unified result linking a BugFindingRef with its matching BugPath."""

    bug_finding: BugFindingRef = Field(description="The original bug finding")
    bug_path: Optional[BugPath] = Field(None, description="The matched CSA path (null if no match)")
    correlation_confidence: Confidence = Field(description="How well they match")
    notes: Optional[str] = Field(None, description="Discrepancies or warnings")


class FileResult(BaseModel):
    """Analysis result for a single source file in a project scan."""

    file_path: str = Field(description="Path to the source file analyzed")
    status: CSAStatus = Field(description="Analysis status for this file")
    duration_ms: int = Field(default=0, ge=0, description="Analysis time for this file")
    bug_paths: list[BugPath] = Field(default_factory=list, description="Bug paths found in this file")
    error: Optional[str] = Field(None, description="Error message if analysis failed")


class ProjectAnalysisResult(BaseModel):
    """Aggregated result from scanning an entire project."""

    project_dir: str = Field(description="Root directory of the project")
    total_files: int = Field(default=0, description="Total files analyzed")
    successful_files: int = Field(default=0, description="Files analyzed successfully")
    failed_files: int = Field(default=0, description="Files that failed to analyze")
    total_duration_ms: int = Field(default=0, ge=0, description="Total analysis time")
    file_results: list[FileResult] = Field(default_factory=list, description="Per-file results")
    all_bug_paths: list[BugPath] = Field(default_factory=list, description="All bug paths across all files")

    @property
    def total_bugs(self) -> int:
        """Total number of bug paths found across all files."""
        return len(self.all_bug_paths)


class ReachabilityTarget(BaseModel):
    """Target location for guided reachability analysis.

    Specifies the file and line to trace execution paths to during
    CSA symbolic execution.
    """

    file_path: str = Field(description="Path to target source file")
    line_number: int = Field(ge=1, description="Target line number")
    compiler_flags: list[str] = Field(
        default_factory=list,
        description="Extra compiler flags (e.g., -I includes)",
    )


class ReachabilityResult(BaseModel):
    """Result from a single reachability analysis run."""

    target: ReachabilityTarget = Field(description="The target location")
    found: bool = Field(description="Whether a path to the target was found")
    bug_paths: list[BugPath] = Field(
        default_factory=list,
        description="Execution paths reaching the target",
    )
    plist_path: Optional[str] = Field(None, description="Raw plist output path")
    duration_ms: int = Field(default=0, ge=0, description="Analysis time")
    error: Optional[str] = Field(None, description="Error message if any")
