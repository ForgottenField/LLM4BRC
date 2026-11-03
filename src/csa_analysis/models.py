"""Data models for CSA path extraction."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EventKind(str, Enum):
    """Type of a path event in a CSA bug path."""

    ERROR_START = "error_start"
    EVENT = "event"
    CONTROL = "control"
    ASSUME = "assume"
    CALL = "call"
    RETURN = "return"
    NOTE = "note"
    REPORT = "report"


class PathEvent(BaseModel):
    """A single step in a bug execution path."""

    event_kind: EventKind = Field(description="Type of event: event, control, or report")
    file_path: str = Field(description="Source file path")
    line_number: int = Field(default=0, ge=0, description="Line number (0 = synthetic event)")
    column_number: Optional[int] = Field(None, ge=1, description="Column number")
    description: str = Field(default="", description="Human-readable event description")
    code_snippet: Optional[str] = Field(None, description="Relevant source code at this point")
    branch_condition: Optional[bool] = Field(None, description="For control events: True=taken, False=not-taken")
    callee: Optional[str] = Field(None, description="Callee name for call/return events")
    assumption: str = Field(default="", description="Constraint asserted by an assume event")
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
