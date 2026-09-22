"""Simplified function summary model for on-demand opaque function analysis.

Retains only the minimal viable summary structure:
- ``ReturnValueConstraint`` for control-flow feasibility checks.
- A single ``FunctionSummary`` model with common fields (no bug-type hierarchy).
- ``CachedFunctionData`` for persistent cache storage.

The bug-type-specific subclasses (MemoryLeakSummary, NullPointerSummary, etc.)
have been removed.  Summarization is no longer automatic in the FP pipeline —
it runs **on-demand** when the post-slice context exceeds size/depth thresholds.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Return value constraint model (for control-flow feasibility analysis)
# ---------------------------------------------------------------------------


class ReturnValueConstraint(BaseModel):
    """Describes the relationship between a function's inputs and its return value.

    Used to determine whether "Assuming condition is true/false" control events
    in a CSA path are feasible — if the constraint shows the condition cannot
    hold given the context, the bug path is a false positive.

    The *failure_nature* field is critical: ``"deterministic"`` means the failure
    condition can be checked against program state (e.g., "queue is full"), while
    ``"non_deterministic"`` means failure depends on system resources (e.g., "malloc
    returns NULL") and can happen at any call, so control events relying on such
    failures are always feasible.
    """

    return_value_semantics: str = Field(
        default="",
        description="What different return values mean, e.g. '0 = success, non-zero = error code WSLAY_ERR_NO_MORE_MSG'",
    )
    success_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that must hold for the function to return its success value",
    )
    failure_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that would cause the function to return an error/failure value",
    )
    failure_nature: str = Field(
        default="deterministic",
        description="Nature of failure: 'deterministic' = predictable from program state (e.g., queue full), 'non_deterministic' = resource/OS failure (e.g., malloc OOM) that can happen at any call",
    )

    def format_for_llm(self) -> str:
        lines = ["  **Return value semantics:** " + self.return_value_semantics]
        nature_label = "Deterministic" if self.failure_nature == "deterministic" else "Non-deterministic (resource/system)"
        lines.append(f"  **Failure nature:** {nature_label}")
        if self.success_conditions:
            for c in self.success_conditions:
                lines.append(f"  - Success when: {c}")
        if self.failure_conditions:
            for c in self.failure_conditions:
                lines.append(f"  - Fails when: {c}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Unified function summary model (no bug-type subclasses)
# ---------------------------------------------------------------------------


class FunctionSummary(BaseModel):
    """Compact summary of a function's behavior for on-demand context compression.

    Used when the post-slice context exceeds thresholds (call depth > 3 or
    code chars > 12 000) to replace deeply-nested or external-library function
    bodies with concise behavioural descriptions.

    The *is_library* flag distinguishes project-internal functions (whose
    bodies exist but are too deep) from external-library functions (no body
    available, summary is a best guess).
    """

    function_name: str = Field(description="Function name")
    signature: str = Field(description="Full function signature")
    file_path: str = Field(default="", description="Source file path (relative to project root)")
    body_available: bool = Field(default=False, description="Whether the function body was available for analysis")
    body_snippet: Optional[str] = Field(default=None, description="Extracted function body snippet")
    is_best_guess: bool = Field(default=False, description="True if summary is a best-guess (no body available)")
    is_library: bool = Field(default=False, description="True if this is an external library function (no project source)")
    preconditions: list[str] = Field(default_factory=list, description="Preconditions for the function behavior")
    analysis: str = Field(default="", description="Free-text analysis of function behavior")
    return_value_constraints: Optional[ReturnValueConstraint] = Field(
        default=None,
        description="Return value semantics and input-output constraints for control-flow feasibility analysis",
    )

    def format_for_llm(self) -> str:
        """Render this summary as compact text for prompt injection."""
        body_flag = " (body available)" if self.body_available else " (BEST GUESS, body NOT available)"
        control_flag = " [CONTROL-CONTEXT]" if self.return_value_constraints else ""
        lib_flag = " [LIBRARY]" if self.is_library else ""
        lines = [
            f"### {self.function_name}{body_flag}{control_flag}{lib_flag}",
            f"* Signature: `{self.signature}`",
            f"* File: {self.file_path}",
        ]
        if self.preconditions:
            lines.append("* Preconditions:")
            for p in self.preconditions:
                lines.append(f"  - {p}")
        if self.return_value_constraints:
            lines.append("* Return Value Constraints:")
            lines.append(self.return_value_constraints.format_for_llm())
        lines.append(f"* Analysis: {self.analysis}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

# Canonical (project-internal) bug type names — used as cache keys.
_CANONICAL_BUG_TYPES: dict[str, str] = {
    "memory leak": "memory_leak",
    "potential memory leak": "memory_leak",
    "dereference of null pointer": "dereference_of_null_pointer",
    "argument with nonnull attribute passed null": "argument_with_nonnull_attribute_passed_null",
    "assigned value is garbage or undefined": "assigned_value_is_garbage_or_undefined",
    "result of operation is garbage or undefined": "result_of_operation_is_garbage_or_undefined",
    "uninitialized argument value": "uninitialized_argument_value",
    "branch condition evaluates to a garbage value": "branch_condition_evaluates_to_a_garbage_value",
    # Short-name variants
    "null pointer": "dereference_of_null_pointer",
    "null": "dereference_of_null_pointer",
    "null_pointer": "dereference_of_null_pointer",
    "nonnull arg": "argument_with_nonnull_attribute_passed_null",
    "nonnull": "argument_with_nonnull_attribute_passed_null",
    "garbage value": "assigned_value_is_garbage_or_undefined",
    "undefined value": "assigned_value_is_garbage_or_undefined",
    "garbage condition": "branch_condition_evaluates_to_a_garbage_value",
    "uninitialized arg": "uninitialized_argument_value",
}


def normalize_bug_type(bug_type: str) -> str:
    """Normalize a CSA bug type string to a canonical underscored key."""
    text = bug_type.strip().lower()
    if text in _CANONICAL_BUG_TYPES:
        return _CANONICAL_BUG_TYPES[text]
    result = re.sub(r"[\s\-\.]+", "_", text)
    result = re.sub(r"[^\w]", "", result)
    if result in _CANONICAL_BUG_TYPES:
        return _CANONICAL_BUG_TYPES[result]
    return result


# ---------------------------------------------------------------------------
# Cache container model
# ---------------------------------------------------------------------------


class CachedFunctionData(BaseModel):
    """Represents one JSON cache file for a single function.

    Stores all bug-type-specific summaries for one function in one file.
    """

    function_name: str = Field(description="Function name")
    project: str = Field(description="Project name")
    file_path: str = Field(default="", description="Source file path (relative)")
    signature: str = Field(default="", description="Function signature")
    body_available: bool = Field(default=False)
    body_snippet: Optional[str] = Field(default=None)
    is_library: bool = Field(default=False, description="External library function")
    summaries: dict[str, dict] = Field(
        default_factory=dict,
        description="Bug type → summary dict mapping",
    )
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
