"""Typed result-document contract shared by pipeline exit paths."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class ReportIdentity(TypedDict):
    report: str
    report_file: str
    finding: str


class AssumptionCheck(TypedDict):
    num_hard_constraints: int
    hard_constraints: list[str]
    num_state_bindings: int
    state_bindings: dict[str, str]
    elapsed_sec: float


class BranchFilter(TypedDict):
    num_segments: int
    total_branches: int
    consistent: int
    contradictory_pruned: int
    insufficient_info: int
    per_segment: list[dict[str, Any]]


class PathSelection(TypedDict):
    verdict: str | None
    elapsed_sec: float
    num_selections: int
    selections: list[dict[str, Any]] | None
    blocked_paths: list[str]
    max_path_attempts: int
    configured_max_path_attempts: int
    full_space_explore: bool
    generalized_fp: bool
    structural_generalized_fp: bool
    leak_consensus_fp: bool
    leak_layer: dict[str, Any]
    fp_exploration: dict[str, Any]


class ResultDocument(TypedDict, total=False):
    """One report result.

    Early exits intentionally carry a smaller field set.  Keeping that fact in
    one typed schema makes the variants explicit without forcing all stages to
    construct the same superset today.
    """

    report: str
    report_file: str
    finding: str
    algorithm: str
    entry_function: str
    bug_type: str
    verdict: str
    verdict_step: str | None
    verdict_reason: str | None
    verdict_steps: list[str]
    total_elapsed_sec: float
    project: NotRequired[str]
    assumption_check: NotRequired[AssumptionCheck]
    branch_filter: NotRequired[BranchFilter]
    path_selection: NotRequired[PathSelection]
    path_space: NotRequired[dict[str, Any]]
    slice_relevance: NotRequired[dict[str, Any]]
    context_insensitive: NotRequired[dict[str, Any]]
    verification_status: NotRequired[str]
    verification_flipped_to_fp: NotRequired[bool]
    poc_verification: NotRequired[dict[str, Any]]
    code_model: NotRequired[dict[str, Any]]
    report_self_contradiction: NotRequired[dict[str, Any]]
    entry_state_fabrication: NotRequired[dict[str, Any]]
    semantic_feasibility: NotRequired[dict[str, Any]]
