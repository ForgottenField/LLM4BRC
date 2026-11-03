"""Typed result-document contract shared by pipeline exit paths."""

from __future__ import annotations

from typing import Any, TypedDict


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
    project: str
    assumption_check: AssumptionCheck
    branch_filter: BranchFilter
    path_selection: PathSelection
    path_space: dict[str, Any]
    slice_relevance: dict[str, Any]
    context_insensitive: dict[str, Any]
    verification_status: str
    verification_flipped_to_fp: bool
    poc_verification: dict[str, Any]
    code_model: dict[str, Any]
    report_self_contradiction: dict[str, Any]
    entry_state_fabrication: dict[str, Any]
    semantic_feasibility: dict[str, Any]
