"""Deterministic checks between a selected whole path and CSA control events."""

from __future__ import annotations

from pathlib import Path


def _decision_line(decision: dict) -> int:
    value = decision.get("source_line", decision.get("line", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _decision_taken(decision: dict) -> bool | None:
    value = decision.get("branch_taken")
    if isinstance(value, bool):
        return value
    branch = str(decision.get("branch", "")).strip().lower()
    if branch == "true":
        return True
    if branch == "false":
        return False
    return None


def _function_key(value: object) -> str:
    return str(value or "").strip().split("::")[-1]


def _file_key(value: object) -> str:
    text = str(value or "").strip()
    return Path(text).as_posix() if text else ""


def _same_location(
    decision: dict,
    constraint: dict,
    line: int,
) -> bool:
    if _decision_line(decision) != line:
        return False
    decision_fn = _function_key(decision.get("function"))
    constraint_fn = _function_key(constraint.get("function"))
    if decision_fn and constraint_fn and decision_fn != constraint_fn:
        return False
    decision_file = _file_key(decision.get("file"))
    constraint_file = _file_key(constraint.get("file"))
    if decision_file and constraint_file:
        if decision_file == constraint_file:
            return True
        return Path(decision_file).name == Path(constraint_file).name
    return True


def selected_path_violations(
    selections: list[dict] | None,
    segments: list,
) -> list[dict]:
    """Return explicit control-direction conflicts between the two paths.

    Matching deliberately spans segments.  The selector commonly places a
    decision on the line that terminates segment N, while the corresponding CSA
    control event starts segment N+1 (the OMSG L694 call/control pair does
    this).  Decisions and constraints are therefore matched by
    ``(file, function, line)`` while retaining every CSA event number.

    A missing branch decision is not a violation because the branch tree may not
    expose every report control event.  A decision at the same location with
    only the opposite direction is an explicit contradiction and is rejected.
    """
    segments_by_index = {
        int(segment.segment_index): segment for segment in (segments or [])
    }
    decisions: list[dict] = []
    for selection in selections or []:
        if not isinstance(selection, dict):
            continue
        try:
            index = int(selection["segment_index"])
        except (KeyError, TypeError, ValueError):
            continue
        segment = segments_by_index.get(index)
        if segment is None:
            continue
        for decision in selection.get("branch_decisions") or []:
            if str(decision.get("kind") or "").strip().lower() == "call":
                continue
            enriched = dict(decision)
            enriched.setdefault("function", segment.function_name)
            enriched.setdefault("file", segment.function_file)
            enriched["selection_segment_index"] = index
            decisions.append(enriched)

    violations: list[dict] = []
    for segment in segments or []:
        for constraint in getattr(segment, "control_constraints", []) or []:
            expected = constraint.get("branch_taken")
            line = int(constraint.get("line") or 0)
            if not isinstance(expected, bool) or line <= 0:
                continue
            at_line = [
                (decision, _decision_taken(decision))
                for decision in decisions
                if _same_location(decision, constraint, line)
            ]
            known = [(d, taken) for d, taken in at_line if taken is not None]
            if not known or any(taken == expected for _, taken in known):
                continue
            decision, actual = known[0]
            violations.append({
                "segment_index": segment.segment_index,
                "selection_segment_index": decision["selection_segment_index"],
                "event_number": constraint.get("event_number"),
                "line": line,
                "function": constraint.get("function") or segment.function_name,
                "file": constraint.get("file") or segment.function_file,
                "reported_branch_taken": expected,
                "selected_branch_taken": actual,
                "node_id": decision.get("node_id", ""),
            })
    return violations
