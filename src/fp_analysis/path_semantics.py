"""Deterministic cross-procedure checks for selected branch paths."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_CALL_NAME_RE = re.compile(
    r"(?:→\s*|callee_[A-Za-z_][A-Za-z0-9_:]*?_)"
    r"(?P<name>[A-Za-z_~][A-Za-z0-9_:]*)\s*\(\)"
)
_BOOL_RETURN_RE = re.compile(r"\breturn\s+(true|false)\s*;", re.I)
_FUNCTION_RANGE_CACHE: dict[str, dict[str, list[tuple[int, int]]]] = {}


@dataclass(frozen=True)
class _Decision:
    selection: dict
    data: dict
    segment_index: int
    source_line: int
    branch_taken: bool | None
    condition: str
    kind: str
    callee: str
    function: str
    source_file: str


def _short_name(value: object) -> str:
    return str(value or "").strip().split("::")[-1].split("<", 1)[0]


def _line(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _taken(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _call_name(decision: dict) -> str:
    explicit = str(decision.get("callee_name") or "").strip()
    if explicit:
        return _short_name(explicit)
    condition = str(
        decision.get("condition_expr", decision.get("condition", "")) or ""
    )
    match = _CALL_NAME_RE.search(condition)
    return _short_name(match.group("name")) if match else ""


def _decision_kind(decision: dict) -> str:
    explicit = str(decision.get("kind") or "").strip().lower()
    if explicit:
        return explicit
    condition = str(
        decision.get("condition_expr", decision.get("condition", "")) or ""
    )
    return "call" if condition.lstrip().startswith("→") else "control"


def _iter_decisions(
    selections: list[dict] | None,
    segments: list | None,
) -> list[_Decision]:
    by_index = {
        _line(getattr(segment, "segment_index", -1)): segment
        for segment in (segments or [])
    }
    result: list[_Decision] = []
    for selection in selections or []:
        if not isinstance(selection, dict):
            continue
        segment_index = _line(selection.get("segment_index"))
        segment = by_index.get(segment_index)
        for decision in selection.get("branch_decisions") or []:
            if not isinstance(decision, dict):
                continue
            result.append(_Decision(
                selection=selection,
                data=decision,
                segment_index=segment_index,
                source_line=_line(
                    decision.get("source_line", decision.get("line", 0))
                ),
                branch_taken=_taken(
                    decision.get(
                        "branch_taken",
                        decision.get("branch"),
                    )
                ),
                condition=str(
                    decision.get(
                        "condition_expr",
                        decision.get("condition", ""),
                    )
                    or ""
                ),
                kind=_decision_kind(decision),
                callee=_call_name(decision),
                function=str(
                    decision.get("function_name")
                    or selection.get("function")
                    or getattr(segment, "function_name", "")
                    or ""
                ),
                source_file=str(
                    decision.get("file")
                    or selection.get("file")
                    or getattr(segment, "function_file", "")
                    or ""
                ),
            ))
    return result


def _required_call_return(call: _Decision, decisions: list[_Decision]) -> bool | None:
    if not call.callee:
        return None
    candidates = [
        decision for decision in decisions
        if decision.kind != "call"
        and decision.source_line == call.source_line
        and decision.function == call.function
        and re.search(
            rf"(?<![\w:]){re.escape(call.callee)}\s*\(",
            decision.condition,
        )
    ]
    for decision in candidates:
        if decision.branch_taken is None:
            continue
        condition = re.sub(r"\s+", " ", decision.condition).strip()
        match = re.search(
            rf"(?P<neg>!?)\s*{re.escape(call.callee)}\s*\(",
            condition,
        )
        if match is None:
            continue
        negated = bool(match.group("neg"))
        has_or = "||" in condition
        has_and = "&&" in condition
        if decision.branch_taken:
            if has_or:
                return None
            return not negated
        if has_and:
            return None
        return negated
    return None


def _strip_line_comments(lines: list[str]) -> list[str]:
    return [re.sub(r"//.*$", "", line) for line in lines]


def _function_ranges(path: Path, name: str) -> list[tuple[int, int]]:
    cache_key = str(path)
    per_path = _FUNCTION_RANGE_CACHE.setdefault(cache_key, {})
    if name in per_path:
        return per_path[name]
    try:
        lines = _strip_line_comments(
            path.read_text(encoding="utf-8", errors="replace").splitlines()
        )
    except OSError:
        per_path[name] = []
        return []
    token = re.compile(rf"(?<![\w:]){re.escape(name)}\s*\(")
    ranges: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        if not token.search(line) or line.lstrip().startswith(("//", "*")):
            continue
        open_line = -1
        for probe in range(index, min(len(lines), index + 16)):
            if ";" in lines[probe] and "{" not in lines[probe]:
                break
            if "{" in lines[probe]:
                open_line = probe
                break
        if open_line < 0:
            continue
        depth = 0
        close_line = -1
        for probe in range(open_line, len(lines)):
            depth += lines[probe].count("{") - lines[probe].count("}")
            if depth == 0 and probe >= open_line:
                close_line = probe
                break
        if close_line >= 0:
            ranges.append((index + 1, close_line + 1))
    per_path[name] = ranges
    return ranges


def _range_containing(
    path: Path,
    name: str,
    line: int,
) -> tuple[int, int] | None:
    ranges = _function_ranges(path, name)
    exact = [item for item in ranges if item[0] <= line <= item[1]]
    if exact:
        return exact[-1]
    return ranges[-1] if ranges else None


def _direct_branch_return(lines: list[str], line: int, end: int) -> bool | None:
    if line <= 0 or line > len(lines):
        return None
    open_line = -1
    for probe in range(line - 1, min(end, line + 12)):
        if "return" in lines[probe]:
            match = _BOOL_RETURN_RE.search(lines[probe])
            if match and "if" in lines[line - 1]:
                return match.group(1).lower() == "true"
        if "{" in lines[probe]:
            open_line = probe
            break
        if ";" in lines[probe]:
            break
    if open_line < 0:
        return None
    depth = 0
    for probe in range(open_line, end):
        depth += lines[probe].count("{") - lines[probe].count("}")
        match = _BOOL_RETURN_RE.search(lines[probe])
        if match and depth == 1:
            return match.group(1).lower() == "true"
        if depth <= 0:
            break
    return None


def _top_level_return(lines: list[str], start: int, end: int) -> bool | None:
    depth = 0
    for probe in range(start - 1, min(end, len(lines))):
        depth += lines[probe].count("{") - lines[probe].count("}")
        match = _BOOL_RETURN_RE.search(lines[probe])
        if match and depth == 1:
            return match.group(1).lower() == "true"
    return None


def _selected_bool_return(
    callee: str,
    decisions: list[_Decision],
    source_root: str | None,
) -> tuple[bool, int, str] | None:
    grouped: dict[str, list[_Decision]] = {}
    for decision in decisions:
        if decision.kind != "control" or not decision.source_file:
            continue
        function = decision.function
        path = _resolve_path(decision.source_file, source_root)
        if not function:
            for candidate_range in _function_ranges(path, callee):
                if candidate_range[0] <= decision.source_line <= candidate_range[1]:
                    function = callee
                    break
        if _short_name(function) == callee:
            grouped.setdefault(str(path), []).append(decision)

    for source_file, selected in grouped.items():
        path = Path(source_file)
        containing = [
            _range_containing(path, callee, decision.source_line)
            for decision in selected
        ]
        valid = [item for item in containing if item is not None]
        if not valid:
            continue
        start = min(item[0] for item in valid)
        end = max(item[1] for item in valid)
        try:
            lines = _strip_line_comments(
                path.read_text(encoding="utf-8", errors="replace").splitlines()
            )
        except OSError:
            continue
        for decision in sorted(selected, key=lambda item: item.source_line):
            if not decision.branch_taken:
                continue
            returned = _direct_branch_return(lines, decision.source_line, end)
            if returned is not None:
                return returned, decision.source_line, source_file
        returned = _top_level_return(lines, start, end)
        if returned is not None:
            return returned, end, source_file
    return None


def _resolve_path(source_file: str, source_root: str | None) -> Path:
    path = Path(source_file).expanduser()
    if path.exists() or path.is_absolute() or not source_root:
        return path
    candidate = Path(source_root).expanduser() / path
    return candidate if candidate.exists() else path


def selected_call_return_violations(
    selections: list[dict] | None,
    segments: list | None = None,
    source_root: str | None = None,
) -> list[dict]:
    """Find selected call/callee-return contradictions.

    The call sentinel records that a call occurs.  Its boolean direction is not
    the callee's return value.  This check derives the return required by the
    caller condition and, when the callee has a simple boolean early-return
    shape, derives the value selected by the callee's own branch decisions.
    """
    decisions = _iter_decisions(selections, segments)
    violations: list[dict] = []
    seen: set[tuple[str, int, bool, bool]] = set()
    for call in decisions:
        if call.kind != "call" or not call.callee:
            continue
        required = _required_call_return(call, decisions)
        if required is None:
            continue
        inferred = _selected_bool_return(call.callee, decisions, source_root)
        if inferred is None:
            continue
        actual, return_line, source_file = inferred
        if actual == required:
            continue
        key = (call.callee, call.source_line, required, actual)
        if key in seen:
            continue
        seen.add(key)
        violations.append({
            "kind": "call_return",
            "segment_index": call.segment_index,
            "callee": call.callee,
            "caller": call.function,
            "call_line": call.source_line,
            "required_return": required,
            "selected_return": actual,
            "return_line": return_line,
            "file": source_file or call.source_file,
            "reason": (
                f"Caller decision at L{call.source_line} requires "
                f"{call.callee}() == {required}, but the selected callee "
                f"branches produce {actual} at L{return_line}."
            ),
        })
    return violations
