"""Bug finding ↔ CSA path correlation.

Matches a BugFindingRef (from feature 002) against parsed CSA bug paths
by file location, line number proximity, and bug type.
"""

from __future__ import annotations

from pathlib import Path

from llm_client.csa_analysis.models import (
    BugFindingRef,
    BugPath,
    Confidence,
    CorrelatedResult,
    PathEvent,
)

# ---------------------------------------------------------------------------
# CWE / bug-type alias mappings
# ---------------------------------------------------------------------------
# Maps a canonical bug name to its known aliases (lowercase).
_TYPE_ALIASES: dict[str, set[str]] = {
    "buffer overflow": {
        "buffer overflow",
        "buffer-overflow",
        "buffer_overflow",
        "cwe-120",
        "cwe-122",
        "cwe-124",
        "out-of-bound",
        "out of bound",
        "array bound",
        "arraybound",
    },
    "use-after-free": {
        "use-after-free",
        "use after free",
        "use_after_free",
        "cwe-416",
        "dangling pointer",
        "use of freed pointer",
        "use of freed",
    },
    "sql injection": {
        "sql injection",
        "sql_injection",
        "cwe-89",
        "sqli",
    },
    "format string": {
        "format string",
        "format_string",
        "cwe-134",
        "insecure format",
    },
    "null pointer": {
        "null pointer",
        "null pointer dereference",
        "null_pointer",
        "null dereference",
        "cwe-476",
        "dereference of null",
        "dereference of null pointer",
    },
    "memory leak": {
        "memory leak",
        "memory_leak",
        "cwe-401",
        "memory never released",
    },
    "divide by zero": {
        "divide by zero",
        "divide_by_zero",
        "division by zero",
        "cwe-369",
    },
    "uninitialized value": {
        "uninitialized value",
        "uninitialized",
        "cwe-457",
        "uninit",
    },
}

# Cache: bug_type → set of canonical names
_type_cache: dict[str, set[str]] = {}


def _normalize_type(text: str) -> str:
    """Lowercase and strip whitespace."""
    return text.strip().lower()


def _get_aliases(bug_type: str) -> set[str]:
    """Resolve a bug type string to the set of all known aliases."""
    normalized = _normalize_type(bug_type)
    if normalized in _type_cache:
        return _type_cache[normalized]

    # Check if it matches any canonical entry
    for canonical, aliases in _TYPE_ALIASES.items():
        if normalized == canonical or normalized in aliases:
            _type_cache[normalized] = aliases | {canonical}
            return _type_cache[normalized]

    # Fallback: just use the normalized form itself
    _type_cache[normalized] = {normalized}
    return _type_cache[normalized]


def _types_match(finding_type: str, csa_type: str) -> bool:
    """Check if two bug type strings refer to the same bug class."""
    f_aliases = _get_aliases(finding_type)
    c_aliases = _get_aliases(csa_type)
    return bool(f_aliases & c_aliases)


def _resolve_path(bug_path: BugPath) -> str:
    """Get the primary file path from a BugPath (first event's file)."""
    if bug_path.events:
        return bug_path.events[0].file_path
    return ""


def _min_line_diff(bug_line: int, path: BugPath) -> int:
    """Compute the minimum absolute line difference between *bug_line*
    and any event in *path*."""
    if not path.events:
        return float("inf")  # type: ignore[return-value]
    return min(
        abs(event.line_number - bug_line) for event in path.events
    )


class BugPathCorrelator:
    """Match a BugFindingRef against parsed CSA bug paths."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correlate(
        self,
        bug_finding: BugFindingRef,
        bug_paths: list[BugPath],
    ) -> CorrelatedResult:
        """Correlate a bug finding with the best-matching CSA bug path.

        Args:
            bug_finding: The bug finding from feature 002 analysis.
            bug_paths: List of parsed CSA bug paths.

        Returns:
            CorrelatedResult with matched bug_path (or None) and confidence.
        """
        if not bug_paths:
            return CorrelatedResult(
                bug_finding=bug_finding,
                bug_path=None,
                correlation_confidence=Confidence.NONE,
                notes="No CSA paths available for correlation.",
            )

        best_path: BugPath | None = None
        best_confidence = Confidence.NONE
        best_line_diff: int | float = float("inf")
        best_notes: list[str] = []

        for path in bug_paths:
            path_file = _resolve_path(path)
            line_diff = _min_line_diff(bug_finding.line_number, path)
            types_match = _types_match(bug_finding.bug_type, path.bug_type)

            score, notes = self._score_match(
                bug_finding=bug_finding,
                path=path,
                path_file=path_file,
                line_diff=line_diff,
                types_match=types_match,
            )

            if self._is_better(score, line_diff, best_confidence, best_line_diff):
                best_path = path
                best_confidence = score
                best_line_diff = line_diff
                best_notes = notes

        return CorrelatedResult(
            bug_finding=bug_finding,
            bug_path=best_path,
            correlation_confidence=best_confidence,
            notes="; ".join(best_notes) if best_notes else None,
        )

    def correlate_all(
        self,
        findings: list[BugFindingRef],
        bug_paths: list[BugPath],
    ) -> list[CorrelatedResult]:
        """Correlate multiple bug findings against CSA paths.

        Each finding is matched independently. A single bug path may be
        matched to at most one finding (first-match wins).
        """
        available = list(bug_paths)
        results: list[CorrelatedResult] = []

        for finding in findings:
            result = self.correlate(finding, available)
            results.append(result)
            # Remove matched path from available pool
            if result.bug_path and result.bug_path in available:
                available.remove(result.bug_path)

        return results

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_match(
        bug_finding: BugFindingRef,
        path: BugPath,
        path_file: str,
        line_diff: int | float,
        types_match: bool,
    ) -> tuple[Confidence, list[str]]:
        """Score a single candidate path and return (confidence, notes)."""
        notes: list[str] = []
        file_match = bug_finding.file_path == path_file

        if not file_match:
            # Check if path_file contains the finding file basename
            bug_basename = Path(bug_finding.file_path).name
            if bug_basename and bug_basename in path_file:
                notes.append(f"File matched by basename: {bug_basename}")
                file_match = True

        if not file_match:
            return Confidence.NONE, [
                f"File mismatch: finding={bug_finding.file_path}, csa={path_file}"
            ]

        if line_diff == 0 and types_match:
            return Confidence.HIGH, notes

        if isinstance(line_diff, int) and line_diff <= 3 and types_match:
            return Confidence.MEDIUM, notes

        if isinstance(line_diff, int) and line_diff <= 10:
            notes.append(f"Line diff={line_diff} within tolerance")
            return Confidence.LOW, notes

        if types_match:
            notes.append(f"Types match but line diff={line_diff} exceeds tolerance")
            return Confidence.LOW if isinstance(line_diff, int) and line_diff <= 20 else Confidence.NONE, notes

        return Confidence.NONE, notes

    @staticmethod
    def _is_better(
        candidate: Confidence,
        candidate_diff: int | float,
        current: Confidence,
        current_diff: int | float,
    ) -> bool:
        """Compare confidence scores — is *candidate* better than *current*?"""
        rank = {"high": 3, "medium": 2, "low": 1, "none": 0}
        if rank.get(candidate.value, 0) > rank.get(current.value, 0):
            return True
        if (
            candidate == current
            and isinstance(candidate_diff, (int, float))
            and isinstance(current_diff, (int, float))
        ):
            return candidate_diff < current_diff
        return False
