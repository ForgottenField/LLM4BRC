"""SourceLineConditionExtractor — map source lines to control-flow conditions.

Phase 1 of the branch extraction fix.  Given a source file and line number,
extracts the condition expression from ``if (...)``, ``while (...)``,
``for (...)``, or ``else if (...)`` statements.

This module provides two entry points:

- :func:`extract_condition` — file-based extraction with source_root support.
- :func:`extract_condition_from_text` — pure string extraction (no I/O).

Both return an :class:`ExtractedCondition` with confidence score (0.0–1.0).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Public data model
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ExtractedCondition:
    """Result of extracting a condition from a source line.

    Attributes:
        expression: The condition text (e.g. ``"r >= 0"``).  Empty string
            when no condition could be extracted.
        source_line: The line number the expression was found at.  May
            differ from the input line when backward/forward scanning.
        confidence: 0.0–1.0.  1.0 = exact control-flow line match.
            0.0 = no condition found.
    """

    expression: str = ""
    source_line: int = 0
    confidence: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Core extraction logic
# ═══════════════════════════════════════════════════════════════════════

# Regex: detect control-flow statement start
_CONTROL_KW_RE = re.compile(
    r"(?:^|})\s*(if|else\s+if|while|for)\s*\(",
    re.IGNORECASE,
)


def extract_condition_from_text(text: str) -> str | None:
    """Extract a condition expression from a single line of source code.

    Handles:
    - ``if (r >= 0)`` → ``"r >= 0"``
    - ``while (len > 0) {`` → ``"len > 0"``
    - ``for (int i = 0; i < n; i++)`` → ``"int i = 0; i < n; i++"``
    - ``} else if (z == 0) {`` → ``"z == 0"``
    - Non-control lines (assignments, returns) → ``None``

    Returns:
        The condition expression, or ``None`` if the line is not a
        control-flow statement.
    """
    if not text:
        return None

    stripped = text.strip()

    # Match control keyword
    m = _CONTROL_KW_RE.search(stripped)
    if not m:
        return None

    # Find the opening paren after the keyword
    paren_start = stripped.index("(", m.end() - 1)
    if paren_start < 0:
        return None

    # Find the matching closing paren
    depth = 0
    close_pos = -1
    for i in range(paren_start, len(stripped)):
        ch = stripped[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close_pos = i
                break

    if close_pos < 0:
        return None  # unmatched paren — multi-line case

    expr = stripped[paren_start + 1:close_pos].strip()
    if not expr:
        return None

    return expr


# ═══════════════════════════════════════════════════════════════════════
# File-based extraction
# ═══════════════════════════════════════════════════════════════════════


def extract_condition(
    source_file: str,
    line: int,
    source_root: str = "",
) -> ExtractedCondition:
    """Read ``source_file`` at ``line`` and extract the condition.

    If ``line`` is not a control-flow statement, scans backward up to 5
    lines to find the nearest ``if`` / ``while`` / ``for``.

    Args:
        source_file: Path to the source file (relative or absolute).
        line: 1-based line number to extract from.
        source_root: Optional root directory to resolve relative paths.

    Returns:
        :class:`ExtractedCondition` with the parsed condition.  On
        failure (file not found, no control flow), confidence is 0.0.
    """
    # Resolve the source path
    path = Path(source_file)
    if not path.is_absolute() and source_root:
        path = Path(source_root) / path

    if not path.exists():
        return ExtractedCondition(expression="", source_line=line, confidence=0.0)

    # Read the file
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return ExtractedCondition(expression="", source_line=line, confidence=0.0)

    if line < 1 or line > len(lines):
        return ExtractedCondition(expression="", source_line=line, confidence=0.0)

    zero_based = line - 1

    # Phase 1: Try the requested line as a single-line condition.
    expr = extract_condition_from_text(lines[zero_based])
    if expr is not None:
        return ExtractedCondition(
            expression=expr, source_line=line, confidence=1.0,
        )

    # Phase 2: If the requested line starts a multi-line control statement
    # (e.g. ``if (a ||\n    b ||\n    c) {``), try forward scan to collect
    # the full condition BEFORE falling back to backward scan.  Otherwise
    # the backward scan would match an enclosing single-line ``if`` and
    # return the wrong condition.
    if _CONTROL_KW_RE.search(lines[zero_based].strip()):
        continued = _try_forward_scan(lines, zero_based)
        if continued:
            return continued

    # Phase 3: Backward scan — the requested line is not a control
    # statement; look for the nearest enclosing single-line condition.
    for offset in range(-1, -6, -1):
        idx = zero_based + offset
        if idx < 0:
            break
        expr = extract_condition_from_text(lines[idx])
        if expr is not None:
            return ExtractedCondition(
                expression=expr,
                source_line=idx + 1,
                confidence=0.7,
            )

    # Phase 4: No control keyword found on or before the line — try forward
    # scan from a closing-paren context (e.g. the line is the end of a
    # multi-line condition like ``c) {``).
    if ")" in lines[zero_based]:
        continued = _try_forward_scan(lines, zero_based)
        if continued:
            return continued
    for offset in range(1, 4):
        idx = zero_based + offset
        if idx >= len(lines):
            break
        text = lines[idx]
        if ")" in text and not text.strip().startswith(")"):
            continued = _try_forward_scan(lines, zero_based)
            if continued:
                return continued

    return ExtractedCondition(expression="", source_line=line, confidence=0.0)


def _try_forward_scan(
    lines: list[str],
    start_idx: int,
) -> ExtractedCondition | None:
    """Try to read a multi-line condition by scanning forward from start_idx.

    Looks for ``if (`` or similar on the current or prior line, then
    reads ahead to collect all text until the matching ``)``.
    """
    # Search backward for the control keyword
    search_start = max(0, start_idx - 3)
    for scan_back in range(start_idx, search_start - 1, -1):
        text = lines[scan_back]
        m = _CONTROL_KW_RE.search(text.strip())
        if m:
            # Found the start — collect everything from the opening paren
            # up to the matching closing paren
            paren_start = text.index("(", m.end() - 1)
            combined = text[paren_start:]
            for ahead in range(scan_back + 1, min(scan_back + 10, len(lines))):
                combined += "\n" + lines[ahead]
                depth = 0
                open_idx = -1
                close_idx = -1
                for i, ch in enumerate(combined):
                    if ch == "(":
                        if depth == 0:
                            open_idx = i
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            close_idx = i
                            # Found the matching closing paren
                            break
                if close_idx >= 0:
                    expr = combined[open_idx + 1:close_idx]
                    # Normalise whitespace in the expression
                    expr = re.sub(r"\s+", " ", expr).strip()
                    return ExtractedCondition(
                        expression=expr,
                        source_line=scan_back + 1,
                        confidence=0.8,
                    )
            break

    return None
