"""Structural Dead-Branch Pruning (Layer 0).

When the CSA analyser takes a **FALSE** branch at an ``if`` statement, every
branch inside the *true-block* of that ``if`` is structurally unreachable.
This module provides deterministic identification and pruning of those dead
branches via heuristic brace matching — no LLM required.

Core concepts
-------------
- **Dead range**: the source-line extent ``(start, end)`` of an if-true-block
  whose enclosing condition is known to be FALSE.
- Any branch whose ``source_line`` falls inside a dead range is tagged
  ``contradictory`` (layer 0) and removed from further consideration.

Algorithm (``IfBodyRangeFinder``)
----------------------------------
1. Start at the ``if`` line.  Scan forward collecting lines, counting
   parenthesis depth to locate the ``)`` that closes the if-condition
   (handles multi-line conditions).
2. After the closing ``)``, skip whitespace / comments to find the body
   delimiter: ``{`` (braced block) or a statement token (single-statement
   body).
3. For ``{...}`` bodies: brace-match to find the closing ``}``.
4. Return the 1-based line range of the true-block body.

Integration
-----------
Used as a **pre-pass** before :class:`ConflictChecker` (layers 1–4)::

    from llm_client.fp_analysis.structural_pruning import (
        StructuralPruner,
        build_pruner_from_segments,
    )

    pruner = build_pruner_from_segments(segments, source_path)
    pruner.prune_tree(branch_tree)   # tags dead branches in-place
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from llm_client.fp_analysis.condition_extractor import extract_condition_from_text


# ── Helpers ──────────────────────────────────────────────────────────────────

_COMMENT_RE = re.compile(r"//.*$")
_CTRL_KW_RE = re.compile(r"\b(if|while|for|switch)\s*\(", re.IGNORECASE)


def _strip_comments(line: str) -> str:
    """Remove ``//`` line comments from *line*."""
    return _COMMENT_RE.sub("", line)


def _line_has_code(line: str) -> bool:
    """Return True if *line* contains anything other than whitespace / comments."""
    return bool(_strip_comments(line).strip())


# ── If-body range finder ─────────────────────────────────────────────────────


class IfBodyRangeFinder:
    """Locate the true-block body extent of an ``if`` statement.

    Uses heuristic brace matching on raw source text.  Handles multi-line
    conditions and both braced / single-statement bodies.
    """

    # Number of lines to scan forward for the if-condition's closing ')'
    # (multi-line conditions are rarely longer than this)
    _MAX_CONDITION_LINES = 20

    @staticmethod
    def find_body_range(
        source_lines: list[str],
        if_line_1based: int,
    ) -> Optional[tuple[int, int]]:
        """Return ``(body_start, body_end)`` (1-based, inclusive) or ``None``.

        Uses line-by-line scanning with dynamic expansion so that even
        large if-blocks are handled correctly.

        Args:
            source_lines: The whole source file split into lines.
            if_line_1based: 1-based line number of the ``if`` keyword.
        """
        n = len(source_lines)
        if if_line_1based < 1 or if_line_1based > n:
            return None

        idx = if_line_1based - 1  # → 0-based

        # ── 1. Locate the if-condition closing ')' ──────────────────────────
        # Scan line-by-line, counting paren depth.  Handle multi-line
        # conditions by continuing until depth returns to 0.
        merged = ""
        line_map: list[tuple[int, int]] = []  # (char_offset, 1based_line)
        paren_start = -1
        depth = 0
        close_pos = -1

        for ahead in range(idx, min(idx + IfBodyRangeFinder._MAX_CONDITION_LINES, n)):
            stripped = _strip_comments(source_lines[ahead])
            line_map.append((len(merged), ahead + 1))
            merged += stripped

            if paren_start < 0:
                m = _CTRL_KW_RE.search(merged)
                if m:
                    paren_start = merged.index("(", m.end() - 1)
                    # Count parens from paren_start within current merged text
                    depth = 0
                    for i in range(paren_start, len(merged)):
                        ch = merged[i]
                        if ch == "(":
                            depth += 1
                        elif ch == ")":
                            depth -= 1
                            if depth == 0:
                                close_pos = i
                                break
            else:
                # Continue counting from previous depth
                for i in range(len(merged) - len(stripped), len(merged)):
                    ch = merged[i]
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                        if depth == 0:
                            close_pos = i
                            break

            if close_pos >= 0:
                break

        if close_pos < 0:
            return None  # malformed — closing ) not found

        # ── 2. Find the body delimiter after ')' ────────────────────────────
        after_paren = merged[close_pos + 1:]
        body_start_offset = close_pos + 1

        # Skip whitespace between ')' and '{' or first statement
        brace_pos = after_paren.find("{")
        stmt_pos = -1
        for i, ch in enumerate(after_paren):
            if ch not in " \t\n\r":
                if ch != "{":
                    stmt_pos = i
                break

        if brace_pos >= 0:
            # ── Braced body ────────────────────────────────────────────────
            body_start_offset += brace_pos
            # Start brace-depth scan from the body '{'
            depth = 1
            # First scan within already-merged text
            close_brace = -1
            for i in range(body_start_offset + 1, len(merged)):
                ch = merged[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        close_brace = i
                        break

            # If not found, continue scanning line by line
            next_line = idx + len(line_map)
            while close_brace < 0 and next_line < n:
                stripped = _strip_comments(source_lines[next_line])
                line_map.append((len(merged), next_line + 1))
                merged += stripped
                for i in range(len(merged) - len(stripped), len(merged)):
                    ch = merged[i]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            close_brace = i
                            break
                next_line += 1

            if close_brace < 0:
                return None

            start_line = _offset_to_line(body_start_offset, line_map)
            end_line = _offset_to_line(close_brace, line_map)
            if start_line and end_line:
                return (start_line, end_line)
            return None
        else:
            # ── Single-statement body (no braces) ───────────────────────────
            if stmt_pos < 0:
                return None
            semi_pos = merged.find(";", body_start_offset + stmt_pos)
            # May need to scan forward for ';'
            next_line = idx + len(line_map)
            while semi_pos < 0 and next_line < n:
                stripped = _strip_comments(source_lines[next_line])
                line_map.append((len(merged), next_line + 1))
                merged += stripped
                semi_pos = merged.find(";", body_start_offset + stmt_pos)
                next_line += 1
            if semi_pos < 0:
                return None
            start_line = _offset_to_line(body_start_offset + stmt_pos, line_map)
            end_line = _offset_to_line(semi_pos, line_map)
            if start_line and end_line:
                return (start_line, end_line)
            return None


def _offset_to_line(
    offset: int,
    line_map: list[tuple[int, int]],
) -> Optional[int]:
    """Map a character offset in merged text to a 1-based line number."""
    line = None
    for char_start, ln in line_map:
        if offset >= char_start:
            line = ln
        else:
            break
    return line


# ── Structural Pruner ────────────────────────────────────────────────────────


@dataclass
class DeadRange:
    """A source-line range where branches are structurally dead."""
    start_line: int
    end_line: int
    if_line: int       # the if-statement that caused this dead range
    segment_index: int  # which segment contributed this range
    is_else_chain: bool = False  # True: if was TRUE → else-chain dead; False: if was FALSE → true-block dead


@dataclass
class ImplicitFalse:
    """An "Assuming..." event at a control-flow line that evaluates to FALSE
    but has no corresponding CONTROL event (typical of ``else if`` chains)."""
    if_line: int         # 1-based line of the if/else if/while/for
    event_number: int    # CSA event number
    assumption_text: str  # the raw "Assuming..." description


class StructuralPruner:
    """Accumulates dead ranges from FALSE-branch segments and prunes branches.

    Usage::

        pruner = StructuralPruner()
        for seg in false_segments:
            pruner.add_false_branch(source_lines, seg.line_end, seg.segment_index)
        pruner.prune_tree(branch_tree)
    """

    def __init__(self) -> None:
        self.dead_ranges: list[DeadRange] = []

    def add_false_branch(
        self,
        source_lines: list[str],
        if_line_1based: int,
        segment_index: int = -1,
    ) -> bool:
        """Compute the dead range for a FALSE branch and register it.

        Returns True if a dead range was successfully computed and added.
        """
        body = IfBodyRangeFinder.find_body_range(source_lines, if_line_1based)
        if body is None:
            return False
        self.dead_ranges.append(DeadRange(
            start_line=body[0],
            end_line=body[1],
            if_line=if_line_1based,
            segment_index=segment_index,
        ))
        return True

    def add_dead_range(
        self,
        start_line: int,
        end_line: int,
        if_line: int = 0,
        segment_index: int = -1,
    ) -> None:
        """Register a pre-computed dead range directly.

        Use this for dead ranges not derived from a simple if-FALSE pattern
        (e.g. else-if chains when the outer if is TRUE).
        """
        self.dead_ranges.append(DeadRange(
            start_line=start_line,
            end_line=end_line,
            if_line=if_line,
            segment_index=segment_index,
            is_else_chain=True,
        ))

    def is_dead(self, source_line: int) -> bool:
        """Return True if *source_line* falls inside any registered dead range."""
        return any(
            dr.start_line <= source_line <= dr.end_line
            for dr in self.dead_ranges
        )

    def prune_tree(self, tree) -> int:
        """Walk *tree* and tag branches in dead ranges as contradictory (layer 0).

        Returns the number of branches pruned.
        """
        return self._prune_nodes(tree.root_branches)

    def prune_checked_branches(self, branches: list[dict]) -> int:
        """Walk a flat/recursive list of checked-branch dicts and prune dead ones.

        Returns the number of branches pruned.
        """
        count = 0
        for b in branches:
            if self._prune_dict_node(b):
                count += 1
        return count

    def _prune_nodes(self, nodes: list) -> int:
        count = 0
        for node in nodes:
            if self._prune_single_node(node):
                count += 1
        return count

    def _prune_single_node(self, node) -> bool:
        """Tag one CFGBranchNode if its source_line is in a dead range."""
        pruned = False
        sl = getattr(node, "source_line", 0) or 0
        if sl and self.is_dead(sl):
            # Only override if not already tagged with a stronger verdict
            from llm_client.fp_analysis.cfg_branch_models import BranchTag
            existing = getattr(node, "tag", None)
            if existing is None or existing.verdict == "insufficient_info":
                # Find which dead range this belongs to
                for dr in self.dead_ranges:
                    if dr.start_line <= sl <= dr.end_line:
                        if dr.is_else_chain:
                            reason = (
                                f"Structurally unreachable: if-condition at "
                                f"L{dr.if_line} (segment {dr.segment_index}) "
                                f"is TRUE; else/else-if clause L{dr.start_line}-"
                                f"L{dr.end_line} is dead; branch at L{sl}"
                            )
                        else:
                            reason = (
                                f"Structurally unreachable: if-condition at "
                                f"L{dr.if_line} (segment {dr.segment_index}) "
                                f"is FALSE; branch at L{sl} is inside "
                                f"true-block L{dr.start_line}-L{dr.end_line}"
                            )
                        node.tag = BranchTag(
                            node.node_id, "contradictory", layer=0,
                            reason=reason,
                        )
                        pruned = True
                        break
        # Recurse into children
        for child in getattr(node, "children", []):
            if self._prune_single_node(child):
                pruned = True
        return pruned

    def _prune_dict_node(self, node: dict) -> bool:
        """Tag one branch dict if its source_line is in a dead range."""
        pruned = False
        sl = node.get("source_line", 0) or 0
        current_verdict = node.get("verdict", "")
        if sl and self.is_dead(sl) and current_verdict in ("insufficient_info", "no_tag", ""):
            for dr in self.dead_ranges:
                if dr.start_line <= sl <= dr.end_line:
                    if dr.is_else_chain:
                        reason = (
                            f"Structurally unreachable: if-condition at "
                            f"L{dr.if_line} (segment {dr.segment_index}) "
                            f"is TRUE; else/else-if clause L{dr.start_line}-"
                            f"L{dr.end_line} is dead; branch at L{sl}"
                        )
                    else:
                        reason = (
                            f"Structurally unreachable: if-condition at "
                            f"L{dr.if_line} (segment {dr.segment_index}) "
                            f"is FALSE; branch at L{sl} is inside "
                            f"true-block L{dr.start_line}-L{dr.end_line}"
                        )
                    node["verdict"] = "contradictory"
                    node["tag"] = {
                        "verdict": "contradictory",
                        "layer": 0,
                        "reason": reason,
                    }
                    pruned = True
                    break
        for child in node.get("children", []):
            if self._prune_dict_node(child):
                pruned = True
        return pruned


# ── Implicit FALSE branch detection from "Assuming..." events ────────────────
# When CSA encounters an if/else if chain, it reports ONE "Taking false branch"
# for the entire chain (at the first ``if``) and uses "Assuming..." events to
# explain why each subsequent ``else if`` also evaluates to FALSE.  These
# implicit FALSE branches must be detected and their true-blocks registered as
# dead ranges.


# Patterns for parsing "Assuming..." event descriptions
_ASSUMING_CONDITION_FALSE_RE = re.compile(r"condition is false", re.IGNORECASE)
_ASSUMING_FIELD_NOT_EQ_RE = re.compile(
    r"Assuming field '(\w+)' is not equal to (\w+)", re.IGNORECASE,
)
_ASSUMING_FIELD_LE_RE = re.compile(
    r"Assuming field '(\w+)' is <=\s*(\w+)", re.IGNORECASE,
)
_ASSUMING_FIELD_EQ_RE = re.compile(
    r"Assuming field '(\w+)' is equal to (\w+)", re.IGNORECASE,
)
# Line-level check for control-flow keyword (not necessarily with '(' on the
# same line — ``else if`` may have the paren on the same line or next).
_IMPLICIT_CTRL_KW_RE = re.compile(
    r"\b(if|else\s+if|while|for)\b", re.IGNORECASE,
)


def _is_condition_falsified(condition_expr: str, assumption_text: str) -> bool:
    """Return True if *assumption_text* definitively contradicts *condition_expr*.

    Conservative: returns False when uncertain (avoids false positives).
    """
    # "Assuming the condition is false" → always FALSE
    if _ASSUMING_CONDITION_FALSE_RE.search(assumption_text):
        return True

    # "Assuming field 'X' is not equal to Y" → look for X == Y in condition
    m = _ASSUMING_FIELD_NOT_EQ_RE.search(assumption_text)
    if m:
        field = m.group(1)
        value = m.group(2)
        eq_pat = re.compile(
            r'\b' + re.escape(field) + r'\s*==\s*' + re.escape(value) + r'\b',
        )
        if eq_pat.search(condition_expr):
            return True
        # Also check bare boolean: condition is just "field" and value is "0"
        # "field is not equal to 0" → field != 0 → TRUE (not falsified)
        return False

    # "Assuming field 'X' is <= N" → look for X > N in condition
    m = _ASSUMING_FIELD_LE_RE.search(assumption_text)
    if m:
        field = m.group(1)
        value = m.group(2)
        gt_pat = re.compile(
            r'\b' + re.escape(field) + r'\s*>\s*' + re.escape(value) + r'\b',
        )
        if gt_pat.search(condition_expr):
            return True
        return False

    # "Assuming field 'X' is equal to Y" → look for X != Y in condition
    m = _ASSUMING_FIELD_EQ_RE.search(assumption_text)
    if m:
        field = m.group(1)
        value = m.group(2)
        ne_pat = re.compile(
            r'\b' + re.escape(field) + r'\s*!=\s*' + re.escape(value) + r'\b',
        )
        if ne_pat.search(condition_expr):
            return True
        return False

    return False


def _path_skips_if_body(
    if_line: int,
    body_range: tuple[int, int],
    path_events: list[dict],
    event_index: int,
) -> bool:
    """Return True if subsequent path events skip over the if-true-block body.

    When the CSA path emits "Assuming..." events at an if-condition line but
    the very next path event lands *after* the true-block body, the condition
    must have evaluated to FALSE — the body was never entered.

    This catches cases where individual "Assuming" events do not textually
    contradict the condition expression, but the path continuation behaviour
    proves the condition is FALSE.
    """
    body_start, body_end = body_range

    for j in range(event_index + 1, len(path_events)):
        next_ln = path_events[j].get("line_number", 0)
        if next_ln < 1:
            continue
        # Still inside the condition span (multi-line conditions) — keep looking
        if next_ln < body_start:
            continue
        # Path entered the body → condition was TRUE, not falsified
        if next_ln <= body_end:
            return False
        # Path jumped past the body → condition was FALSE
        return True

    # Reached end of events without evidence either way
    return False


def find_implicit_false_branches(
    path_events: list[dict],
    source_lines: list[str],
) -> list[ImplicitFalse]:
    """Scan "Assuming..." events for control-flow lines whose condition is FALSE
    but have no corresponding CONTROL event at the same line.

    Uses two detection methods:

    1. **Textual falsification**: the "Assuming..." text directly contradicts
       the condition expression (e.g. "field X is not equal to 0" vs ``X == 0``).
    2. **Path-continuation**: subsequent path events skip over the true-block
       body entirely, proving the condition was FALSE even when individual
       assumption texts are inconclusive.
    """
    # Build set of line numbers that HAVE a control event
    control_lines: set[int] = set()
    for evt in path_events:
        if evt.get("event_kind") == "control":
            ln = evt.get("line_number", 0)
            if ln:
                control_lines.add(ln)

    implicit: list[ImplicitFalse] = []
    seen_if_lines: set[int] = set()  # dedup: one ImplicitFalse per if-line

    for idx, evt in enumerate(path_events):
        if evt.get("event_kind") != "event":
            continue
        desc = evt.get("description", "")
        if "assuming" not in desc.lower():
            continue
        ln = evt.get("line_number", 0)
        if ln < 1 or ln > len(source_lines):
            continue

        # Must be at a control-flow line
        src_line = source_lines[ln - 1]
        if not _IMPLICIT_CTRL_KW_RE.search(_strip_comments(src_line)):
            continue

        # Must NOT already have a CONTROL event at this line
        if ln in control_lines:
            continue

        # Dedup: one ImplicitFalse per if-line (multi-line conditions may
        # produce multiple "Assuming" events at the same if-line)
        if ln in seen_if_lines:
            continue

        # ── Method 1: textual falsification ──────────────────────────────────
        # Requires a successfully extracted condition expression (single-line
        # conditions only; multi-line conditions return None).
        is_false = False
        cond = extract_condition_from_text(src_line)
        if cond is not None and _is_condition_falsified(cond, desc):
            is_false = True

        # ── Method 2: path-continuation ──────────────────────────────────────
        # Works even when condition extraction fails (multi-line conditions).
        # If the next path event after the condition lands past the true-block
        # body, the condition must be FALSE.
        if not is_false:
            body_range = IfBodyRangeFinder.find_body_range(source_lines, ln)
            if body_range is not None and _path_skips_if_body(
                ln, body_range, path_events, idx,
            ):
                is_false = True

        if is_false:
            seen_if_lines.add(ln)
            implicit.append(ImplicitFalse(
                if_line=ln,
                event_number=evt.get("path_number", 0),
                assumption_text=desc,
            ))

    return implicit


# ── Short-circuit OR / AND detection ─────────────────────────────────────────
# When the CSA path includes an "Assuming..." event that confirms the first
# operand of an ``||`` chain is TRUE (or the first operand of an ``&&`` chain
# is FALSE), the remaining operands are *short-circuited* — they are never
# evaluated at runtime.  Sub-expression PDG nodes inside those later operands
# should be pruned.

# Patterns that indicate a TRUE value in "Assuming..." event descriptions.
# Order matters: _ASSUMING_FALSE_PATTERNS is checked first to avoid "is equal to 0"
# being mis-classified as TRUE by the generic "is equal to" pattern.
_ASSUMING_TRUE_PATTERNS = [
    re.compile(r"is non-null", re.IGNORECASE),
    re.compile(r"is not equal to 0", re.IGNORECASE),
    re.compile(r"condition is true", re.IGNORECASE),
    # "is equal to <non-zero>" — must exclude 0 and NULL
    re.compile(r"is equal to (?!0\b|null\b|false\b)", re.IGNORECASE),
]

# Patterns that indicate a FALSE value (checked before TRUE patterns).
_ASSUMING_FALSE_PATTERNS = [
    re.compile(r"is equal to 0", re.IGNORECASE),
    re.compile(r"is null", re.IGNORECASE),
    re.compile(r"condition is false", re.IGNORECASE),
    re.compile(r"is not equal to (?!0\b)", re.IGNORECASE),  # "not equal to <non-zero>" → variable differs from expected → could be 0
]


@dataclass
class ShortCircuitedBranch:
    """A PDG sub-expression node that is short-circuited."""
    source_line: int        # 1-based line of the sub-expression
    chain_start_line: int   # 1-based line of the OR/AND chain start
    event_number: int       # CSA event that proves short-circuit
    chain_type: str         # "or" | "and"


def find_short_circuited_branches(
    path_events: list[dict],
    source_lines: list[str],
    event_range: tuple[int, int] | None = None,
) -> list[ShortCircuitedBranch]:
    """Find sub-expression nodes that are short-circuited by prior operands.

    Scans "Assuming..." events within *event_range* (or all events if None).
    When an event confirms the first operand of an ``||`` chain is TRUE (or
    the first operand of an ``&&`` chain is FALSE), all subsequent operands
    in the chain are short-circuited.

    Returns a list of :class:`ShortCircuitedBranch` records, one per
    short-circuited operand line.
    """
    start_evt = event_range[0] if event_range else 1
    end_evt = event_range[1] if event_range else 999999

    # Collect "Assuming..." events in range
    assumed: list[dict] = []
    for evt in path_events:
        pn = evt.get("path_number", 0)
        if start_evt <= pn <= end_evt:
            if evt.get("event_kind") == "event":
                if "assuming" in evt.get("description", "").lower():
                    assumed.append(evt)

    results: list[ShortCircuitedBranch] = []
    n = len(source_lines)

    for evt in assumed:
        evt_line = evt.get("line_number", 0)
        if evt_line < 1 or evt_line > n:
            continue

        desc = evt.get("description", "")

        # Determine if the assumption implies TRUE or FALSE.
        # FALSE patterns checked first — "is equal to 0" must not be
        # mis-classified as TRUE by the generic "is equal to" pattern.
        implies_false = any(p.search(desc) for p in _ASSUMING_FALSE_PATTERNS)
        implies_true = any(p.search(desc) for p in _ASSUMING_TRUE_PATTERNS) if not implies_false else False
        if not implies_true and not implies_false:
            continue

        # Check if the event line is at an OR/AND chain start
        src = _strip_comments(source_lines[evt_line - 1])
        chain_type = ""
        if "||" in src:
            chain_type = "or"
        elif "&&" in src:
            chain_type = "and"
        else:
            continue  # Not part of a short-circuit-able expression

        # Short-circuit condition: OR + first-op TRUE, or AND + first-op FALSE
        if chain_type == "or" and not implies_true:
            continue
        if chain_type == "and" and not implies_false:
            continue

        # Scan forward from evt_line to find subsequent ||/&& operands.
        # The last operand in the chain may NOT contain ||/&& — it ends
        # with the closing ``)``.  We detect it as the line immediately
        # following a || / && line that is not itself a new statement.
        prev_had_op = False
        for ahead in range(evt_line, min(evt_line + 10, n)):
            line_src = _strip_comments(source_lines[ahead])
            line_num = ahead + 1

            if line_num == evt_line:
                # First operand line — record whether it continues
                prev_had_op = chain_type == "or" and "||" in line_src
                if not prev_had_op:
                    prev_had_op = chain_type == "and" and "&&" in line_src
                continue

            has_op = (chain_type == "or" and "||" in line_src) or \
                     (chain_type == "and" and "&&" in line_src)

            if has_op:
                results.append(ShortCircuitedBranch(
                    source_line=line_num,
                    chain_start_line=evt_line,
                    event_number=evt.get("path_number", 0),
                    chain_type=chain_type,
                ))
                prev_had_op = True
            elif prev_had_op:
                # Immediate successor of a ||/&& line: last operand.
                # It does NOT contain the operator itself but is still
                # part of the chain (e.g. ``L689: opcode == PING) {``
                # following ``L688: ... CONNECTION_CLOSE ||``).
                if not _CTRL_KW_RE.match(line_src):
                    results.append(ShortCircuitedBranch(
                        source_line=line_num,
                        chain_start_line=evt_line,
                        event_number=evt.get("path_number", 0),
                        chain_type=chain_type,
                    ))
                break
            else:
                break  # left the chain

    return results


# ── Else-chain dead range detection (TRUE if → dead else if/else) ─────────
# When an ``if`` condition is known to be TRUE, all subsequent ``else if`` and
# ``else`` clauses in the same chain are structurally unreachable.  This
# function detects those clauses and returns their line extents so their
# true-blocks can be registered as dead ranges.


def _find_matching_brace_line(
    source_lines: list[str],
    open_brace_line: int,
    open_brace_col: int = 0,
) -> int | None:
    """Find the 1-based line number of the '}' matching a '{'.

    Args:
        source_lines: Source file lines.
        open_brace_line: 1-based line containing the opening '{'.
        open_brace_col: 0-based column of the '{' (0 = first char).

    Returns:
        1-based line of matching '}', or None if not found.
    """
    depth = 0
    for i in range(open_brace_line - 1, len(source_lines)):
        stripped = _strip_comments(source_lines[i])
        for j, ch in enumerate(stripped):
            if i == open_brace_line - 1 and j < open_brace_col:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return i + 1
    return None


# Regex: matches '}' followed by optional whitespace then 'else'
_ELSE_AFTER_BRACE_RE = re.compile(r'\}\s*else\b')
# Regex: matches 'else' followed by optional whitespace then 'if'
_ELSE_IF_RE = re.compile(r'\belse\s+if\b', re.IGNORECASE)
_ELSE_BARE_RE = re.compile(r'\belse\b', re.IGNORECASE)


def find_else_chain_for_true_if(
    source_lines: list[str],
    if_line: int,
) -> list[tuple[int, int]]:
    """Find dead ranges for else/else-if clauses when *if_line* is known TRUE.

    When an ``if`` condition takes the TRUE branch, every ``else if`` and
    ``else`` clause belonging to the same if-chain is structurally
    unreachable.  This function returns ``(dead_start, dead_end)`` line
    ranges (inclusive, 1-based) for each clause.  The ranges include the
    keyword line and the full body, so that both the condition-check branch
    and any nested branches inside the body are pruned.

    Args:
        source_lines: Source file split by line.
        if_line: 1-based line of the ``if`` keyword known to be TRUE.

    Returns:
        List of ``(start, end)`` dead ranges.  Empty if no else-chain exists.
    """
    n = len(source_lines)
    if if_line < 1 or if_line > n:
        return []

    # 1. Find the if-true-block — we need its closing '}'
    body = IfBodyRangeFinder.find_body_range(source_lines, if_line)
    if body is None:
        return []
    body_end = body[1]  # 1-based line of '}'

    # 2. Find the 'else' keyword after the if-true-block's closing '}'
    # The '}' may share a line with 'else', or 'else' may be on the next line.
    else_keyword_line = -1
    else_keyword_col = -1
    for ahead in range(body_end - 1, min(body_end + 2, n)):
        stripped = _strip_comments(source_lines[ahead])
        m = _ELSE_AFTER_BRACE_RE.search(stripped)
        if m:
            # The 'else' starts after the '}'
            else_keyword_line = ahead + 1
            # Find 'else' within the match
            else_pos = m.group().find('else')
            else_keyword_col = m.start() + else_pos
            break

    if else_keyword_line < 0:
        return []

    # 3. Scan forward, tracking the else-chain.
    # Each clause in the chain follows the pattern:
    #   } else if (...) { body }   or   } else { body }
    # The chain ends when we hit a '}' that is NOT followed by another 'else'.
    dead_ranges: list[tuple[int, int]] = []
    current_line = else_keyword_line

    while current_line <= n and else_keyword_line > 0:
        kw_line = else_keyword_line
        stripped = _strip_comments(source_lines[kw_line - 1])

        # Find the opening '{' for this clause's body
        # It may be on the keyword line or a subsequent line
        open_brace_line = -1
        open_brace_col = -1
        for scan in range(kw_line - 1, min(kw_line + IfBodyRangeFinder._MAX_CONDITION_LINES, n)):
            scan_stripped = _strip_comments(source_lines[scan])
            brace_idx = scan_stripped.find('{')
            if brace_idx >= 0:
                open_brace_line = scan + 1
                open_brace_col = brace_idx
                break

        if open_brace_line < 0:
            break

        # Find matching closing '}'
        close_brace_line = _find_matching_brace_line(
            source_lines, open_brace_line, open_brace_col,
        )
        if close_brace_line is None:
            break

        # The dead range runs from the keyword line to the closing '}'
        dead_ranges.append((kw_line, close_brace_line))

        # Check if there's another 'else' after this clause's '}'
        next_else_line = -1
        for scan in range(close_brace_line - 1, min(close_brace_line + 2, n)):
            scan_stripped = _strip_comments(source_lines[scan])
            m = _ELSE_AFTER_BRACE_RE.search(scan_stripped)
            if m:
                next_else_line = scan + 1
                else_pos = m.group().find('else')
                else_keyword_col = m.start() + else_pos
                break

        else_keyword_line = next_else_line
        if else_keyword_line < 0:
            break

    return dead_ranges


# ── Factory ──────────────────────────────────────────────────────────────────


def build_pruner_from_segments(
    segments: list,
    source_path: str,
    path_events: list[dict] | None = None,
) -> StructuralPruner:
    """Build a :class:`StructuralPruner` from all FALSE-branch segments.

    When *path_events* is provided, also scans "Assuming..." events for
    implicit FALSE branches at ``else if`` / ``while`` / ``for`` lines that
    lack a corresponding CONTROL event (see :func:`find_implicit_false_branches`).

    Args:
        segments: List of ``SegmentInfo`` objects (from ``segment_path()``).
        source_path: Resolved local path to the source file.
        path_events: Optional list of parsed CSA path event dicts (from
            ``ParsedReport.path_events``).  When provided, implicit FALSE
            branches from "Assuming..." events are registered.

    Returns:
        A configured ``StructuralPruner`` ready for ``prune_tree()``.
    """
    path = Path(source_path)
    if not path.exists():
        return StructuralPruner()

    source_lines = path.read_text().splitlines()
    pruner = StructuralPruner()

    # ── Track which if-lines we've already processed for else-chain ─────
    seen_if_lines: set[int] = set()

    for seg in segments:
        if seg.control_branch_taken is False:
            # FALSE-branch: true-block is dead (existing logic)
            pruner.add_false_branch(
                source_lines,
                seg.line_end,
                segment_index=seg.segment_index,
            )

        elif seg.control_branch_taken is True:
            # TRUE-branch: if this is an ``if``, its else-chain is dead.
            # Only process each if-line once (a single if can appear as the
            # boundary of multiple consecutive segments).
            if_line = seg.line_end
            if if_line in seen_if_lines:
                continue
            src_line = source_lines[if_line - 1] if 1 <= if_line <= len(source_lines) else ""
            if not _CTRL_KW_RE.search(_strip_comments(src_line)):
                continue
            seen_if_lines.add(if_line)

            else_ranges = find_else_chain_for_true_if(source_lines, if_line)
            for dead_start, dead_end in else_ranges:
                pruner.add_dead_range(
                    dead_start, dead_end,
                    if_line=if_line,
                    segment_index=seg.segment_index,
                )

    # ── Implicit FALSE branches from "Assuming..." events ─────────────────────
    if path_events:
        implicit = find_implicit_false_branches(path_events, source_lines)
        for imp in implicit:
            # Find which segment this event belongs to
            seg_idx = -1
            for seg in segments:
                start_evt, end_evt = seg.event_range
                if start_evt <= imp.event_number <= end_evt:
                    seg_idx = seg.segment_index
                    break

            pruner.add_false_branch(
                source_lines,
                imp.if_line,
                segment_index=seg_idx,
            )

    return pruner
