"""LLM-powered FP report path analyzer — identifies gaps, completes paths, generates POCs."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from llm_client.base import LLMConfig, LLMRequest, LLMProvider
from llm_client.factory import ProviderFactory

from llm_client.fp_analysis.compile_checker import CompileChecker, CompileResult
from llm_client.fp_analysis.condition_extractor import extract_condition
from llm_client.fp_analysis.html_parser import FPReportParser
from llm_client.fp_analysis.models import (
    AnalysisResult,
    CallChainEntry,
    Classification,
    CompletedPath,
    CompletedStep,
    Confidence,
    FPGap,
    FPGapType,
    GapSeverity,
    LifecycleAssessment,
    ParsedReport,
    PathAnalysis,
    POCPlan,
)
from llm_client.fp_analysis.prompt_templates import (
    CFG_FEASIBILITY_PROMPT,
    CFG_FEASIBILITY_SYSTEM_PROMPT,
    COMPILE_FIX_PROMPT,
    COMPILE_FIX_SYSTEM_PROMPT,
    CONSTRAINT_COMPLETION_PROMPT,
    CONSTRAINT_COMPLETION_SYSTEM_PROMPT,
    DATA_FLOW_PHASE1_DEFAULT,
    DATA_FLOW_PHASE1_MEMORY_LEAK,
    DATA_FLOW_SECTION_DEFAULT,
    DATA_FLOW_SECTION_MEMORY_LEAK,
    EARLY_FEASIBILITY_PROMPT,
    EARLY_FEASIBILITY_SYSTEM_PROMPT,
    SEMANTIC_FEASIBILITY_PROMPT,
    SEMANTIC_FEASIBILITY_SYSTEM_PROMPT,
    ERROR_START_LEGEND_DEFAULT,
    ERROR_START_LEGEND_MEMORY_LEAK,
    FEASIBILITY_CHECK_PROMPT,
    FEASIBILITY_CHECK_SYSTEM_PROMPT,
    GAP_ANALYSIS_PROMPT,
    GAP_ANALYSIS_SYSTEM_PROMPT,
    OOM_RULES,
    OOM_TRIGGER_INSTRUCTION,
    PATH_COMPLETION_PROMPT,
    PATH_COMPLETION_SYSTEM_PROMPT,
    PATH_MERGE_CHECK_PROMPT,
    PATH_MERGE_CHECK_SYSTEM_PROMPT,
    POC_GENERATION_PROMPT,
    POC_GENERATION_SYSTEM_PROMPT,
    SEGMENT_PATH_SELECTION_PROMPT,
    SEGMENT_PATH_SELECTION_SYSTEM_PROMPT,
    SINGLE_PATH_FEASIBILITY_PROMPT,
    SINGLE_PATH_FEASIBILITY_SYSTEM_PROMPT,
    STATE_LIFECYCLE_PROMPT,
    STATE_LIFECYCLE_SYSTEM_PROMPT,
    VERIFICATION_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)
from llm_client.fp_analysis.source_context import SourceContextExtractor
from llm_client.fp_analysis.summary_cache import SummaryCache
from llm_client.fp_analysis.pdg_augment import augment_pdg_sdg
from llm_client.fp_analysis.pdg_loader import load_pdg
from llm_client.fp_analysis.pdg_models import SDG
from llm_client.fp_analysis.domain_facts import (
    domain_facts_from_lines,
    format_domain_facts,
    _is_macro_noise,
)
from llm_client.fp_analysis.state_lifecycle import CallChainExtractor
from llm_client.fp_analysis.slicer import SlicerConfig, compute_slice_from_seeds
from llm_client.fp_analysis.path_seed_extractor import (
    build_caller_map,
    build_path_annotations,
    extract_seeds_from_bug_path,
)
from llm_client.fp_analysis.code_reducer import reduce_code

from llm_client.fp_analysis.slice_mask import SliceMask
from llm_client.fp_analysis.iterative_selector import IterativePathSelector

# CFG-based feasibility analysis
from llm_client.fp_analysis.cfg_feasibility import (
    _get_function_summary,
    extract_assumptions_from_events,
    run_feasibility_analysis,
    segment_path,
)
from llm_client.fp_analysis.cfg_models import (
    AssumptionFact,
    AssumptionVerdict,
    CFGAnalysisResult,
)

# CFG branch filter integration (Step 3+4)
from llm_client.fp_analysis.cfg_branch_models import (
    CFGBranchTree,
    Postcondition,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "deepseek-chat"

# Project-specific hints for LLM prompts (compile-fix and POC generation)
_PROJECT_HINTS: dict[str, dict[str, str]] = {
    "default": {
        "compile_fix": """- Target project: unknown project
- Compiler: clang++-14 with -std=c++17
- Test framework: gtest
- Compilation mode: -fsyntax-only (type/syntax check only, no linking)
- Fix includes, types, namespaces, and API usage as needed.""",
        "poc_context": """Key headers: project-specific headers and standard C++ headers.""",
    },
    "folly": {
        "compile_fix": """- Target project: folly (Facebook Open Source Library)
- Compiler: clang++-14 with -std=c++17
- Test framework: gtest
- Compilation mode: -fsyntax-only (type/syntax check only, no linking)
- Include paths available: folly/, gtest/, standard C++ headers
- For Endian conversion: use `#include <folly/lang/Bits.h>` (NOT `folly/Endian.h` which does not exist)
- For `testing::MatcherBase`: it lives in `testing::internal::MatcherBase` (in `/usr/include/gtest/gtest-matchers.h`)
- For `RWPrivateCursor`: it is `folly::io::RWCursor<folly::io::CursorAccess::PRIVATE>` defined in `folly/io/Cursor.h`
- For `SocketAddress::setFromLocalAddr`: this method is **private** (accessible only within the class). Use the public wrapper `SocketAddress::setFromLocalAddress(NetworkSocket)` instead, or call through a public function that internally invokes `setFromLocalAddr`.
- For `testing::MatcherBase<T>`: this class has **protected** constructor/destructor. You cannot instantiate it directly. Use `testing::Matcher<T>` (which inherits from MatcherBase) or `testing::internal::MatcherBase<T>` through a valid subclass like `testing::Matcher<T>`.""",
        "poc_context": """folly (Facebook Open Source Library). Use `#include <folly/...>` paths.
Key headers: folly/SocketAddress.h, folly/io/Cursor.h, folly/lang/Bits.h.
Test framework: gtest with `#include <gtest/gtest.h>`.""",
    },
    "aria2": {
        "compile_fix": """- Target project: aria2 (lightweight multi-protocol download utility)
- Compiler: clang++-14 with -std=c++17
- Test framework: gtest or standalone main()
- Compilation mode: -fsyntax-only (type/syntax check only, no linking)
- Include paths: deps/wslay/lib/includes/ (for wslay headers)
- Define WSLAY_VERSION="1.1.0" to avoid missing wslayver.h
- For wslay structures: struct wslay_event_msg_source has a union with field "source"
- wslay_event_queue_fragmented_msg_ex takes: ctx, msg_type, source, user_data
- wslay_queue_push returns: 0 on success, non-zero (WSLAY_ERR_NOMEM) on malloc failure
- The opaque function wslay_queue_push internally calls wslay_queue_cell which calls malloc
- wslay_event_recv(ctx) takes ONLY 1 argument: wslay_event_context_ptr ctx
- wslay_event_recv does NOT take data/buffer arguments — it reads WS frames through callbacks.recv_callback
- wslay_event_send(ctx) also takes ONLY 1 argument: wslay_event_context_ptr ctx""",
        "poc_context": """aria2 project with wslay WebSocket library. Use `#include <wslay/wslay.h>` paths.
Key headers: wslay/wslay.h.
Define WSLAY_VERSION="1.1.0" before including wslay headers.
For wslay callbacks: the on_msg_recv callback signature is:
  int callback(wslay_event_context_ptr ctx, const struct wslay_event_on_msg_recv_arg *arg, void *user_data);
struct wslay_event_msg_source is used for event sources (union with "source" field).
IMPORTANT API signatures:
- wslay_event_recv(wslay_event_context_ptr ctx) — takes ONLY ctx (NO data/buffer args). It reads WS frames from callbacks.recv_callback internally.
- wslay_event_send(wslay_event_context_ptr ctx) — takes ONLY ctx. It sends queued messages via callbacks.send_callback.
- The recv_callback you set in struct wslay_event_callbacks feeds raw WS frame bytes to the library.
- wslay_event_context_server_init(ctx, &callbacks, user_data) — callbacks must be non-null, all fields zero-initialized.
Test with standalone main() or gtest.""",
    },
}


# ---------------------------------------------------------------------------
# Step 3.5 helpers — deterministic anchor-vs-constraint feasibility check
# ---------------------------------------------------------------------------


def _find_variable_in_condition(condition: str) -> str | None:
    """Extract the primary variable name from a branch condition expression.

    Handles:
    - Simple names: ``"blockLength_ <= 0"`` → ``"blockLength_"``
    - Member access: ``"obj->field > 5"`` → ``"obj->field"``
    - Pointer: ``"*ptr == 0"`` → ``"ptr"``
    - Function calls: ``"func(arg) == true"`` → ``None`` (too complex)

    Returns ``None`` when extraction is unreliable (fall back to LLM).
    """
    if not condition:
        return None

    # Strip leading negation operators
    text = condition.lstrip("! ")

    # Try to find a simple pattern: name [operator [value]]
    # The variable name is usually the first identifier before a comparison
    # operator, not preceded by a function call.
    import re

    # Heuristic: match first identifier token before a comparison operator,
    # but NOT if preceded by a function call pattern.
    call_match = re.match(r'^[a-zA-Z_]\w*\s*\(', text)
    if call_match:
        return None  # function call — too complex

    # Match: optionally-dereferenced qualified name + operator
    m = re.match(
        r'^(\*?\s*[a-zA-Z_][\w.]*(?:->[a-zA-Z_][\w.]*)*)\s*(==|!=|<=|>=|<|>)\s',
        text,
    )
    if m:
        return m.group(1).strip().lstrip("*").strip()

    # Simple boolean variable (just a name, no operator)
    # Only return if it looks like a C++ identifier
    simple = re.match(r'^[a-zA-Z_]\w*$', text)
    if simple:
        return None  # Too simple — could be a flag, but no comparison to anchor

    return None


def _close_open_structures(s: str) -> str:
    """Append the closing ``}``/``]`` needed to balance a (possibly truncated)
    JSON string, respecting string literals and escapes.  Used to salvage an
    LLM response that was cut off at the token limit."""
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
    return s + "".join("}" if ch == "{" else "]" for ch in reversed(stack))


def _repair_truncated_json(text: str) -> str | None:
    """Best-effort repair of a JSON document truncated mid-output (typically an
    unterminated string at the token limit).  Returns the shortest repaired
    string that parses, or ``None`` if unrecoverable.

    Strategy: find where ``json.loads`` first failed, then try progressively
    earlier structural cut points (the unterminated-string opener first, then
    surrounding ``, { [ } ]``), rebalancing braces each time, until a prefix
    parses."""
    if not text:
        return None
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    try:
        json.loads(text)
    except json.JSONDecodeError as ex:
        epos = ex.pos
    else:
        epos = len(text)

    # Candidate cut points, best-effort first (nearest the error, going back).
    cuts: list[int] = []
    seen: set[int] = set()

    def _add(i: int) -> None:
        if 0 < i < len(text) and i not in seen:
            seen.add(i)
            cuts.append(i)

    _add(epos)
    for i in range(min(epos, len(text)) - 1, -1, -1):
        if text[i] in ",{[]}" and (i == 0 or text[i - 1] != "\\"):
            _add(i)
        if len(cuts) > 250:
            break
    cuts.sort(reverse=True)

    for cut in cuts:
        prefix = text[:cut].rstrip().rstrip(",").rstrip()
        # If a value was cut right after its key's colon, the prefix ends with
        # a dangling ``"key":`` — drop that trailing key so the object closes
        # cleanly instead of leaving a key with no value.
        if prefix.endswith(":"):
            m = re.search(r'"([^"\\]|\\.)*"\s*$', prefix.rstrip()[:-1].rstrip())
            if m:
                prefix = prefix[: m.start()].rstrip()
        closed = _close_open_structures(prefix)
        try:
            json.loads(closed)
            return closed
        except json.JSONDecodeError:
            continue
    return None


def _evaluate_constraint_against_branch(
    anchor_branch: bool,
    condition: str,
    constraint_str: str,
) -> tuple[str, str]:
    """Check whether a concrete constraint is compatible with a branch decision.

    Args:
        anchor_branch: ``True`` = branch taken is TRUE, ``False`` = FALSE.
        condition: The condition expression, e.g. ``"blockLength_ <= 0"``.
        constraint_str: The concretized constraint value, e.g. ``"> 0"``.

    Returns:
        ``("consistent", msg)`` — constraint supports the branch direction.
        ``("contradictory", msg)`` — constraint contradicts the branch direction.
        ``("insufficient_info", msg)`` — cannot determine (fall back to LLM).
    """
    import re

    # Normalise
    cond_stripped = condition.strip()
    constraint_norm = constraint_str.strip()

    # Extract comparison operator and value from condition
    # Patterns: "var [=!]= value", "var <[=] value", "var >[=] value"
    cond_match = re.match(
        r'^[a-zA-Z_][\w.]*(?:->[a-zA-Z_][\w.]*)*\s*(==|!=|<=|>=|<|>)\s*(.+)$',
        cond_stripped,
    )
    if not cond_match:
        return ("insufficient_info", f"Cannot parse condition: {condition}")

    cond_op = cond_match.group(1)
    cond_rhs = cond_match.group(2).strip().rstrip(".")

    # Extract comparison from constraint string.
    # Handle four forms:
    #   1. "> 0"                   — standard range constraint
    #   2. "> 0, default 1M"       — range with extras (parse first token)
    #   3. ">= 10 (from config)"   — range with parenthetical
    #   4. "42"                    — plain numeric (treat as "== 42")
    #   5. "allocated"             — non-numeric (LLM fallback)
    constr_match = re.match(
        r'^\s*(!=|==|<=|>=|<|>)\s*(.+)$',
        constraint_norm,
    )
    if not constr_match:
        # Check for plain numeric value (treat as "== value")
        plain_num = re.match(r'^\s*(\d+(?:\.\d+)?)\s*$', constraint_norm)
        if plain_num:
            constr_op = "=="
            constr_rhs = plain_num.group(1).strip()
        else:
            # Non-numeric constraint like "allocated", "non-null pointer"
            return _check_non_numeric_constraint(
                anchor_branch, cond_op, cond_rhs, constraint_norm,
            )
    else:
        constr_op = constr_match.group(1)
        constr_rhs = constr_match.group(2).strip().rstrip(".")

        # Strip parenthetical and trailing extras from RHS
        # e.g. "0, default 1M" → "0", "0 (from config)" → "0"
        constr_rhs = re.sub(
            r'[,\s(].*$', '', constr_rhs
        ).strip()

    # Parse both sides numerically — if both succeed, do range-overlap check.
    # If one side is non-numeric, fall back to heuristic.
    def _to_num(s: str):
        try:
            return float(s)
        except ValueError:
            return None

    cond_val = _to_num(cond_rhs)
    constr_val = _to_num(constr_rhs)

    if cond_val is None or constr_val is None:
        return ("insufficient_info",
                f"Cannot parse numeric values: condition '{cond_rhs}' vs "
                f"constraint '{constr_rhs}'")

    # Range-overlap check:
    # Both the constraint and the (possibly negated) condition define numeric ranges.
    # We check whether the intersection of the two ranges is non-empty.

    def _range_intersects(
        op_a: str, val_a: float,
        op_b: str, val_b: float,
    ) -> bool:
        """Check if the ranges defined by ``x op_a val_a`` and ``x op_b val_b`` intersect."""
        # Each constraint defines a half-line or point: x < v, x <= v, x > v, x >= v, x == v, x != v
        # Two half-lines intersect iff their directions overlap.
        # E.g. (x > 0) and (x > 5) intersect at (x > 5).
        #      (x > 0) and (x <= 0) do NOT intersect.

        # Handle "!=" specially: it's all values except one point.
        if op_a == "!=":
            if op_b == "!=":
                return True  # two excluded points can't prevent intersection
            if op_b == "==":
                return val_a != val_b  # !=a and ==b intersect iff a != b
            # !=a and {<, <=, >, >=} b: always intersect unless the only intersection is at val_a
            # We need to check if there's a value ≠ val_a that satisfies op_b/val_b
            # For efficiency, return True (the edge case of val_a being the only value
            # satisfying op_b/val_b is rare and handled by LLM fallback)
            return True
        if op_b == "!=":
            return _range_intersects(op_b, val_b, op_a, val_a)  # symmetric

        if op_a == "==":
            if op_b == "==":
                return abs(val_a - val_b) < 1e-9
            # ==a and {<, <=, >, >=} b
            if op_b == "<": return val_a < val_b
            if op_b == "<=": return val_a <= val_b
            if op_b == ">": return val_a > val_b
            if op_b == ">=": return val_a >= val_b
            return False
        if op_b == "==":
            return _range_intersects(op_b, val_b, op_a, val_a)

        # Both are {<, <=, >, >=}
        # Map each to a canonical half-line
        def _to_interval(op, val):
            if op in ("<", "<="):
                return "left", val
            else:  # >, >=
                return "right", val

        side_a, bound_a = _to_interval(op_a, val_a)
        side_b, bound_b = _to_interval(op_b, val_b)

        if side_a == side_b:
            # Both point same direction: x > 0 and x >= 5 → intersect if max(bound) allows
            if side_a == "right":
                return True  # x > 0 and x > 5 always intersect (just take max bound)
            else:
                return True  # x < 0 and x < 5 always intersect (just take min bound)
        else:
            # Opposite directions: x > 0 and x <= 10 → intersect if bounds cross
            right_bound = bound_a if side_a == "right" else bound_b
            left_bound = bound_b if side_b == "left" else bound_a
            # Need the right bound to be STRICTLY LESS than the left bound
            # (for open intervals), or LESS-OR-EQUAL (for closed).
            # x > 5 and x < 5 → no intersection (5 is not > 5)
            # x >= 5 and x <= 5 → intersection at 5
            # x > 5 and x <= 5 → no intersection
            if (op_a in (">=", "<=") or op_b in (">=", "<=")) and right_bound == left_bound:
                return True
            return right_bound < left_bound

    # The condition as written: x cond_op cond_val
    # If anchor_branch is TRUE, we need x to satisfy the condition.
    # If anchor_branch is FALSE, we need x to satisfy the NEGATION.

    if anchor_branch:
        # Need: x satisfies constraint AND x satisfies cond_op cond_val
        return _check_range_intersection(
            constr_op, constr_val, cond_op, cond_val,
            constraint_norm, cond_stripped,
        )
    else:
        # Need: x satisfies constraint AND x does NOT satisfy cond_op cond_val
        # Negation map:
        negation_map = {
            "==": "!=",
            "!=": "==",
            "<": ">=",
            "<=": ">",
            ">": "<=",
            ">=": "<",
        }
        neg_op = negation_map.get(cond_op)
        if neg_op is None:
            return ("insufficient_info",
                    f"Cannot negate operator '{cond_op}'")
        return _check_range_intersection(
            constr_op, constr_val, neg_op, cond_val,
            constraint_norm, f"NOT ({cond_stripped})",
        )


def _check_range_intersection(
    op_a: str, val_a: float,
    op_b: str, val_b: float,
    label_a: str, label_b: str,
) -> tuple[str, str]:
    """Check whether ``x op_a val_a`` and ``x op_b val_b`` have intersecting ranges."""
    if _range_intersects(op_a, val_a, op_b, val_b):
        return ("consistent",
                f"Constraint '{label_a}' and required state '{label_b}' "
                f"have overlapping ranges — feasible")
    else:
        return ("contradictory",
                f"Constraint '{label_a}' requires range {_describe_range(op_a, val_a)}, "
                f"but required state '{label_b}' requires "
                f"{_describe_range(op_b, val_b)} — no overlap")


def _describe_range(op: str, val: float) -> str:
    """Human-readable description of a numeric range."""
    if op == "==":
        return f"== {val}"
    if op == "!=":
        return f"!= {val}"
    if op == "<":
        return f"< {val}"
    if op == "<=":
        return f"<= {val}"
    if op == ">":
        return f"> {val}"
    if op == ">=":
        return f">= {val}"
    return f"{op} {val}"


# Keep the original _range_intersects name for the helper but define it locally
# inside _evaluate_constraint_against_branch.  We also need it at module level
# for testability.  Let's lift it out.

def _range_intersects(
    op_a: str, val_a: float,
    op_b: str, val_b: float,
) -> bool:
    """Check if the ranges defined by ``x op_a val_a`` and ``x op_b val_b`` intersect."""
    if op_a == "!=":
        if op_b == "!=":
            return True
        if op_b == "==":
            return abs(val_a - val_b) > 1e-9
        return True
    if op_b == "!=":
        return _range_intersects(op_b, val_b, op_a, val_a)

    if op_a == "==":
        if op_b == "==":
            return abs(val_a - val_b) < 1e-9
        if op_b == "<":
            return val_a < val_b
        if op_b == "<=":
            return val_a <= val_b
        if op_b == ">":
            return val_a > val_b
        if op_b == ">=":
            return val_a >= val_b
        return False
    if op_b == "==":
        return _range_intersects(op_b, val_b, op_a, val_a)

    # Both are {<, <=, >, >=}
    if op_a in ("<", "<=") and op_b in ("<", "<="):
        return True  # same direction, always intersect
    if op_a in (">", ">=") and op_b in (">", ">="):
        return True  # same direction, always intersect

    # Opposite directions: one is < or <=, the other is > or >=
    right_bound = val_a if op_a in ("<", "<=") else val_b
    left_bound = val_b if op_b in (">", ">=") else val_a

    # Intersection at boundary point is only valid if BOTH sides are inclusive
    if abs(right_bound - left_bound) < 1e-9:
        return "=" in op_a and "=" in op_b

    return left_bound < right_bound


def _check_non_numeric_constraint(
    anchor_branch: bool,
    cond_op: str,
    cond_rhs: str,
    constraint_norm: str,
) -> tuple[str, str]:
    """Handle non-numeric constraints like ``"allocated"``, ``"NULL"``."""
    cond_rhs_upper = cond_rhs.upper()
    is_null_check = cond_rhs_upper in ("NULL", "NULLPTR", "0", "FALSE")
    is_nonnull_check = cond_op in ("!=", "==") and is_null_check

    if is_nonnull_check:
        constraint_lower = constraint_norm.lower().strip()

        # Helper: does the constraint clearly indicate NULL?
        def _clearly_null(s: str) -> bool:
            return s in ("null", "nullptr", "none", "0")
        def _clearly_nonnull(s: str) -> bool:
            return s.startswith("non-") or "non-null" in s or "allocated" in s or "valid" in s

        if _clearly_null(constraint_lower):
            constraint_has_null = True
        elif _clearly_nonnull(constraint_lower):
            constraint_has_null = False
        else:
            # Ambiguous — let LLM handle it
            return ("insufficient_info",
                    f"Constraint '{constraint_norm}' is ambiguous for "
                    f"null check '{cond_op} {cond_rhs}' — needs LLM check")

        if cond_op == "==":
            # "ptr == NULL"
            if anchor_branch:
                # Branch TRUE: ptr IS null
                if constraint_has_null:
                    return ("consistent",
                            f"constraint '{constraint_norm}' indicates NULL, "
                            f"supports TRUE branch of '== NULL'")
                return ("contradictory",
                        f"condition requires NULL, but constraint "
                        f"'{constraint_norm}' does not indicate NULL")
            else:
                # Branch FALSE: ptr is NOT null
                if constraint_has_null:
                    return ("contradictory",
                            f"condition is FALSE (ptr not NULL), but "
                            f"constraint '{constraint_norm}' indicates NULL")
                return ("consistent",
                        f"non-null constraint '{constraint_norm}' supports "
                        f"FALSE branch of '== NULL'")
        else:  # "!="
            # "ptr != NULL"
            if anchor_branch:
                # Branch TRUE: ptr is NOT null
                if constraint_has_null:
                    return ("contradictory",
                            f"condition is TRUE (ptr != NULL), but "
                            f"constraint '{constraint_norm}' indicates NULL")
                return ("consistent",
                        f"non-null constraint '{constraint_norm}' supports "
                        f"TRUE branch of '!= NULL'")
            else:
                # Branch FALSE: ptr IS null
                if constraint_has_null:
                    return ("consistent",
                            f"constraint '{constraint_norm}' indicates NULL, "
                            f"supports FALSE branch of '!= NULL'")
                return ("contradictory",
                        f"condition requires NULL (FALSE branch of '!= NULL'), "
                        f"but constraint '{constraint_norm}' does not indicate NULL")

    return ("insufficient_info",
            f"Non-numeric constraint '{constraint_norm}' for "
            f"condition '{cond_op} {cond_rhs}' — needs LLM check")


def _check_anchor_against_constraints(
    anchor_branch: bool,
    condition_text: str,
    variable: str | None,
    concretized_values: dict[str, str],
) -> tuple[str, str]:
    """Check one anchor event against the concretized constraints.

    Looks up *variable* (if provided) in *concretized_values*, then calls
    :func:`_evaluate_constraint_against_branch` to determine consistency.

    Returns ``("consistent", ...)``, ``("contradictory", ...)``, or
    ``("insufficient_info", ...)``.
    """
    if variable is None:
        return ("insufficient_info",
                f"No variable extracted from condition '{condition_text}' "
                f"— needs LLM check")

    # Look up variable in concretized values
    constraint = concretized_values.get(variable)
    if constraint is None:
        # Try partial match: check if variable is a suffix of any key
        for k, v in concretized_values.items():
            if k.endswith(variable) or variable.endswith(k):
                constraint = v
                break

    if constraint is None:
        return ("insufficient_info",
                f"Variable '{variable}' not found in concretized values "
                f"— needs LLM check")

    return _evaluate_constraint_against_branch(
        anchor_branch=anchor_branch,
        condition=condition_text,
        constraint_str=constraint,
    )


# ---------------------------------------------------------------------------
# Step 4 helper — merge LLM-established state into cumulative dict
# ---------------------------------------------------------------------------


def _merge_state(cumulative: dict[str, str], new_state_text: str | dict) -> None:
    """Parse LLM ``established_state`` into ``{var: constraint}``.

    Handles two formats:

    - **dict** (preferred): ``{"ctx->write_enabled": "TRUE", ...}``.
      Each entry becomes ``var == val`` in *cumulative*.
    - **str** (legacy): free-text like ``"x >= 10, ptr == NULL"``.
      Extracts ``var [=|!=|<|>] value`` patterns.

    The *cumulative* dict is updated **in-place**.
    """
    if not new_state_text:
        return

    if isinstance(new_state_text, dict):
        for var, val in new_state_text.items():
            var_str = str(var).strip()
            val_str = str(val).strip()
            cumulative[var_str] = f"== {val_str}"
        return

    if not isinstance(new_state_text, str):
        return

    import re

    # Pattern: "var [=|!=|<|>] value" possibly comma/period/line-separated
    pattern = re.compile(
        r'([a-zA-Z_][\w.]*)\s*(==|!=|<=|>=|<|>)\s*([^,;.\n]+)'
    )

    for m in pattern.finditer(new_state_text):
        var = m.group(1).strip()
        op = m.group(2)
        val = m.group(3).strip().rstrip(".,;")
        cumulative[var] = f"{op} {val}"


# ---------------------------------------------------------------------------
# Step 3+4 helper — apply contradictory CFG branch tags to SliceMask
# ---------------------------------------------------------------------------


def _apply_branch_tags_to_mask(
    tree: CFGBranchTree,
    mask: SliceMask,
) -> SliceMask:
    """Apply contradictory branch tags back to the SliceMask.

    Contradictory-verified branches have their CFG blocks removed
    from the mask so subsequent pipeline stages (constraint completion,
    POC generation) also skip them.
    """
    for node in _walk_branch_nodes(tree.root_branches):
        if node.tag and node.tag.verdict == "contradictory":
            fn = node.function_name
            if fn and fn in mask.cfg_retained_blocks:
                # Parse block ID from node_id like "seg0__f_B3_taken"
                import re as _re
                m = _re.search(r'_B(\d+)(?:_taken|_not_taken)?$', node.node_id)
                if m:
                    bid = int(m.group(1))
                    mask.cfg_retained_blocks[fn].discard(bid)
    return mask


def _walk_branch_nodes(nodes: list) -> list:
    """Walk all branch nodes in a tree (BFS)."""
    result: list = []
    stack = list(nodes)
    while stack:
        n = stack.pop(0)
        result.append(n)
        stack.extend(n.children)
    return result


def _resolve_source_path(bug_file: str, source_root: str) -> str:
    """Resolve a bug report source file path to a local path under *source_root*.

    Tries, in order:
    1. The absolute path as-is (if it exists locally).
    2. If *bug_file* is relative, ``source_root / bug_file``.
    3. Suffix matching — iterates the parts of the absolute *bug_file* path
       (which may be from a different build machine) and checks each suffix
       under *source_root*.

    Returns the first path that exists, or the original *bug_file* if none
    does (caller must handle gracefully).
    """
    path = Path(bug_file)
    if path.exists():
        return str(path)

    if source_root:
        src_root = Path(source_root)
        if not path.is_absolute():
            test = src_root / path
            if test.exists():
                return str(test)

        # Absolute path from a different build machine — match suffix
        parts = path.parts
        for i in range(1, len(parts)):
            test = src_root.joinpath(*parts[i:])
            if test.exists():
                return str(test)

    return bug_file  # give up


# ---------------------------------------------------------------------------
# Constraint helpers (module-level, used by FPAnalyzer + tests)
# ---------------------------------------------------------------------------


def _negate_constraint(variable: str, operator: str, value: str) -> str | None:
    """Produce the logical negation of ``variable operator value`` as a string.

    Used to convert a **VIOLATED** assumption into a hard constraint.

    >>> _negate_constraint("x", "<=", "0")
    'x > 0'
    >>> _negate_constraint("x", "==", "0")
    'x != 0'
    """
    negation_map = {
        "==": "!=",
        "!=": "==",
        "<": ">=",
        "<=": ">",
        ">": "<=",
        ">=": "<",
    }
    neg_op = negation_map.get(operator)
    if neg_op is None:
        return None
    return f"{variable} {neg_op} {value}"


def _to_number(s: str) -> float | None:
    """Parse *s* as a float, or return ``None`` if it isn't numeric."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _as_interval(op: str, value: str) -> tuple[float | None, float | None] | None:
    """Map a same-variable constraint to a closed interval ``[lo, hi]``.

    Numeric values are epsilon-padded so strict inequalities don't falsely
    overlap.  Returns ``None`` for ``!=`` (non-interval) or non-numeric values.
    """
    num = _to_number(value)
    if num is None:
        return None
    eps = abs(num) * 1e-9 if num else 1e-9
    if op == "==":
        return (num, num)
    if op == "<":
        return (None, num - eps)
    if op == "<=":
        return (None, num)
    if op == ">":
        return (num + eps, None)
    if op == ">=":
        return (num, None)
    return None  # !=


def _intervals_conflict(ia, ib) -> bool:
    """Return True if two closed intervals have an empty intersection."""
    lo_a, hi_a = ia
    lo_b, hi_b = ib
    lo = lo_a if lo_a is not None else lo_b
    if lo_a is not None and lo_b is not None:
        lo = max(lo_a, lo_b)
    hi = hi_a if hi_a is not None else hi_b
    if hi_a is not None and hi_b is not None:
        hi = min(hi_a, hi_b)
    return lo is not None and hi is not None and lo > hi


def _constraint_sets_conflict(op_a: str, val_a: str, op_b: str, val_b: str) -> bool:
    """Return True if two single-variable constraints can never both hold.

    Handles same-variable value conflicts such as ``x < 10`` vs ``x > 20``,
    ``x == 5`` vs ``x == 7``, and ``x == 0`` vs ``x != 0``.  Non-numeric values
    (e.g. ``i < current_group_size``) conservatively return ``False``.
    """
    # Same value, complementary ==/!= → conflict
    if op_a == "==" and op_b == "!=":
        av = _to_number(val_a)
        return av is not None and av == _to_number(val_b)
    if op_a == "!=" and op_b == "==":
        bv = _to_number(val_b)
        return bv is not None and _to_number(val_a) == bv
    if op_a == "!=" and op_b == "!=":
        return False  # two != constraints can both hold

    ia = _as_interval(op_a, val_a)
    ib = _as_interval(op_b, val_b)
    if ia is None or ib is None:
        return False
    return _intervals_conflict(ia, ib)


def _constraints_conflict(fact: AssumptionFact, hard_constraint: str) -> bool:
    """Check whether *fact* (a CSA assumption) contradicts *hard_constraint*.

    Two detection paths:

    1. **Exact negation** — the hard constraint is literally the negation of
       the fact (e.g. fact ``x <= 0`` vs ``x > 0``).
    2. **Same-variable bound conflict** — the hard constraint constrains the
       same variable to a disjoint value/range (e.g. fact ``x < 10`` vs
       ``x > 20``).

    >>> _constraints_conflict(
    ...     AssumptionFact(event_number=5, line_number=62, segment_index=0,
    ...                    variable="x", operator="<=", value="0",
    ...                    raw_description="", is_error_start=False),
    ...     "x > 0")
    True
    """
    # Fast path: exact negation of the fact.
    negated = _negate_constraint(fact.variable, fact.operator, fact.value)
    if negated is not None and negated.replace(" ", "") == hard_constraint.replace(" ", ""):
        return True

    # Slower path: parse the hard constraint and compare same-variable bounds.
    m = re.match(
        r"^\s*(\w+(?:\.\w+)*)\s*(==|!=|<=|>=|<|>)\s*(.+?)\s*$", hard_constraint,
    )
    if not m:
        return False
    hc_var, hc_op, hc_val = m.group(1), m.group(2), m.group(3).strip()
    if hc_var != fact.variable:
        return False
    return _constraint_sets_conflict(
        fact.operator, fact.value, hc_op, hc_val,
    )


# ---------------------------------------------------------------------------
# FPAnalyzer
# ---------------------------------------------------------------------------


# How many branch decisions of a candidate are rendered into the prompt.
# A candidate is one concrete combination over a segment's branches, so its
# decision list has one entry per branch — for a segment inside a large
# function that is tens of thousands of entries per candidate (faiss'
# ``read_index`` reaches ~28K branches per segment), which overflows the model
# context many times over.  The LLM picks by candidate INDEX and sees the
# segment's metrics, so the tail adds prompt weight without adding information.
_MAX_RENDERED_DECISIONS = 40

# How many characters of a segment's ``established_state`` are echoed into the
# next prompt.  The state is LLM-written and, for a segment inside a large
# function, comes back as one entry per branch — faiss' ``read_index`` returns
# ~1.1 MB for a single segment, and echoing that for every segment overflowed
# the model context.  The state is still kept in full for constraint merging;
# only the prompt rendering is bounded.
_MAX_RENDERED_STATE_CHARS = 2000


class PathSpaceExhausted(Exception):
    """Raised when every candidate of a segment is already blocked, so no NEW
    distinct whole-path is available within the explored top-k subspace.

    This is intentionally DISTINCT from returning ``None`` from path selection:
    ``None`` means the conflict checker found the path genuinely infeasible
    (a sound, conflict-based FP).  Exhaustion only means "we ran out of distinct
    top-k paths to try" — it does NOT, by itself, prove the report is FP (there
    may be untried free-segment combinations that reach the bug).  The caller
    must route exhaustion to bounded-exploration aggregation (FP only if every
    explored distinct path was decisive-FP with no unresolved), never to an
    immediate hard FP.
    """


class FPAnalyzer:
    """Analyze CSA reports, complete path constraints, check feasibility, and generate POCs.

    Feasibility-driven pipeline:
    1. **analyze** — Parse HTML + extract source context (no LLM gap analysis)
    2. **complete_and_verify** — LLM constraint completion + feasibility check
    3. **generate_poc** — LLM POC code generation (if path is feasible)
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
        model: str = _DEFAULT_MODEL,
        source_root: str | Path = "",
        temperature: float = 0.0,
        # Completion budget per call.  The pipeline's verbose stages were being cut
        # off at the old 4096 default: on read_index-484 five of eight
        # simulated-execution replies ended mid-string at ~12.2k chars, and
        # constraint completion asked for more than the 8192 it was then given
        # (26,279 chars ≈ 8.3k tokens) and had to be salvaged from a repaired
        # prefix.  Measured on this account: `deepseek-chat` returns
        # finish_reason="stop" after 11,000 completion tokens, so the model's cap
        # is well above 8192 — the budget is ours to set.
        max_tokens: int = 16384,
        output_dir: str | Path = "",
        cache_root: str | Path = "summary_cache",
        pdg_path: str | Path | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._source_root = str(Path(source_root).resolve()) if source_root else ""
        self._output_dir = Path(output_dir).resolve() if output_dir else Path("output/intermediate")
        self._html_parser = FPReportParser()
        self._source_extractor = SourceContextExtractor(
            project_root=self._source_root or None
        )
        self._summary_cache = SummaryCache(cache_root=cache_root)
        self._sdg: SDG | None = None

        # Load PDG if provided
        if pdg_path:
            pdg_file = Path(pdg_path)
            if pdg_file.exists():
                try:
                    pdgs = load_pdg(str(pdg_file))
                    self._sdg = SDG(pdgs=pdgs)
                    augment_pdg_sdg(self._sdg)
                    logger.info(
                        "Loaded PDG from %s (%d functions)",
                        pdg_file, len(self._sdg.pdgs),
                    )
                except Exception as e:
                    logger.warning("Failed to load PDG from %s: %s", pdg_file, e)
            else:
                logger.warning("PDG file not found: %s", pdg_file)
        self._project_name: str = ""

        if llm_config:
            self._provider: LLMProvider = ProviderFactory.create(llm_config)
        else:
            self._provider = None

    # ------------------------------------------------------------------
    # Step 1: Parse report and extract context (no LLM gap analysis)
    # ------------------------------------------------------------------

    def analyze(
        self,
        report_path: str | Path,
        source_root: str | Path | None = None,
    ) -> AnalysisResult:
        """Parse an FP report and extract context without LLM gap analysis.

        In the new feasibility-driven flow, this method only parses the HTML
        report and extracts source context.  LLM calls happen later in
        ``_complete_constraints()`` and ``_check_feasibility()``.

        Args:
            report_path: Path to an FP HTML report.
            source_root: Optional project root for source resolution.

        Returns:
            AnalysisResult with parsed report and a neutral PathAnalysis
            (empty gaps — no LLM gap analysis performed).

        Raises:
            ValueError: If report is missing required metadata.
        """
        source_root = source_root or self._source_root
        if source_root:
            self._source_extractor = SourceContextExtractor(
                project_root=str(source_root)
            )

        # Step A: Parse the HTML report
        parsed = self._html_parser.parse(str(report_path))
        bug_path = self._html_parser.parse_to_bug_path(str(report_path))

        # Step B: Extract source context
        source_ctx = self._source_extractor.extract(parsed)
        source_ctx_str = source_ctx.format_for_llm(max_lines=80)

        # Step B2: Extract call chain context for data flow tracing
        call_chain = self._source_extractor.extract_call_chain(parsed)
        call_chain_str = self._format_call_chain(call_chain)

        # Step C: Format path events with annotations
        # ⚠ marks the Error Start event (value origin), → marks function calls
        path_str_lines = []
        for evt in parsed.path_events:
            line_info = f" L{evt['line_number']}" if evt['line_number'] else ""
            prefix = "⚠ " if evt.get("is_error_start") else "  "
            path_str_lines.append(
                f"{prefix}#{evt['path_number']} [{evt['event_kind']}]{line_info}: {evt['description']}"
            )
        path_str = "\n".join(path_str_lines)

        # Error Start info section (used by constraint completion prompt)
        error_start_info_str = self._format_error_start_info(
            parsed.error_start_event, bug_type=parsed.metadata.bug_type,
        )

        # Build a neutral PathAnalysis (no gap analysis — gaps filled later
        # by _complete_constraints())
        neutral_analysis = PathAnalysis(
            path_id=Path(str(report_path)).stem,
            is_likely_fp=False,
            gaps=[],
            confidence=Confidence.HIGH,
            reasoning=f"Path through {parsed.metadata.function_name} "
                      f"with {len(parsed.path_events)} events. "
                      f"Constraint completion and feasibility check follow.",
        )

        result = AnalysisResult(
            parsed_report=parsed,
            bug_path=bug_path,
            analysis=neutral_analysis,
            source_root=str(source_root) if source_root else "",
            annotated_path=path_str,
            call_chain_context=call_chain_str,
        )

        # Save intermediate JSON for debugging
        self._save_intermediate_json(result, str(report_path))

        return result

    def _run_pdg_slicing(self, result: AnalysisResult) -> SliceResult | None:
        """Run PDG-based backward slicing and store reduced_code in AnalysisResult.

        Best-effort: if PDG is not loaded, slicing fails, or no seeds are
        found, falls back silently (reduced_code stays empty).

        Returns:
            The ``SliceResult`` if slicing succeeded, ``None`` on failure.
        """
        if self._sdg is None:
            return None

        parsed = result.parsed_report
        bug_path = result.bug_path
        source_root = self._source_root or result.source_root
        if not source_root:
            return None

        try:
            seeds = extract_seeds_from_bug_path(
                bug_path, self._sdg,
                source_root=source_root,
                fallback_trigger_fn=parsed.metadata.function_name,
                fallback_trigger_line=parsed.metadata.bug_line,
            )
        except Exception:
            logger.debug("Seed extraction failed for %s", parsed.metadata.function_name, exc_info=True)
            return None

        if not seeds:
            logger.debug("No seeds extracted for %s", parsed.metadata.function_name)
            return None

        annotations = build_path_annotations(bug_path, self._sdg, source_root=source_root)
        caller_map = build_caller_map(bug_path, self._sdg, entry_fn=parsed.metadata.function_name)

        slicer_cfg = SlicerConfig(max_depth=10)
        slice_result = compute_slice_from_seeds(
            self._sdg, seeds, slicer_cfg,
            annotations=annotations,
            caller_map=caller_map,
        )

        if not slice_result.nodes:
            logger.debug("Slice produced 0 nodes for %s", parsed.metadata.function_name)
            return None

        # Compute reduced code (clean mode: slice + structural anchors)
        # IMPORTANT: extract control-flow event lines from the bug path and
        # force-preserve them in the reduced code.  Without this, the backward
        # slice can remove branch predicates (like ``if (c == -1)``) that are
        # NOT control-dependent on the trigger but ARE essential for
        # understanding the CSA path events.
        preserve_lines: set[int] = set()
        for evt in parsed.path_events:
            if evt.get("is_control_flow") and evt.get("branch_condition") in (True, False):
                ln = evt.get("line_number", 0)
                if ln > 0:
                    preserve_lines.add(ln)
        if preserve_lines:
            logger.info(
                "Preserving %d control-flow line(s) from path events: %s",
                len(preserve_lines), sorted(preserve_lines),
            )

        try:
            reduced = reduce_code(
                slice_result,
                source_root=source_root,
                include_search_roots=[Path(source_root)],
                preserve_lines=preserve_lines or None,
            )
        except Exception:
            logger.debug("Code reduction failed for %s", parsed.metadata.function_name, exc_info=True)
            return None

        if not reduced.strip():
            logger.debug("Reduced code is empty for %s", parsed.metadata.function_name)
            return None

        # Compute reduction ratio
        orig_size = 0
        for fn_name in slice_result.involved_functions:
            fn_pdg = self._sdg.get_pdg(fn_name)
            if fn_pdg:
                lines = {n.source_line for n in fn_pdg.nodes.values() if n.source_line > 0}
                orig_size += len(lines)

        meaningful = sum(1 for sn in slice_result.nodes
                         if sn.node.kind != "recovery" and "recovery" not in (sn.node.expression or ""))
        reduced_lines = len([l for l in reduced.split("\n") if l.strip()])
        reduction_ratio = 1 - (reduced_lines / max(orig_size, 1)) if orig_size > 0 else 0.0

        result.reduced_code = reduced
        result.slice_info = {
            "num_seeds": len(seeds),
            "num_slice_nodes": len(slice_result.nodes),
            "meaningful_nodes": meaningful,
            "recovery_nodes": len(slice_result.nodes) - meaningful,
            "involved_functions": sorted(slice_result.involved_functions),
            "input_variables": sorted(slice_result.input_variables),
            "truncated": slice_result.truncated,
            "orig_line_estimate": orig_size,
            "reduced_lines": reduced_lines,
            "reduction_ratio": round(reduction_ratio, 3),
        }

        logger.info(
            "PDG slice for %s: %d nodes, %d functions, reduction ratio=%.0f%%",
            parsed.metadata.function_name,
            len(slice_result.nodes),
            len(slice_result.involved_functions),
            reduction_ratio * 100,
        )

        return slice_result

    # ------------------------------------------------------------------
    # SliceMask builder
    # ------------------------------------------------------------------

    def _build_slice_mask(
        self, slice_result: SliceResult | None,
    ) -> SliceMask | None:
        """Build a ``SliceMask`` from a ``SliceResult`` and the SDG.

        Returns ``None`` if the slice or SDG is unavailable.
        """
        if slice_result is None or self._sdg is None:
            return None
        return SliceMask.from_slice_result(slice_result, self._sdg)

    # ------------------------------------------------------------------
    # Slice variable analysis for constraint completion
    # ------------------------------------------------------------------

    @staticmethod
    def _analyze_slice_variables(reduced_code: str) -> str:
        """Scan reduced (sliced) code for library calls, member fields, globals.

        Returns a formatted string suitable for injection into the constraint
        completion prompt.  Returns an empty string if no variables of interest
        are found.
        """
        if not reduced_code.strip():
            return ""

        findings: list[str] = []

        # ── Library function calls ──
        lib_funcs: set[str] = set()
        for m in re.finditer(
            r'\b(memset|memcpy|memmove|malloc|calloc|realloc|free|'
            r'strcpy|strncpy|sprintf|snprintf|printf|fprintf|assert'
            r'|getopt_long|mmap|munmap|socket|bind|listen|accept|close'
            r'|read|write|open|fopen|fclose|fread|fwrite)\s*\(',
            reduced_code,
        ):
            lib_funcs.add(m.group(1))
        if lib_funcs:
            findings.append(
                f"- Library functions in slice: {', '.join(sorted(lib_funcs))}"
            )

        # ── Member field accesses (this->foo or this.foo) ──
        field_accesses: set[str] = set()
        for m in re.finditer(r'(?<!\w)this\s*[.-]>\s*(\w+)', reduced_code):
            field_accesses.add(m.group(1))
        for m in re.finditer(r'\[this\]\.(\w+)', reduced_code):
            field_accesses.add(m.group(1))
        if field_accesses:
            findings.append(
                f"- Member field accesses in slice: {', '.join(sorted(field_accesses))}"
            )

        # ── Potential global / member variables (trailing-underscore heuristic) ──
        globals_found: set[str] = set()
        for m in re.finditer(r'\b([a-z][a-zA-Z0-9_]+_)\b', reduced_code):
            name = m.group(1)
            if name not in ('this_',) and len(name) > 2:
                globals_found.add(name)
        if globals_found:
            findings.append(
                f"- Potential member/global vars: {', '.join(sorted(globals_found)[:10])}"
            )

        if not findings:
            return ""

        return (
            "## Slice Variable Analysis (variables found in PDG-sliced code)\n"
            + "\n".join(findings)
            + "\n"
        )

    # ------------------------------------------------------------------
    # Library API contract formatting (inject into LLM prompts)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_library_api_contracts(reduced_code: str) -> str:
        """Scan reduced code for known library function calls and format API contracts.

        Uses the static knowledge base in ``lib_knowledge.py`` — if a function
        call found in the reduced code is registered there, its API documentation
        (return semantics, side effects, preconditions) is injected.

        Returns a formatted section suitable for inclusion in any LLM prompt,
        or an empty string if no library functions are detected.

        This method stays in sync automatically: adding a new entry to
        ``_LIB_FUNCTIONS`` in ``lib_knowledge.py`` makes it available here
        without updating any regex.
        """
        if not reduced_code.strip():
            return ""

        # Strip C++ comments before scanning to avoid false positives from
        # function names mentioned in comments.  Line comments are stripped
        # per-line (re.MULTILINE); block comments need re.DOTALL since they
        # can span multiple lines.  String/char literals are also stripped
        # to avoid matching function names inside string constants.
        code_no_comments = re.sub(r'//.*', '', reduced_code, flags=re.MULTILINE)
        code_no_comments = re.sub(r'/\*.*?\*/', '', code_no_comments, flags=re.DOTALL)
        code_no_comments = re.sub(r"'[^']*'", '', code_no_comments)
        code_no_comments = re.sub(r'"[^"]*"', '', code_no_comments)

        from llm_client.fp_analysis.lib_knowledge import list_registered, lookup

        found_contracts: list[str] = []
        for fn_name in list_registered():
            if re.search(rf'\b{re.escape(fn_name)}\s*\(', code_no_comments):
                entry = lookup(fn_name)
                if entry:
                    found_contracts.append(entry.format_for_llm())

        if not found_contracts:
            return ""

        section = (
            "## Library Function API Contracts (from knowledge base)\n"
            "The following are the API contracts for library functions "
            "detected in the sliced code. Use these as ground truth:\n\n"
            + "\n\n".join(found_contracts)
            + "\n"
        )
        return section

    # ------------------------------------------------------------------
    # Domain-specific knowledge injection (union type-punning,
    # loop-carried initialization, path-merging detection)
    # ------------------------------------------------------------------

    _UNION_TYPES = frozenset({
        "sockaddr_union", "pthread_mutex_t", "pthread_cond_t",
        "epoll_data_t", "sigval", "event",
    })

    @staticmethod
    def _detect_union_type_punning(reduced_code: str) -> str:
        """Detect union type-punning patterns and return knowledge injection.

        POSIX networking code commonly uses ``sockaddr_union`` where
        ``memcpy(&ad.storage, ...)`` writes to one union member and
        ``ad.in.sin_addr.s_addr`` reads from another member of the same
        union.  The CSA's type tracking treats these as separate variables,
        but in practice the members share the same storage bytes.

        Returns a formatted instruction string, or empty string if no
        union pattern is detected.
        """
        if not reduced_code.strip():
            return ""

        # Strip comments/strings before scanning
        code = re.sub(r'//.*', '', reduced_code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r"'[^']*'", '', code)
        code = re.sub(r'"[^"]*"', '', code)

        # (A) Check for union type declarations or known union type names
        has_union_keyword = bool(re.search(r'\bunion\s+\w+\s*\{', code))
        has_known_union_type = bool(
            re.search(
                r'\b(' + '|'.join(re.escape(t) for t in FPAnalyzer._UNION_TYPES) + r')\b',
                code,
            )
        )
        # (B) Check for memcpy(&var.member1, ...) followed by var.member2 access
        memcpy_targets: set[str] = set()
        for m in re.finditer(r'memcpy\s*\(\s*&(\w+)\.(\w+)', code):
            memcpy_targets.add(m.group(1))

        union_member_reads: set[str] = set()
        for var in memcpy_targets:
            for m in re.finditer(rf'\b{re.escape(var)}\.(\w+)', code):
                union_member_reads.add(f"{var}.{m.group(1)}")

        has_memcpy_type_punning = bool(union_member_reads)

        if not (has_union_keyword or has_known_union_type or has_memcpy_type_punning):
            return ""

        lines = [
            "\n\n## ⚠ Union Type-Punning (C/C++ Domain Knowledge)",
            "The code above uses a C/C++ `union` where different members share the same storage.",
            "This is STANDARD PRACTICE in POSIX networking code and other low-level C/C++.",
            "CRITICAL: When `memcpy` (or any write) targets one union member (e.g. `var.storage`),",
            "it INITIALIZES the bytes shared by ALL union members — even though",
            "the CSA's type-based tracking treats them as separate variables.",
            "Reading from another union member (e.g. `var.in.sin_addr.s_addr`) after writing",
            "to a different member (`var.storage`) is valid and well-defined at the byte level.",
            "",
            "If the CSA path claims a union member is \"garbage\" or \"uninitialized\" but another",
            "member was written to before the read, this is LIKELY A FALSE POSITIVE caused by",
            "the CSA's conservative union type modeling.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _detect_loop_initialized_vars(reduced_code: str) -> str:
        """Detect loop-carried initialization patterns.

        A local variable declared inside a loop body (e.g.
        ``uint32_t codep;`` inside ``for(...) { ... }``) may be
        initialized on the first iteration by a function call that
        writes through a pointer.  On subsequent iterations the
        variable retains the value written by the previous call,
        but the CSA (which treats each loop iteration independently)
        may incorrectly flag it as uninitialized.
        """
        if not reduced_code.strip():
            return ""

        code = re.sub(r'//.*', '', reduced_code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        code = re.sub(r"'[^']*'", '', code)
        code = re.sub(r'"[^"]*"', '', code)
        code = re.sub(r'\bauto\s+', '', code)

        # Look for loops containing local var + address-taking function call
        # Pattern: for/while (...) { ... type var; ... func(&var, ...) ... }
        loop_pattern = re.compile(
            r'\b(for|while)\s*\(.*?\)\s*\{'
            r'(?:(?!\}).)*?'                              # non-greedy skip
            r'\b(\w[\w\s*&]*)\s+(\w+)\s*(?:=\s*[^;,]+)?;'  # local var decl
            r'(?:(?!\}).)*?'
            r'&\3\b',                                      # &var in call
            re.DOTALL | re.IGNORECASE,
        )

        # Simpler heuristic: find loops with a local variable that is
        # passed by address to a nearby function call
        has_loop_init_pattern = bool(
            re.search(
                r'\b(for|while)\s*\(.*\)\s*\{[^}]*?'
                r'\b\w[\w\s*&]*\s+\w+\s*;\s*[^}]*?'
                r'&\w+\s*[,)]',
                code,
                re.DOTALL,
            )
        )

        if not has_loop_init_pattern:
            return ""

        return (
            "\n\n## ⚠ Loop-Carried Variable Initialization (C/C++ Domain Knowledge)\n"
            "A local variable declared INSIDE a loop body has its lifetime reset on each "
            "iteration, but the CSA's path-sensitive analysis treats each iteration's "
            "variable as independent — it does NOT model the initialization from a "
            "previous iteration's function call.\n\n"
            "If the code pattern is:\n"
            "```\n"
            "for(...) {\n"
            "    Type var;           // declared each iteration\n"
            "    func(&var, ...);     // writes to var via pointer\n"
            "}\n"
            "```\n"
            "Then `var` IS initialized by the FIRST call (when the function's internal "
            "state starts in its initial state, e.g. `UTF8_ACCEPT`). On subsequent "
            "iterations, the function may READ `var` before writing it — but `var` "
            "retains the value written by the previous call. The CSA may incorrectly "
            "report `var` as uninitialized.\n\n"
            "**If the CSA path shows a loop-carried variable read as \"garbage\" but "
            "the function initialization is guaranteed on the first call, this is "
            "likely a FALSE POSITIVE.**\n"
        )

    @staticmethod
    def _inject_domain_specific_knowledge(reduced_code: str) -> str:
        """Combine all domain-specific knowledge injections.

        Returns a string to inject into the LLM prompt, or empty string.
        """
        parts: list[str] = []
        union_knowledge = FPAnalyzer._detect_union_type_punning(reduced_code)
        if union_knowledge:
            parts.append(union_knowledge)
        loop_knowledge = FPAnalyzer._detect_loop_initialized_vars(reduced_code)
        if loop_knowledge:
            parts.append(loop_knowledge)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Branch decision formatting for feasibility checks
    # ------------------------------------------------------------------

    @staticmethod
    def _format_branch_decisions(
        path_events: list[dict],
        source_lines: dict[int, str] | None = None,
    ) -> str:
        """Extract and format the reported path's branch decisions.

        When *source_lines* is provided (``{line_number: source_line}``),
        appends the actual source code as a comment for each branch decision.

        Returns a multi-line string with each control event's line, direction
        (TRUE/FALSE), and description.
        """
        formatted: list[str] = []
        for evt in path_events:
            if evt.get("is_control_flow") and evt.get("branch_condition") in (True, False):
                direction = "TRUE" if evt["branch_condition"] else "FALSE"
                desc = evt.get("description", "")
                line = evt.get("line_number", 0)
                source_annotation = ""
                if source_lines and line in source_lines:
                    src = source_lines[line].strip()
                    if src:
                        source_annotation = f"  // {src}"
                formatted.append(f"  L{line}: {direction} — {desc}{source_annotation}")
        if not formatted:
            return "  (no branch decisions in reported path)"
        return "\n".join(formatted)

    # ------------------------------------------------------------------
    # CFG branch filter: build postcondition from segment + parsed events
    # ------------------------------------------------------------------

    def _build_postcondition(
        self,
        segment: object,
        parsed: object,
    ) -> Postcondition:
        """Build a :class:`Postcondition` from a segment's control event info.

        The postcondition is derived from the segment's boundary control event.
        It serves as the "anchor" for CFG branch filtering in Step 4.

        Args:
            segment: A ``SegmentInfo`` (or compatible) with
                ``control_event_number``, ``control_branch_taken``,
                ``function_name``, and ``line_end``.
            parsed: A ``ParsedReport`` with ``path_events`` list.

        Returns:
            A :class:`Postcondition` for the segment.  When no control event
            is found, a default with available metadata is returned.
        """
        if segment.control_event_number is not None:
            for evt in parsed.path_events:
                if evt.get("path_number") == segment.control_event_number:
                    source_line = evt.get("line_number", segment.line_end)
                    condition_expr = evt.get("description", "")

                    # Phase 1: Try to extract concrete condition from source code
                    if self._source_root:
                        source_file = (
                            evt.get("file_path", "") or parsed.metadata.bug_file
                        )
                        resolved = _resolve_source_path(source_file, self._source_root)
                        extracted = extract_condition(
                            resolved, source_line,
                            source_root=self._source_root,
                        )
                        if extracted.confidence > 0:
                            condition_expr = extracted.expression

                    return Postcondition(
                        function_name=segment.function_name or parsed.metadata.function_name,
                        source_line=source_line,
                        condition_expr=condition_expr,
                        branch_taken=(
                            evt.get("branch_condition")
                            if evt.get("branch_condition") is not None
                            else True
                        ),
                    )
        # Fallback: build from segment's available fields
        return Postcondition(
            function_name=segment.function_name or parsed.metadata.function_name,
            source_line=segment.line_end,
            condition_expr="",
            branch_taken=(
                segment.control_branch_taken
                if segment.control_branch_taken is not None
                else True
            ),
        )

    # ------------------------------------------------------------------
    # Constraints formatting helper
    # ------------------------------------------------------------------

    @staticmethod
    def _format_constraints_for_prompt(completed: CompletedPath) -> str:
        """Format completed constraints as a string for LLM prompts."""
        constraints_lines = [
            f"  - [{s.new_event_kind}] {s.description}"
            + (f"  (concretized: {s.concretization})" if s.concretization else "")
            for s in completed.completed_steps
        ]
        constraints_str = "\n".join(constraints_lines) if constraints_lines else "  (none)"

        preconditions_str = "\n".join(
            f"  - {p}" for p in completed.preconditions
        ) if completed.preconditions else "  (none)"

        concretized_str = "\n".join(
            f"  - {k} = {v}" for k, v in completed.concretized_values.items()
        ) if completed.concretized_values else "  (none)"

        return constraints_str, preconditions_str, concretized_str

    # ------------------------------------------------------------------
    # Step 3: Single-Path Feasibility Check (reported path only)
    # ------------------------------------------------------------------

    def _check_single_path_feasibility(
        self,
        analysis_result: AnalysisResult,
        completed: CompletedPath,
        slice_mask: SliceMask | None = None,
        previous_conflicts: list[str] | None = None,
    ) -> tuple[bool, str, list[str]]:
        """Check the **reported path** branch conditions against constraints.

        Unlike the older ``_check_feasibility``, this method does NOT evaluate
        CSA warning validity (``is_real_bug``).  It only checks whether the
        specific branch decisions in the CSA bug report can be satisfied given
        the completed constraints.

        This is the refactored Step 3: it replaces the old CFG-enumeration
        approach with a focused constraint-consistency query.

        Returns:
            (is_feasible, explanation, conflicts) triple.
        """
        parsed = analysis_result.parsed_report

        # Build source line index and annotate branch decisions with source code
        source_lines = self._build_event_source_index(parsed)
        branch_decisions = self._format_branch_decisions(
            parsed.path_events, source_lines=source_lines,
        )

        # Format constraints
        constraints_str, preconditions_str, concretized_str = (
            self._format_constraints_for_prompt(completed)
        )

        # Source context (prefer sliced code)
        if analysis_result.reduced_code:
            source_ctx_str = analysis_result.reduced_code
        else:
            source_ctx = self._source_extractor.extract(parsed)
            source_ctx_str = source_ctx.format_for_llm(max_lines=80)

        # Error Start info
        error_start_info_str = self._format_error_start_info(
            parsed.error_start_event, bug_type=parsed.metadata.bug_type,
        )

        # Previous conflicts for retry
        previous_conflicts_str = ""
        if previous_conflicts:
            previous_conflicts_str = (
                "## Previously Found Conflicts (to be resolved)\n"
                + "\n".join(f"  - {c}" for c in previous_conflicts)
                + "\n\nRe-check whether these are still conflicts given the updated constraints.\n"
            )

        prompt = SINGLE_PATH_FEASIBILITY_PROMPT.format(
            bug_type=parsed.metadata.bug_type,
            bug_location=parsed.metadata.bug_file,
            bug_line=parsed.metadata.bug_line,
            bug_function=parsed.metadata.function_name,
            bug_description=parsed.metadata.bug_description,
            error_start_info=error_start_info_str,
            source_context=source_ctx_str,
            branch_decisions=branch_decisions,
            completed_constraints=constraints_str,
            preconditions=preconditions_str,
            concretized_values=concretized_str,
            previous_conflicts=previous_conflicts_str,
        )

        # ── Inject library function API contracts (from static knowledge base) ──
        if analysis_result.reduced_code:
            api_contracts = self._format_library_api_contracts(
                analysis_result.reduced_code,
            )
            if api_contracts:
                prompt += api_contracts + "\n"

        # ── Inject domain-specific knowledge (union type-punning, loop init, etc.) ──
        if analysis_result.reduced_code:
            domain_knowledge = self._inject_domain_specific_knowledge(
                analysis_result.reduced_code,
            )
            if domain_knowledge:
                prompt += domain_knowledge + "\n"

        # Inject project-wide constraints as hard bounds
        source_root_for_constraints = self._source_root or (
            analysis_result.source_root or None
        )
        target_classes = self._extract_class_names_from_path(parsed)
        global_constraints = self._load_global_constraints(
            source_root_for_constraints, target_classes=target_classes,
        )
        if global_constraints and global_constraints != "(no project-wide constraints)":
            prompt += (
                "\n\n## Project-Wide Real-World Constraints (from codebase analysis)\n"
                "The following constraints are KNOWN to hold in real-world execution:\n"
                f"{global_constraints}\n"
            )

        # ── Memory leak specific: disambiguate return values at different levels ──
        if self._is_memory_leak(parsed.metadata.bug_type):
            prompt += (
                "\n\n## Memory Leak Report: Disambiguating Return Values\n"
                "IMPORTANT: In a memory leak path, different functions at different "
                "call levels have DIFFERENT return values:\n"
                "- The ALLOCATION function (e.g., `wslay_event_omsg_non_fragmented_init`) "
                "returns 0 (success) and allocates memory.\n"
                "- A SUBSEQUENT operation (e.g., `wslay_queue_push`) may return "
                "non-zero (failure) — the actual error that triggers the leak.\n"
                "- The OUTER function propagates the error from the subsequent "
                "operation.\n\n"
                "An 'Allocated memory' event followed by a 'returning non-zero' "
                "event at a DIFFERENT call level is NOT a contradiction — "
                "it is exactly the leak path! The CSA path events for memory leaks "
                "mix success and failure across function boundaries.\n"
                "Do NOT flag different return values at different function levels "
                "as infeasible contradictions.\n"
            )

        try:
            raw = self._call_llm(SINGLE_PATH_FEASIBILITY_SYSTEM_PROMPT, prompt)
            data = self._parse_json_response(raw, "single_path_feasibility")
            is_feasible = bool(data.get("is_feasible", True))
            explanation = str(data.get("explanation", ""))
            conflicts = list(data.get("conflicts", []))
            # Safety-net: override LLM contradictions
            # The LLM sometimes correctly identifies a known CSA FP pattern
            # (union type-punning, memcpy init, loop-carried init) but then
            # still outputs is_feasible=True.  Detect and override.
            if is_feasible:
                _explanation_lower = explanation.lower()
                _KNOWN_FP_KEYWORDS = [
                    "union type-punning",
                    "memcpy to ad.storage",
                    "initializes all bytes",
                    "initializes all union members",
                    "makes the read well-defined",
                    "shared-storage initialization",
                    "loop-carried",
                    "UTF8_ACCEPT",
                    "imsg_reset",
                    "initializes the state",
                    "initialized by the first call",
                    "previous iteration",
                    "state reset function",
                    "cross-iteration initialization",
                    "first call with fresh state",
                ]
                for keyword in _KNOWN_FP_KEYWORDS:
                    if keyword in _explanation_lower or keyword in explanation:
                        logger.info(
                            "Safety-net: LLM output feasible=True but explanation "
                            "mentions known FP pattern '%s'. Overriding to FP.",
                            keyword,
                        )
                        is_feasible = False
                        fp_override = (
                            "Known CSA false positive pattern detected in explanation: "
                            "'" + keyword + "'. The value IS "
                            "actually initialized, contradicting the CSA report."
                        )
                        conflicts = [fp_override] + (conflicts or [])
                        if explanation:
                            explanation = "[OVERRIDE to FP] " + explanation
                        break

        except (RuntimeError, json.JSONDecodeError) as e:
            logger.warning("Single-path feasibility LLM call failed: %s", e)
            is_feasible = True  # optimistic on error
            explanation = f"Feasibility check failed: {e}"
            conflicts = []

        return is_feasible, explanation, conflicts

    # ------------------------------------------------------------------
    # Step 2.7: State Lifecycle Analysis — cross-function state tracking
    # ------------------------------------------------------------------

    def _analyze_state_lifecycle(
        self,
        analysis_result: AnalysisResult,
        completed: CompletedPath,
    ) -> LifecycleAssessment:
        """Run Step 2.7 state lifecycle analysis.

        Combines static analysis (call chain extraction, PDG-based state
        variable identification, known pattern detection) with a focused
        LLM prompt to determine whether the CSA-flagged Error Start variable
        is actually initialized through a cross-function state lifecycle
        that the CSA does not model.

        When static analysis finds no state variables or no PDG is available,
        the LLM call is skipped and a conservative default is returned
        (``error_start_initialized_by_lifecycle=False`` — meaning no override).
        """
        parsed = analysis_result.parsed_report

        # ── Step 1: Static analysis — extract call chain ──
        call_chain = CallChainExtractor.extract(parsed.path_events)

        if not call_chain:
            logger.debug(
                "Step 2.7: no call chain found (no Calling events in path)."
            )
            return LifecycleAssessment()

        # ── Step 2: Identify state variables via PDG ──
        error_start_var = None
        if parsed.error_start_event:
            error_start_var = parsed.error_start_event.get("variable")
        state_vars = CallChainExtractor.identify_state_variables(
            call_chain,
            error_start_var=error_start_var,
            sdg=self._sdg,
        )

        # ── Step 3: Detect known patterns (static) ──
        reduced_code = analysis_result.reduced_code or ""
        known_patterns = CallChainExtractor.detect_known_patterns(reduced_code)

        # ── Guard: skip LLM if no state variables to analyze ──
        if not state_vars:
            logger.debug(
                "Step 2.7: no state variables identified; skipping LLM."
            )
            return LifecycleAssessment(call_chain=call_chain)

        # ── Step 4: Build function behavior table ──
        function_table = self._build_function_behavior_table(
            call_chain, state_vars,
        )

        # ── Step 5: Source context ──
        if analysis_result.reduced_code:
            source_ctx_str = analysis_result.reduced_code
        else:
            source_ctx = self._source_extractor.extract(parsed)
            source_ctx_str = source_ctx.format_for_llm(max_lines=80)

        # ── Step 6: Format call sequence for prompt ──
        call_seq_lines: list[str] = []
        for entry in call_chain:
            direction = "→" if entry.is_entry else "←"
            callee_str = f" {entry.callee}" if entry.callee else ""
            call_seq_lines.append(
                f"  #{entry.event_index}: {direction} {entry.function_name}"
                f"{callee_str} (L{entry.source_line})"
            )
        call_sequence_text = "\n".join(call_seq_lines)

        # Format known patterns
        patterns_text = "\n".join(
            f"  - {p}" for p in known_patterns
        ) if known_patterns else "  (none detected)"

        # ── Step 7: LLM analysis (focused prompt) ──
        prompt = STATE_LIFECYCLE_PROMPT.format(
            bug_type=parsed.metadata.bug_type,
            bug_location=parsed.metadata.bug_file,
            bug_line=parsed.metadata.bug_line,
            bug_function=parsed.metadata.function_name,
            error_start_var=error_start_var or "(unknown)",
            call_sequence=call_sequence_text,
            source_context=source_ctx_str,
            known_patterns=patterns_text,
            function_behavior_table=function_table,
        )

        try:
            raw = self._call_llm(STATE_LIFECYCLE_SYSTEM_PROMPT, prompt)
            data = self._parse_json_response(raw, "state_lifecycle")
            return LifecycleAssessment(
                call_chain=call_chain,
                state_variables=data.get("state_variables", state_vars),
                reset_functions=data.get("reset_functions", []),
                non_reset_functions=data.get("non_reset_functions", []),
                error_start_initialized_by_lifecycle=bool(
                    data.get("error_start_initialized_by_lifecycle", False)
                ),
                lifecycle_type=data.get("lifecycle_type"),
                explanation=data.get("explanation", ""),
            )
        except (RuntimeError, json.JSONDecodeError) as e:
            logger.warning("Step 2.7 LLM call failed: %s", e)
            return LifecycleAssessment(
                call_chain=call_chain,
                state_variables=state_vars,
                explanation=f"LLM error: {e}",
            )

    def _build_function_behavior_table(
        self,
        call_chain: list[CallChainEntry],
        state_vars: list[str],
    ) -> str:
        """Build a structured function behavior table for the LLM prompt.

        Uses PDG FunctionSummary to determine which functions write to which
        state variables.  Falls back to listing all functions in the chain
        when PDG data is unavailable.
        """
        if not call_chain or not self._sdg:
            return "(PDG data unavailable — analyze from source code)"

        rows: list[str] = []
        seen: set[str] = set()

        for entry in call_chain:
            if not entry.is_entry or not entry.callee:
                continue
            if entry.callee in seen:
                continue
            seen.add(entry.callee)

            callee_pdg = None
            output_vars: list[str] = []
            get_pdg = getattr(self._sdg, "get_pdg", None)
            if get_pdg is not None:
                try:
                    callee_pdg = get_pdg(entry.callee)
                except Exception:
                    pass
            if callee_pdg is not None:
                output_vars = callee_pdg.summary.output_variables or []

            # Match state vars against this function's output vars
            writes_state = any(
                sv.split("->")[-1] in output_vars or
                any(sv in ov or ov in sv for ov in output_vars)
                for sv in state_vars
            )

            rows.append(
                f"  - {entry.callee}: "
                f"state_params={entry.state_params or '(none)'}"
                f", writes_state={'YES' if writes_state else 'No'}"
            )

        if not rows:
            return "(no unique called functions to analyze)"

        return "Function behaviors:\n" + "\n".join(rows)

    # ------------------------------------------------------------------
    # Step 3.5: Early Feasibility Check — anchor events vs constraints
    # ------------------------------------------------------------------

    def _early_feasibility_check(
        self,
        analysis_result: AnalysisResult,
        completed: CompletedPath,
    ) -> tuple[bool, list[str]]:
        """Check reported-path anchor events against completed constraints.

        Two-pass approach:
        - **Pass A**: deterministic check for each anchor event — extract
          variable, look up in ``concretized_values``, check range overlap.
        - **Pass B**: LLM fallback for anchors where Pass A returns
          ``insufficient_info``.

        Returns:
            ``(is_early_feasible, conflicts)``.  If ``is_early_feasible`` is
            ``False``, the path is contradictory — skip Step 4.
        """
        parsed = analysis_result.parsed_report
        concretized = (completed.concretized_values or {})

        # Collect anchor events from the CSA report
        anchors: list[dict] = []
        for evt in parsed.path_events:
            if evt.get("is_control_flow") and evt.get("branch_condition") in (True, False):
                anchors.append({
                    "line": evt.get("line_number", 0),
                    "branch": evt.get("branch_condition"),
                    "description": evt.get("description", ""),
                })

        if not anchors:
            # No anchor events → nothing to check early
            return True, []

        logger.info(
            "Step 3.5: Early feasibility check on %d anchor event(s) ...",
            len(anchors),
        )

        conflicts: list[str] = []
        llm_fallback_anchors: list[dict] = []

        # Pass A: deterministic check
        for anchor in anchors:
            cond_text = anchor["description"]  # e.g. "blockLength_ <= 0"
            variable = _find_variable_in_condition(cond_text)
            result, msg = _check_anchor_against_constraints(
                anchor_branch=anchor["branch"],
                condition_text=cond_text,
                variable=variable,
                concretized_values=concretized,
            )

            anchor["deterministic_result"] = result
            anchor["variable"] = variable

            if result == "contradictory":
                conflicts.append(
                    f"Anchor L{anchor['line']} ({'TRUE' if anchor['branch'] else 'FALSE'} "
                    f"branch of '{cond_text}'): {msg}"
                )
                logger.info(
                    "  Anchor L%d: CONTRADICTORY — %s",
                    anchor["line"], msg,
                )
            elif result == "insufficient_info":
                llm_fallback_anchors.append(anchor)
                logger.info(
                    "  Anchor L%d: insufficient_info — needs LLM fallback",
                    anchor["line"],
                )
            else:
                logger.info(
                    "  Anchor L%d: consistent", anchor["line"],
                )

        # Pass B: LLM fallback for insufficient_info anchors
        for anchor in llm_fallback_anchors:
            constraint_str = concretized.get(anchor["variable"], "(no constraint)")
            try:
                prompt = EARLY_FEASIBILITY_PROMPT.format(
                    bug_type=parsed.metadata.bug_type,
                    bug_location=parsed.metadata.bug_file,
                    bug_line=parsed.metadata.bug_line,
                    anchor_line=anchor["line"],
                    anchor_branch="TRUE" if anchor["branch"] else "FALSE",
                    anchor_condition=anchor["description"],
                    anchor_variable=anchor["variable"] or "(unknown)",
                    completed_constraint=constraint_str,
                    deterministic_result=anchor["deterministic_result"],
                )
                raw = self._call_llm(EARLY_FEASIBILITY_SYSTEM_PROMPT, prompt)
                data = self._parse_json_response(
                    raw, "early_feasibility_fallback"
                )
                determination = data.get("determination", "insufficient_info")
                if determination == "contradictory":
                    reason = data.get("detected_contradiction")
                    conflicts.append(
                        f"Anchor L{anchor['line']} "
                        f"({'TRUE' if anchor['branch'] else 'FALSE'} branch): "
                        f"{reason or data.get('reasoning', 'LLM detected contradiction')}"
                    )
                    logger.info(
                        "  Anchor L%d: LLM fallback → CONTRADICTORY",
                        anchor["line"],
                    )
                else:
                    logger.info(
                        "  Anchor L%d: LLM fallback → consistent",
                        anchor["line"],
                    )
            except (RuntimeError, json.JSONDecodeError) as e:
                logger.warning(
                    "LLM fallback for anchor L%d failed: %s",
                    anchor["line"], e,
                )
                # Optimistic: assume consistent on error

        is_early_feasible = len(conflicts) == 0
        if not is_early_feasible:
            logger.info(
                "Step 3.5: %d conflict(s) found — path is infeasible (FP).",
                len(conflicts),
            )

        return is_early_feasible, conflicts

    # ------------------------------------------------------------------
    # Semantic finite-domain (range-variable) conflict check
    # ------------------------------------------------------------------
    def _semantic_range_conflict_check(self, parsed: ParsedReport) -> list[str]:
        """Check the reported path as a whole for a range/finite-domain conflict.

        CSA may report as "feasible" a path that forces a guard-pinned variable
        (one compared against a small set of fixed constants, often produced by an
        opaque constant factory such as ``fourcc``) to be simultaneously inside and
        outside its pinned set — e.g. a null deref reached by taking every
        pointer-assigning dispatch arm FALSE, while the enclosing region that gates
        the deref is only entered when the variable equals one of those same tags.

        The variable's recovered finite domain (see domain_facts) is injected as
        semantic-completion evidence and the LLM is asked (SEMANTIC_FEASIBILITY)
        whether the reported path constraints conflict on such a variable.  On
        ``contradictory`` the report is FP; the conflict is returned so the caller's
        existing FP short-circuit (before Step 4 path selection) fires.

        Best effort and conservative: if no guard-pinned variable is recovered, or
        the LLM is not decisive, no conflict is returned.
        """
        snippets = getattr(parsed, "source_snippets", None) or {}
        if not snippets:
            return []
        bug_fn = getattr(parsed.metadata, "function_name", "") or ""
        bug_line = int(getattr(parsed.metadata, "bug_line", 0) or 0)

        # Clean control skeleton (drop READ1/READVECTOR macro-expanded noise).
        clean = {
            ln: txt
            for ln, txt in snippets.items()
            if not _is_macro_noise(txt)
        }
        if not clean:
            return []
        skeleton = "\n".join(f"{ln}: {clean[ln]}" for ln in sorted(clean))
        facts_text = format_domain_facts(
            domain_facts_from_lines(clean, function_name=bug_fn)
        )
        if not facts_text:
            # No guard-pinned / finite-domain variable => nothing to conflict.
            return []

        # Reconstruct the reported control decisions over control lines only, so
        # macro-internal READ loops (about `ret`, not the guard variable) are excluded.
        clean_lines = set(clean)
        events: list[str] = []
        for evt in getattr(parsed, "path_events", []) or []:
            if not evt.get("is_control_flow"):
                continue
            line = int(evt.get("line_number", 0) or 0)
            if line <= 0 or line not in clean_lines:
                continue
            desc = evt.get("description", "") or ""
            bc = evt.get("branch_condition")
            if bc in (True, False):
                events.append(f"L{line}: {'TRUE' if bc else 'FALSE'}")
            elif "condition is true" in desc or "is true" in desc:
                events.append(f"L{line}: TRUE (region entry)")
        if not events:
            # No usable control decision over the guard variable.
            return []

        # LLM prompt — scalars are .format()-substituted; skeleton + facts are
        # appended RAW afterwards because C++ source contains '{' / '}'.
        prompt = SEMANTIC_FEASIBILITY_PROMPT.format(
            bug_type=getattr(parsed.metadata, "bug_type", "") or "",
            bug_function=bug_fn or "(unknown)",
            bug_location=getattr(parsed.metadata, "bug_file", "") or "",
            bug_line=bug_line,
            path_decisions="\n".join(events),
        )
        prompt += (
            "\n\n### Bug-Function Control Source (report region)\n"
            f"```cpp\n{skeleton}\n```\n"
        )
        prompt += (
            "\n\n### Recovered Guard-Pinned Finite-Domain Facts (semantic completion)\n"
            f"{facts_text}\n"
        )
        try:
            raw = self._call_llm(SEMANTIC_FEASIBILITY_SYSTEM_PROMPT, prompt)
            data = self._parse_json_response(raw, "semantic_feasibility")
        except (RuntimeError, json.JSONDecodeError) as e:
            logger.warning("Semantic feasibility LLM call failed: %s", e)
            return []

        determination = data.get("determination", "consistent")
        logger.info(
            "Semantic feasibility → %s (guard var: %s)",
            determination,
            data.get("guard_variable", "(unknown)"),
        )
        if determination == "contradictory":
            reason = (
                data.get("detected_contradiction")
                or data.get("reasoning")
                or "guard variable forced into contradictory memberships on the path"
            )
            return [
                "Semantic finite-domain conflict: "
                f"({data.get('guard_variable', 'guard var')}) {reason}"
            ]
        return []

    def semantic_fp_reason(self, parsed: ParsedReport) -> str | None:
        """Decisive pre-enumeration FP gate: reason string or None.

        Runs :meth:`_semantic_range_conflict_check` on the reported path.  When the
        reported path forces a guard-pinned / finite-domain variable into a
        self-contradiction (admitted into a deref region only when ``h ∈ S``, yet
        every pointer-assigning dispatch arm taken FALSE so the pointer stays null),
        returns a human-readable FP explanation so the caller can short-circuit
        BEFORE enumerating the (huge) path space.  Returns ``None`` otherwise
        (no conflict, or the LLM was not decisive) — the pipeline proceeds normally.
        """
        conflicts = self._semantic_range_conflict_check(parsed)
        return " ".join(conflicts) if conflicts else None

    # ------------------------------------------------------------------
    # Step 4: Segment-by-Segment Feasible Path Selection
    # ------------------------------------------------------------------

    def _select_single_segment_path(
        self,
        analysis_result: AnalysisResult,
        completed: CompletedPath,
        segment: object,  # SegmentInfo from cfg_feasibility
        segment_index: int,
        num_segments: int,
        preceding_state: str,
        constraints_str: str,
        global_constraints: str,
    ) -> dict | None:
        """Ask LLM to select a feasible path through a single segment.

        Returns the LLM response dict (with ``is_feasible``, ``branch_decisions``,
        ``established_state``), or ``None`` on failure.
        """
        parsed = analysis_result.parsed_report

        # Source context (prefer sliced; fall back to segment code from entry)
        if analysis_result.reduced_code:
            source_ctx_str = analysis_result.reduced_code
        else:
            source_ctx_str = segment.code_from_entry if hasattr(segment, 'code_from_entry') else ""
            if not source_ctx_str:
                source_ctx = self._source_extractor.extract(parsed)
                source_ctx_str = source_ctx.format_for_llm(max_lines=60)

        # Format control events in this segment from the reported path
        source_lines = self._build_event_source_index(parsed)
        branch_events = self._format_branch_decisions(
            parsed.path_events, source_lines=source_lines,
        )

        # Format preconditions
        preconditions_str = "\n".join(
            f"  - {p}" for p in completed.preconditions
        ) if completed.preconditions else "  (none)"

        prompt = SEGMENT_PATH_SELECTION_PROMPT.format(
            segment_index=segment_index,
            num_segments=num_segments,
            bug_type=parsed.metadata.bug_type,
            bug_location=parsed.metadata.bug_file,
            bug_line=parsed.metadata.bug_line,
            bug_function=parsed.metadata.function_name or "unknown",
            segment_label=getattr(segment, 'label', f'S{segment_index}'),
            line_start=getattr(segment, 'line_start', 0),
            line_end=getattr(segment, 'line_end', 0),
            function_name=getattr(segment, 'function_name', ''),
            source_context=source_ctx_str,
            completed_constraints=constraints_str,
            preceding_state=preceding_state,
            branch_events=branch_events,
            global_constraints=global_constraints or "(no project-wide constraints)",
        )

        # ── Inject library function API contracts (from static knowledge base) ──
        if analysis_result.reduced_code:
            api_contracts = self._format_library_api_contracts(
                analysis_result.reduced_code,
            )
            if api_contracts:
                prompt += api_contracts + "\n"

        # ── Inject domain-specific knowledge (union type-punning, loop init, etc.) ──
        if analysis_result.reduced_code:
            domain_knowledge = self._inject_domain_specific_knowledge(
                analysis_result.reduced_code,
            )
            if domain_knowledge:
                prompt += domain_knowledge + "\n"

        try:
            raw = self._call_llm(SEGMENT_PATH_SELECTION_SYSTEM_PROMPT, prompt)
            data = self._parse_json_response(raw, f"segment_{segment_index}_selection")
            return data
        except (RuntimeError, json.JSONDecodeError) as e:
            logger.warning("Segment %d path selection failed: %s", segment_index, e)
            return None

    def _check_global_path_consistency(
        self,
        analysis_result: AnalysisResult,
        completed: CompletedPath,
        selections: list[dict],
        constraints_str: str,
        global_constraints: str,
    ) -> dict:
        """Check whether merged segment selections are globally consistent.

        Calls the LLM to verify that the combined state across all segments
        has no contradictions.
        """
        parsed = analysis_result.parsed_report

        # Format segment selections for the prompt
        segment_parts: list[str] = []
        for sel in selections:
            seg_idx = sel.get("segment_index", "?")
            decisions = sel.get("branch_decisions", [])
            state = sel.get("established_state", "(unknown)")
            decisions_str = self._format_selected_decisions(decisions)
            segment_parts.append(
                f"### Segment {seg_idx}\n"
                f"- Branch decisions: {decisions_str}\n"
                f"- Established state: {self._clip_state(state)}\n"
            )

        segment_selections_text = "\n".join(segment_parts)

        prompt = PATH_MERGE_CHECK_PROMPT.format(
            bug_type=parsed.metadata.bug_type,
            bug_location=parsed.metadata.bug_file,
            bug_line=parsed.metadata.bug_line,
            bug_description=parsed.metadata.bug_description,
            completed_constraints=constraints_str,
            global_constraints=global_constraints or "(no project-wide constraints)",
            segment_selections=segment_selections_text,
        )

        try:
            raw = self._call_llm(PATH_MERGE_CHECK_SYSTEM_PROMPT, prompt)
            return self._parse_json_response(raw, "path_merge_check")
        except (RuntimeError, json.JSONDecodeError) as e:
            logger.warning("Path merge check failed: %s", e)
            return {"is_globally_feasible": False}

    def _select_combination_segments(
        self,
        analysis_result: AnalysisResult,
        completed: CompletedPath,
        segments: list,
        path_space: dict,
        combo: tuple,
    ) -> list[dict] | None:
        """Deterministically build + conflict-check ONE whole-path combination.

        ``combo`` is a 0-based candidate index per segment into
        ``path_space["per_segment"][i]["candidates"]``.  The selections are built
        without any LLM pick (from ``combination_enum.build_selections_from_combo``)
        and then run through the same global merge check as the LLM selection.
        Returns the
        ``selections`` list if the combination is globally feasible, else ``None``
        — meaning THAT whole-path combination is infeasible (a whole-path FP), NOT
        that the report is FP: the caller blocks the combination and enumerates the
        next one.
        """
        from llm_client.fp_analysis.combination_enum import build_selections_from_combo

        parsed = analysis_result.parsed_report
        constraints_str, _preconditions_str, _concretized_str = (
            self._format_constraints_for_prompt(completed)
        )
        global_constraints = self._load_global_constraints(
            self._source_root or analysis_result.source_root,
            target_classes=self._extract_class_names_from_path(parsed),
        ) or "(no project-wide constraints)"

        selections = build_selections_from_combo(path_space, combo, segments)

        merge = self._check_global_path_consistency(
            analysis_result=analysis_result,
            completed=completed,
            selections=selections,
            constraints_str=constraints_str,
            global_constraints=global_constraints,
        )
        if merge and merge.get("is_globally_feasible", False):
            return selections
        logger.info(
            "combination: combo %s globally infeasible (whole-path FP).",
            tuple(combo),
        )
        return None

    def _select_feasible_path_segments(
        self,
        analysis_result: AnalysisResult,
        completed: CompletedPath,
        slice_mask: SliceMask | None = None,
        max_rounds: int = 3,
        selector: IterativePathSelector | None = None,
        segments: list | None = None,
        cfg_cache: dict | None = None,
        hard_constraints: list[str] | None = None,
        blocked_paths: list[str] | None = None,
        blocked_candidates: list[dict] | None = None,
    ) -> list[dict] | None:
        """Select a globally feasible execution path with conflict-driven learning.

        *blocked_paths*: optional list of previously-tried whole-path signatures
        (branch-combination strings) that were later abandoned because their
        simulation was conflicted/unresolved.  When non-empty, the LLM is told to
        pick a DIFFERENT feasible combination rather than re-derive a blocked path.

        *blocked_candidates*: optional list of ``{segment_index: selected_candidate}``
        maps, one per previously-tried path.  This is the *deterministic* half of
        blocked-path avoidance: each blocked candidate index is excluded from the
        per-segment candidate list before the LLM sees it, so re-selection is
        guaranteed to pick a genuinely different branch combination (the LLM-only
        ``blocked_paths`` prompt hint alone is advisory — the LLM keeps re-picking
        the top-ranked candidate).

        How it differs from a plain best-first walk:

        - **ShortestPathComputer**: ranks branch candidates by (S, C, V) metrics
          so the LLM sees structured options instead of raw branch trees.
        - **ConflictPatternLearner**: extracts 8-layer conflict patterns from
          merge failures, guides re-selection with avoidance rules.
        - **Frozen segments**: non-conflicting segments are carried forward;
          only conflicting segments are re-selected.
        - **Stuck detection**: exits early when the same conflict repeats.

        Args:
            analysis_result: Parsed report and metadata.
            completed: Completed path constraints from Step 2-3.
            slice_mask: Post-slice retained/removed node tracking.
            max_rounds: Maximum re-selection rounds (default 3).
            selector: Optional IterativePathSelector for PDG pruning.
            segments: Pre-built segments with optional ``cfg_branch_tree``.
                If ``None``, segments are created from the parsed report.
            cfg_cache: Optional CFG cache for branch extraction fallback.

        Returns:
            List of segment selection dicts, or ``None`` if infeasible.
        """
        from llm_client.fp_analysis.shortest_path import (
            CalleePathCache,
            ShortestPathComputer,
        )
        from llm_client.fp_analysis.conflict_learner import (
            ConflictPatternLearner,
        )
        from llm_client.fp_analysis.cfg_feasibility import (
            segment_path as split_into_segments,
        )

        parsed = analysis_result.parsed_report

        # Step 1: Use provided segments or split internally
        if segments is None:
            segments = split_into_segments(parsed, sdg=self._sdg)
        if not segments:
            logger.info("Linear path — no segment selection needed.")
            return [{
                "segment_index": 0,
                "branch_decisions": [],
                "established_state": "linear path",
            }]

        logger.info("%d segment(s) for conflict-driven path selection.", len(segments))

        # Step 2: Initialise shortest-path and conflict-learning infrastructure
        callee_cache = CalleePathCache(self._sdg) if self._sdg else CalleePathCache(None)
        shortest_pc = ShortestPathComputer(self._sdg, callee_cache)
        learner = ConflictPatternLearner(
            provider=self._provider,
            model=self._model,
        )

        # Step 3: Prepare shared context
        constraints_str, preconditions_str, concretized_str = (
            self._format_constraints_for_prompt(completed)
        )
        global_constraints = self._load_global_constraints(
            self._source_root or analysis_result.source_root,
            target_classes=self._extract_class_names_from_path(parsed),
        ) or "(no project-wide constraints)"

        # Step 4: Round-based re-selection loop
        current_selections: dict[int, dict] = {}
        accumulated_patterns: list = []
        conflicting_segs: list[int] = []

        for round_num in range(1, max_rounds + 1):
            logger.info("Round %d/%d ...", round_num, max_rounds)

            if round_num == 1:
                # ── First round: select all segments ──
                new_selections = self._select_all_segments(
                    segments=segments,
                    shortest_pc=shortest_pc,
                    constraints_str=constraints_str,
                    parsed=parsed,
                    global_constraints=global_constraints,
                    hard_constraints=hard_constraints,
                    blocked_paths=blocked_paths,
                    blocked_candidates=blocked_candidates,
                )
                if new_selections is None:
                    return None
                current_selections = new_selections
            else:
                # ── Re-select only conflicting segments ──
                if not conflicting_segs:
                    logger.info("No conflicting segments identified — accepting current")
                    break

                new_selections = self._reselect_conflicting(
                    segments=segments,
                    current_selections=current_selections,
                    conflict_segs=set(conflicting_segs),
                    shortest_pc=shortest_pc,
                    accumulated_patterns=accumulated_patterns,
                    learner=learner,
                    round_num=round_num,
                    constraints_str=constraints_str,
                    parsed=parsed,
                    blocked_paths=blocked_paths,
                )
                if new_selections is None:
                    return None
                current_selections = new_selections

            # ── Global merge check ──
            merge_result = self._check_global_path_consistency(
                analysis_result=analysis_result,
                completed=completed,
                selections=list(current_selections.values()),
                constraints_str=constraints_str,
                global_constraints=global_constraints,
            )

            if merge_result and merge_result.get("is_globally_feasible", False):
                logger.info("Globally consistent path found in round %d", round_num)
                return list(current_selections.values())

            # ── Track conflicting segments for next round ──
            conflicting_segs = merge_result.get("conflicting_segments", []) if merge_result else []
            logger.info(
                "Round %d global conflict — %d segment(s) conflicting.",
                round_num, len(conflicting_segs) if conflicting_segs else len(segments),
            )

            # ── Learn from the conflict ──
            segment_states = {
                seg_idx: {"postcondition": getattr(seg, "postcondition", ""),
                          "function_name": getattr(seg, "function_name", ""),
                          "line_start": getattr(seg, "line_start", 0),
                          "line_end": getattr(seg, "line_end", 0)}
                for seg_idx, seg in enumerate(segments)
            }
            patterns = learner.learn(
                merge_result=merge_result,
                segment_selections=current_selections,
                segment_states=segment_states,
            )

            # ── Stuck detection ──
            prev_sigs = {p.pattern_signature for p in accumulated_patterns}
            curr_sigs = {p.pattern_signature for p in patterns}
            new_hints = any(
                p.repair_hints for p in patterns
                if not accumulated_patterns
                or p.pattern_signature not in prev_sigs
            )
            if learner._is_stuck(prev_sigs, curr_sigs, round_num, new_hints):
                logger.info(
                    "Stuck after round %d — path infeasible (FP).", round_num,
                )
                return None

            accumulated_patterns.extend(patterns)

        # Exhausted all rounds
        logger.info("Exhausted max_rounds=%d — path infeasible (FP).", max_rounds)
        return None

    # ------------------------------------------------------------------
    # Assumption feasibility checking (between branch filter and path selection)
    # ------------------------------------------------------------------

    def check_assumptions_feasibility(
        self,
        segments: list,
        parsed,
        global_constraints: str = "",
    ) -> tuple[list[str], dict[str, str]]:
        """Check CSA assumptions across segments and produce **hard constraints**
        for the path selection pipeline.

        Unlike the previous cascading-failure design, this method always
        returns (the path selection itself discovers conflicts and returns
        FP).  The returned hard constraints are injected into path selection
        so it can detect contradictions against the CSA path's own assumptions.

        Returns:
            ``(hard_constraints, cumulative_state)`` where *hard_constraints*
            is a list of constraint strings (e.g. ``["blockLength_ > 0"]``)
            that must be satisfied by any feasible path.
        """
        hard_constraints: list[str] = []
        cumulative_state: dict[str, str] = {}

        for seg_idx, segment in enumerate(segments):
            if parsed is None or not hasattr(parsed, "path_events"):
                continue

            facts = extract_assumptions_from_events(
                parsed.path_events,
                getattr(segment, "event_range", (0, 0)),
                segment_index=seg_idx,
            )
            if not facts:
                continue

            # ── Tier 1: rule-based check ──
            for fact in facts:
                result = self._layer1_rule_check(fact, global_constraints)
                fact.layer1_result = result
                if result == "VIOLATED":
                    logger.info(
                        "Assumption check (Tier 1): Seg %d, Event #%d: '%s %s %s' "
                        "→ VIOLATED.  Adding negation constraint.",
                        seg_idx, fact.event_number,
                        fact.variable, fact.operator, fact.value,
                    )

            # ── Tier 2: LLM-based check for UNCERTAIN facts ──
            uncertain = [f for f in facts if f.layer1_result == "UNCERTAIN"]
            tier2_result = None
            if uncertain:
                preceding_state = "\n".join(
                    f"  - {k}: {v}" for k, v in cumulative_state.items()
                ) or "(no prior state)"

                tier2_result = self._layer2_llm_check(
                    uncertain_facts=uncertain,
                    cumulative_state=preceding_state,
                    global_constraints=global_constraints,
                    segment=segment,
                    parsed=parsed,
                )
                # Merge state bindings
                for binding in tier2_result.get("state_bindings", []):
                    if isinstance(binding, dict):
                        cumulative_state[binding.get("var", "")] = binding.get("value", "")

            # ── Build hard constraints from Tier-1 + Tier-2 results ──
            segment_hc = self._build_hard_constraints_from_assumptions(
                facts, tier2_results=tier2_result,
            )
            hard_constraints.extend(segment_hc)
            if segment_hc:
                logger.info(
                    "Assumption check: Seg %d → %d hard constraint(s): %s",
                    seg_idx, len(segment_hc),
                    "; ".join(segment_hc[:3]),
                )

            # ── Propagate bindings to cumulative_state ──
            for fact in facts:
                if fact.layer1_result not in ("VIOLATED",):
                    key = f"{fact.variable}@L{fact.line_number}"
                    cumulative_state[key] = (
                        f"{fact.variable} {fact.operator} {fact.value}"
                    )

        return hard_constraints, cumulative_state

    def _build_hard_constraints_from_assumptions(
        self,
        facts: list[AssumptionFact],
        tier2_results: dict | None = None,
    ) -> list[str]:
        """Convert assumption-check results into hard constraint strings.

        - VIOLATED facts → negation constraint
          (e.g. ``blockLength_ <= 0`` violated → ``blockLength_ > 0``)
        - CONSISTENT ``==`` facts → equality binding
          (e.g. ``count == 128``)
        - UNCERTAIN facts → constraints from tier2_results (LLM-provided)
        """
        constraints: list[str] = []

        for fact in facts:
            if fact.layer1_result == "VIOLATED":
                # Negate the violated assumption
                negation = _negate_constraint(fact.variable, fact.operator, fact.value)
                if negation:
                    constraints.append(negation)
            elif fact.layer1_result == "CONSISTENT" and fact.operator == "==":
                constraints.append(f"{fact.variable} == {fact.value}")

        # Merge Tier-2 constraints
        if tier2_results:
            for c in tier2_results.get("hard_constraints", []):
                if c and c not in constraints:
                    constraints.append(c)

        return constraints

    def _check_constraint_conflict(
        self,
        segment,
        parsed,
        hard_constraints: list[str],
    ) -> str | None:
        """Check whether CSA assumptions in *segment* conflict with *hard_constraints*.

        Returns a human-readable conflict description if a contradiction is
        found, or ``None`` if all assumptions are compatible (or no assumptions
        are extractable).

        This is called from :meth:`_select_all_segments` **before** branch
        candidate computation.  When a conflict is detected, the path selection
        returns ``None`` (FP) — the conflict detection is *part of* path
        selection, not a separate pre-filter.
        """
        if not hard_constraints:
            return None
        if parsed is None or not hasattr(parsed, "path_events"):
            return None

        facts = extract_assumptions_from_events(
            parsed.path_events,
            getattr(segment, "event_range", (0, 0)),
            segment_index=getattr(segment, "segment_index", 0),
        )
        if not facts:
            return None

        for fact in facts:
            for hc in hard_constraints:
                if _constraints_conflict(fact, hc):
                    return (
                        f"Hard constraint '{hc}' contradicts CSA assumption "
                        f"'{fact.variable} {fact.operator} {fact.value}' "
                        f"at Event #{fact.event_number} L{fact.line_number}"
                    )

        return None

    # ------------------------------------------------------------------
    # Tier-1 + Tier-2 (unchanged internals)
    # ------------------------------------------------------------------

    def _layer1_rule_check(
        self,
        fact: AssumptionFact,
        global_constraints: str,
    ) -> str:
        """Tier-1: deterministic rule-based check against the constraint cache.

        Returns one of ``"VIOLATED"``, ``"CONSISTENT"``, or ``"UNCERTAIN"``.

        Rules (applied in order):
        1. **No constraints** → UNCERTAIN.
        2. **Exact equality match** (high confidence) → CONSISTENT.
        3. **Equality conflict** (high confidence, different value) → VIOLATED.
        4. **Multiple conflicting constraints** → UNCERTAIN.
        5. **Low-confidence constraint** → UNCERTAIN.
        6. **Non-equality operator** → UNCERTAIN (range constraints need
           semantic reasoning beyond Tier-1).
        """
        if not global_constraints:
            return "UNCERTAIN"

        # Build a simple index from the constraints text:
        # field name → [(value, confidence), ...]
        field_values: dict[str, list[tuple[str, str]]] = {}
        for m in re.finditer(
            r"field\s+'(\w+(?:\.\w+)*)'\s+initial_value\s+'([^']+)'\s+"
            r"\(confidence:\s*(high|low|medium)\)",
            global_constraints,
        ):
            fname = m.group(1)
            fval = m.group(2)
            conf = m.group(3).lower()
            field_values.setdefault(fname, []).append((fval, conf))

        var = fact.variable
        if var not in field_values:
            return "UNCERTAIN"

        entries = field_values[var]

        # Only equality checks can produce definitive results at Tier 1
        if fact.operator == "==":
            for val, conf in entries:
                if conf != "high":
                    continue
                if val == fact.value:
                    return "CONSISTENT"
                # Different value → conflict
                return "VIOLATED"

            # No high-confidence match found
            return "UNCERTAIN"

        # For non-equality operators (<=, <, >, >=, !=):
        # We can detect clear violations (e.g. known=128, assumption <=0)
        # but CANNOT confirm consistency — range constraints need semantic
        # reasoning beyond Tier-1's pure value matching.
        high_vals = [v for v, c in entries if c == "high"]
        if len(high_vals) == 0:
            return "UNCERTAIN"

        # Try numeric comparison for a single known value
        if len(high_vals) == 1:
            try:
                known = int(high_vals[0])
                assumed = int(fact.value)
            except (ValueError, TypeError):
                return "UNCERTAIN"

            # Check if the assumption is _contradicted_ by the known value
            op = fact.operator
            contradicted = False
            if op == "<=" and known > assumed:
                contradicted = True
            elif op == "<" and known >= assumed:
                contradicted = True
            elif op == ">" and known <= assumed:
                contradicted = True
            elif op == ">=" and known < assumed:
                contradicted = True
            elif op == "!=" and known == assumed:
                contradicted = True

            if contradicted:
                return "VIOLATED"
            # Even if compatible, we cannot prove CONSISTENT —
            # the known value is just one possible input.
            return "UNCERTAIN"

        # Multiple high-confidence values → cannot decide
        return "UNCERTAIN"

    def _layer2_llm_check(
        self,
        uncertain_facts: list[AssumptionFact],
        cumulative_state: str,
        global_constraints: str,
        segment,
        parsed,
    ) -> dict:
        """Tier-2: LLM-based feasibility check for assumptions Tier-1 cannot decide.

        Batches all uncertain assumptions for a segment into a single LLM
        call.  Returns a structured dict with ``is_infeasible``,
        ``hard_constraints``, and ``state_bindings`` keys.

        When the LLM provider is unavailable (e.g. in test contexts), returns
        a safe default (not infeasible, no constraints), falling back to the
        existing branch-only behavior.
        """
        if not uncertain_facts:
            return {
                "is_infeasible": False,
                "hard_constraints": [],
                "state_bindings": [],
            }

        # Build the assumption list for the prompt
        fact_lines: list[str] = []
        for f in uncertain_facts:
            fact_lines.append(
                f"  - Event #{f.event_number} L{f.line_number}: {f.raw_description}\n"
                f"    → {f.variable} {f.operator} {f.value}\n"
                f"    (error_start={f.is_error_start})"
            )
        fact_text = "\n".join(fact_lines)

        # Get source context from the segment
        source_ctx = getattr(segment, "code_from_entry", "") or ""

        fn_name = getattr(segment, "function_name", "?")
        ls = getattr(segment, "line_start", 0)
        le = getattr(segment, "line_end", 0)

        bug_meta = ""
        if parsed is not None and hasattr(parsed, "metadata"):
            m = parsed.metadata
            bug_meta = (
                f"- **Bug Type**: {getattr(m, 'bug_type', '?')}\n"
                f"- **Bug Location**: {getattr(m, 'bug_file', '?')}:"
                f"{getattr(m, 'bug_line', '?')}\n"
            )

        prompt = (
            f"## Source Context ({fn_name} L{ls}-L{le})\n"
            f"```cpp\n{source_ctx[:8000]}\n```\n\n"
            f"## Prior State from Preceding Segments\n"
            f"{cumulative_state}\n\n"
            f"## Uncertain Assumptions to Evaluate\n"
            f"{fact_text}\n\n"
            f"## External Constraints\n"
            f"{global_constraints or '(no project-wide constraints)'}\n\n"
            f"## Bug Metadata\n"
            f"{bug_meta}\n"
        )

        # Try LLM; if unavailable, return safe default
        try:
            if self._provider is None:
                logger.debug(
                    "Layer2: no LLM provider — returning safe default "
                    "(not infeasible)."
                )
                return {
                    "is_infeasible": False,
                    "hard_constraints": [],
                    "state_bindings": [],
                }

            system = (
                "You are a C++ static analysis expert. Evaluate whether "
                "the CSA (Clang Static Analyzer) assumptions listed below "
                "are feasible under real-world constraints.\n\n"
                "For each assumption, determine:\n"
                "- **FEASIBLE**: real-world inputs exist where this "
                "assumption holds.\n"
                "- **INFEASIBLE**: this assumption contradicts real-world "
                "constraints, type semantics, or the known behavior of the "
                "program.\n\n"
                "For INFEASIBLE assumptions, provide **hard_constraints**: "
                "what MUST be true instead (e.g. if CSA assumes "
                "'blockLength_ <= 0' is infeasible, the hard constraint is "
                "'blockLength_ > 0').\n\n"
                "Output JSON:\n"
                "{\"assumptions\": [{\"event\": N, \"is_feasible\": bool, "
                "\"reasoning\": \"...\", "
                "\"hard_constraints\": [\"constraint1\", ...], "
                "\"state_bindings\": [{\"var\": \"...\", \"value\": \"...\"}]}], "
                "\"overall_infeasible\": bool}"
            )

            raw = self._call_llm(system, prompt)
            data = self._parse_json_response(raw, "assumption_feasibility")
            is_infeasible = data.get("overall_infeasible", False)
            bindings: list[dict] = []
            hard_constraints: list[str] = []
            for a in data.get("assumptions", []):
                if not a.get("is_feasible", True):
                    is_infeasible = True
                    for c in a.get("hard_constraints", []):
                        if c and c not in hard_constraints:
                            hard_constraints.append(c)
                for b in a.get("state_bindings", []):
                    bindings.append(b)

            if is_infeasible:
                logger.info(
                    "Layer2 LLM: %d assumption(s) → INFEASIBLE "
                    "(%d hard constraint(s))",
                    len(uncertain_facts), len(hard_constraints),
                )
            else:
                logger.info(
                    "Layer2 LLM: %d assumption(s) → feasible "
                    "(%d state binding(s))",
                    len(uncertain_facts), len(bindings),
                )
            return {
                "is_infeasible": is_infeasible,
                "hard_constraints": hard_constraints,
                "state_bindings": bindings,
            }
        except (RuntimeError, json.JSONDecodeError) as e:
            logger.warning(
                "Layer2 LLM call failed: %s — returning safe default.", e,
            )
            return {
                "is_infeasible": False,
                "hard_constraints": [],
                "state_bindings": [],
            }

    # ------------------------------------------------------------------
    # conflict-driven path selection
    # ------------------------------------------------------------------

    def _select_all_segments(
        self,
        segments: list,
        shortest_pc: object,
        constraints_str: str,
        parsed: object,
        global_constraints: str,
        hard_constraints: list[str] | None = None,
        blocked_paths: list[str] | None = None,
        blocked_candidates: list[dict] | None = None,
    ) -> dict[int, dict] | None:
        """First-round selection: compute shortest-path candidates for each segment
        and let LLM choose the best via ``PATH_CANDIDATE_SELECTION_PROMPT``.

        *blocked_paths* (whole-path signatures tried and abandoned earlier) is
        appended to each segment prompt so the LLM avoids re-deriving them.

        *blocked_candidates* (list of ``{segment_index: selected_candidate}`` maps,
        one per blocked path) is the deterministic enforcement: candidate indices
        that a blocked path already chose for this segment are excluded from the
        candidate list *before* the LLM sees it, guaranteeing re-selection picks a
        genuinely different branch combination (not merely "prompted" to).

        *hard_constraints* are injected by the assumption feasibility check
        (Step 5.5).  If a segment's CSA assumptions contradict these
        constraints, ``None`` is returned (FP) — the conflict is discovered
        **by path selection**, not by a separate pre-filter.

        Returns ``{{seg_idx: selection_dict}}`` or ``None`` on failure.
        """
        from llm_client.fp_analysis.prompt_templates import (
            PATH_CANDIDATE_SELECTION_SYSTEM_PROMPT,
            PATH_CANDIDATE_SELECTION_PROMPT,
        )

        selections: dict[int, dict] = {}
        cumulative_state: dict[str, str] = {}

        for seg_idx, segment in enumerate(segments):
            # ═══ Constraint conflict check (from assumption feasibility) ═══
            if hard_constraints:
                conflict = self._check_constraint_conflict(
                    segment, parsed, hard_constraints,
                )
                if conflict is not None:
                    logger.info(
                        "Seg %d — CONSTRAINT CONFLICT: %s → no feasible path (FP).",
                        seg_idx, conflict,
                    )
                    return None  # ← Path selection discovers the FP!

            # Build preceding state from cumulative state (same pattern as v1)
            if selections:
                last_state = selections[seg_idx - 1].get("established_state", "")
                state_lines = "\n".join(
                    f"  - {var} {val}" for var, val in cumulative_state.items()
                )
                preceding_state = (
                    "## Cumulative State from Previous Segments\n"
                    f"{state_lines}\n\n"
                    f"## Latest Segment State\n{self._clip_state(last_state)}\n"
                )
            else:
                preceding_state = "Function entry / no prior events."

            # Compute shortest-path candidates from the CFG branch tree.
            # The postcondition is passed to compute_topk so filtering
            # happens inside (before the k-cut), guaranteeing the top-k
            # are constraint-respecting.
            tree = getattr(segment, "cfg_branch_tree", None)
            candidates = []
            if tree is not None and getattr(tree, "root_branches", None):
                pc = getattr(segment, "postcondition", None) or getattr(
                    tree, "postcondition", None,
                )
                for rb in tree.root_branches:
                    tag = rb.tag.verdict if rb.tag else "NONE"
                    logger.info(
                        "Seg %d root: L%d %s | tag=%s | children=%d",
                        seg_idx, rb.source_line, rb.condition_expr[:60],
                        tag, len(rb.children),
                    )
                candidates = shortest_pc.compute_topk(tree, k=3, postcondition=pc)
            else:
                candidates = [([], None)]  # Linear segment — no branches
                pc = None

            # ── Deterministic blocked-candidate exclusion (stable indices) ──
            # Previously-tried paths chose a specific candidate index for this
            # segment.  We keep the FULL candidate list with STABLE 1-based
            # indices (never drop/renumber, which would make the recorded
            # ``selected_candidate`` ambiguous across attempts).  Instead we
            # compute the set of already-tried indices, surface them to the LLM,
            # and later deterministically remap any LLM pick that lands on a
            # blocked index to the next unblocked one.  This guarantees
            # re-selection yields a genuinely different branch combination.
            excluded_indices: set[int] = set()
            if blocked_candidates and len(candidates) > 1:
                for cmap in blocked_candidates:
                    c = cmap.get(seg_idx)
                    if isinstance(c, int) and c >= 1 and c <= len(candidates):
                        excluded_indices.add(c)
                if excluded_indices:
                    logger.info(
                        "Seg %d — %d previously-tried candidate index(es) %s "
                        "will be avoided (stable indices, %d total candidates).",
                        seg_idx, len(excluded_indices), sorted(excluded_indices),
                        len(candidates),
                    )

            candidate_text = self._format_candidates(candidates, postcondition=pc)

            # If linear segment, confirm it directly without LLM call
            if not candidates or (len(candidates) == 1 and not candidates[0][0]):
                # Build a meaningful state description from the segment metadata
                fn = getattr(segment, "function_name", "?")
                label = getattr(segment, "label", f"S{seg_idx}")
                ls = getattr(segment, "line_start", 0)
                le = getattr(segment, "line_end", 0)
                pc_text = f", postcondition: {pc.condition_expr}" if pc and getattr(pc, "condition_expr", None) else ""
                state_desc = (
                    f"Linear segment {label} ({fn} L{ls}-L{le}){pc_text}"
                )
                selection = {
                    "segment_index": seg_idx,
                    "branch_decisions": [],
                    "established_state": state_desc,
                    "is_feasible": True,
                }
                selections[seg_idx] = selection
                _merge_state(cumulative_state, state_desc)
                logger.info(
                    "Segment %d — linear (no branches), auto-confirmed.", seg_idx,
                )
                continue

            # Format prompt with candidates, constraints, and preceding state
            prompt = PATH_CANDIDATE_SELECTION_PROMPT.format(
                constraints=constraints_str,
                preceding_state=preceding_state,
                candidates=candidate_text,
            )

            # Add bug-level context
            bug_meta = getattr(parsed, "metadata", None)
            if bug_meta:
                prompt += (
                    f"\n\n### Bug Context\n"
                    f"- **Type**: {getattr(bug_meta, 'bug_type', 'unknown')}\n"
                    f"- **File**: {getattr(bug_meta, 'bug_file', '?')}:"
                    f"{getattr(bug_meta, 'bug_line', '?')}\n"
                )

            # Add blocked-path avoidance (previously-tried whole-path combos)
            if blocked_paths:
                prompt += self._format_blocked_paths(blocked_paths)

            # Per-segment stable-index avoidance: tell the LLM exactly which
            # candidate numbers were already tried for THIS segment.  (The
            # deterministic remap below enforces it regardless of the LLM.)
            if excluded_indices:
                prompt += (
                    "\n\n### AVOID these already-tried candidate numbers "
                    "for this segment\n"
                    "The following candidate number(s) were selected in an "
                    "earlier attempt and their simulated execution reached no bug "
                    "(blocked). You MUST NOT select these numbers; pick a "
                    f"different candidate.\n  - {sorted(excluded_indices)}\n"
                )

            # Call LLM
            try:
                raw = self._call_llm(
                    PATH_CANDIDATE_SELECTION_SYSTEM_PROMPT, prompt,
                )
                data = self._parse_json_response(
                    raw, f"segment_{seg_idx}_selection",
                )
            except (RuntimeError, json.JSONDecodeError) as e:
                logger.warning(
                    "Segment %d LLM selection failed: %s", seg_idx, e,
                )
                return None

            # Map selected_candidate → branch_decisions
            selected_idx = data.get("selected_candidate", 1)
            # ── Deterministic remap (stable indices) ──
            # If the LLM picked an index that a previous attempt already tried for
            # this segment, bump it to the smallest unblocked index.  This is the
            # enforcement half of blocked-candidate avoidance — it cannot be
            # skipped by the LLM re-picking the top-ranked candidate.  If every
            # index is blocked, no new distinct path exists through this segment
            # → return None (signals the exploration has exhausted the space).
            if excluded_indices and selected_idx in excluded_indices:
                alt = next(
                    (i for i in range(1, len(candidates) + 1)
                     if i not in excluded_indices),
                    None,
                )
                if alt is not None:
                    logger.info(
                        "Seg %d — LLM re-picked blocked candidate #%d; "
                        "deterministically remapped to #%d.",
                        seg_idx, selected_idx, alt,
                    )
                    selected_idx = alt
                else:
                    # All candidates of this segment are blocked.  This is NOT a
                    # conflict-based infeasibility (it doesn't prove the report is
                    # FP), so do NOT return None (which the caller treats as a hard
                    # FP).  Signal exhaustion; the caller must route it to
                    # bounded-exploration aggregation instead.
                    logger.info(
                        "Seg %d — all %d candidate(s) already tried; "
                        "no new distinct path through this segment (exhaustion).",
                        seg_idx, len(candidates),
                    )
                    raise PathSpaceExhausted(
                        f"seg {seg_idx}: all {len(candidates)} candidate(s) "
                        "already tried"
                    )
            candidate_idx = max(0, min(selected_idx - 1, len(candidates) - 1))
            chosen_decisions, _chosen_metrics = candidates[candidate_idx]

            branch_decisions = []
            for d in chosen_decisions:
                condition = getattr(d, "condition_expr", "")
                line = getattr(d, "source_line", 0)
                taken = getattr(d, "branch_taken", True)
                branch_decisions.append({
                    # v1-compatible keys (used by merge check)
                    "line": line,
                    "branch": "true" if taken else "false",
                    # selection keys (used by shortest_path)
                    "node_id": getattr(d, "node_id", ""),
                    "branch_taken": taken,
                    "condition": condition,
                    "source_line": line,
                })

            selection = {
                "segment_index": seg_idx,
                "branch_decisions": branch_decisions,
                "established_state": data.get("established_state", ""),
                "reasoning": data.get("reasoning", ""),
                "selected_candidate": selected_idx,
                "num_candidates": len(candidates),
                "is_feasible": True,
            }
            selections[seg_idx] = selection

            # Merge established_state into cumulative state
            _merge_state(cumulative_state, selection["established_state"])
            logger.info(
                "Segment %d — candidate #%d/%d selected (%d branch decisions).",
                seg_idx, selected_idx, len(candidates), len(branch_decisions),
            )

        return selections

    def _reselect_conflicting(
        self,
        segments: list,
        current_selections: dict[int, dict],
        conflict_segs: set[int],
        shortest_pc: object,
        accumulated_patterns: list,
        learner: object,
        round_num: int,
        constraints_str: str,
        parsed: object,
        blocked_paths: list[str] | None = None,
    ) -> dict[int, dict] | None:
        """Re-select only conflicting segments, freezing the rest.

        Injects avoidance rules from learned conflict patterns and frozen
        segment context into ``SEGMENT_RESELECT_PROMPT``.

        *blocked_paths* (abandoned whole-path signatures) is appended so the
        re-selection also avoids re-deriving a blocked path.

        Returns updated selections dict or ``None`` on failure.
        """
        from llm_client.fp_analysis.prompt_templates import (
            SEGMENT_RESELECT_PROMPT,
            PATH_CANDIDATE_SELECTION_SYSTEM_PROMPT,
        )

        # Freeze non-conflicting segments
        new_selections = {
            i: sel for i, sel in current_selections.items()
            if i not in conflict_segs
        }

        for seg_idx in sorted(conflict_segs):
            segment = segments[seg_idx]

            # Re-compute candidates with postcondition filtering
            # inside compute_topk (before k-cut).
            tree = getattr(segment, "cfg_branch_tree", None)
            candidates = []
            if tree is not None and getattr(tree, "root_branches", None):
                pc = getattr(segment, "postcondition", None) or getattr(
                    tree, "postcondition", None,
                )
                candidates = shortest_pc.compute_topk(tree, k=3, postcondition=pc)
            else:
                candidates = [([], None)]
                pc = None

            # Filter relevant avoidance rules for this segment
            relevant_rules = [
                p for p in accumulated_patterns
                if seg_idx in p.conflicting_segments
            ]

            avoidance_text = learner.generate_avoidance_rules(
                patterns=relevant_rules,
                current_selections=new_selections,
            ) if relevant_rules else "(no previously learned rules — select a DIFFERENT candidate)"

            # Build frozen context from non-conflicting segments
            frozen_text = "\n".join(
                f"Segment {i}: {self._clip_state(sel.get('established_state', 'N/A'))}"
                for i, sel in sorted(new_selections.items())
            )

            # Build re-selection prompt  (pc already resolved above)
            candidate_text = self._format_candidates(candidates, postcondition=pc)

            # Gather all preceding segments' established_state for context
            preceding_parts = []
            for i, sel in sorted(new_selections.items()):
                preceding_parts.append(
                    f"Segment {i}: {self._clip_state(sel.get('established_state', 'N/A'))}"
                )
            preceding_context = "\n".join(preceding_parts) if preceding_parts else "(none)"

            prompt = SEGMENT_RESELECT_PROMPT.format(
                segment_index=seg_idx,
                round=round_num,
                avoidance_rules=avoidance_text,
                candidates=candidate_text,
                frozen_context=frozen_text,
            )
            if blocked_paths:
                prompt += self._format_blocked_paths(blocked_paths)

            # Call LLM
            try:
                raw = self._call_llm(
                    PATH_CANDIDATE_SELECTION_SYSTEM_PROMPT, prompt,
                )
                data = self._parse_json_response(
                    raw, f"reselect_segment_{seg_idx}_round{round_num}",
                )
            except (RuntimeError, json.JSONDecodeError) as e:
                logger.warning(
                    "Segment %d re-selection failed: %s", seg_idx, e,
                )
                return None

            # Map selected_candidate → branch_decisions
            selected_idx = data.get("selected_candidate", 1)
            candidate_idx = max(0, min(selected_idx - 1, len(candidates) - 1))
            chosen_decisions, _chosen_metrics = candidates[candidate_idx]

            branch_decisions = []
            for d in chosen_decisions:
                condition = getattr(d, "condition_expr", "")
                line = getattr(d, "source_line", 0)
                taken = getattr(d, "branch_taken", True)
                branch_decisions.append({
                    "line": line,
                    "branch": "true" if taken else "false",
                    "node_id": getattr(d, "node_id", ""),
                    "branch_taken": taken,
                    "condition": condition,
                    "source_line": line,
                })

            new_selections[seg_idx] = {
                "segment_index": seg_idx,
                "branch_decisions": branch_decisions,
                "established_state": data.get("established_state", ""),
                "reasoning": data.get("reasoning", ""),
                "selected_candidate": selected_idx,
                "num_candidates": len(candidates),
                "is_feasible": True,
                "re_selected": True,
                "round": round_num,
            }

            logger.info(
                "Segment %d RE-SELECTED — candidate #%d/%d (round %d).",
                seg_idx, selected_idx, len(candidates), round_num,
            )

        return new_selections

    @staticmethod
    def _format_blocked_paths(blocked_paths: list[str]) -> str:
        """Build the 'AVOID previously-tried conflict paths' prompt section.

        *blocked_paths* are whole-path branch-combination signatures that were
        selected in an earlier attempt but whose simulation was conflicted /
        unresolved, so they were abandoned.  Instructs the LLM to choose a
        different feasible combination instead of re-deriving one of these.
        """
        lines = "\n".join(f"  - {p}" for p in blocked_paths)
        return (
            "\n\n### AVOID previously-tried conflict paths\n"
            "The following complete branch combinations were already selected "
            "and their simulated execution was conflicted / could not reach the "
            "bug (abandoned). You MUST NOT re-derive any of these exact "
            f"combinations. Choose a DIFFERENT feasible branch combination.\n{lines}\n"
        )

    @staticmethod
    def _filter_candidates_by_postcondition(
        candidates: list,
        postcondition,
    ) -> list:
        """Remove candidates whose branch decisions violate the postcondition.

        When the postcondition specifies a required branch direction at a
        specific source line (e.g. "L392 must take FALSE"), any candidate
        that makes the opposite choice at that line is pruned.

        Args:
            candidates: ``[(list[PathBranchDecision], PathMetrics), ...]``.
            postcondition: The segment's Postcondition (or None).

        Returns:
            Filtered candidate list.  If ALL candidates are pruned, the
            original list is returned (best-effort — let the LLM handle it).
        """
        if postcondition is None:
            logger.debug("Postcondition filter: pc=None, skipping")
            return candidates

        pc_line = getattr(postcondition, "source_line", 0)
        pc_taken = getattr(postcondition, "branch_taken", None)
        logger.debug(
            "Postcondition filter: checking %d candidates against "
            "L%d must be %s",
            len(candidates), pc_line,
            "TRUE" if pc_taken else "FALSE" if pc_taken is not None else "?",
        )
        if not pc_line or pc_taken is None:
            return candidates

        filtered: list = []
        for cand in candidates:
            decisions, metrics = cand
            violation = False
            for d in decisions:
                d_line = getattr(d, "source_line", 0)
                d_taken = getattr(d, "branch_taken", True)
                # A branch at the postcondition's line must match the required direction
                if d_line == pc_line and d_taken != pc_taken:
                    violation = True
                    break
            if not violation:
                filtered.append(cand)

        if not filtered:
            logger.warning(
                "Postcondition filter pruned ALL %d candidates "
                "(L%d must be %s) — returning original list",
                len(candidates), pc_line,
                "TRUE" if pc_taken else "FALSE",
            )
            return candidates  # best-effort fallback

        pruned = len(candidates) - len(filtered)
        if pruned > 0:
            logger.info(
                "Postcondition filter: pruned %d/%d candidates "
                "(L%d must be %s)",
                pruned, len(candidates), pc_line,
                "TRUE" if pc_taken else "FALSE",
            )
        return filtered

    @staticmethod
    def _clip_state(state: object) -> str:
        """Bound an ``established_state`` before rendering it into a prompt."""
        text = state if isinstance(state, str) else str(state or "")
        if len(text) <= _MAX_RENDERED_STATE_CHARS:
            return text
        clipped = text[:_MAX_RENDERED_STATE_CHARS]
        return (f"{clipped}\n… [state truncated for the prompt: "
                f"{len(text)} characters total]")

    @staticmethod
    def _format_selected_decisions(decisions: list) -> str:
        """Render one segment's selected branch directions, capped.

        A selection carries one decision per branch of its segment, so the
        rendering grows with the segment's branch count — a segment inside a
        big function (faiss ``read_index``) has tens of thousands, which used
        to push the whole-path merge prompt past the model's context limit.
        The count is what carries the meaning here, so the tail is summarized.
        """
        if not decisions:
            return "linear (no branches)"
        shown = decisions[:_MAX_RENDERED_DECISIONS]
        parts = [f"L{d['line']}:{d['branch']}" for d in shown]
        hidden = len(decisions) - len(shown)
        if hidden:
            parts.append(f"(+{hidden} more, {len(decisions)} total)")
        return "; ".join(parts)

    @staticmethod
    def _format_candidates(
        candidates: list,
        postcondition=None,
    ) -> str:
        """Format shortest-path candidates for LLM prompt display.

        Args:
            candidates: ``[(list[PathBranchDecision], PathMetrics), ...]``.
            postcondition: Optional Postcondition — when provided, the required
                branch direction is included as a constraint hint.

        Returns:
            Formatted string showing each candidate with metrics.
        """
        lines: list[str] = []
        # If postcondition specifies a branch direction, show it as a hard constraint
        if postcondition and getattr(postcondition, "condition_expr", None):
            pc_line = getattr(postcondition, "source_line", 0)
            pc_dir = "TRUE" if getattr(postcondition, "branch_taken", True) else "FALSE"
            lines.append(
                f"**⚠ Hard constraint**: Branch at L{pc_line} MUST take the "
                f"**{pc_dir}** branch (CSA report direction). "
                f"Candidates violating this are INFEASIBLE."
            )
            lines.append("")
        for i, (decisions, metrics) in enumerate(candidates):
            s = getattr(metrics, "S", 0)
            c = getattr(metrics, "C", 0)
            v = getattr(metrics, "V", 0)
            lines.append(f"**Candidate #{i + 1}** (S={s}, C={c}, V={v})")
            if decisions:
                shown = decisions[:_MAX_RENDERED_DECISIONS]
                for d in shown:
                    side = "TRUE" if getattr(d, "branch_taken", True) else "FALSE"
                    cond = getattr(d, "condition_expr", "") or getattr(d, "node_id", "?")
                    lines.append(f"  - {cond} → {side}")
                hidden = len(decisions) - len(shown)
                if hidden:
                    lines.append(
                        f"  - … (+{hidden} more branch decisions in this "
                        f"candidate — {len(decisions)} total)"
                    )
            else:
                lines.append("  (no branch decisions — linear path)")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _is_memory_leak(bug_type: str) -> bool:
        """Check if the bug type is a memory leak."""
        return bug_type.lower().strip() == "memory leak"

    @staticmethod
    def _get_error_start_legend(bug_type: str) -> str:
        """Get the bug-type-specific ⚠ legend for the gap analysis prompt."""
        if FPAnalyzer._is_memory_leak(bug_type):
            return ERROR_START_LEGEND_MEMORY_LEAK
        return ERROR_START_LEGEND_DEFAULT

    @staticmethod
    def _get_data_flow_section(bug_type: str) -> str:
        """Get the bug-type-specific data flow analysis section for the system prompt."""
        if FPAnalyzer._is_memory_leak(bug_type):
            return DATA_FLOW_SECTION_MEMORY_LEAK
        return DATA_FLOW_SECTION_DEFAULT

    @staticmethod
    def _get_data_flow_phase1(bug_type: str) -> str:
        """Get the bug-type-specific Phase 1 instructions for the user prompt."""
        if FPAnalyzer._is_memory_leak(bug_type):
            return DATA_FLOW_PHASE1_MEMORY_LEAK
        return DATA_FLOW_PHASE1_DEFAULT

    def _call_llm_gap_analysis(
        self,
        bug_type: str,
        bug_location: str,
        bug_function: str,
        bug_description: str,
        source_context: str,
        num_events: int,
        path_events: str,
        report_name: str,
        error_start_info: str = "",
        call_chain_context: str = "",
    ) -> PathAnalysis:
        """Call LLM for gap analysis, parse and validate the response.

        The prompt uses bug-type-specific templates for the Error Start
        description and data flow analysis instructions.  For Memory Leak
        reports the ⚠ Error Start is described as the *allocation site*
        rather than an undefined/garbage value origin.
        """
        data_flow_section = self._get_data_flow_section(bug_type)
        error_start_legend = self._get_error_start_legend(bug_type)
        data_flow_phase1 = self._get_data_flow_phase1(bug_type)

        system_prompt = GAP_ANALYSIS_SYSTEM_PROMPT.format(
            data_flow_section=data_flow_section,
        )
        prompt = GAP_ANALYSIS_PROMPT.format(
            bug_type=bug_type,
            bug_location=bug_location,
            bug_function=bug_function,
            bug_description=bug_description,
            source_context=source_context,
            num_events=num_events,
            path_events=path_events,
            error_start_info=error_start_info,
            error_start_legend=error_start_legend,
            data_flow_phase1=data_flow_phase1,
            call_chain_context=call_chain_context,
        )

        raw = self._call_llm(system_prompt, prompt)
        data = self._parse_json_response(raw, "gap_analysis")

        return self._parse_gap_analysis(data, report_name)

    # ------------------------------------------------------------------
    # Step 2: Complete path via LLM
    # ------------------------------------------------------------------

    def complete_path(self, analysis: AnalysisResult) -> CompletedPath:
        """Complete the path based on gap analysis.

        Args:
            analysis: The result from analyze().

        Returns:
            CompletedPath with additional steps and concretizations.

        Raises:
            RuntimeError: If LLM completion fails.
        """
        parsed = analysis.parsed_report
        gaps = analysis.analysis

        # Get the source context from the extractor
        source_ctx = self._source_extractor.extract(parsed)
        source_ctx_str = source_ctx.format_for_llm(max_lines=80)

        # Format path events
        path_str_lines = []
        for evt in parsed.path_events:
            line_info = f" L{evt['line_number']}" if evt['line_number'] else ""
            path_str_lines.append(
                f"  #{evt['path_number']} [{evt['event_kind']}]{line_info}: {evt['description']}"
            )
        path_str = "\n".join(path_str_lines)

        # Format gap analysis
        gaps_str_lines = [
            f"  - [{g.gap_type.value}] ({g.severity.value}) {g.description} @ {g.location}"
            for g in gaps.gaps
        ]
        gaps_str = "\n".join(gaps_str_lines) if gaps_str_lines else "  No gaps identified."

        # Call LLM
        prompt = PATH_COMPLETION_PROMPT.format(
            original_summary=gaps.reasoning,
            bug_type=parsed.metadata.bug_type,
            bug_location=parsed.metadata.bug_file,
            bug_line=parsed.metadata.bug_line,
            bug_function=parsed.metadata.function_name,
            gap_analysis=gaps_str,
            source_context=source_ctx_str,
            path_events=path_str,
        )

        raw = self._call_llm(PATH_COMPLETION_SYSTEM_PROMPT, prompt)
        data = self._parse_json_response(raw, "path_completion")

        return self._parse_completed_path(data, gaps)

    # ------------------------------------------------------------------
    # Step 2a (new): Constraint Completion — identify + fill ALL implicit
    #                constraints in one LLM pass (no gap analysis needed)
    # ------------------------------------------------------------------

    def _format_selected_branch_blueprint(
        self, selections: list[dict] | None,
    ) -> str:
        """Serialize the path selections into a human-readable "branch
        blueprint" that forces POC/constraint generation down a SPECIFIC
        feasible path.

        Each selection is a per-segment dict carrying the branch decisions
        (``node_id`` / ``condition`` / ``branch_taken`` / ``line``) the
        selector picked to make the whole path globally consistent.  By
        injecting these into the constraint-completion and POC prompts, a
        *different* selection (a different feasible path in Tier-2 exploration)
        yields a genuinely different POC — so re-verifying a new path is not a
        no-op replay of the same test case.

        Returns an empty string when *selections* is None/empty.
        """
        if not selections:
            return ""
        lines = []
        for sel in selections:
            seg_idx = sel.get("segment_index", "?")
            decisions = sel.get("branch_decisions", []) or []
            state = sel.get("established_state", "")
            lines.append(f"Segment {seg_idx}:")
            if decisions:
                for d in decisions[:_MAX_RENDERED_DECISIONS]:
                    cond = d.get("condition", "") or d.get("condition_expr", "")
                    line = d.get("line") or d.get("source_line") or "?"
                    taken = "TRUE" if d.get("branch_taken", True) else "FALSE"
                    lines.append(
                        f"  - L{line}: condition `{cond}` → {taken}"
                    )
                hidden = len(decisions) - _MAX_RENDERED_DECISIONS
                if hidden > 0:
                    lines.append(
                        f"  - … (+{hidden} more decisions on this segment's "
                        f"path — {len(decisions)} in total)"
                    )
            else:
                lines.append("  (linear — no branch decisions)")
            if state:
                lines.append(f"  Established state: {self._clip_state(state)}")
            else:
                lines.append("  Established state: (none)")
        blueprint = "\n".join(lines)
        return (
            "\n\n## Selected Feasible Path (Branch Blueprint)\n"
            "The path selector chose the following per-segment branch "
            "decisions as a globally-consistent feasible path. You MUST "
            "concretize values / write the test case to force EXACTLY these "
            "decisions (no other branch combination). If these decisions are "
            "mutually impossible, state so rather than silently substituting "
            "a different combination:\n\n"
            f"{blueprint}"
        )

    def _complete_constraints(
        self, analysis_result: AnalysisResult,
        previous_conflicts: list[str] | None = None,
        selected_branches: list[dict] | None = None,
    ) -> CompletedPath:
        """Identify and fill ALL implicit constraints in the CSA path.

        This replaces the old gap-analysis + completion two-step.  Instead
        of filling pre-identified gaps, the LLM enumerates every implicit
        constraint, assumption, and concretization in a single pass.

        Slice-aware: when PDG-sliced code is available, the method also
        scans the reduced code for library calls, member fields, and
        global variables, injecting that analysis into the prompt so the
        LLM focuses only on constraints relevant to the sliced code.

        Args:
            analysis_result: The result from analyze() (parsed report + context).
            previous_conflicts: Optional list of conflicts from a prior
                feasibility check, enabling targeted re-completion.

        Returns:
            CompletedPath with filled constraints and concretized values.
        """
        parsed = analysis_result.parsed_report

        # Get source context (prefer PDG-sliced reduced code if available)
        if analysis_result.reduced_code:
            source_ctx_str = analysis_result.reduced_code
            logger.info(
                "Using PDG-sliced code (%d lines) as source context for %s",
                len(analysis_result.reduced_code.splitlines()),
                parsed.metadata.function_name,
            )
        else:
            source_ctx = self._source_extractor.extract(parsed)
            source_ctx_str = source_ctx.format_for_llm(max_lines=80)

        # Pre-index companion .cc/.cpp files so we can annotate path events
        # with their source lines (the event dicts don't carry file_path).
        source_file_index = self._build_event_source_index(parsed)

        # Format path events (same format as analyze(), but with source annotation)
        path_str_lines = []
        for evt in parsed.path_events:
            line_info = f" L{evt['line_number']}" if evt['line_number'] else ""
            prefix = "⚠ " if evt.get("is_error_start") else "  "

            # Annotate control-flow events with the actual source line
            source_annotation = ""
            line = evt.get("line_number", 0)
            if line and evt.get("is_control_flow"):
                src_line = source_file_index.get(line)
                if src_line:
                    source_annotation = f"  // {src_line.strip()}"

            path_str_lines.append(
                f"{prefix}#{evt['path_number']} [{evt['event_kind']}]{line_info}: "
                f"{evt['description']}{source_annotation}"
            )
        path_str = "\n".join(path_str_lines)

        # Error Start info for the prompt
        error_start_info_str = self._format_error_start_info(
            parsed.error_start_event, bug_type=parsed.metadata.bug_type,
        )

        # Bug-type-specific legend
        error_start_legend = self._get_error_start_legend(parsed.metadata.bug_type)

        prompt = CONSTRAINT_COMPLETION_PROMPT.format(
            bug_type=parsed.metadata.bug_type,
            bug_location=parsed.metadata.bug_file,
            bug_line=parsed.metadata.bug_line,
            bug_function=parsed.metadata.function_name,
            bug_description=parsed.metadata.bug_description,
            source_context=source_ctx_str,
            path_events=path_str,
            error_start_legend=error_start_legend,
            num_events=len(parsed.path_events),
            error_start_info=error_start_info_str,
        )

        # ── Inject slice variable analysis (library calls, members, globals) ──
        if analysis_result.reduced_code:
            slice_var_analysis = self._analyze_slice_variables(
                analysis_result.reduced_code,
            )
            if slice_var_analysis:
                prompt += "\n" + slice_var_analysis + "\n"

        # ── Inject library function API contracts (from static knowledge base) ──
        if analysis_result.reduced_code:
            api_contracts = self._format_library_api_contracts(
                analysis_result.reduced_code,
            )
            if api_contracts:
                prompt += api_contracts + "\n"

        # ── Inject domain-specific knowledge (union type-punning, loop init, etc.) ──
        if analysis_result.reduced_code:
            domain_knowledge = self._inject_domain_specific_knowledge(
                analysis_result.reduced_code,
            )
            if domain_knowledge:
                prompt += domain_knowledge + "\n"

        # ── Inject previous conflicts for targeted re-completion ──
        if previous_conflicts:
            prompt += (
                "\n\n## Previously Found Conflicts (to be resolved)\n"
                "The completion below was flagged with these conflicts. "
                "Please adjust the constraints to RESOLVE them:\n"
                + "\n".join(f"  - {c}" for c in previous_conflicts)
                + (
                    "\n\n"
                    "**IMPORTANT**: The CSA path events (listed above under "
                    "\"Execution Path\") are GROUND TRUTH — they represent the "
                    "actual execution path the analyzer traced. "
                    "You MUST NOT assume conditions that contradict explicit path "
                    "events. For example, if a path event says \"Loop condition "
                    "is false\", the loop body was NOT entered — do not assume "
                    "it was. If the path is infeasible given the CSA's own events, "
                    "acknowledge that and set preconditions/constraints accordingly "
                    "rather than inventing assumptions that override the path.\n"
                    "\n"
                    "**CRITICAL: Check variable QUALIFICATION (which object?).**\n"
                    "If a conflict says \"iocb.opcode == PING contradicts FALSE branch "
                    "at L694 (wslay_is_ctrl_frame)\", check whether the concretized "
                    "variable name is CORRECT:\n"
                    "- `iocb.opcode` is the CURRENT frame's opcode\n"
                    "- `ctx->imsg->opcode` or `imsg->opcode` is the ACCUMULATED "
                    "message's opcode\n"
                    "- These are DIFFERENT variables! If the CSA path shows "
                    "`imsg->opcode == WSLAY_PING` but the concretization says "
                    "`iocb.opcode == WSLAY_PING`, the concretization uses the "
                    "WRONG variable name. Correct it by using the right object.\n"
                )
                + "\n"
            )

        # Inject project-wide real-world constraints (from call-site analysis)
        source_root_for_constraints = self._source_root or (
            analysis_result.source_root or None
        )
        target_classes = self._extract_class_names_from_path(parsed)
        global_constraints = self._load_global_constraints(
            source_root_for_constraints, target_classes=target_classes,
        )
        if global_constraints and global_constraints != "(no project-wide constraints)":
            prompt += (
                "\n\n## Project-Wide Real-World Constraints (from codebase analysis)\n"
                "The following constraints are KNOWN to hold in real execution. "
                "Use them as HARD PRECONDITIONS when completing the path:\n"
                f"{global_constraints}\n"
            )

        # ── Inject the selected feasible-path branch blueprint so the completed
        # concretizations are anchored to THIS specific branch combination (not
        # just the CSA path's default).  Makes a different selected path yield a
        # genuinely different CompletedPath (P1: path-specific POC). ──
        if selected_branches:
            blueprint = self._format_selected_branch_blueprint(selected_branches)
            if blueprint:
                prompt += (
                    "\n\n## Selected Feasible Path — Concretize Along These Branches\n"
                    "The path selector chose a SPECIFIC feasible path for this "
                    "report. Resolve every concretized value / precondition so it "
                    "forces exactly the branch decisions below. Do not re-derive a "
                    "different branch combination:\n"
                    f"{blueprint}\n"
                )

        try:
            raw = self._call_llm(CONSTRAINT_COMPLETION_SYSTEM_PROMPT, prompt)
            data = self._parse_json_response_tolerant(raw, "constraint_completion")
        except (RuntimeError, json.JSONDecodeError) as first_err:
            logger.warning(
                "Constraint completion failed (will retry with increased max_tokens=%d): %s",
                self._max_tokens * 2, first_err,
            )
            # One retry with doubled token budget
            old_max = self._max_tokens
            self._max_tokens = old_max * 2
            try:
                raw = self._call_llm(CONSTRAINT_COMPLETION_SYSTEM_PROMPT, prompt)
                data = self._parse_json_response_tolerant(raw, "constraint_completion")
            except (RuntimeError, json.JSONDecodeError) as retry_err:
                self._max_tokens = old_max
                raise RuntimeError(
                    f"Constraint completion failed even with max_tokens={old_max * 2}: {retry_err}"
                ) from retry_err
            finally:
                self._max_tokens = old_max

        return self._parse_completed_constraints(data, analysis_result)

    # ------------------------------------------------------------------
    # Step 2b: Verification — check if gaps are resolved after completion
    # ------------------------------------------------------------------

    def _call_llm_verification(
        self,
        path_events: str,
        gap_analysis: str,
        completed_steps: str,
        bug_type: str,
        bug_location: str,
        bug_description: str,
    ) -> dict:
        """Call LLM to verify whether completed path is TP or FP."""
        prompt = VERIFICATION_PROMPT.format(
            path_events=path_events,
            gap_analysis=gap_analysis,
            completed_steps=completed_steps,
            bug_type=bug_type,
            bug_location=bug_location,
            bug_description=bug_description,
        )
        raw = self._call_llm(VERIFICATION_SYSTEM_PROMPT, prompt)
        return self._parse_json_response(raw, "verification")

    # ------------------------------------------------------------------
    # Step 2c (new): Feasibility Check — verify completed constraints
    #                have no logical contradictions
    # ------------------------------------------------------------------

    def _check_feasibility(
        self,
        analysis_result: AnalysisResult,
        completed: CompletedPath,
        previous_conflicts: list[str] | None = None,
    ) -> tuple[bool, bool, str, list[str]]:
        """Check whether the CSA warning is a TRUE BUG or FALSE POSITIVE.

        Two-part check:
        (A) Constraint consistency: are any path constraints contradictory?
        (B) CSA warning validity: is the flagged variable actually
            uninitialized/invalid at the point of use?

        If (A) fails → FP (infeasible path).
        If (A) passes but (B) fails → FP (CSA limitation, e.g. memcpy
        initializes the variable but CSA doesn't model it).
        Only if (A) AND (B) pass → TP (real bug).

        Returns:
            (is_feasible, is_real_bug, explanation, conflicts) tuple.
        """
        parsed = analysis_result.parsed_report

        # Format completed constraints for the prompt
        constraints_lines = []
        for s in completed.completed_steps:
            line = f"  - [{s.new_event_kind}] {s.description}"
            if s.concretization:
                line += f"  (concretized: {s.concretization})"
            constraints_lines.append(line)
        constraints_str = "\n".join(constraints_lines) if constraints_lines \
            else "  (no additional constraints)"

        preconditions_str = "\n".join(
            f"  - {p}" for p in completed.preconditions
        ) if completed.preconditions else "  (none)"

        concretized_str = "\n".join(
            f"  - {k} = {v}" for k, v in completed.concretized_values.items()
        ) if completed.concretized_values else "  (none)"

        previous_conflicts_str = ""
        if previous_conflicts:
            previous_conflicts_str = (
                "## Previously Found Conflicts (to be resolved)\n" +
                "\n".join(f"  - {c}" for c in previous_conflicts) +
                "\n\nRe-check whether these are still conflicts after "
                "the updated completion.\n"
            )

        # Source context: prefer reduced code, fall back to raw source
        if analysis_result.reduced_code:
            source_ctx_str = analysis_result.reduced_code
        else:
            source_ctx = self._source_extractor.extract(parsed)
            source_ctx_str = source_ctx.format_for_llm(max_lines=80)

        # Error Start info for the prompt
        error_start_info_str = self._format_error_start_info(
            parsed.error_start_event, bug_type=parsed.metadata.bug_type,
        )

        prompt = FEASIBILITY_CHECK_PROMPT.format(
            bug_type=parsed.metadata.bug_type,
            bug_location=parsed.metadata.bug_file,
            bug_line=parsed.metadata.bug_line,
            bug_function=parsed.metadata.function_name,
            bug_description=parsed.metadata.bug_description,
            error_start_info=error_start_info_str,
            source_context=source_ctx_str,
            original_path_summary=completed.original_path_summary,
            completed_constraints=constraints_str,
            preconditions=preconditions_str,
            concretized_values=concretized_str,
            previous_conflicts=previous_conflicts_str,
        )

        # Inject project-wide real-world constraints as hard feasibility bounds
        source_root_for_constraints = self._source_root or (
            analysis_result.source_root or None
        )
        target_classes = self._extract_class_names_from_path(
            analysis_result.parsed_report
        )
        global_constraints = self._load_global_constraints(
            source_root_for_constraints, target_classes=target_classes,
        )
        if global_constraints and global_constraints != "(no project-wide constraints)":
            prompt += (
                "\n\n## Known External Constraints (project-wide, from codebase analysis)\n"
                "The following constraints are KNOWN to hold in the real-world execution "
                "environment (from construction-site analysis). Use them as HARD "
                "PRECONDITIONS — a path that violates these is INFEASIBLE:\n"
                f"{global_constraints}\n"
            )

        try:
            raw = self._call_llm(FEASIBILITY_CHECK_SYSTEM_PROMPT, prompt)
            data = self._parse_json_response(raw, "feasibility_check")
            is_feasible = bool(data.get("is_feasible", True))
            is_real_bug = bool(data.get("is_real_bug", is_feasible))
            explanation = str(data.get("explanation", ""))
            conflicts = list(data.get("conflicts", []))
        except (RuntimeError, json.JSONDecodeError) as e:
            logger.warning("Feasibility check LLM call failed: %s", e)
            # On error, assume feasible but not real bug to avoid false TP
            is_feasible = True
            is_real_bug = False
            explanation = f"Feasibility check failed: {e}"
            conflicts = []

        return is_feasible, is_real_bug, explanation, conflicts

    # ------------------------------------------------------------------
    # Global constraints loading
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_class_names_from_path(parsed_report: ParsedReport) -> list[str]:
        """Extract likely class names from path events (heuristic).

        Looks for ``ClassName::methodName`` patterns in event descriptions,
        bug file paths, and the function name.
        """
        class_names: set[str] = set()
        for evt in parsed_report.path_events:
            desc = evt.get("description", "")
            # Match "ClassName::methodName" or "'ClassName::methodName'"
            for m in re.finditer(r"(\w+)::\w+", desc):
                class_names.add(m.group(1))

        # Extract from bug file name (remove .cc/.cpp)
        bug_file = Path(parsed_report.metadata.bug_file).stem
        if bug_file:
            class_names.add(bug_file)

        # Extract class prefix from function name (if it has ::)
        func = parsed_report.metadata.function_name
        if "::" in func:
            for part in func.split("::"):
                if part and part[0].isupper():
                    class_names.add(part)

        return sorted(class_names)

    def _load_global_constraints(
        self, source_root: str | None = None,
        target_classes: list[str] | None = None,
    ) -> str:
        """Load or mine project-wide constraints for the current project.

        *target_classes* filters the output to include only constraints
        relevant to those classes, keeping the prompt compact.

        Returns formatted text suitable for inclusion in the LLM prompt.
        """
        try:
            from llm_client.fp_analysis.constraint_miner import (
                mine_all_constraints,
            )
            from llm_client.fp_analysis.constraint_cache import (
                ConstraintCache,
            )

            project_root = self._source_root or source_root
            if not project_root:
                # Try using the local project/ directory
                local = Path("project")
                if local.exists():
                    project_root = str(local.resolve())
                else:
                    return ""

            project_name = self._project_name or "unknown"
            cache = ConstraintCache()

            db = cache.get(project_name)
            if db is None:
                db = mine_all_constraints(project_root, project_name)
                cache.set(project_name, db)

            # If parameter_constraints are empty but target_classes provided,
            # run the targeted scanner and augment the DB (don't persist, as
            # the static scanners may not have captured them)
            if not db.parameter_constraints and target_classes:
                from llm_client.fp_analysis.constraint_miner import (
                    scan_parameter_constraints,
                )
                param_pcs = scan_parameter_constraints(
                    project_root, target_classes,
                )
                db.parameter_constraints.extend(param_pcs)

            return db.format_relevant_constraints(target_classes)
        except Exception as e:
            logger.debug("Could not load global constraints: %s", e)
            return ""

    # ------------------------------------------------------------------
    # CFG-based feasibility check (In-Report FP detection)
    # ------------------------------------------------------------------

    def _cfg_feasibility_check(
        self,
        analysis_result: AnalysisResult,
    ) -> CFGAnalysisResult | None:
        """Run CFG-based feasibility analysis and LLM verification.

        For each segment of the CSA path, enumerates alternative branch
        combinations via the actual clang CFG, and asks the LLM to judge
        feasibility of each variant.

        Returns ``CFGAnalysisResult`` if the analysis runs successfully,
        ``None`` if it fails (e.g., clang unavailable, CFG parse failure)
        so the caller can fall back to the existing LLM-only flow.
        """
        cfg_result = None
        try:
            cfg_result = run_feasibility_analysis(
                analysis_result.parsed_report,
                source_root=self._source_root or None,
                project_name=self._project_name or "default",
                max_variants=16,
            )
        except Exception as e:
            logger.warning("CFG feasibility analysis failed: %s", e)
            return None

        if cfg_result is None or not cfg_result.segments:
            logger.info("CFG analysis: no segments to analyze.")
            return None

        parsed = analysis_result.parsed_report
        bug_type = parsed.metadata.bug_type
        bug_file = parsed.metadata.bug_file
        bug_line = parsed.metadata.bug_line
        bug_function = parsed.metadata.function_name

        # Load project-wide constraints (mined or cached)
        global_constraints_text = self._load_global_constraints()
        if not global_constraints_text:
            global_constraints_text = "(no project-wide constraints available)"

        # Chain state from one segment to the next
        accumulated_state = ""
        prev_post_str = "Function entry / no prior events."
        for seg_analysis in cfg_result.segments:
            # Find the matching SegmentInfo (from path segmentation)
            segment_info = next(
                (s for s in segment_path(parsed)
                 if s.segment_index == seg_analysis.segment_index),
                None,
            )
            if segment_info is None:
                continue

            # Build source context (container code only, no called-function bodies)
            source_parts = ["--- Container function code ---"]
            if segment_info.code_from_entry:
                source_parts.append(segment_info.code_from_entry)
            source_context = "\n".join(source_parts)

            # Build function summaries for called functions (on-demand, cached)
            summary_parts: list[str] = []
            for fn_name in segment_info.called_functions:
                if fn_name in (segment_info.function_name, "memset", "malloc",
                               "free", "memcpy", "printf", "fprintf", "snprintf"):
                    continue
                try:
                    summary = _get_function_summary(
                        fn_name, bug_type, parsed, self._provider,
                        self._summary_cache, self._source_root,
                    )
                    if summary:
                        summary_parts.append(summary)
                except ValueError as e:
                    logger.warning(
                        "Skipping summary for '%s': %s",
                        fn_name, str(e).split('\n')[0],
                    )
                    summary_parts.append(
                        f"### {fn_name}  ⚠ (no summary available — "
                        f"not in KB and no body found)"
                    )

            function_summaries_text = "\n\n".join(summary_parts) if summary_parts \
                else "(no function summaries available)"

            # Build three-part precondition
            acc_state = accumulated_state if accumulated_state else "(function entry, no prior state)"
            post_str = segment_info.format_postcondition()

            prompt = CFG_FEASIBILITY_PROMPT.format(
                bug_type=bug_type,
                bug_file=bug_file,
                bug_line=bug_line,
                bug_function=bug_function or "unknown",
                accumulated_state=acc_state,
                prev_postcondition=prev_post_str,
                function_summaries=function_summaries_text,
                global_constraints=global_constraints_text,
                source_context=source_context,
                postcondition=post_str,
            )

            try:
                raw = self._call_llm(CFG_FEASIBILITY_SYSTEM_PROMPT, prompt)
                data = self._parse_json_response(raw, "cfg_feasibility")
                feasible = data.get("feasible", True)
                reasoning = data.get("reasoning", "")
                seg_analysis.variant_feasibility[0] = feasible
                seg_analysis.variant_reasoning[0] = reasoning
                seg_analysis.has_infeasible_segment = not feasible
                seg_analysis.established_state = data.get("established_state", "")
                # Update accumulated state AND previous postcondition for the next segment
                if feasible and seg_analysis.established_state:
                    accumulated_state = seg_analysis.established_state
                prev_post_str = post_str

            except Exception as e:
                logger.warning(
                    "LLM feasibility check failed for segment %d: %s",
                    seg_analysis.segment_index, e,
                )
                continue

        # Overall: if ANY segment is infeasible → confirmed FP
        cfg_result.overall_infeasible = any(
            s.has_infeasible_segment for s in cfg_result.segments
        )

        logger.info(
            "CFG feasibility: %s infeasible segment(s) found.",
            "✅ no" if not cfg_result.overall_infeasible else "⚠ at least one",
        )

        return cfg_result

    def complete_and_verify(
        self,
        analysis_result: AnalysisResult,
        max_iterations: int = 3,
    ) -> tuple[AnalysisResult, CompletedPath | None]:
        """Iteratively complete constraints + check feasibility.

        Refactored 4-step flow (replaces the old CFG-first flow):

        1. **PDG Backward Slicing** (front-most) — slice at bug trigger,
           build ``SliceMask`` to track retained/removed nodes.
        2. **Constraint Completion** (on sliced code) — LLM fills all
           implicit constraints; slice variable analysis is injected.
        3. **Single-Path Feasibility Check** — LLM checks whether the
           **reported path's** branch conditions conflict with constraints.
           If YES → FP (retry with conflict context, max *max_iterations*).
        4. **Segment-by-Segment Path Selection** — LLM selects a feasible
           path through each segment, then merges and checks global
           consistency.  If consistently feasible → TP.

        Returns:
            (updated AnalysisResult with is_feasible and verification_history,
             last CompletedPath from the final iteration, or None if FP)
        """
        # ──────────────────────────────────────────────────────────────────
        # Step 1: PDG Backward Slicing (front-most)
        # ──────────────────────────────────────────────────────────────────
        logger.info("Step 1: PDG backward slicing ...")
        slice_result = self._run_pdg_slicing(analysis_result)

        # Build SliceMask to track retained/removed PDG nodes and CFG blocks
        slice_mask = self._build_slice_mask(slice_result)
        analysis_result.slice_mask = slice_mask.to_dict() if slice_mask else None

        if slice_result and slice_result.nodes:
            logger.info(
                "Slice: %d nodes, %d function(s), input vars: %s",
                len(slice_result.nodes),
                len(slice_result.involved_functions),
                sorted(slice_result.input_variables)[:8],
            )
        else:
            logger.info("No PDG slice available; using full source code context.")

        # ⭐ Phase 2: create IterativePathSelector for anchor-driven pruning
        selector: IterativePathSelector | None = None
        if self._sdg is not None:
            source_root = self._source_root or analysis_result.source_root or ""
            selector = IterativePathSelector(
                sdg=self._sdg,
                source_root=source_root,
                max_depth=10,
            )
            logger.info("Phase 2: IterativePathSelector created for anchor-driven pruning.")

        # ──────────────────────────────────────────────────────────────────
        # Steps 2 + 3: Constraint Completion → Single-Path Feasibility Check
        # ──────────────────────────────────────────────────────────────────
        verification_history: list[dict] = []
        completed: CompletedPath | None = None
        is_feasible = False
        feasibility_explanation = ""
        feasibility_conflicts: list[str] = []
        previous_conflicts: list[str] | None = None

        for iteration in range(1, max_iterations + 1):
            logger.info(
                "Step 2: Constraint completion iteration %d/%d ...",
                iteration, max_iterations,
            )

            # Step 2: Complete constraints (slice-aware: injects variable
            #         analysis for lib calls, members, globals)
            try:
                completed = self._complete_constraints(
                    analysis_result, previous_conflicts=previous_conflicts,
                )
            except (RuntimeError, json.JSONDecodeError) as e:
                logger.error("Constraint completion failed: %s", e)
                verification_history.append({
                    "iteration": iteration,
                    "step": "constraint_completion",
                    "error": str(e),
                })
                break

            # ── Check: Is the Error Start value truly uninitialized? ──
            if not completed.error_start_is_real:
                logger.info(
                    "Step 2 error_start_is_real=False: the CSA-flagged value IS actually "
                    "initialized (memcpy/union/loop-init). Classifying as FP."
                )
                is_feasible = False
                exp = (
                    "The Error Start value is not truly uninitialized: the completed "
                    "constraints show it IS initialized through a mechanism the CSA "
                    "does not model (union type-punning via memcpy, loop-carried "
                    "variable initialization, or memcpy to overlapping storage). "
                    "This is a known CSA false positive pattern."
                )
                conflicts = [exp]
                verification_history.append({
                    "iteration": iteration,
                    "step": "error_start_is_real",
                    "is_feasible": False,
                    "num_conflicts": 1,
                    "conflicts": [exp],
                })
                break

            # ══════════════════════════════════════════════════════════════
            # Step 2.7: State Lifecycle Analysis (NEW)
            # ══════════════════════════════════════════════════════════════
            # Phase 2.5 may miss complex cross-function state lifecycle
            # patterns.  This step runs a focused analysis using a dedicated
            # prompt + static call-chain extraction to double-check.
            lifecycle = self._analyze_state_lifecycle(
                analysis_result, completed,
            )
            if lifecycle.error_start_initialized_by_lifecycle:
                logger.info(
                    "Step 2.7 detected lifecycle initialization: "
                    "Error Start variable IS initialized through a state "
                    "lifecycle pattern. Overriding to FP."
                )
                completed.error_start_is_real = False
                is_feasible = False
                exp = (
                    "State lifecycle analysis: "
                    + lifecycle.explanation
                )
                conflicts = [exp]
                verification_history.append({
                    "iteration": iteration,
                    "step": "state_lifecycle",
                    "is_feasible": False,
                    "num_conflicts": 1,
                    "conflicts": [exp],
                    "lifecycle_type": lifecycle.lifecycle_type,
                    "state_variables": lifecycle.state_variables,
                    "reset_functions": lifecycle.reset_functions,
                    "non_reset_functions": lifecycle.non_reset_functions,
                })
                break

            # Step 3: Single-path feasibility check (reported path only)
            logger.info("Step 3: Single-path feasibility check ...")
            is_feasible, exp, conflicts = self._check_single_path_feasibility(
                analysis_result, completed, slice_mask,
                previous_conflicts=previous_conflicts,
            )

            verification_history.append({
                "iteration": iteration,
                "step": "single_path_feasibility",
                "is_feasible": is_feasible,
                "num_conflicts": len(conflicts),
                "explanation": exp,
                "conflicts": conflicts,
                "completed_steps_count": len(completed.completed_steps),
            })

            # Loop detection: identical explanation → no progress
            if previous_conflicts is not None and exp == feasibility_explanation:
                logger.warning(
                    "Feasibility loop detected (identical explanation); stopping."
                )
                feasibility_explanation = exp
                feasibility_conflicts = conflicts
                break

            if is_feasible:
                feasibility_explanation = exp
                feasibility_conflicts = []
                logger.info(
                    "Step 3 passed: reported path constraints are consistent."
                )
                break

            # Infeasible — save and re-complete with conflict context
            feasibility_explanation = exp
            feasibility_conflicts = conflicts
            previous_conflicts = conflicts
            logger.info(
                "Step 3 failed (%d conflict(s)). Re-completing ...",
                len(conflicts),
            )

        # If Step 3 never passed → FP
        if not is_feasible or completed is None:
            analysis_result.is_feasible = False
            analysis_result.feasibility_explanation = feasibility_explanation
            analysis_result.feasibility_conflicts = feasibility_conflicts
            analysis_result.verification_history = verification_history
            analysis_result.classification = Classification.FALSE_POSITIVE
            analysis_result.fp_explanation = feasibility_explanation
            return analysis_result, None

        # ──────────────────────────────────────────────────────────────────
        # Step 3.5: Early Feasibility Check — anchor events vs constraints
        # ──────────────────────────────────────────────────────────────────
        early_feasible, early_conflicts = self._early_feasibility_check(
            analysis_result, completed,
        )

        verification_history.append({
            "iteration": 1,
            "step": "early_feasibility",
            "is_feasible": early_feasible,
            "num_conflicts": len(early_conflicts),
            "conflicts": early_conflicts,
        })

        if not early_feasible:
            logger.info(
                "Step 3.5 failed: %d anchor conflict(s) found — path is FP.",
                len(early_conflicts),
            )
            analysis_result.is_feasible = False
            analysis_result.feasibility_explanation = (
                "Early feasibility check found contradictory anchor events. "
                "Completed constraints conflict with the reported path's "
                "branch decisions."
            )
            analysis_result.feasibility_conflicts = early_conflicts
            analysis_result.verification_history = verification_history
            analysis_result.classification = Classification.FALSE_POSITIVE
            analysis_result.fp_explanation = analysis_result.feasibility_explanation
            return analysis_result, None

        logger.info("Step 3.5 passed: all anchor events are consistent.")

        # ──────────────────────────────────────────────────────────────────
        # Step 4: Segment-by-Segment Feasible Path Selection
        # ──────────────────────────────────────────────────────────────────
        logger.info("Step 4: Segment-by-segment feasible path selection ...")

        selected_path = self._select_feasible_path_segments(
            analysis_result, completed, slice_mask,
            selector=selector,
        )

        if selected_path is None:
            # Could not construct a globally consistent path → FP
            logger.info("Step 4 failed: no globally consistent path.")
            verification_history.append({
                "iteration": 1,
                "step": "segment_selection",
                "is_feasible": False,
                "num_conflicts": 1,
                "conflicts": [
                    "No globally consistent combination of branch decisions "
                    "across path segments",
                ],
                "num_segments": "?",
            })
            analysis_result.is_feasible = False
            analysis_result.feasibility_explanation = (
                "Could not construct a globally feasible execution path. "
                "Segment-by-segment path selection found persistent "
                "conflicts across one or more segments."
            )
            analysis_result.feasibility_conflicts = [
                "No globally consistent combination of branch decisions "
                "across path segments",
            ]
            analysis_result.verification_history = verification_history
            analysis_result.classification = Classification.FALSE_POSITIVE
            analysis_result.fp_explanation = analysis_result.feasibility_explanation
            return analysis_result, None

        # ── Globally feasible path found → TP ──
        logger.info(
            "Step 4 passed: globally feasible path with %d segment(s).",
            len(selected_path),
        )
        verification_history.append({
            "iteration": 1,
            "step": "segment_selection",
            "is_feasible": True,
            "num_conflicts": 0,
            "num_segments": len(selected_path),
        })
        analysis_result.is_feasible = True
        analysis_result.feasibility_explanation = (
            "All segment paths are consistent. A globally feasible "
            "execution path that triggers the bug exists."
        )
        analysis_result.feasibility_conflicts = []
        analysis_result.verification_history = verification_history
        analysis_result.classification = Classification.TRUE_POSITIVE
        analysis_result.fp_explanation = ""

        return analysis_result, completed

    # ------------------------------------------------------------------
    # Step 3: Generate POC via LLM
    # ------------------------------------------------------------------

    def generate_poc(
        self, completed: CompletedPath, analysis: AnalysisResult,
        selected_branches: list[dict] | None = None,
    ) -> str:
        """Generate a POC test harness from the completed path.

        Args:
            completed: The result from complete_path().
            analysis: The original analysis result (for source context).
            selected_branches: Optional path selections (per-segment branch
                decisions).  When provided, the generated POC is forced to
                drive the inputs along THIS specific feasible path, so a
                different selection yields a genuinely different test case.

        Returns:
            C++ source code for a POC test case.
        """
        # Prefer PDG-sliced reduced code as source context
        if analysis.reduced_code:
            source_ctx_str = analysis.reduced_code
        else:
            source_ctx = self._source_extractor.extract(analysis.parsed_report)
            source_ctx_str = source_ctx.format_for_llm(max_lines=60)

        preconditions_str = "\n".join(
            f"  - {p}" for p in completed.preconditions
        ) if completed.preconditions else "  (none)"

        concretized_str = "\n".join(
            f"  - {k} = {v}" for k, v in completed.concretized_values.items()
        ) if completed.concretized_values else "  (none)"

        project_name = self._project_name or "unknown"
        hints = _PROJECT_HINTS.get(project_name, _PROJECT_HINTS["default"])
        project_specific_context = hints["poc_context"]

        # Format the system prompt with project name and OOM rules
        system_prompt = POC_GENERATION_SYSTEM_PROMPT.format(
            project_name=project_name,
            oom_rules=OOM_RULES,
        )

        prompt = POC_GENERATION_PROMPT.format(
            bug_type=analysis.parsed_report.metadata.bug_type,
            bug_location=analysis.parsed_report.metadata.bug_file,
            bug_line=analysis.parsed_report.metadata.bug_line,
            bug_function=analysis.parsed_report.metadata.function_name,
            completed_path_summary=completed.final_path_summary,
            preconditions=preconditions_str,
            concretized_values=concretized_str,
            source_context=source_ctx_str,
            project_name=project_name,
            project_specific_context=project_specific_context,
            oom_trigger_instruction=OOM_TRIGGER_INSTRUCTION,
        )

        # Inject the selected feasible-path branch blueprint so the POC drives
        # the inputs along THIS specific path (P1: path-specific POC).  A
        # different selection yields a genuinely different test case, making
        # Tier-2 re-verification of a different path meaningful.
        if selected_branches:
            blueprint = self._format_selected_branch_blueprint(selected_branches)
            if blueprint:
                prompt += (
                    "\n\n## Selected Feasible Path — Force These Branches\n"
                    "The path selector chose a SPECIFIC feasible path. Write "
                    "the test case so its inputs drive execution along EXACTLY "
                    "these branch decisions below (no other combination). If "
                    "forcing them is impossible, say so rather than silently "
                    "substituting a different path:\n"
                    f"{blueprint}\n"
                )

        raw = self._call_llm(system_prompt, prompt)
        return self._extract_code(raw)

    # ------------------------------------------------------------------
    # Step 4: POC with Compilation Fix Loop
    # ------------------------------------------------------------------

    def _make_compile_checker(self) -> CompileChecker:
        """Create a CompileChecker with proper project root and project-specific config."""
        return CompileChecker(
            project_root=self._source_root or os.getcwd(),
            project_name=self._project_name,
            source_root=self._source_root,
        )

    def generate_poc_with_compile_fix(
        self,
        completed: CompletedPath,
        analysis: AnalysisResult,
        max_iterations: int = 3,
    ) -> tuple[str, list[dict]]:
        """Generate POC and iteratively fix compilation errors via LLM.

        Loops:
            generate (or fix) → compile → if errors → LLM fix → repeat

        Args:
            completed: Completed path from complete_path().
            analysis: Analysis result (for source context).
            max_iterations: Max compile-fix iterations (default 3).

        Returns:
            (final_code, iteration_log) where iteration_log has entries:
            {"iteration": int, "status": str, "errors": str, "code": str}
        """
        checker = self._make_compile_checker()
        iteration_log: list[dict] = []

        # Step 1: Generate initial POC
        current_code = self.generate_poc(completed, analysis)
        iteration_log.append({
            "iteration": 0,
            "status": "generated",
            "errors": "",
            "code": current_code,
        })

        # Step 2: Iteratively fix compilation errors
        for attempt in range(1, max_iterations + 1):
            logger.info("Compile-fix iteration %d/%d ...", attempt, max_iterations)
            result = checker.check(current_code)

            if result.success:
                logger.info("Compilation successful on iteration %d.", attempt)
                iteration_log.append({
                    "iteration": attempt,
                    "status": "success",
                    "errors": "",
                    "code": current_code,
                })
                return current_code, iteration_log

            # Compilation failed — ask LLM to fix
            error_text = result.format_errors_for_llm(max_errors=10)
            logger.warning(
                "Compilation failed (%d errors), sending to LLM for fix...",
                len(result.errors),
            )

            fixed_code = self._fix_code_with_llm(
                current_code, error_text, iteration_log=iteration_log,
            )
            iteration_log.append({
                "iteration": attempt,
                "status": "fixed",
                "errors": error_text,
                "code": fixed_code,
            })

            # If the code didn't change, avoid infinite loop
            if fixed_code == current_code:
                logger.warning("LLM returned identical code; stopping fix loop.")
                break

            current_code = fixed_code

        # Final compilation check
        logger.info("Final compilation check after %d iterations ...", max_iterations)
        result = checker.check(current_code)
        iteration_log.append({
            "iteration": max_iterations + 1,
            "status": "success" if result.success else "failed",
            "errors": result.format_errors_for_llm(max_errors=10) if not result.success else "",
            "code": current_code,
        })

        return current_code, iteration_log

    @staticmethod
    def _extract_types_from_errors(compile_errors: str) -> list[str]:
        """Extract type/struct/union names from compilation error messages.

        Looks for patterns like:
        - ``no member named 'X' in 'TypeName'``
        - ``incomplete type 'TypeName'``
        - ``variable has incomplete type 'TypeName'``
        """
        types: list[str] = []
        # Pattern 1: "no member named 'X' in 'TypeName'"
        for m in re.finditer(r"no member named '\w+' in '(\w+(?:::\w+)*)'", compile_errors):
            if m.group(1) not in types:
                types.append(m.group(1))
        # Pattern 2: "incomplete type 'TypeName'"
        for m in re.finditer(r"(?:incomplete|has incomplete) type '(\w+(?:::\w+)*)'", compile_errors):
            if m.group(1) not in types:
                types.append(m.group(1))
        # Pattern 3: "candidate function not viable" + type name in parentheses
        for m in re.finditer(r"cannot convert argument of incomplete type '(\w+)'", compile_errors):
            if m.group(1) not in types:
                types.append(m.group(1))
        return types

    @staticmethod
    def _extract_function_names_from_errors(compile_errors: str) -> list[str]:
        """Extract function names from "no matching function for call to" errors."""
        funcs: list[str] = []
        for m in re.finditer(r"no matching function for call to '(\w+)'", compile_errors):
            if m.group(1) not in funcs:
                funcs.append(m.group(1))
        return funcs

    def _extract_types_from_source(self, source_code: str) -> list[str]:
        """Extract type names from struct/union/class variable declarations in source.

        Catches cases like ``struct wslay_event_callbacks callbacks = {``
        so their definitions can be looked up proactively, even if the error
        message doesn't name the type (e.g. "excess elements").
        """
        types: list[str] = []
        # Pattern: (struct|class|union) TypeName variableName = or ;
        for m in re.finditer(
            r'(?:struct|class|union)\s+(\w+(?:::\w+)*)\s+\w+\s*(?:=|;|\[)',
            source_code,
        ):
            if m.group(1) not in types:
                types.append(m.group(1))
        # Pattern: TypeName variable = {  (for typedef'd structs used without keyword)
        for m in re.finditer(r'(?<!\w)(\w+)\s+\w+\s*=\s*\{', source_code):
            t = m.group(1)
            # Skip obvious non-type names like "if", "switch", common keywords
            if t not in ("if", "switch", "while", "for", "return", "catch") and t not in types:
                types.append(t)
        return types

    def _find_type_definitions(self, type_names: list[str]) -> str:
        """Search project source headers for definitions of the given types.

        Searches ``self._source_root`` and the local ``project/`` directory.
        Returns formatted text with each type's definition and source file.
        """
        if not type_names:
            return ""

        # Collect search roots
        search_roots: list[Path] = []
        if self._source_root:
            p = Path(self._source_root)
            if p.exists():
                search_roots.append(p)
        local_project = Path("project")
        if local_project.exists():
            search_roots.append(local_project)

        if not search_roots:
            return ""

        results: list[str] = []
        checked_types: set[str] = set()

        for type_name in type_names:
            if type_name in checked_types:
                continue
            checked_types.add(type_name)

            for root in search_roots:
                found = False
                for h_file in sorted(root.rglob("*.h")):
                    try:
                        content = h_file.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue

                    # Match: (struct|class|union) type_name { or : (inheritance)
                    m = re.search(
                        rf'(struct|class|union)\s+{re.escape(type_name)}\s*(?::\s*\w+(?:\s*,\s*\w+)*\s*)?\{{',
                        content,
                    )
                    if not m:
                        # Also try forward declaration or typedef
                        m = re.search(
                            rf'(?:typedef\s+)?(struct|class|union)\s+{re.escape(type_name)}\s*;',
                            content,
                        )
                    if not m:
                        continue

                    # Extract definition using brace matching
                    start = m.start()
                    brace_start = content.find('{', start)
                    if brace_start == -1:
                        # Forward declaration or typedef
                        end = content.find(';', start)
                        snippet = content[start:end + 1] if end != -1 else content[start:start + 200]
                    else:
                        depth = 0
                        end = brace_start
                        for i in range(brace_start, min(brace_start + 5000, len(content))):
                            if content[i] == '{':
                                depth += 1
                            elif content[i] == '}':
                                depth -= 1
                                if depth == 0:
                                    end = i + 1
                                    # Include trailing semicolon
                                    while end < len(content) and content[end] in ' \t':
                                        end += 1
                                    if end < len(content) and content[end] == ';':
                                        end += 1
                                    break
                        snippet = content[start:end]

                    rel_path = h_file.relative_to(root) if root != h_file else h_file
                    results.append(f"// From {rel_path}\n{snippet}")
                    found = True
                    break  # Found in this root

                if found:
                    break  # Don't search other roots for this type

        if not results:
            return ""

        return "## Relevant Type Definitions\n" + "\n\n".join(results) + "\n"

    def _find_function_declarations(self, function_names: list[str]) -> str:
        """Search project source headers for function declarations.

        When the compiler reports "no matching function for call to 'X'",
        the function name X is extracted and searched in header files to
        find its actual declaration (return type, parameters).  This gives
        the LLM the information needed to fix the call site.
        """
        if not function_names:
            return ""

        search_roots: list[Path] = []
        if self._source_root:
            p = Path(self._source_root)
            if p.exists():
                search_roots.append(p)
        local_project = Path("project")
        if local_project.exists():
            search_roots.append(local_project)

        if not search_roots:
            return ""

        results: list[str] = []
        checked: set[str] = set()

        for func_name in function_names:
            if func_name in checked:
                continue
            checked.add(func_name)

            for root in search_roots:
                found = False
                for h_file in sorted(root.rglob("*.h")):
                    try:
                        content = h_file.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue

                    lines = content.split("\n")
                    for i, line in enumerate(lines):
                        # Skip comments and preprocessor lines
                        stripped = line.strip()
                        if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                            continue
                        if stripped.startswith("#"):
                            continue

                        if f"{func_name}(" not in stripped:
                            continue

                        # Check this is a declaration (return type + name + params), not
                        # a reference in a comment or a typedef / function pointer field
                        # Look for a simple declaration pattern: type name(params);
                        # or a multi-line declaration where this line ends with '(', ','
                        if ");" in stripped or stripped.rstrip().endswith(";"):
                            # Single-line declaration: int foo(int x);
                            rel_path = h_file.relative_to(root) if root != h_file else h_file
                            results.append(f"// From {rel_path}\n{stripped}")
                            found = True
                            break

                        if stripped.endswith(",") or stripped.endswith("("):
                            # Multi-line declaration — collect lines until ');'
                            decl_lines = [stripped]
                            depth = stripped.count("(") - stripped.count(")")
                            j = i + 1
                            while j < len(lines) and depth > 0:
                                l = lines[j].strip()
                                decl_lines.append(l)
                                depth += l.count("(") - l.count(")")
                                if depth <= 0 and ";" in l:
                                    break
                                j += 1
                            rel_path = h_file.relative_to(root) if root != h_file else h_file
                            results.append(f"// From {rel_path}\n{' '.join(decl_lines)}")
                            found = True
                            break

                if found:
                    break

        if not results:
            return ""

        return "## Relevant Function Declarations\n" + "\n\n".join(results) + "\n"

    def _build_fix_history_context(self, iteration_log: list[dict]) -> str:
        """Format previous fix attempts to prevent oscillation.

        Skips iteration 0 (initial generation). Summarises what was tried
        and what errors remained.
        """
        if len(iteration_log) <= 1:
            return ""

        parts = ["## Previous Fix Attempts (DO NOT repeat these mistakes)"]
        for entry in iteration_log[1:]:
            iter_num = entry.get("iteration", "?")
            errors = entry.get("errors", "")
            status = entry.get("status", "")

            if status == "fixed" and errors:
                # Extract just the error messages (skip location/code context)
                messages: list[str] = []
                for line in errors.split("\n"):
                    line_stripped = line.strip()
                    if line_stripped.startswith("Message:"):
                        messages.append(line_stripped.replace("Message:", "").strip())

                if messages:
                    parts.append(f"- Attempt {iter_num}: {len(messages)} error(s)")
                    for msg in messages[:3]:
                        parts.append(f"  - {msg}")
                else:
                    parts.append(f"- Attempt {iter_num}: compilation failed")

        return "\n".join(parts) + "\n"

    def _fix_code_with_llm(
        self,
        source_code: str,
        compile_errors: str,
        iteration_log: list | None = None,
    ) -> str:
        """Send broken code + compilation errors to LLM and get a fixed version.

        Args:
            source_code: The C++ POC code that failed to compile.
            compile_errors: Formatted compilation errors.
            iteration_log: Optional log of previous fix attempts, used to
                provide type context and fix history for the next attempt.

        Returns:
            Fixed C++ source code.
        """
        project_name = self._project_name or "unknown"
        hints = _PROJECT_HINTS.get(project_name, _PROJECT_HINTS["default"])
        project_hints = hints["compile_fix"]

        # Enrich errors with type definitions, function declarations, and fix history
        type_context = ""
        fix_history = ""
        try:
            types_from_errors = self._extract_types_from_errors(compile_errors)
            types_from_source = self._extract_types_from_source(source_code)
            # Merge both sources, preserving order (errors first, then source)
            all_types = list(dict.fromkeys(types_from_errors + types_from_source))
            if all_types:
                type_context = self._find_type_definitions(all_types)

            # Also look up function declarations for "no matching function" errors
            func_names = self._extract_function_names_from_errors(compile_errors)
            if func_names:
                func_decl_context = self._find_function_declarations(func_names)
                if func_decl_context:
                    type_context += "\n" + func_decl_context

            if iteration_log:
                fix_history = self._build_fix_history_context(iteration_log)
        except Exception as e:
            logger.warning("Error enriching compile-fix prompt: %s", e)

        system_prompt = COMPILE_FIX_SYSTEM_PROMPT.format(
            project_hints=project_hints,
        )
        prompt = COMPILE_FIX_PROMPT.format(
            source_code=source_code,
            compile_errors=compile_errors,
            type_context=type_context,
            fix_history=fix_history,
        )
        raw = self._call_llm(system_prompt, prompt)
        extracted = self._extract_code(raw)
        # Debug: log if LLM returned the same code
        if extracted.strip() == source_code.strip():
            logger.warning(
                "LLM returned identical code (raw=%d chars, extracted=%d chars)",
                len(raw), len(extracted),
            )
            logger.debug("Raw LLM response: %s", raw[:2000])
        return extracted

    # ------------------------------------------------------------------
    # One-shot: analyze → complete → generate POC
    # ------------------------------------------------------------------

    def process_report(
        self,
        report_path: str | Path,
        source_root: str | Path | None = None,
    ) -> tuple[AnalysisResult, CompletedPath | None, str | None]:
        """One-shot: analyze → complete constraints → check feasibility → POC.

        Feasibility-driven flow (replaces old analyze+classify+complete flow):
        1. Parse report + extract context (no LLM gap analysis).
        2. Complete constraints: identify ALL implicit constraints in one LLM
           call.
        3. Check feasibility: verify no contradictory conditions.
        4. If feasible → generate POC with compile-fix loop (POC IS the proof).
        5. If infeasible → explain contradictions (FP evidence), no POC.

        Args:
            report_path: Path to an FP HTML report.
            source_root: Optional project root for source resolution.

        Returns:
            (analysis, completed_path, poc_code) triple.
            completed_path is None if the path was found infeasible.
            poc_code is None if POC generation was skipped (infeasible path).
        """
        # Step 1: Parse + context extraction (no LLM gap analysis)
        analysis = self.analyze(report_path, source_root)

        # Step 2: Complete constraints + check feasibility.
        # CFG runs first (quick FP detection); slicing runs only after
        # CFG confirms no contradictions; constraint completion follows.
        result, completed = self.complete_and_verify(analysis, max_iterations=3)

        # Step 3: If truly TP (feasible AND bug is real) → generate POC
        if result.classification == Classification.TRUE_POSITIVE:
            logger.info(
                "Path constraints are consistent and bug is real. Generating POC ..."
            )
            if completed is None:
                completed = self._complete_constraints(result)
            poc_code, iter_log = self.generate_poc_with_compile_fix(completed, result)
            result.poc_iteration_log = iter_log
            result.poc_code = poc_code
            return result, completed, poc_code

        # FP path → explain why (contradictions or CSA limitation)
        reason = result.feasibility_explanation or result.fp_explanation or "no explanation"
        logger.info("Path classified as FALSE_POSITIVE: %s", reason[:200])
        return result, None, None

    # ------------------------------------------------------------------
    # Helper: build event source index
    # ------------------------------------------------------------------

    def _build_event_source_index(
        self, parsed: ParsedReport,
    ) -> dict[int, str]:
        """Build ``{line_number: source_line}`` for path events.

        Since path events don't carry ``file_path``, we infer the source file
        from the calling-context class name (e.g. ``Calling 'BitfieldMan::method'``)
        and look up ``ClassName.cc`` / ``ClassName.cpp`` in the project root.

        Returns a dict keyed by line number (for control-flow events).
        """
        source_root = self._source_root
        if not source_root:
            return {}
        src_dir = Path(source_root) / "src"
        if not src_dir.exists():
            return {}

        # 1. Collect class names from calling events
        class_names: set[str] = set()
        for evt in parsed.path_events:
            desc = evt.get("description", "")
            m = re.search(r"(?:Calling|← Calling)\s+'(\w+)::", desc)
            if m:
                class_names.add(m.group(1))
        # Also try the bug function's scope
        fn_name = parsed.metadata.function_name
        if "::" in fn_name:
            class_names.add(fn_name.split("::")[0])

        if not class_names:
            return {}

        # 2. Find .cc/.cpp files for those classes
        file_cache: dict[str, list[str]] = {}  # stem → lines
        for cn in class_names:
            for ext in (".cc", ".cpp"):
                cf = src_dir / f"{cn}{ext}"
                if cf.exists():
                    try:
                        file_cache[cn] = cf.read_text(encoding="utf-8",
                                                       errors="replace").splitlines()
                    except OSError:
                        pass
                    break

        if not file_cache:
            return {}

        # 3. For each event line, look up the source line from the most
        #    recently associated class file
        result: dict[int, str] = {}
        current_class: str | None = None

        for evt in parsed.path_events:
            desc = evt.get("description", "")
            line = evt.get("line_number", 0)
            if not line:
                continue

            # Track which class we're in based on calling events
            m = re.search(r"(?:Calling|← Calling)\s+'(\w+)::", desc)
            if m:
                current_class = m.group(1)

            # Control flow events: look up source line
            if evt.get("is_control_flow") and current_class and current_class in file_cache:
                lines = file_cache[current_class]
                idx = line - 1
                if 0 <= idx < len(lines):
                    result[line] = lines[idx]

        # Also search all cached files for lines not yet found
        found_lines = set(result.keys())
        for evt in parsed.path_events:
            line = evt.get("line_number", 0)
            if not line or line in found_lines or not evt.get("is_control_flow"):
                continue
            # Try each class file
            for cn, lines in file_cache.items():
                idx = line - 1
                if 0 <= idx < len(lines):
                    result[line] = lines[idx]
                    found_lines.add(line)
                    break

        return result

    # ------------------------------------------------------------------
    # LLM call helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  max_tokens: int | None = None) -> str:
        """Make an LLM call and return the raw response text.

        *max_tokens* overrides this analyzer's budget for this one call (use it
        for stages whose reply is intrinsically long).  A reply cut off at the
        budget is logged as a warning: without it a truncated response looks
        exactly like a model producing malformed JSON, which is how five of
        eight simulated-execution rounds on read_index-484 lost their verdict and
        silently fell back to a regex over an incomplete document.
        """
        if not self._provider:
            raise RuntimeError(
                "No LLM provider configured. Pass llm_config to FPAnalyzer constructor."
            )

        budget = max_tokens or self._max_tokens
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request = LLMRequest(
            messages=messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=budget,
        )

        logger.info("Sending LLM request (model=%s, max_tokens=%d)...",
                    self._model, budget)
        start = time.monotonic()
        response = self._provider.send(request)
        elapsed = time.monotonic() - start
        logger.info(
            "LLM response received in %.1fs (%d chars, finish_reason=%s)",
            elapsed, len(response.content or ""), response.finish_reason,
        )
        if response.finish_reason in ("length", "max_tokens"):
            logger.warning(
                "LLM response TRUNCATED at max_tokens=%d (%d chars): the tail is "
                "missing, so any JSON/code in it is incomplete. Raise the budget "
                "for this call (max_tokens= is accepted by _call_llm) or make the "
                "caller parse tolerantly.",
                budget, len(response.content or ""),
            )

        return response.content

    @staticmethod
    def _parse_json_response(raw: str, context: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences.

        Also handles truncated responses where the closing `` ``` `` fence
        is missing (response cut off at token limit).
        """
        text = raw.strip()

        # Match complete ```json ... ``` block
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            # No closing fence — try matching opening fence only (truncated response)
            match = re.match(r"```(?:json)?\s*\n?(.*)", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM %s output: %s\nRaw: %s", context, e, raw[:500])
            raise RuntimeError(f"LLM returned invalid JSON for {context}: {e}")

        return data

    @staticmethod
    def _parse_json_response_tolerant(raw: str, context: str) -> dict:
        """Like :meth:`_parse_json_response`, but attempts to salvage an LLM
        response truncated at the token limit (e.g. an unterminated string) by
        repairing the cut-off JSON instead of failing.  Falls back to the strict
        parse first so fully-valid responses are handled identically."""
        text = raw.strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        else:
            match = re.match(r"```(?:json)?\s*\n?(.*)", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            repaired = _repair_truncated_json(text)
            if repaired is not None:
                try:
                    logger.warning(
                        "LLM %s output was truncated at token limit; salvaged "
                        "a repaired JSON prefix (%d chars).", context, len(repaired),
                    )
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
            logger.error(
                "Failed to parse LLM %s output: %s\nRaw: %s",
                context, e, raw[:500],
            )
            raise RuntimeError(f"LLM returned invalid JSON for {context}: {e}")

    @staticmethod
    def _parse_gap_analysis(data: dict, path_id: str) -> PathAnalysis:
        """Parse validated gap analysis from LLM JSON response."""
        raw_gaps = data.get("gaps", [])
        gaps: list[FPGap] = []

        for g in raw_gaps:
            try:
                gap_type = FPGapType(g.get("gap_type", "overly_simple_assumption"))
            except ValueError:
                gap_type = FPGapType.OVERLY_SIMPLE_ASSUMPTION

            try:
                severity = GapSeverity(g.get("severity", "medium"))
            except ValueError:
                severity = GapSeverity.MEDIUM

            gaps.append(FPGap(
                gap_type=gap_type,
                location=g.get("location", "unknown"),
                description=g.get("description", ""),
                severity=severity,
                source_context=g.get("source_context"),
                suggestion=g.get("suggestion", ""),
            ))

        try:
            confidence = Confidence(data.get("confidence", "medium"))
        except ValueError:
            confidence = Confidence.MEDIUM

        return PathAnalysis(
            path_id=path_id,
            is_likely_fp=bool(data.get("is_likely_fp", True)),
            gaps=gaps,
            confidence=confidence,
            reasoning=data.get("reasoning", ""),
        )

    @staticmethod
    def _parse_completed_path(
        data: dict, analysis: PathAnalysis
    ) -> CompletedPath:
        """Parse validated completed path from LLM JSON response."""
        raw_steps = data.get("completed_steps", [])
        steps: list[CompletedStep] = []

        for s in raw_steps:
            steps.append(CompletedStep(
                original_event_index=s.get("original_event_index"),
                new_event_kind=s.get("new_event_kind", "event"),
                description=s.get("description", ""),
                file_path=s.get("file_path", ""),
                line_number=int(s["line_number"]) if s.get("line_number") is not None else 0,
                code_snippet=s.get("code_snippet"),
                concretization=s.get("concretization"),
            ))

        return CompletedPath(
            original_path_summary=data.get("original_path_summary", ""),
            analysis=analysis,
            completed_steps=steps,
            final_path_summary=data.get("final_path_summary", ""),
            preconditions=data.get("preconditions", []),
            concretized_values=data.get("concretized_values", {}),
            error_start_is_real=data.get("error_start_is_real", True),
        )

    @staticmethod
    def _parse_completed_constraints(
        data: dict, analysis_result: AnalysisResult
    ) -> CompletedPath:
        """Parse the LLM's constraint-completion response into a CompletedPath.

        Similar to ``_parse_completed_path()`` but builds its own neutral
        PathAnalysis rather than relying on a pre-existing gap analysis.
        """
        raw_steps = data.get("completed_steps", [])
        steps: list[CompletedStep] = []
        for s in raw_steps:
            # Sanitize concretization: LLM sometimes returns nested objects
            # (e.g. {"filterBitfield_": {"str": null, "int": null}})
            # instead of flat key-value pairs. Flatten or skip empty entries.
            raw_conc = s.get("concretization")
            conc: dict[str, str | int | float | bool] | None = None
            if isinstance(raw_conc, dict):
                cleaned: dict[str, str | int | float | bool] = {}
                for k, v in raw_conc.items():
                    if isinstance(v, dict):
                        # Nested dict — extract the first non-None leaf value
                        leaf = next((lv for lv in v.values() if lv is not None), None)
                        if leaf is not None:
                            cleaned[k] = leaf
                    elif v is not None:
                        cleaned[k] = v
                if cleaned:
                    conc = cleaned

            steps.append(CompletedStep(
                original_event_index=s.get("original_event_index"),
                new_event_kind=s.get("new_event_kind", "event"),
                description=s.get("description", ""),
                file_path=s.get("file_path", ""),
                line_number=int(s["line_number"]) if s.get("line_number") is not None else 0,
                code_snippet=s.get("code_snippet"),
                concretization=conc,
            ))

        # Build a neutral PathAnalysis (no gap analysis performed)
        report_stem = Path(analysis_result.parsed_report.raw_html_path).stem
        neutral_analysis = PathAnalysis(
            path_id=report_stem,
            is_likely_fp=False,
            gaps=[],
            confidence=Confidence.HIGH,
            reasoning=data.get("original_path_summary", ""),
        )

        return CompletedPath(
            original_path_summary=data.get("original_path_summary", ""),
            analysis=neutral_analysis,
            completed_steps=steps,
            final_path_summary=data.get("final_path_summary", ""),
            preconditions=data.get("preconditions", []),
            concretized_values=data.get("concretized_values", {}),
            error_start_is_real=data.get("error_start_is_real", True),
        )

    @staticmethod
    def _extract_code(raw: str) -> str:
        """Extract C++ code from LLM response, stripping markdown fences."""
        text = raw.strip()

        # Try to extract code from markdown fence
        match = re.search(r"```(?:cpp|c\+\+)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If no fence found, assume the entire response is code
        return text

    # ------------------------------------------------------------------
    # Enhanced context formatting & validation
    # ------------------------------------------------------------------

    @staticmethod
    def _format_error_start_info(
        error_start_event: dict | None,
        bug_type: str = "",
    ) -> str:
        """Format Error Start event info for the LLM prompt.

        Returns a string section to be included in the prompt, or
        a note that no explicit Error Start marker was found.

        The description is bug-type-aware: for Memory Leak reports the
        Error Start is described as the allocation site rather than an
        undefined/garbage value origin.
        """
        is_memory_leak = FPAnalyzer._is_memory_leak(bug_type)

        if error_start_event:
            if is_memory_leak:
                return (
                    f"\n## Error Start Event (Allocation Site)\n"
                    f"The memory that could be leaked is allocated at:\n"
                    f"- Event #{error_start_event.get('event_number', '?')}\n"
                    f"- Line {error_start_event.get('line_number', '?')}\n"
                    f"- Description: {error_start_event.get('description', '')}\n"
                    f"- File: {error_start_event.get('file_path', '')}\n"
                    f"Trace this allocated pointer through subsequent events "
                    f"to determine if it reaches a free or is leaked.\n"
                )
            return (
                f"\n## Error Start Event (Value Origin)\n"
                f"The undefined/garbage value originates at:\n"
                f"- Event #{error_start_event.get('event_number', '?')}\n"
                f"- Line {error_start_event.get('line_number', '?')}\n"
                f"- Description: {error_start_event.get('description', '')}\n"
                f"- File: {error_start_event.get('file_path', '')}\n"
                f"Trace this value through subsequent events to identify root cause.\n"
            )

        if is_memory_leak:
            return (
                "\n## Error Start (No Explicit Marker)\n"
                "No explicit 'Error Start' event was found in this path. "
                "Infer the allocation site from context:\n"
                "- Look for malloc/calloc/realloc calls near the beginning of the path\n"
                "- The allocated memory that could be leaked must be identified\n"
            )
        return (
            "\n## Error Start (No Explicit Marker)\n"
            "No explicit 'Error Start' event was found in this path. "
            "You MUST infer the origin of the undefined/garbage value from context:\n"
            "- Look for uninitialized local variables near the start of the path\n"
            "- Look for opaque function return values treated as uninitialized\n"
            "- Look for allocated memory assumed to contain garbage\n"
            "- The earliest point where the value could become undefined\n"
        )

    @staticmethod
    def _format_call_chain(call_chain: dict[str, dict[int, str]]) -> str:
        """Format call chain source context for the LLM prompt."""
        if not call_chain:
            return ""

        parts = ["\n## Call Chain Context (source code around each call site)"]
        for key, snippets in call_chain.items():
            if not snippets:
                continue
            parts.append(f"\n### {key}")
            lines = sorted(snippets.keys())
            for ln in lines:
                parts.append(f"{ln:>6}: {snippets[ln]}")

        return "\n".join(parts) + "\n"

    @staticmethod
    def _validate_gap_analysis(
        analysis: PathAnalysis, path_events: list[dict]
    ) -> list[str]:
        """Validate that the LLM's gap analysis covers the data flow.

        Checks whether the LLM's gaps mention the Error Start event.
        Returns a list of warning strings (empty if analysis is adequate).
        """
        warnings: list[str] = []

        # Find the Error Start event
        error_events = [e for e in path_events if e.get("is_error_start")]
        if not error_events:
            return warnings

        es = error_events[0]
        es_num = es.get("path_number", 0)
        es_desc = es.get("description", "")

        # Concatenate all gap descriptions, suggestions, locations AND reasoning
        gap_text = " ".join(
            g.description + " " + g.suggestion + " " + g.location
            for g in analysis.gaps
        ).lower() + " " + analysis.reasoning.lower()

        # Check if Error Start event number or key terms are mentioned
        es_key_terms = es_desc.lower().split()[:5]  # first few words of description
        if str(es_num) not in gap_text and not any(
            term in gap_text for term in es_key_terms if len(term) > 3
        ):
            warnings.append(
                f"LLM gap analysis did not mention Error Start (event #{es_num}). "
                f"The true root cause (how the undefined value originates) may be missed. "
                f"Error Start description: {es_desc}"
            )

        return warnings

    def _save_intermediate_json(
        self, result: AnalysisResult, report_path: str
    ) -> None:
        """Save intermediate analysis data to a JSON file for debugging."""
        report_name = Path(report_path).stem
        output_dir = self._output_dir / report_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build a serializable snapshot
        intermediate = {
            "report_name": report_name,
            "metadata": {
                "bug_type": result.parsed_report.metadata.bug_type,
                "bug_category": result.parsed_report.metadata.bug_category,
                "bug_file": result.parsed_report.metadata.bug_file,
                "bug_line": result.parsed_report.metadata.bug_line,
                "bug_column": result.parsed_report.metadata.bug_column,
                "bug_description": result.parsed_report.metadata.bug_description,
                "function_name": result.parsed_report.metadata.function_name,
                "path_length": result.parsed_report.metadata.path_length,
            },
            "error_start_event": result.parsed_report.error_start_event,
            "calling_events": result.parsed_report.calling_events,
            "path_events_annotated": [
                {
                    "path_number": e.get("path_number"),
                    "event_kind": e.get("event_kind"),
                    "description": e.get("description"),
                    "line_number": e.get("line_number"),
                    "branch_condition": e.get("branch_condition"),
                    "file_path": e.get("file_path"),
                    "is_error_start": e.get("is_error_start"),
                    "is_calling_event": e.get("is_calling_event"),
                    "called_function": e.get("called_function"),
                }
                for e in result.parsed_report.path_events
            ],
            "annotated_path": result.annotated_path,
            "call_chain_context": result.call_chain_context,
        }

        out_path = output_dir / "analysis_intermediate.json"
        try:
            out_path.write_text(
                json.dumps(intermediate, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Intermediate analysis saved to %s", out_path)
        except Exception as e:
            logger.warning("Failed to save intermediate JSON: %s", e)

    # ------------------------------------------------------------------
    # Classification: TP (real bug) vs FP (false positive)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(analysis: PathAnalysis) -> Classification:
        """Rule-based classification from gap analysis results.

        No gaps → TP (the analyzer found nothing wrong → real bug path).
        Has gaps with any HIGH/MEDIUM severity → FP suspicion.
        All gaps LOW severity → UNCERTAIN.
        """
        if not analysis.gaps:
            return Classification.TRUE_POSITIVE

        severities = {g.severity for g in analysis.gaps}
        if GapSeverity.HIGH in severities or GapSeverity.MEDIUM in severities:
            return Classification.FALSE_POSITIVE

        return Classification.UNCERTAIN

    @staticmethod
    def _make_minimal_completed_path(analysis: AnalysisResult) -> CompletedPath:
        """Create a minimal CompletedPath for TP reports (no gaps to fill).

        This skips the LLM completion call — the raw path IS the bug path.
        """
        return CompletedPath(
            original_path_summary=analysis.analysis.reasoning,
            analysis=analysis.analysis,
            completed_steps=[],
            final_path_summary=analysis.analysis.reasoning,
            preconditions=[],
            concretized_values={},
        )

    # ------------------------------------------------------------------
    # convenience: skip LLM and use mock data for testing
    # ------------------------------------------------------------------

    def analyze_with_mock(
        self, report_path: str | Path, mock_gaps: list[FPGap] | None = None
    ) -> AnalysisResult:
        """Analyze with mock gap analysis (no LLM call needed)."""
        parsed = self._html_parser.parse(str(report_path))
        bug_path = self._html_parser.parse_to_bug_path(str(report_path))

        if mock_gaps is None:
            mock_gaps = [
                FPGap(
                    gap_type=FPGapType.OVERLY_SIMPLE_ASSUMPTION,
                    location="event #16",
                    description="Mock: analyzer assumes pointer is null without checking caller guarantees",
                    severity=GapSeverity.HIGH,
                    suggestion="Check the caller's precondition; the pointer is guaranteed non-null.",
                ),
            ]

        analysis = PathAnalysis(
            path_id=Path(str(report_path)).stem,
            is_likely_fp=True,
            gaps=mock_gaps,
            confidence=Confidence.HIGH,
            reasoning="Mock analysis for testing",
        )

        # Set classification based on whether gaps exist
        classification = self._classify(analysis)

        return AnalysisResult(
            parsed_report=parsed,
            bug_path=bug_path,
            analysis=analysis,
            classification=classification,
            annotated_path="",
            call_chain_context="",
        )

    def complete_path_with_mock(
        self, analysis: AnalysisResult
    ) -> CompletedPath:
        """Generate a mock completed path (no LLM call needed)."""
        return CompletedPath(
            original_path_summary="Mock original path summary",
            analysis=analysis.analysis,
            completed_steps=[
                CompletedStep(
                    original_event_index=None,
                    new_event_kind="event",
                    description="Mock: Caller passes valid pointer (from getAddrInfo result)",
                    file_path=analysis.parsed_report.metadata.bug_file,
                    line_number=analysis.parsed_report.metadata.bug_line,
                    concretization={"ptr": "valid_pointer"},
                ),
            ],
            final_path_summary="Mock completed path",
            preconditions=["ptr != nullptr"],
            concretized_values={"ptr": "valid_pointer"},
        )

    def generate_poc_mock(self) -> str:
        """Generate a mock POC (no LLM call needed)."""
        return """#include <gtest/gtest.h>
#include <folly/SocketAddress.h>

TEST(SocketAddressTest, SetFromLocalAddr) {
    folly::SocketAddress addr;
    // Mock POC for testing
    EXPECT_TRUE(true);
}
"""
