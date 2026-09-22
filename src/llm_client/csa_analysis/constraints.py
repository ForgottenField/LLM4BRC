"""Path constraint extraction and Z3-based solving.

Extracts branch assumptions from CSA plist output, converts them to
Z3 constraints, and solves for concrete POC input values.
"""

from __future__ import annotations

import plistlib
import re
from pathlib import Path
from typing import Any, Optional

from llm_client.csa_analysis.models import BugPath, EventKind


# ---------------------------------------------------------------------------
# Constraint extraction
# ---------------------------------------------------------------------------

# Matches "Assuming 'var_name' is 0" style messages
_ASSUMING_RE = re.compile(
    r"Assuming\s+'(\w+)'\s+is\s+(\d+)"
)


def extract_constraints_from_plist(plist_path: str) -> list[dict[str, Any]]:
    """Extract per-path constraints from a CSA plist file.

    Returns a list of dicts, one per execution path, each containing:
      - path_index: int
      - constraints: list of (variable, value) tuples
      - events: list of raw path event dicts
    """
    with open(plist_path, "rb") as f:
        data = plistlib.load(f)

    diagnostics = data.get("diagnostics", [])
    result: list[dict[str, Any]] = []

    for diag_idx, diag in enumerate(diagnostics):
        path = diag.get("path", [])
        constraints: list[tuple[str, str]] = []
        events: list[dict[str, Any]] = []

        for node in path:
            kind = node.get("kind", "")
            msg = node.get("message", "")

            if kind == "event" and msg:
                events.append({
                    "kind": "event",
                    "message": msg,
                    "location": node.get("location", {}),
                })
                m = _ASSUMING_RE.match(msg)
                if m:
                    constraints.append((m.group(1), m.group(2)))

            elif kind == "control":
                events.append({
                    "kind": "control",
                    "location": node.get("location", {}),
                    "edges": node.get("edges", []),
                })

        result.append({
            "path_index": diag_idx,
            "constraints": constraints,
            "events": events,
        })

    return result


def format_constraints_as_smtlib(
    constraints: list[tuple[str, str]],
    var_sorts: Optional[dict[str, str]] = None,
) -> str:
    """Format constraints as SMT-LIB v2 script for Z3.

    Args:
        constraints: List of (variable, value) pairs.
        var_sorts: Optional map of variable name → sort name ("Int", "BitVec", etc.)

    Returns:
        SMT-LIB v2 script string.
    """
    if var_sorts is None:
        var_sorts = {}

    lines = [
        "(set-logic QF_LIA)",
        "(set-info :status unknown)",
    ]

    # Declare variables
    seen_vars: set[str] = set()
    for var_name, _ in constraints:
        if var_name in seen_vars:
            continue
        seen_vars.add(var_name)
        sort = var_sorts.get(var_name, "Int")
        if sort == "Int":
            lines.append(f"(declare-fun {var_name} () Int)")
        else:
            lines.append(f"(declare-fun {var_name} () {sort})")

    # Assert constraints
    for var_name, value in constraints:
        lines.append(f"(assert (= {var_name} {value}))")

    lines.append("(check-sat)")
    lines.append("(get-model)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Z3-based solving
# ---------------------------------------------------------------------------

def solve_with_z3(
    constraints: list[tuple[str, str]],
) -> Optional[dict[str, Any]]:
    """Solve constraints using Z3 and return a model.

    Args:
        constraints: List of (variable, value) pairs, e.g.
                     [("sym_dst_size", "0"), ("sym_src_size", "0")]

    Returns:
        Dict mapping variable name → Z3 model value, or None if unsat.
    """
    import z3

    solver = z3.Solver()
    var_map: dict[str, Any] = {}

    for var_name, value in constraints:
        if var_name not in var_map:
            var_map[var_name] = z3.Int(var_name)
        val = int(value)
        solver.add(var_map[var_name] == val)

    result = solver.check()
    if result == z3.sat:
        model = solver.model()
        return {name: model[var] for name, var in var_map.items()}
    return None


def generate_poc_inputs(
    constraints: list[tuple[str, str]],
    harness_params: list[str],
) -> dict[str, Any]:
    """Generate concrete POC input values from constraints.

    For variables not constrained by CSA, assigns default values
    (1 for sizes, to keep mallocs valid).

    Args:
        constraints: List of (variable, value) pairs from CSA.
        harness_params: List of symbolic variable names expected by the harness.

    Returns:
        Dict of variable → concrete value.
    """
    import z3

    solver = z3.Solver()
    var_map: dict[str, Any] = {}

    for var_name, value in constraints:
        if var_name not in var_map:
            var_map[var_name] = z3.Int(var_name)
        val = int(value)
        solver.add(var_map[var_name] == val)

    # For unconstrained size variables, constrain to range [0, 256]
    for p in harness_params:
        if p not in var_map:
            var_map[p] = z3.Int(p)
            solver.add(var_map[p] >= 0)
            solver.add(var_map[p] <= 256)

    result = solver.check()
    if result == z3.sat:
        model = solver.model()
        return {
            name: model[var].as_long() if model[var] is not None else 0
            for name, var in var_map.items()
        }
    return {}


def get_harness_symbolic_vars(harness_code: str) -> list[str]:
    """Extract symbolic variable names from a generated harness.

    Looks for ``scanf`` calls on ``sym_*`` variables.
    """
    vars_found: list[str] = []
    for line in harness_code.splitlines():
        m = re.match(r"\s*scanf\(.*&\s*(sym_\w+)\s*\)", line)
        if m:
            vars_found.append(m.group(1))
        # Also match size variables used with malloc
        m2 = re.match(r"\s*scanf\(.*&\s*(sym_\w+_size)\s*\)", line)
        if m2 and m2.group(1) not in vars_found:
            vars_found.append(m2.group(1))
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for v in vars_found:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique
