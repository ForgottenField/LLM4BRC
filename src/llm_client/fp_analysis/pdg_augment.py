"""Post-processing augmentation for PDG def-use chains.

Adds synthetic def-use edges that the C++ PDGBuilder cannot derive from
Clang CFG/AST reaching-definitions alone.  Three augmentation passes:

1. **Union alias analysis** (Gap B → A)
   Members of the same C++ union share the same memory — a write through
   one member is a reachable definition for a read through another.
   This pass detects union-typed variables and connects member accesses.

2. **Class member field tracking** (Gap C)
   ``this->field_`` accesses are not modelled as explicit variable
   definitions at the PDG level (the CFG reaching-definitions analysis
   only tracks local variables).  This pass introduces pseudo-variables
   ``[this].field_`` so that intra-procedural def-use chains for member
   fields work correctly.

3. **Opaque function pointer side effects** (Gap A)
   ``memcpy(&dst, src, n)``, ``memset(&dst, 0, n)``, and similar
   standard-library functions modify memory through pointer arguments.
   This pass recognises these calls and adds synthetic definition edges
   for the pointed-to variable.

The passes are independent and run in sequence.  Each is conservative:
it only adds edges when it can be *certain* a definition exists.  False
positives (spurious edges) are harmless — they widen the slice slightly
but never exclude a relevant node.

Usage::

    from llm_client.fp_analysis.pdg_augment import augment_pdg_sdg

    sdg = augment_pdg_sdg(sdg)  # adds synthetic def-use edges in-place
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from llm_client.fp_analysis.pdg_models import (
    DefUseEdge,
    FunctionPDG,
    PDGNode,
    SDG,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Standard-library functions that modify memory through pointer arguments.
_OPAQUE_MODIFIERS: frozenset[str] = frozenset({
    "memcpy", "memmove", "memset", "memcmp",
    "bzero", "bcopy", "explicit_bzero",
    "strcpy", "strncpy", "strcat", "strncat",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Regex: capture base variable from "&var.member..." or "var.member..."
_RE_MEMBER_ACCESS = re.compile(
    r"^(?:&\s*)?(\w[\w_]*)\s*\.\s*\w[\w_]*"
)
# Regex: capture base variable from "this->member"
_RE_THIS_MEMBER = re.compile(
    r"this\s*->\s*(\w[\w_]*)"
)
# Regex: capture base variable from "&var" (top-level addr-of)
_RE_ADDR_OF = re.compile(
    r"^&\s*(\w[\w_]*)\s*$"
)


def _collect_union_decls(fn_pdg: FunctionPDG) -> set[str]:
    """Return set of variable names declared as union types."""
    unions: set[str] = set()
    for node in fn_pdg.nodes.values():
        if node.kind != "decl":
            continue
        expr = node.expression or ""
        # Detect union declarations: "union ... var ;" or "sockaddr_union var;"
        if "union" in expr.lower() and node.variable:
            unions.add(node.variable)
    return unions


def _collect_member_accesses(
    fn_pdg: FunctionPDG,
) -> dict[str, dict[str, list[tuple[str, bool]]]]:
    """Map ``(base_var, member_path) → [(node_id, is_def)]``.

    ``is_def=True`` means the node *writes to* the member (assignment target,
    memcpy destination, etc.).  ``is_def=False`` means the node *reads* it.

    Detects:
    - ``var.member`` — struct/union member access
    - ``&var.member`` — address-of member (memcpy destination)
    - ``var.member.sub`` — nested member access
    """
    result: dict[str, dict[str, list[tuple[str, bool]]]] = {}

    for nid, node in fn_pdg.nodes.items():
        expr = node.expression or ""
        base, member = _parse_base_member(expr)
        if base and member:
            is_def = _is_def_site(node, expr)
            result.setdefault(base, {}).setdefault(member, []).append((nid, is_def))
        # Also handle raw var access when var is a known union
        # e.g. if var is a union and the expression uses just "var" as an arg
    return result


def _parse_base_member(expr: str) -> tuple[Optional[str], Optional[str]]:
    """Extract ``(base_var, member)`` from a member-access expression.

    Examples::

        "ad.in.sin_addr.s_addr"  →  ("ad", "in")
        "&ad.storage"             →  ("ad", "storage")
        "ifa->ifa_addr"           →  ("ifa", "ifa_addr")
        "this->bitfield_"         →  ("this", "bitfield_")  →  special
    """
    m = _RE_MEMBER_ACCESS.match(expr)
    if m:
        base = m.group(1)
        # Skip numeric / constexpr bases like "1025"
        if base.isdigit() or base.startswith("0x"):
            return None, None
        # Extract the first member name
        rest = expr[m.end():]
        next_member = rest.split(".")[0].split("->")[0].strip(" )];,")
        return base, next_member if next_member else None
    m2 = _RE_THIS_MEMBER.search(expr)
    if m2:
        return "this", m2.group(1)
    return None, None


def _is_def_site(node: PDGNode, expr: str) -> bool:
    """Heuristic: is *node* a *definition* (write) of a variable?

    ``True`` for:
    - assignment nodes (``var = ...``)
    - decl nodes with initialiser
    - call_expr that is a known opaque modifier (memcpy/memset as first arg)
    - ``&var`` as address-of (the variable is the *destination*)
    """
    if node.kind == "assign":
        return True
    if node.kind == "decl" and "=" in (expr or ""):
        return True
    if node.is_call and node.callee in _OPAQUE_MODIFIERS:
        return True
    return False


def _extract_union_member_path(expr: str, base: str) -> Optional[str]:
    """Extract the full member path for a union variable access.

    E.g. ``ad.in.sin_addr.s_addr`` → ``"in.sin_addr.s_addr"``
    """
    pattern = re.compile(r"(?:^|(?<=\W))" + re.escape(base) + r"\.(.+)$")
    m = pattern.search(expr)
    if m:
        # Trim trailing punctuation / whitespace
        path = m.group(1).rstrip(")]; ,")
        return path
    return None


# ---------------------------------------------------------------------------
# Pass 1: Union alias analysis
# ---------------------------------------------------------------------------


def _augment_union_aliases(fn_pdg: FunctionPDG) -> list[DefUseEdge]:
    """Add synthetic def-use edges for union member accesses.

    When a variable is declared as a union (or detected as one through
    multi-member access patterns), all member accesses share the same
    memory.  This pass connects definition sites (assignments, memcpy
    calls, etc.) through any member to use sites through any *other*
    member.

    Uses line-level targeting: if any node on line L references a union
    variable, the edge connects to ALL nodes on L.  This ensures
    sub-expression nodes (e.g. a bare ``htonl(...)`` call that is an
    operand of a larger expression involving the union) are reached.
    """
    if fn_pdg.is_empty:
        return []

    new_edges: list[DefUseEdge] = []

    # Step 1: collect union variables from decl nodes
    union_vars = _collect_union_decls(fn_pdg)

    # Precompute: for each union variable, collect all target lines and nodes
    var_targets: dict[str, set[str]] = {}
    for union_var in union_vars:
        lines = _collect_lines_referencing_var(fn_pdg, union_var)
        var_targets[union_var] = _collect_node_ids_on_lines(fn_pdg, lines)

    # Step 2: for each union variable, find def and use nodes
    for union_var in union_vars:
        def_nodes: list[str] = []
        targets = var_targets.get(union_var, set())

        for nid, node in fn_pdg.nodes.items():
            if nid not in targets:
                continue
            if _is_def_site(node, node.expression or ""):
                def_nodes.append(nid)

        # Step 3: add edges from each def to all other target nodes on
        # lines that reference the union variable
        for def_id in def_nodes:
            for use_id in targets:
                if def_id != use_id:
                    new_edges.append(
                        DefUseEdge(
                            def_node_id=def_id,
                            use_node_id=use_id,
                            variable=union_var,
                        )
                    )

    if new_edges:
        logger.debug(
            "Union alias: %d new edge(s) for %d union variable(s) in %s",
            len(new_edges),
            len(union_vars),
            fn_pdg.function_name,
        )

    return new_edges


# ---------------------------------------------------------------------------
# Pass 2: Class member field tracking (this->X as pseudo-variable)
# ---------------------------------------------------------------------------


def _collect_this_member_accesses(
    fn_pdg: FunctionPDG,
) -> dict[str, list[tuple[str, bool]]]:
    """Map ``field_name → [(node_id, is_def)]`` for ``this->field`` accesses."""
    accesses: dict[str, list[tuple[str, bool]]] = {}

    for nid, node in fn_pdg.nodes.items():
        expr = node.expression or ""
        # Find all "this->field" patterns
        for m in _RE_THIS_MEMBER.finditer(expr):
            field = m.group(1)
            is_def = _is_def_site(node, expr)
            # Also: if this->field is the FIRST argument of memcpy/memset,
            # treat it as a def (memcpy writes TO this->field)
            if node.is_call and node.callee in _OPAQUE_MODIFIERS:
                is_def = True
            accesses.setdefault(field, []).append((nid, is_def))

    return accesses


def _augment_member_fields(fn_pdg: FunctionPDG) -> list[DefUseEdge]:
    """Add synthetic def-use edges for class member fields (``this->X``).

    Introduces pseudo-variables ``[this].field_name`` and connects
    definition sites to use sites within the same function.

    Intra-procedural only: if a field is used but never defined in this
    function, it becomes an implicit input (tracked as ``[this].field``
    with no def), which cross-function propagation can connect later.
    """
    if fn_pdg.is_empty:
        return []

    new_edges: list[DefUseEdge] = []
    accesses = _collect_this_member_accesses(fn_pdg)

    for field, sites in accesses.items():
        def_ids = [nid for nid, is_def in sites if is_def]
        use_ids = [nid for nid, is_def in sites if not is_def]

        pseudo_var = f"[this].{field}"

        for def_id in def_ids:
            for use_id in use_ids:
                if def_id != use_id:
                    new_edges.append(
                        DefUseEdge(
                            def_node_id=def_id,
                            use_node_id=use_id,
                            variable=pseudo_var,
                        )
                    )

    if new_edges:
        logger.debug(
            "Member fields: %d new edge(s) for %d field(s) in %s",
            len(new_edges),
            len(accesses),
            fn_pdg.function_name,
        )

    return new_edges


# ---------------------------------------------------------------------------
# Pass 3: Opaque function pointer side effects
# ---------------------------------------------------------------------------

# Regex: first argument of a call like memcpy(&var, ...) or memcpy(&var.member, ...)
_RE_FIRST_ARG_ADDR = re.compile(
    r"^(?:&\s*)?(\w[\w_]*)(?:\s*\.\s*\w[\w_]*)?\s*[,)]"
)


def _collect_lines_referencing_var(
    fn_pdg: FunctionPDG, base_var: str,
) -> set[int]:
    """Return all source lines where *any* PDG node references *base_var*.

    Uses expression-text matching: searches for ``var.`` member-access
    patterns.  This is intentionally broad to cover sub-expression nodes
    (e.g. ``htonl(...)`` that is part of ``ad.in.sin_addr.s_addr != htonl(...)``)
    that don't themselves contain the variable name.
    """
    lines: set[int] = set()
    pattern = re.compile(r"\b" + re.escape(base_var) + r"\s*\.")
    for node in fn_pdg.nodes.values():
        if pattern.search(node.expression or ""):
            lines.add(node.source_line)
    return lines


def _collect_node_ids_on_lines(
    fn_pdg: FunctionPDG, lines: set[int],
) -> set[str]:
    """Return all node IDs whose ``source_line`` is in *lines*."""
    ids: set[str] = set()
    for nid, node in fn_pdg.nodes.items():
        if node.source_line in lines:
            ids.add(nid)
    return ids


def _augment_opaque_calls(fn_pdg: FunctionPDG) -> list[DefUseEdge]:
    """Add synthetic definition edges for opaque modifier calls.

    Recognises ``memcpy``, ``memset``, and similar functions that modify
    memory through their first pointer argument.

    The algorithm is **line-based**: when an opaque call writes to variable
    ``v`` (detected via ``&v`` or ``&v.member``), synthetic def-use edges
    are added to ALL nodes on any source line that references ``v``.
    This ensures sub-expression nodes (like a call to ``htonl`` that is
    an operand of ``v.member != htonl(...)``) are also reachable.
    """
    if fn_pdg.is_empty:
        return []

    new_edges: list[DefUseEdge] = []
    # Precompute: for each variable referenced via member access, collect
    # all source lines and node IDs on those lines.
    var_lines: dict[str, set[int]] = {}
    var_targets: dict[str, set[str]] = {}

    for node in fn_pdg.nodes.values():
        expr = node.expression or ""
        for m in re.finditer(r"\b(\w[\w_]*)\s*\.\s*\w", expr):
            base = m.group(1)
            if base.isdigit() or base.startswith("0x"):
                continue
            var_lines.setdefault(base, set()).add(node.source_line)

    for var, lines in var_lines.items():
        var_targets[var] = _collect_node_ids_on_lines(fn_pdg, lines)

    for nid, node in fn_pdg.nodes.items():
        if not node.is_call:
            continue
        if node.callee not in _OPAQUE_MODIFIERS:
            continue

        expr = node.expression or ""
        first_arg = _extract_first_arg(expr)
        if not first_arg:
            continue

        # ---- Case A: First argument is &var or &var.member ----
        addr_match = re.match(r"^\s*&\s*(\w[\w_]*)", first_arg)
        if addr_match:
            base_var = addr_match.group(1)
            if base_var.isdigit():
                continue
            targets = var_targets.get(base_var, set())
            for other_nid in targets:
                if other_nid != nid:
                    new_edges.append(
                        DefUseEdge(
                            def_node_id=nid,
                            use_node_id=other_nid,
                            variable=base_var,
                        )
                    )

        # ---- Case B: First argument is this->field ----
        this_match = _RE_THIS_MEMBER.search(first_arg)
        if this_match:
            field = this_match.group(1)
            pseudo_var = f"[this].{field}"
            # Find all lines where this->field is referenced
            field_lines: set[int] = set()
            for onode in fn_pdg.nodes.values():
                if f"this->{field}" in (onode.expression or ""):
                    field_lines.add(onode.source_line)
            targets = _collect_node_ids_on_lines(fn_pdg, field_lines)
            for other_nid in targets:
                if other_nid != nid:
                    new_edges.append(
                        DefUseEdge(
                            def_node_id=nid,
                            use_node_id=other_nid,
                            variable=pseudo_var,
                        )
                    )

    if new_edges:
        logger.debug(
            "Opaque calls: %d new edge(s) in %s",
            len(new_edges),
            fn_pdg.function_name,
        )

    return new_edges


def _extract_first_arg(expr: str) -> Optional[str]:
    """Extract the first argument from a function-call expression.

    For ``memcpy(&ad.storage, ifa->ifa_addr, addrlen)`` returns
    ``"&ad.storage"``.

    Uses a simple paren-balance scan (not a full parser).
    """
    # Find the opening paren after the function name
    paren_idx = expr.find("(")
    if paren_idx == -1:
        return None
    rest = expr[paren_idx + 1:]

    # Scan until we hit an unparenthesised comma or end
    depth = 0
    end = 0
    for i, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                end = i
                break
            depth -= 1
        elif ch == "," and depth == 0:
            end = i
            break
    else:
        end = len(rest)

    return rest[:end].strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def augment_pdg_sdg(
    sdg: SDG,
    enable_union_aliases: bool = True,
    enable_member_fields: bool = True,
    enable_opaque_calls: bool = True,
) -> SDG:
    """Post-process an SDG to synthesise missing def-use edges.

    Runs three independent augmentation passes on each function's PDG.

    Args:
        sdg: The System Dependence Graph to augment (modified *in-place*).
        enable_union_aliases: Enable union member alias analysis (Pass 1).
        enable_member_fields: Enable ``this->X`` pseudo-variable tracking
            (Pass 2).
        enable_opaque_calls: Enable memcpy/memset pointer-effect modelling
            (Pass 3).

    Returns:
        The same ``SDG`` instance (with augmented ``def_use_edges`` in
        each :class:`FunctionPDG` and rebuilt traversal indices).
    """
    total_edges = 0
    for fn_name, fn_pdg in sdg.pdgs.items():
        new_edges: list[DefUseEdge] = []

        if enable_union_aliases:
            new_edges.extend(_augment_union_aliases(fn_pdg))
        if enable_member_fields:
            new_edges.extend(_augment_member_fields(fn_pdg))
        if enable_opaque_calls:
            new_edges.extend(_augment_opaque_calls(fn_pdg))

        if new_edges:
            fn_pdg.def_use_edges.extend(new_edges)
            fn_pdg.__post_init__()  # rebuild traversal indices
            total_edges += len(new_edges)

    logger.info(
        "PDG augmentation complete: added %d synthetic def-use edge(s) "
        "across %d function(s)",
        total_edges,
        sum(1 for fn_pdg in sdg.pdgs.values()
            if any((
                enable_union_aliases and bool(_collect_union_decls(fn_pdg)),
                enable_member_fields and bool(_collect_this_member_accesses(fn_pdg)),
                enable_opaque_calls,
            ))),
    )
    return sdg
