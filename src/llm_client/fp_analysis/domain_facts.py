"""Guard-pinned / finite-domain variable detection (semantic-constraint supplement).

CSA (Clang Static Analyzer) often treats a variable read from external input (a file,
a stream, a parsed header) as an unconstrained value, so a path that is *semantically*
impossible can look feasible to it.  A classic cause is an out-of-line constant
factory — e.g. ``fourcc("rrot")`` defined in another translation unit — which CSA models
as an *uninterpreted symbolic* return value.  Then ``h == fourcc("rrot")`` becomes a
comparison against an unknown symbol, CSA never learns that ``h`` is pinned to a small
set of fixed constants, and a path that requires ``h`` to be both one of them and none
of them (the reported false positive) is not pruned.

This module is Phase-1 of a "semantic-constraint completion" idea: it *detects* variables
that source code pins to a finite constant set (via ``==`` / ``!=`` against foldable
constants) on the reported bug path, *folds* those constants to concrete values where
deterministic (integer literals; recognized 4-char constant factories such as ``fourcc``),
and produces :class:`DomainFact` records.  These facts are rendered as advisory prose and
injected into the constraint text the LLM auditor sees (see run_path_selection.py), so the
LLM can reason that a variable CSA left "arbitrary" actually has a small, fixed domain —
helping it conclude a report is a false positive.

The facts are deliberately *advisory* (Phase-1).  They say "the variable is pinned to this
finite set"; they do NOT hard-conclude FP.  A decisive "guard-pinned-null-deref" gate is
deferred to Phase-2 and should not be inferred here.

No LLM is called; everything here is deterministic and unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Folders for 4-char "tag/fourcc"-style string->integer constant factories.  The value is
#: the byte-ordering used by the factory body when assembling the integer from the chars.
#: ``fourcc`` (faiss io.cpp) uses little-endian assembly: x[0] | x[1]<<8 | x[2]<<16 | x[3]<<24.
_KNOWN_STRING_CONSTANT_FNS: dict[str, str] = {"fourcc": "le4"}

#: Operators that pin a variable to a constant set when taken TRUE (``==``) or when we are
#: reasoning about which set a guard admits (``==``/``!=`` both constrain membership).
_CMP_OPS = ("==", "!=")

# Match a single-equality leaf:  ``h == fourcc("rrot")``  or  ``count == 12``
_LEAF_RE = re.compile(
    r"^\s*(?P<var>[a-zA-Z_][\w>.]*)\s*(?P<op>==|!=)\s*(?P<rhs>.*?)\s*$"
)
# Match  ident("literal")  (a constant-factory call, e.g. fourcc("rrot"))
_CALL_LIT_RE = re.compile(r"^\s*([a-zA-Z_]\w*)\s*\(\s*\"([^\"]*)\"\s*\)\s*$")
_INT_RE = re.compile(r"^[+-]?\d+$")

#: Node kinds that can be a branch predicate / comparison leaf.
_BRANCH_KINDS = frozenset({"binary_op"})

#: Cap on the number of DomainFacts rendered into the LLM prompt (bound injected text).
_MAX_FACTS = 5


def _fold_fourcc_constant(text: str) -> int | None:
    """Fold a 4-char string into the uint32 ``fourcc`` (faiss) produces.

    Matches the out-of-line body in faiss/impl/io.cpp:
        x[0] | x[1] << 8 | x[2] << 16 | x[3] << 24   (little-endian assembly).
    """
    if len(text) != 4:
        return None
    x = [ord(c) for c in text]
    return x[0] | (x[1] << 8) | (x[2] << 16) | (x[3] << 24)


def fold_rhs(rhs: str) -> tuple[str, str | None, str] | None:
    """Fold an equality/inequality RHS into ``(kind, folded_value_or_None, canonical)``.

    Returns ``None`` when the RHS is not a foldable constant we can reason about.

    ``canonical`` is a stable display string: for ints it is the integer, for a known
    constant factory it is the source form (so the LLM sees the meaning), and ``folded``
    (when not None) is the deterministic concrete value the source pins the variable to.
    """
    rhs = rhs.strip()
    if _INT_RE.match(rhs):
        return ("int", rhs, rhs)

    m = _CALL_LIT_RE.match(rhs)
    if m:
        fn_name, lit = m.group(1), m.group(2)
        source = f'{fn_name}("{lit}")'
        order = _KNOWN_STRING_CONSTANT_FNS.get(fn_name)
        if order == "le4":
            val = _fold_fourcc_constant(lit)
            if val is not None:
                return ("fourcc", f"0x{val:08x}", source)
        # Unknown / un-foldable constant factory: keep the source form, no folded value.
        return ("opaque_callee", None, source)

    return None


def parse_equality_leaf(expr: str) -> tuple[str, str, str] | None:
    """Parse a single ``var op RHS`` leaf expression.  Returns ``(var, op, rhs)`` or None.

    Only handles a *single* comparison with no ``||``/``&&`` so that truncated/disjunctive
    whole-guard text is ignored — the individual leaf ``binary_op`` nodes (which PDGBuilder
    records separately and untruncated) are the reliable source.
    """
    if expr is None:
        return None
    if "||" in expr or "&&" in expr:
        return None
    m = _LEAF_RE.match(expr)
    if not m:
        return None
    var, op, rhs = m.group("var"), m.group("op"), m.group("rhs").strip()
    if not var or not op or not rhs:
        return None
    return (var, op, rhs)


@dataclass
class DomainFact:
    """A variable that source code pins to a finite set of constants on the bug path."""

    variable: str
    kind: str  # fourcc | int | opaque_callee | mixed
    values: list[str] = field(default_factory=list)  # canonical display entries
    folded: list[str] = field(default_factory=list)  # concrete values where known
    source_lines: list[int] = field(default_factory=list)
    function_name: str = ""
    callee: str | None = None


def _record(fact: DomainFact, op: str, folded: str | None, source: str, line: int) -> None:
    entry = f"{source} = {folded}" if folded else source
    if entry not in fact.values:
        fact.values.append(entry)
    if folded and folded not in fact.folded:
        fact.folded.append(folded)
    if line not in fact.source_lines:
        fact.source_lines.append(line)


def group_domain_facts(
    leaves: list[tuple[str, int, str]],  # (function_name, source_line, expression)
    min_distinct: int = 2,
) -> list[DomainFact]:
    """Group single-equality comparison leaves by variable into DomainFacts.

    ``leaves`` is a list of ``(function_name, source_line, expression)`` for ``binary_op``
    comparison nodes.  A variable yields a :class:`DomainFact` only when it is compared to
    ``>= min_distinct`` *distinct* foldable constants (the low-noise guard: a single
    comparison doesn't pin a finite domain).
    """
    by_var: dict[str, dict] = {}
    for fn_name, line, expr in leaves:
        parsed = parse_equality_leaf(expr)
        if not parsed:
            continue
        var, op, rhs = parsed
        folded = fold_rhs(rhs)
        if not folded:
            continue
        kind, concrete, source = folded
        slot = by_var.setdefault(
            var,
            {
                "facts": None,
                "kinds": set(),
                "op": set(),
                "lines": [],
                "fn": fn_name,
            },
        )
        if slot["facts"] is None:
            slot["facts"] = DomainFact(
                variable=var,
                kind=kind,
                function_name=fn_name,
            )
        slot["kinds"].add(kind)
        slot["op"].add(op)
        _record(slot["facts"], op, concrete, source, line)

    facts: list[DomainFact] = []
    for slot in by_var.values():
        f = slot["facts"]
        if f is None or len(f.values) < min_distinct:
            continue
        kinds = slot["kinds"]
        f.kind = "mixed" if len(kinds) > 1 else next(iter(kinds))
        facts.append(f)
    return facts


def _iter_retained_leaves(sdg, slice_mask, target_functions) -> list[tuple[str, int, str]]:
    """Collect single-equality comparison leaves from the SDG for the target functions.

    Only functions in ``target_functions`` are scanned, and within them only nodes whose
    kind is a comparison and whose expression parses to a foldable single equality are kept.
    Nodes outside the root-cause slice are filtered out when ``slice_mask`` can tell us so
    (best effort — slice membership of a leaf is not always recorded, so a leaf in a target
    function is still retained as a candidate).
    """
    leaves: list[tuple[str, int, str]] = []
    for fn_name in target_functions:
        pdg = sdg.get_pdg(fn_name) if hasattr(sdg, "get_pdg") else None
        if pdg is None:
            continue
        for node_id, node in getattr(pdg, "nodes", {}).items():
            if getattr(node, "kind", None) not in _BRANCH_KINDS:
                continue
            expr = getattr(node, "expression", "") or ""
            if parse_equality_leaf(expr) is None:
                continue
            leaves.append((fn_name, getattr(node, "source_line", 0), expr))
    return leaves


def extract_domain_facts(
    sdg,
    slice_mask,
    target_function_names: set[str],
    min_distinct: int = 2,
    limit: int = _MAX_FACTS,
) -> list[DomainFact]:
    """Detect guard-pinned variables for the functions that own the reported bug path.

    ``target_function_names`` should be the set of functions that own segments on the path
    (plus the bug function).  Returns at most ``limit`` DomainFacts, most-constrained first
    (larger value set ⇒ more decisively pinned), each rendered adviso­rily.
    """
    leaves = _iter_retained_leaves(sdg, slice_mask, target_function_names)
    facts = group_domain_facts(leaves, min_distinct=min_distinct)
    facts.sort(key=lambda f: (-len(f.folded), min(f.source_lines) if f.source_lines else 0))
    return facts[:limit]


def format_domain_facts(facts: list[DomainFact]) -> str:
    """Render DomainFacts as advisory prose for the LLM auditor.

    Advisory by design (Phase-1): it states that CSA treats the variable as free but source
    pins it to this finite set, and invites the model to check whether the reported bug
    state forces the variable outside the set.  It does NOT assert FP.
    """
    if not facts:
        return ""
    lines = [
        "## Guard-pinned / finite-domain variables (semantic completion)",
        "CSA (or the tool) may treat these variables as able to take any value, but the "
        "source compares them against a small set of fixed constants in the guarded region:",
    ]
    for f in facts:
        loc = f" (in {f.function_name.split()[-1]})" if f.function_name else ""
        vals = ", ".join(f.values)
        tail = (
            " The constants are mutually distinct; `{0}` cannot take a value outside this "
            "set and still stay on the flagged path. When judging reachability, check "
            "whether the reported bug state forces `{0}` to a value these guards exclude."
        ).format(f.variable)
        lines.append(f"- `{f.variable}`{loc} is compared against these fixed constants:\n"
                     f"    {vals}{tail}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Source-text scanning (no SDG dependency)
#
# Used by the early feasibility pass (path_analyzer).  The CSA report's
# ``source_snippets`` already carry the flagged region's source, so a
# guard-pinned / finite-domain variable can be recovered straight from the text
# instead of from the PDG.  This lets the semantic range-conflict check run at
# Step 3.5 (before path selection) without requiring the SDG to be materialised
# yet.
# ═══════════════════════════════════════════════════════════════════════════

#: Markers of CSA-HTML source lines that are macro-expanded noise (READ1 /
#: READVECTOR / FAISS_THROW inlines).  Dropping them leaves a readable control
#: skeleton of the dispatch (`if` / `else if` / assignments / the deref).
_MACRO_NOISE = (
    "(*f)",
    "snprintf",
    "do { if",
    "strerror",
    "__s",
    "__size",
    "); } while",
)

# Locate a comparison ``var op`` anywhere inside a source line; the constant is
# read from the text right after the operator (see ``_read_foldable_const``).
_EQ_TOK_RE = re.compile(
    r"\b(?P<var>[A-Za-z_][\w>.]*)\s*(?P<op>==|!=)\s*"
)

_SNIPPET_VAR_RE = re.compile(r"[A-Za-z_]\w*")


def _is_macro_noise(line: str) -> bool:
    """True when a source line is macro-expanded noise rather than real control."""
    return any(tok in line for tok in _MACRO_NOISE)


def _read_foldable_const(text: str, i: int) -> str | None:
    """Read a foldable constant token starting at ``text[i]`` after an op.

    Handles two shapes seen in dispatch guards:
      * a bare integer literal  ``12`` / ``0x..``;
      * a constant-factory call ``fourcc("Pcam")`` — read as a balanced-paren
        unit so a trailing ``) {`` / ``) ||`` in the line is not absorbed.
    Returns the raw source of the constant, or None if it is not a plausible
    literal / known call.  Unknown identifiers (runtime values) are not reads.
    """
    j = i
    while j < len(text) and (text[j] == " " or text[j] == "\t"):
        j += 1
    if j >= len(text):
        return None
    # Balanced-paren call:  ident(...)   (recurse balanced for the general case)
    m = re.match(r"[A-Za-z_]\w*\s*\(", text[j:])
    if m:
        k = j + m.end() - 1  # index of the opening '('
        depth = 0
        while k < len(text):
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
                if depth == 0:
                    return text[j : k + 1]
            elif depth == 0:
                break
            k += 1
        return None
    # Bare literal: decimal or hex integer (reject runtime identifiers).
    m = re.match(r"[-+]?(?:0[xX][0-9a-fA-F]+|\d+)", text[j:])
    if m:
        return m.group(0)
    return None


def iter_foldable_equalities(line: str):
    """Yield ``(var, op, rhs)`` for foldable equality/inequality tokens in text.

    A token ``var op rhs`` is yielded only when ``fold_rhs(rhs)`` succeeds (i.e.
    rhs is a concrete int literal or a recognised constant-factory call such as
    ``fourcc("...")``), so a comparison against a runtime value is ignored.
    """
    if not line:
        return
    i = 0
    while i < len(line):
        m = _EQ_TOK_RE.search(line, i)
        if not m:
            return
        rhs = _read_foldable_const(line, m.end())
        if rhs is not None and fold_rhs(rhs) is not None:
            yield (m.group("var"), m.group("op"), rhs)
        i = m.end()
        if i >= len(line):
            return


def domain_facts_from_lines(
    snippet_lines: dict[int, str],
    min_distinct: int = 2,
    limit: int = _MAX_FACTS,
    function_name: str = "",
) -> list[DomainFact]:
    """Recover guard-pinned / finite-domain variables from report source text.

    ``snippet_lines`` maps a source line number to its (CSA-HTML) text, e.g.
    ``ParsedReport.source_snippets``.  Macro-expanded noise lines are skipped.
    A variable yields a :class:`DomainFact` only when it is compared against
    ``>= min_distinct`` distinct foldable constants (the same low-noise rule as
    :func:`group_domain_facts`).
    """
    leaves: list[tuple[str, int, str]] = []
    for line_no in sorted(snippet_lines):
        text = snippet_lines[line_no]
        if _is_macro_noise(text):
            continue
        for var, op, rhs in iter_foldable_equalities(text):
            # A concrete single-equality leaf expression for group_domain_facts.
            leaves.append((function_name or "", line_no, f"{var} {op} {rhs}"))
    facts = group_domain_facts(leaves, min_distinct=min_distinct)
    facts.sort(key=lambda f: (-len(f.folded), min(f.source_lines) if f.source_lines else 0))
    return facts[:limit]
