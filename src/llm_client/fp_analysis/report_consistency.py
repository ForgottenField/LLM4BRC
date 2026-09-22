"""Report self-contradiction check (deterministic, no LLM).

A CSA bug report's path asserts a *direction* for every branch condition it
walks.  When the same condition is asserted in **both** directions on one path,
with no write to the condition's variables in between, the reported path cannot
be realized by any input: the report is a false positive, and that conclusion
needs no path enumeration, no POC, and no model call.

Why the existing layers miss it
-------------------------------
* The report's own step text carries no values — CSA prints ``Assuming the
  condition is true/false`` and ``Taking false branch``, so
  :func:`~llm_client.fp_analysis.cfg_feasibility.extract_assumptions_from_events`
  extracts nothing for those steps (:func:`_check_constraint_conflict` then has
  no constraint to compare, and ``_check_global_path_consistency`` is asked
  about the *selected* combination, not the report's path).
* The branch conditions themselves live in the CFG/PDG layer, where
  :class:`~llm_client.fp_analysis.conflict_checker.ConflictChecker` compares a
  branch against its segment postcondition / established state — never against
  another branch of the report path.

The characterization this module exploits: a condition's *text* is available
from the CFG/PDG branch nodes (keyed by source line), and its *direction* is
stated by the report's own events.  Joining the two yields, for the first time,
the report's branch decisions as (condition, direction) pairs — see
:func:`condition_claims`.  Two claims with the same condition text and opposite
directions, on different lines of the same function, with no intervening write,
are contradictory.

Measured examples (both labelled FPs in the faiss corpus, both verified in
``index_read.cpp``):

* 484-1: the path reports ``h == fourcc("IHNs")`` **false** at L901 (the
  short-circuit operand that admits the HNSW arm) and the same
  ``h == fourcc("IHNs")`` **true** at L908 (``if (h == fourcc("IHNs")) idxhnsw
  = new IndexHNSWSQ();``), while ``h`` is written only once, at L506.
* 484-2: ``h == fourcc("INSp")`` **false** at L921 (an operand of the
  ``INSf || INSp || INSs`` chain) and **true** at L925 (``if (h ==
  fourcc("INSp")) idxnsg = new IndexNSGPQ();``), which sits *inside* that
  chain's arm — it can only be reached with the chain true.

CSA could not refute either because ``fourcc`` is an out-of-line function
(``faiss/impl/io.cpp``), so its two call results are independent unconstrained
symbols to the analyzer's solver.

Soundness guards (all of them must hold before a contradiction is reported):

1. **Same function and file.**  A different frame may legitimately hold a
   different value for a same-named local.
2. **Different source lines.**  A loop re-entry reads the *same* line twice, so
   same-line occurrences are never paired.
3. **No intervening write** to any variable of the condition — checked against
   the source lines between the two claims (assignments, ``++``/``--``,
   address-of, and subscript stores) *and* against the report's own events
   (``… stored to 'x'``).  Any doubt (unreadable source, missing file) means no
   claim.
4. **No call step between the two claims.**  A call can make the two
   occurrences *different symbol instances*: CSA re-creates an invalidated
   global as a fresh symbol after a call whose effects it cannot model, and a
   recursive re-entry evaluates the condition in a new frame.  Two ``Assuming
   the condition is …`` steps for one condition text then describe two symbols,
   not a contradiction.  Any ``Calling 'x'`` step between the claims vetoes the
   pair (484-1 has none: its two assumptions are adjacent branch steps).
5. **Unambiguous line pairing.**  A source line may own several conditions (one
   per ``||`` operand).  Claims are paired positionally (report event order ↔
   PDG column order) and only when the counts match exactly; otherwise the line
   is skipped.  The two orders agree because ``||``/``&&`` operands are
   evaluated left to right, which is also the order Clang emits their blocks
   in — and a short-circuit that never evaluates an operand yields *fewer*
   report steps than nodes, i.e. a count mismatch, i.e. skip.

Conservative by construction: every guard fails *closed* (no claim), so a
missed contradiction degrades to the pipeline's existing behaviour, while a
reported one is a proof over the report's own path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: CSA's own wording for a branch outcome on a reported path step:
#: ``← Assuming the condition is true →``.  The control events
#: (``← Taking false branch →``) restate the same decision, usually on the
#: enclosing statement's line rather than the condition's, so they are never
#: used as a direction source here.
_DIRECTION_RE = re.compile(r"Assuming\s+the\s+condition\s+is\s+(true|false)", re.I)

#: PDG node ids look like ``S_901_13`` — line 901, column 13.  The column is
#: what orders several conditions sharing one line (``a || b || c``).
_PDG_NODE_RE = re.compile(r"_S_(\d+)_(\d+)$")

#: Branch expressions that carry no condition: the callee leaf sentinel and the
#: ``branch=True`` fallback that :func:`_infer_condition_from_event` produces
#: when the event text holds no expression.
_NO_CONDITION_RE = re.compile(r"^(\s*)$|^(branch=(True|False|None))$|^→\s")

_IDENT_RE = re.compile(r"[A-Za-z_]\w*")

#: ``Calling 'read_index'`` — CSA's wording for a call step on the path.
_CALLING_RE = re.compile(r"Calling '([^']+)'")
_STRING_LIT_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
_HEX_OR_NUM_RE = re.compile(r"^(0[xX][0-9a-fA-F]+|\d+)$")

_KEYWORDS = frozenset({
    "if", "else", "elif", "while", "for", "do", "switch", "case", "default",
    "return", "break", "continue", "goto", "sizeof", "alignof", "typeof",
    "const", "static", "inline", "volatile", "mutable", "extern", "register",
    "void", "int", "long", "short", "char", "float", "double", "bool", "auto",
    "unsigned", "signed", "struct", "class", "enum", "union", "namespace",
    "true", "false", "null", "nullptr", "NULL", "new", "delete", "operator",
    "and", "or", "not", "this", "throw", "try", "catch", "using", "typedef",
})


# ─── Data ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConditionClaim:
    """One direction the report claims for one branch condition."""

    event_number: int
    line: int
    direction: bool
    condition: str
    function_name: str
    source_file: str

    def describe(self) -> str:
        return (f"{self.condition} → "
                f"{'TRUE' if self.direction else 'FALSE'} "
                f"(event #{self.event_number}, L{self.line})")


@dataclass(frozen=True)
class ReportContradiction:
    """Two claims of one condition that cannot both hold on this path."""

    condition: str
    first: ConditionClaim
    second: ConditionClaim
    variables: list[str] = field(default_factory=list)

    def reason(self) -> str:
        f, s = self.first, self.second
        subjects = _display_vars(self.condition, self.variables)
        return (
            f"报告路径自相矛盾：{self.condition!r} 在 L{f.line} 被判为 "
            f"{'TRUE' if f.direction else 'FALSE'}（事件 #{f.event_number}），"
            f"又在 L{s.line} 被判为 "
            f"{'TRUE' if s.direction else 'FALSE'}（事件 #{s.event_number}），"
            f"而 {', '.join(subjects) or '该条件变量'} 在此期间无写入"
            f"（函数 {f.function_name}，{Path(f.source_file).name}）—— "
            f"该路径不可实现，CSA 无法关联同一条件（判定由 fourcc() 一类跨 TU "
            f"函数/表达式产生，分析器视其为自由符号）"
        )


# ─── Claim extraction ─────────────────────────────────────────────────────────


def condition_claims(segments: list, path_events: list) -> list[ConditionClaim]:
    """Join the report's branch directions with the CFG/PDG condition texts.

    Returns the report path's branch decisions as ``(condition, direction)``
    pairs.  Steps whose line owns no condition node, or whose line's node count
    disagrees with the number of reports steps on it, are dropped (guard 5).
    """
    line_conditions = _line_condition_index(segments)
    if not line_conditions:
        return []

    # Direction records, grouped by line, in path order.
    by_line: dict[int, list[tuple[int, bool]]] = {}
    for evt in path_events or []:
        desc = evt.get("description", "") if isinstance(evt, dict) else ""
        m = _DIRECTION_RE.search(desc)
        if not m:
            continue
        line = int(evt.get("line_number", 0) or 0)
        if line <= 0:
            continue
        by_line.setdefault(line, []).append(
            (int(evt.get("path_number", 0) or 0), m.group(1).lower() == "true"),
        )

    claims: list[ConditionClaim] = []
    for line, records in by_line.items():
        nodes = line_conditions.get(line)
        if not nodes:
            continue
        if len(nodes) != len(records):
            # Ambiguous attribution — several conditions on one line and a
            # different number of steps.  Pairing them by position would guess.
            logger.debug(
                "report-consistency: L%d has %d condition(s) but %d report "
                "step(s) — skipping the line", line, len(nodes), len(records),
            )
            continue
        records.sort(key=lambda r: r[0])
        for (event_number, direction), node in zip(
            records, sorted(nodes, key=lambda n: n.column),
        ):
            claims.append(ConditionClaim(
                event_number=event_number,
                line=line,
                direction=direction,
                condition=node.condition,
                function_name=node.function_name,
                source_file=node.source_file,
            ))
    return claims


@dataclass(frozen=True)
class _LineCondition:
    """One condition a source line owns, with the column that orders it."""

    condition: str
    function_name: str
    source_file: str
    column: int

    @property
    def key(self) -> str:
        """Identity for comparing two conditions: whitespace-insensitive text."""
        return _normalize(self.condition)


def _line_condition_index(segments: list) -> dict[int, list[_LineCondition]]:
    """``line → [_LineCondition]`` from the segments' branch trees.

    Each segment's tree covers its own (file, function); the file is taken from
    the segment because the branch nodes themselves carry no ``source_file``.
    The callee leaf sentinel and the ``branch=…`` fallback are dropped — they
    are not conditions.
    """
    index: dict[int, list[_LineCondition]] = {}
    for seg in segments or []:
        tree = getattr(seg, "cfg_branch_tree", None)
        if tree is None:
            continue
        src = getattr(seg, "function_file", "") or ""
        for node in _walk_tree(list(getattr(tree, "root_branches", []) or [])):
            cond = (getattr(node, "condition_expr", "") or "").strip()
            if _NO_CONDITION_RE.match(cond):
                continue
            if not _IDENT_RE.search(_STRING_LIT_RE.sub("", cond)):
                continue
            line = int(getattr(node, "source_line", 0) or 0)
            if line <= 0:
                continue
            fn = (getattr(node, "function_name", "")
                  or getattr(tree, "function_name", "") or "")
            entry = _LineCondition(
                condition=cond,
                function_name=fn,
                source_file=src,
                column=_node_column(getattr(node, "node_id", "") or ""),
            )
            bucket = index.setdefault(line, [])
            if entry not in bucket:
                bucket.append(entry)
    return index


def _walk_tree(nodes: list) -> list:
    """Depth-first flattening of a segment's branch tree."""
    out: list = []
    for n in nodes:
        out.append(n)
        children = getattr(n, "children", None) or []
        if children:
            out.extend(_walk_tree(children))
    return out


def _node_column(node_id: str) -> int:
    m = _PDG_NODE_RE.search(node_id)
    return int(m.group(2)) if m else 0


def _normalize(condition: str) -> str:
    """Condition identity: whitespace-insensitive text."""
    return re.sub(r"\s+", "", condition)


# ─── Contradiction detection ──────────────────────────────────────────────────


def detect_report_contradictions(
    segments: list, path_events: list, source_root: str | None = None,
) -> list[ReportContradiction]:
    """All same-condition-opposite-direction pairs on the report's own path.

    *source_root* is only used to locate the source file when a segment carries
    no absolute ``function_file``; when it cannot be read the pair is skipped.
    """
    claims = condition_claims(segments, path_events)
    if not claims:
        return []

    by_condition: dict[str, list[ConditionClaim]] = {}
    for c in claims:
        by_condition.setdefault(_normalize(c.condition), []).append(c)

    events = list(path_events or [])
    contradictions: list[ReportContradiction] = []
    seen: set[tuple] = set()
    for cond, group in by_condition.items():
        if len(group) < 2:
            continue
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a.direction == b.direction or a.line == b.line:
                    continue
                if (a.function_name, a.source_file) != (b.function_name, b.source_file):
                    continue
                lo, hi = (a, b) if a.line <= b.line else (b, a)
                key = (cond, lo.line, hi.line)
                if key in seen:
                    continue
                variables = _condition_variables(a.condition)
                if not variables:
                    continue
                called = _intervening_call(events, lo, hi)
                if called:
                    logger.debug(
                        "report-consistency: call to %r between L%d and L%d — "
                        "the two assumptions may be about different symbols, "
                        "not a contradiction", called, lo.line, hi.line,
                    )
                    continue
                if not _no_write_between(lo, hi, variables, events, source_root):
                    continue
                seen.add(key)
                contradictions.append(ReportContradiction(
                    condition=a.condition,
                    first=a,
                    second=b,
                    variables=variables,
                ))
    contradictions.sort(key=lambda c: c.second.event_number)
    return contradictions


def _display_vars(condition: str, variables: list[str]) -> list[str]:
    """The variables of *variables* that are not callee names in *condition*.

    ``fourcc`` in ``h == fourcc("IHNs")`` is read by the condition but is a
    function, so it is a poor answer to "what could change this value" — the
    guard still watches it, only the human-readable reason drops it.
    """
    stripped = _STRING_LIT_RE.sub("", condition)
    calls = {
        m.group(1) for m in re.finditer(r"\b(\w+)\s*\(", stripped)
    }
    named = [v for v in variables if v not in calls]
    return named or variables


def _condition_variables(condition: str) -> list[str]:
    """Identifiers a write to which would change the condition's value.

    String literals are dropped first (``fourcc("IHNs")`` must not contribute
    ``IHNs``), as are keywords, macro-style all-caps names (``MAX_LEN``, rarely
    assignable) and numeric literals.  A one-letter upper-case identifier is
    kept — ``T`` is a template parameter, and keeping it is the conservative
    side of the guard.
    """
    stripped = _STRING_LIT_RE.sub("", condition)
    out: list[str] = []
    for m in _IDENT_RE.finditer(stripped):
        w = m.group(0)
        if w in _KEYWORDS or (w.isupper() and len(w) > 1) or _HEX_OR_NUM_RE.match(w):
            continue
        if w not in out:
            out.append(w)
    return out


def _no_write_between(
    lo: ConditionClaim, hi: ConditionClaim, variables: list[str],
    events: list, source_root: str | None,
) -> bool:
    """True when nothing between the two claims can change the condition.

    Two independent sources of evidence, both required to agree that there is
    no write: the source text of the function between the two lines, and the
    report's own events between the two steps (CSA records ``… stored to 'x'``).
    Unreadable source ⇒ ``False`` (no claim).
    """
    if _reported_store_between(events, lo, hi, variables):
        logger.debug(
            "report-consistency: report records a store to %s between "
            "events #%d and #%d", variables, lo.event_number, hi.event_number,
        )
        return False

    lines = _read_source(lo.source_file, source_root)
    if lines is None:
        return False
    start = max(0, min(lo.line, hi.line) - 1)
    end = min(len(lines), max(lo.line, hi.line))
    for raw in lines[start:end]:
        for var in variables:
            if _writes_var(raw, var):
                logger.debug(
                    "report-consistency: L? writes %r between L%d and L%d — "
                    "not a contradiction", var, lo.line, hi.line,
                )
                return False
    return True


def _reported_store_between(
    events: list, lo: ConditionClaim, hi: ConditionClaim, variables: list[str],
) -> bool:
    """True when a report step between the two names a store to one of them."""
    first = min(lo.event_number, hi.event_number)
    last = max(lo.event_number, hi.event_number)
    for evt in events:
        if not isinstance(evt, dict):
            continue
        try:
            pn = int(evt.get("path_number", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not (first < pn < last):
            continue
        desc = evt.get("description", "") or ""
        for var in variables:
            if re.search(rf"stored\s+to\s+'{re.escape(var)}'", desc):
                return True
    return False


def _intervening_call(
    events: list, lo: ConditionClaim, hi: ConditionClaim,
) -> str | None:
    """The callee of the first call step between the two claims, if any.

    *Any* call is a write source for the condition: a callee with effects the
    analyzer cannot model invalidates globals (CSA re-creates them as fresh
    symbols), and a recursive re-entry reads the condition in a new frame.
    Either way the second ``Assuming …`` is about a different symbol than the
    first, so the pair proves nothing.
    """
    first = min(lo.event_number, hi.event_number)
    last = max(lo.event_number, hi.event_number)
    for evt in events:
        if not isinstance(evt, dict):
            continue
        try:
            pn = int(evt.get("path_number", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not (first < pn < last):
            continue
        # ``called_function`` is set by the parser's annotation pass; read the
        # description too, so the guard also holds for event lists built
        # elsewhere (and for tests).
        called = evt.get("called_function") or ""
        if not called:
            m = _CALLING_RE.search(evt.get("description", "") or "")
            called = m.group(1) if m else ""
        if called:
            return called
    return None


_SOURCE_CACHE: dict[str, list[str] | None] = {}


def _read_source(source_file: str, source_root: str | None) -> list[str] | None:
    """Read the claim's source file, resolving a relative path against the root."""
    if not source_file:
        return None
    candidates = [source_file]
    if not Path(source_file).is_absolute() and source_root:
        candidates.append(str(Path(source_root) / source_file))
    for cand in candidates:
        if cand in _SOURCE_CACHE:
            cached = _SOURCE_CACHE[cand]
            if cached is not None:
                return cached
            continue
        try:
            text = Path(cand).read_text(encoding="utf-8", errors="replace")
        except OSError:
            _SOURCE_CACHE[cand] = None
            continue
        _SOURCE_CACHE[cand] = text.splitlines()
        return _SOURCE_CACHE[cand]
    return None


_WRITE_PATTERN_SRCS = (
    # assignment, but never the ``==`` of a comparison
    r"(?<![=!<>+\-*/%&|^])\b{var}\b\s*(?<![=!<>])=(?!=)",
    r"(\+\+|--)\s*\b{var}\b",
    r"\b{var}\b\s*(\+\+|--)",
    # address taken — a callee may write through the pointer
    r"&\s*\b{var}\b",
    # subscript store: ``v[i] = …``
    r"\b{var}\b\s*\[[^\]]*\]\s*(?<![=!<>])=(?!=)",
)

_WRITE_CACHE: dict[str, tuple] = {}


def _write_patterns(var: str) -> tuple:
    cached = _WRITE_CACHE.get(var)
    if cached is None:
        esc = re.escape(var)
        cached = tuple(re.compile(s.format(var=esc)) for s in _WRITE_PATTERN_SRCS)
        _WRITE_CACHE[var] = cached
    return cached


def _writes_var(line: str, var: str) -> bool:
    """Whether *line* can write *var* (conservatively).

    Covers assignment, ``++``/``--``, address-of (a callee may write through it)
    and subscript stores.  A comparison (``h == …``) is not a write.
    """
    return any(p.search(line) for p in _write_patterns(var))
