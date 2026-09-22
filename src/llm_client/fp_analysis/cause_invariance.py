"""Structural-FP cause-invariance — deterministically prove a decisive FP on one
path generalizes to the whole (relevant) branch space.

The verifier can conclude a report is a *decisive structural* FP when the CSA
path is internally self-contradictory (``csa_contradiction``) or the LLM marks it
``definite_fp`` (e.g. "the deref'd pointer cannot be null under the real API
contract").  That contradiction is report-level: it concerns the *root-cause
variable* (the deref'd pointer), not the branch directions of the enumerated
space.  So if none of the *relevant* branches that vary across the reduced space
writes that root-cause pointer, the pointer's null-ness (and hence the
contradiction) is identical on every combination → a single decisive FP
generalizes to the whole space (16→1 simulation).

Pure functions; no LLM calls.

Soundness note: enabling a generalization relies on the *caller-side* call node
not writing the pointer (C++ passes pointers by value, so a callee cannot
reassign the caller's pointer — it can only write through it).  A cross-function
pass-by-reference-to-pointer reassignment would be missed; we mitigate with the
callee ``output_variables`` blocker and by requiring a positive caller-side
resolution.  Combined with the structural-FP gate, the truncation guard, and a
successful root-cause-variable extraction, generalization only fires on positive
evidence.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from llm_client.fp_analysis.pdg_models import SDG
from llm_client.fp_analysis.slicer import find_trigger_node
from llm_client.fp_analysis.combination_enum import _ci_dims
from llm_client.fp_analysis.branch_relevance import _CONTROL_KINDS, _line_matches


# ---------------------------------------------------------------------------
# Root-cause pointer extraction
# ---------------------------------------------------------------------------


# Keywords/literals that can never name the dereferenced pointer.  A bug reported at
# a function's closing brace (a leak report) resolves its trigger node to the
# ``return`` statement, whose expression is ``return idx;`` — without this guard
# ``_base_identifier`` hands back the keyword ``return``.  A "root-cause pointer"
# that no branch writes makes ``cause_invariant`` trivially true, which would
# generalize a decisive FP (and, for a leak report, for a reason that has nothing to
# do with the report at all — the structural contradiction there is about branch
# directions, not about a pointer).  Refusing here is conservative: it only ever
# blocks a generalization.
_NON_POINTER_WORDS = frozenset({
    "return", "if", "else", "for", "while", "do", "switch", "case", "default",
    "break", "continue", "goto", "throw", "try", "catch", "sizeof", "new",
    "delete", "this", "nullptr", "NULL", "true", "false", "void", "const",
    "static", "struct", "class", "enum", "typedef", "using", "namespace",
    "auto", "bool", "char", "short", "int", "long", "float", "double",
    "unsigned", "signed", "operator",
})


def _base_identifier(text: Optional[str]) -> Optional[str]:
    """Leading identifier of a dereference expression: ``info->ai_addr`` /
    ``*results`` / ``(info)`` → ``info`` / ``results``.  Skips a ``this``
    receiver (``this->info_->ai_addr`` → ``info_``).  ``None`` if no identifier
    is present, or if the leading word is a keyword/literal
    (``_NON_POINTER_WORDS``) rather than a variable."""
    if not text:
        return None
    s = text.strip()
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    s = s.lstrip("&*").strip()
    for recv in ("this->", "this."):
        if s.startswith(recv):
            s = s[len(recv):]
            break
    m = re.match(r"([A-Za-z_]\w*)", s)
    if not m:
        return None
    base = m.group(1)
    return None if base in _NON_POINTER_WORDS else base


def _descriptions(parsed) -> list[str]:
    out: list[str] = []
    es = getattr(parsed, "error_start_event", None) or {}
    if es.get("description"):
        out.append(str(es["description"]))
    for evt in getattr(parsed, "path_events", []) or []:
        d = evt.get("description") if isinstance(evt, dict) else getattr(evt, "description", None)
        if d:
            out.append(str(d))
    return out


def extract_deref_pointer(parsed, sdg: SDG) -> Optional[str]:
    """The base pointer V that is dereferenced at the bug (e.g. ``info`` for
    ``info->ai_addr``).  ``None`` if it cannot be recovered (do not generalize).

    Sources, in order: the trigger/deref PDG node's expression base identifier;
    the CSA ``'<X>' is null`` path-event text; the trigger node's ``variable``.
    """
    meta = getattr(parsed, "metadata", None)
    fn = getattr(meta, "function_name", None) or None
    line = getattr(meta, "bug_line", None) or 0
    trigger_nid: Optional[str] = None
    trigger_node = None
    if fn:
        trigger_nid = find_trigger_node(fn, line, sdg)
        if trigger_nid:
            pdg = sdg.get_pdg(fn)
            trigger_node = pdg.nodes.get(trigger_nid) if pdg is not None else None

    # 1. CSA deref event text names the null pointer directly (most reliable):
    #    "dereference of a null pointer (loaded from variable 'info')".
    for desc in _descriptions(parsed):
        m = re.search(
            r"dereference of a null pointer\s*\(loaded from variable '([A-Za-z_]\w*)'\)",
            desc,
        )
        if m:
            return m.group(1)
    # 2. CSA ``'<X>' is null`` event text.
    for desc in _descriptions(parsed):
        m = re.search(r"'([A-Za-z_]\w*(?:\.\w+|->\w+)*)'\s+is\s+null", desc)
        if m:
            base = _base_identifier(m.group(1))
            if base:
                return base
    # 3. trigger node expression base (skips a ``this`` receiver).
    if trigger_node is not None:
        base = _base_identifier(trigger_node.expression)
        if base:
            return base
    # 4. trigger node variable.
    if trigger_node is not None and trigger_node.variable:
        base = _base_identifier(trigger_node.variable)
        if base:
            return base
    return None


# ---------------------------------------------------------------------------
# "Does this branch write the pointer?" — deterministic
# ---------------------------------------------------------------------------


def _node_writes(node, var: str) -> bool:
    """True if *node* (re)assigns the pointer *var* itself (``var = ...``),
    NOT writes through it (``var->field = ...``, ``callee(var, ...)``)."""
    if not var:
        return False
    if (getattr(node, "variable", None) or "") == var:
        return True
    expr = getattr(node, "expression", None) or ""
    # ``var = `` with a RHS char that isn't ``=``/``>`` excludes ``==`` / ``!=``.
    return re.search(r"\b" + re.escape(var) + r"\s*=\s*[^=>]", expr) is not None


def _parse_pdg(node_id: str) -> Optional[tuple[str, str]]:
    """``pdg_<fn>_S_<line>_<col>`` → ``(fn, predicate_id)``."""
    rest = node_id[len("pdg_"):]
    mark = rest.find("_S_")
    if mark == -1:
        return None
    return rest[:mark], rest[mark + 1:]


def _parse_callee(decision: dict) -> str:
    """Short callee name from a call-edge decision (reuse branch_relevance rule)."""
    cond = (decision.get("condition_expr") or "").strip()
    if cond.startswith("→ ") and "()" in cond:
        name = cond[2:].split("()", 1)[0].strip()
        if name:
            return name
    node_id = (decision.get("node_id") or "").strip()
    if node_id.startswith("callee_"):
        parts = node_id.split("_")
        for i, p in enumerate(parts):
            if p.startswith("L") and p[1:].isdigit() and i >= 2:
                return "_".join(parts[2:i])
    return ""


def _pdg_writes(pdg, pred: str, var: str) -> bool:
    """True if the predicate *pred* (or any statement it controls transitively)
    writes the pointer *var*."""
    node = pdg.nodes.get(pred)
    if node is not None and _node_writes(node, var):
        return True
    seen: set[str] = set()
    stack = list(pdg.ctrl_succs_by_node.get(pred, set()))
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        nd = pdg.nodes.get(nid)
        if nd is not None and _node_writes(nd, var):
            return True
        stack.extend(pdg.ctrl_succs_by_node.get(nid, set()))
    return False


def _resolve_seg(sdg: SDG, decision: dict, bug_file: Optional[str]) -> Optional[tuple[str, str]]:
    """``seg_`` CFG fallback → ``(canon_fn, predicate_id)`` via line index."""
    if not bug_file:
        return None
    line = decision.get("source_line", 0) or 0
    cond = (decision.get("condition_expr") or "").strip()
    matches = _line_matches(sdg.build_line_index(), bug_file, line)
    best = None
    for fn, nid in matches:
        pdg = sdg.get_pdg(fn)
        if pdg is None:
            continue
        node = pdg.nodes.get(nid)
        if node is None or node.kind not in _CONTROL_KINDS:
            continue
        if cond and (node.expression or "").strip() and \
                cond not in (node.expression or "") and \
                (node.expression or "") not in cond:
            continue
        best = (pdg.function_name, nid)
        if not cond:
            break
    return best


def _call_writes(sdg: SDG, decision: dict, var: str, bug_file: Optional[str]) -> bool:
    """True if a call-edge branch writes the pointer *var*.

    Conservative: only returns False (non-write) when we positively resolved the
    caller-side call site and neither it nor the callee's declared outputs write
    *var*.  Any failed resolution → True (do not generalize).
    """
    callee = _parse_callee(decision)
    if callee:
        cpdg = sdg.get_pdg(callee)
        if cpdg is not None:
            out = (cpdg.summary.output_variables if getattr(cpdg, "summary", None)
                   else None) or []
            if var in out:
                return True
    line = decision.get("source_line", 0) or 0
    if bug_file:
        matches = _line_matches(sdg.build_line_index(), bug_file, line)
        if matches:
            for fn, nid in matches:
                pdg = sdg.get_pdg(fn)
                node = pdg.nodes.get(nid) if pdg is not None else None
                if node is not None and _node_writes(node, var):
                    return True
            return False  # found the line's nodes; none writes *var*
    return True  # could not resolve caller-side → cannot prove non-write


def branch_writes_pointer(
    sdg: SDG, decision: dict, var: str, *, bug_file: Optional[str] = None,
) -> bool:
    """Deterministic: does *decision* (a branch decision) write the pointer *var*?

    Conservative — any unparseable / unresolvable case returns True (do not
    generalize)."""
    node_id = (decision.get("node_id") or "").strip()
    if not node_id:
        return True
    if node_id.startswith("pdg_"):
        res = _parse_pdg(node_id)
        if res is None:
            return True
        fn, pred = res
        pdg = sdg.get_pdg(fn)
        return _pdg_writes(pdg, pred, var) if pdg is not None else True
    if node_id.startswith("seg_"):
        res = _resolve_seg(sdg, decision, bug_file)
        if res is None:
            return True
        fn, pred = res
        pdg = sdg.get_pdg(fn)
        return _pdg_writes(pdg, pred, var) if pdg is not None else True
    if node_id.startswith("callee_") or node_id.startswith("call_"):
        return _call_writes(sdg, decision, var, bug_file)
    return True


# ---------------------------------------------------------------------------
# Whole-space invariance
# ---------------------------------------------------------------------------


def _iter_decisions(selections):
    for sel in selections or []:
        for d in sel.get("branch_decisions", []) or []:
            yield d


def _dimensions(path_space: dict) -> list[list[dict]]:
    """``path_space``'s enumeration dimensions, as candidate lists.

    A dimension is either one segment or a context-insensitive group (identical
    branch sets collapsed together, forced to share a candidate).  Only a
    dimension's first ``count`` candidates are enumerable (``iter_combinations``
    walks ``range(count)``, and ``build_selections_from_combo`` indexes the
    representative's list), so the scans below use exactly those.
    """
    per_segment = path_space["per_segment"]
    out: list[list[dict]] = []
    for dim in _ci_dims(path_space):
        cands = per_segment[dim["members"][0]]["candidates"]
        out.append(list(cands[: dim["count"]]))
    return out


def cause_invariant(
    sdg: SDG,
    reference_selections,
    reduced_ps: dict,
    segments: Sequence,
    node_relevance: dict,
    var: Optional[str],
    *,
    truncated: bool = False,
    bug_file: Optional[str] = None,
) -> bool:
    """True iff the root-cause pointer *var* is identical on every reduced
    combination and on the reference path.

    Generalizes a decisive structural FP on the reference path to the whole
    reduced space iff no *relevant* branch that writes *var* takes a different
    value anywhere in the space (including being absent in one path but present
    in another).  Irrelevant branches (slice-confirmed removed) have no
    data/control path to the trigger, so they cannot write *var* and are ignored.

    The scan is EXHAUSTIVE without ever enumerating the cross-product.  It used
    to iterate ``iter_combinations(reduced_ps, segments)`` twice — but that
    helper bounds itself (``_ENUM_LIMIT`` = 200_000 enumerated, ``_YIELD_CAP`` =
    100_000 yielded) for memory safety, so above those bounds it visited only a
    lexicographic *prefix* of the combos and could report "invariant" from an
    incomplete scan.  That is exactly what the caller's
    ``_STRUCTURAL_GP_MAX = 1000`` cap existed to keep out of a huge space's
    reach, at the price of never generalizing inside one.  Instead: every combo
    picks exactly one candidate per *dimension*, and a writer branch's value on a
    combo is the one from the HIGHEST dimension whose chosen candidate carries it
    at all (``None`` when none does — how the cross-product scan saw an absent
    branch).  So a per-dimension scan of each dimension's candidate list, plus
    which dimensions can ever be the highest supplier, decides exactly what
    enumerating every combo decided — linear in the materialized path space
    (dimensions × candidates × decisions), never in the cross-product, so the cap
    is gone.

    Conservative where it must be: an unresolvable write analysis returns True
    (do not generalize); a branch carrying several values inside one candidate
    (a loop re-entry recorded twice) is kept as several values, so any of them
    failing to match the reference blocks the generalization.
    """
    if truncated or not var:
        return False

    cache: dict[tuple[str, int], bool] = {}

    def is_relevant_writer(d) -> bool:
        nid = (d.get("node_id") or "").strip()
        if not nid:
            return False
        if not node_relevance.get(nid, True):  # unknown treated as relevant
            return False
        key = (nid, d.get("source_line", 0) or 0)
        if key not in cache:
            cache[key] = branch_writes_pointer(sdg, d, var, bug_file=bug_file)
        return cache[key]

    # Reference values on the decisive path (None when a writer branch is absent there).
    ref: dict[str, Optional[bool]] = {}
    for d in _iter_decisions(reference_selections):
        nid = (d.get("node_id") or "").strip()
        if nid:
            ref[nid] = bool(d.get("branch_taken", True))

    writer_nids: set[str] = set()
    for d in _iter_decisions(reference_selections):
        if is_relevant_writer(d):
            writer_nids.add((d.get("node_id") or "").strip())

    dims = _dimensions(reduced_ps)
    if any(not cands for cands in dims):
        # An enumerable dimension with no candidate makes the cross-product empty,
        # and `iter_combinations` then yields nothing — a vacuous "every combo
        # agrees".  Keep that reading rather than inventing a counter-example.
        return True

    # Per dimension: nid → (values the branch can take there, how many of the
    # dimension's candidates carry it at all).
    dim_info: list[dict[str, tuple[set, int]]] = []
    for cands in dims:
        info: dict[str, tuple[set, int]] = {}
        vals: dict[str, set] = {}
        counts: dict[str, int] = {}
        for cand in cands:
            for d in cand.get("branch_decisions") or []:
                nid = (d.get("node_id") or "").strip()
                if not nid:
                    continue
                if is_relevant_writer(d):
                    writer_nids.add(nid)
                    vals.setdefault(nid, set()).add(bool(d.get("branch_taken", True)))
                counts[nid] = counts.get(nid, 0) + 1
        for nid in counts:
            info[nid] = (vals.get(nid, set()), counts[nid])
        dim_info.append(info)

    if not writer_nids:
        return True  # no relevant branch writes the pointer anywhere → invariant

    def is_partial(di: int, nid: str) -> bool:
        """Does *di* carry the branch in only SOME of its candidates?"""
        _vals, cnt = dim_info[di][nid]
        return cnt < len(dims[di])

    for nid in writer_nids:
        rv = ref.get(nid)
        suppliers = [di for di in range(len(dims)) if nid in dim_info[di]]
        if not suppliers:
            # The reference path takes this writer branch; no combo does.
            return False
        if all(is_partial(di, nid) for di in suppliers):
            # Some combo picks a non-carrying candidate everywhere → the branch is
            # absent there, which is not the same value as on the reference path.
            return False
        # A combo's value is the one from the HIGHEST dimension whose candidate
        # carries the branch.  So `di` supplies a combo only when every higher
        # supplier has a non-carrying candidate to fall through from; otherwise
        # `di` is always overwritten and its values never reach a combo.
        for di in suppliers:
            if all(is_partial(di2, nid) for di2 in suppliers if di2 > di):
                if dim_info[di][nid][0] != {rv}:
                    return False
    return True
