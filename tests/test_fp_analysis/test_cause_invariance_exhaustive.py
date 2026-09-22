"""`cause_invariant` must be exhaustive without enumerating the cross-product (task B).

The structural-FP generalization is only sound if the scan really covers every
reduced combination.  It used to iterate ``iter_combinations`` (twice), which
bounds itself for memory safety (``_ENUM_LIMIT`` = 200_000 enumerated,
``_YIELD_CAP`` = 100_000 yielded) — above those bounds it visited a lexicographic
*prefix* of the combos, so "invariant" could come from a partial scan.  That is
why the caller carried a ``_STRUCTURAL_GP_MAX = 1000`` cap: it kept a huge space
out of reach of the incomplete answer, at the price of never generalizing inside
one (a decisive structural FP in a huge space could then only ever be UNKNOWN).

`cause_invariant` now decomposes the same predicate over the enumeration
dimensions (a combo picks exactly one candidate per dimension, so a writer
branch's value depends only on its own dimension's candidate), which is linear in
the materialized path space.  These tests pin (a) equivalence with the
cross-product formulation on small spaces, (b) exhaustiveness past the old
enumeration bound, and (c) the deliberately conservative corners.
"""

import random

from llm_client.fp_analysis.combination_enum import iter_combinations
from llm_client.fp_analysis.cause_invariance import (
    _iter_decisions,
    branch_writes_pointer,
    cause_invariant,
)


# ── a tiny fake SDG: a PDG node "writes" the pointer when its `variable` is it ──

class _Node:
    def __init__(self, variable=None, expression=""):
        self.variable = variable
        self.expression = expression


class _Pdg:
    def __init__(self, writers):
        # `_parse_pdg("pdg_f_S_1_3")` → ("f", "S_1_3"), which is the nodes key
        self.nodes = {f"S_1_{i}": _Node(variable="p") for i in writers}
        self.ctrl_succs_by_node = {}
        self.function_name = "f"


class _Sdg:
    def __init__(self, writers):
        self._pdg = _Pdg(writers)

    def get_pdg(self, fn):
        return self._pdg if fn == "f" else None


def _decision(node_key: str, taken: bool) -> dict:
    """``node_key`` is ``(line, col)`` of a ``pdg_f_S_1_<col>`` node."""
    return {"node_id": f"pdg_f_S_1_{node_key}", "branch_taken": taken,
            "source_line": 1, "condition_expr": "p"}


def _candidate(decisions) -> dict:
    sig = ",".join(f"L1:{'T' if d['branch_taken'] else 'F'}" for d in decisions)
    return {"signature": sig or "linear", "rank": 1, "branch_decisions": list(decisions)}


def _path_space(candidate_lists) -> dict:
    return {"per_segment": [
        {"candidates": cands, "num_feasible_candidates": len(cands)}
        for cands in candidate_lists
    ]}


class _Seg:
    """The segment attributes ``build_selections_from_combo`` reads."""

    def __init__(self, i):
        self.function_name = f"f{i}"
        self.label = f"S{i}"
        self.line_start = 1
        self.line_end = 2


def _reference_cause_invariant(sdg, reference_selections, path_space, var):
    """The pre-rewrite formulation: iterate ``iter_combinations`` (the real one).

    Using the production helper keeps the reference faithful — it honours
    ``ci_dims`` (a context-insensitive group is one dimension) and the
    ``num_feasible_candidates`` cap exactly like the old code did.
    """
    segments = [_Seg(i) for i in range(len(path_space["per_segment"]))]

    def writer_of(d):
        nid = (d.get("node_id") or "").strip()
        if not nid:
            return None
        return nid if branch_writes_pointer(sdg, d, var) else None

    writer_ids = set()
    for _combo, sel in iter_combinations(path_space, segments):
        for d in _iter_decisions(sel):
            nid = writer_of(d)
            if nid:
                writer_ids.add(nid)
    for d in _iter_decisions(reference_selections):
        nid = writer_of(d)
        if nid:
            writer_ids.add(nid)
    if not writer_ids:
        return True

    ref = {nid: None for nid in writer_ids}
    for d in _iter_decisions(reference_selections):
        nid = (d.get("node_id") or "").strip()
        if nid in writer_ids:
            ref[nid] = bool(d.get("branch_taken", True))

    for _combo, sel in iter_combinations(path_space, segments):
        vals = {nid: None for nid in writer_ids}
        for d in _iter_decisions(sel):
            nid = (d.get("node_id") or "").strip()
            if nid in writer_ids:
                vals[nid] = bool(d.get("branch_taken", True))
        for nid in writer_ids:
            if ref[nid] != vals[nid]:
                return False
    return True


def _run(path_space, reference_selections, writers, var="p"):
    sdg = _Sdg(writers)
    new = cause_invariant(sdg, reference_selections, path_space, [], {}, var)
    old = _reference_cause_invariant(sdg, reference_selections, path_space, var)
    return new, old


def _random_space(rng, node_pool):
    n_segments = rng.randint(1, 3)
    space = []
    for _ in range(n_segments):
        cands = []
        for _ in range(rng.randint(1, 3)):
            keys = rng.sample(node_pool, rng.randint(0, 2))
            cands.append(_candidate([_decision(k, rng.random() < 0.5) for k in keys]))
        space.append(cands)
    return _path_space(space)


def _reference_selections(path_space, rng):
    """One concrete path through the space, as the decisive attempt's selections."""
    return [{"segment_index": i,
             "branch_decisions": ps["candidates"][rng.randrange(len(ps["candidates"]))]
                                     .get("branch_decisions") or []}
            for i, ps in enumerate(path_space["per_segment"])]


# ── equivalence with the cross-product formulation ──────────────────────────

def test_matches_the_cross_product_formulation_on_random_small_spaces():
    rng = random.Random(20260921)
    node_pool = [1, 2, 3, 4]
    for _ in range(300):
        space = _random_space(rng, node_pool)
        ref_sel = _reference_selections(space, rng)
        writers = [k for k in node_pool if rng.random() < 0.5]
        new, old = _run(space, ref_sel, writers)
        assert new == old, (space, ref_sel, writers, new, old)


def test_matches_the_cross_product_formulation_with_ci_grouping():
    """Same equivalence with ``ci_dims`` present (a group is ONE dimension).

    Segments 0 and 1 are collapsed into a group forced to share a candidate, so
    segment 1's own candidate list is never enumerated — both formulations have to
    read the representative's, and the reference reaches that through the real
    ``iter_combinations``.
    """
    rng = random.Random(7)
    node_pool = [1, 2, 3, 4]
    for _ in range(200):
        space = _random_space(rng, node_pool)
        per = space["per_segment"]
        per.insert(1, {"candidates": list(per[0]["candidates"]),
                       "num_feasible_candidates": per[0]["num_feasible_candidates"]})
        space["ci_dims"] = [{"members": [0, 1], "count": len(per[0]["candidates"])}] + [
            {"members": [i], "count": ps["num_feasible_candidates"]}
            for i, ps in enumerate(per) if i != 0
        ]
        ref_sel = _reference_selections(space, rng)
        writers = [k for k in node_pool if rng.random() < 0.5]
        new, old = _run(space, ref_sel, writers)
        assert new == old, (space, ref_sel, writers, new, old)


def test_no_writer_anywhere_is_invariant():
    space = _path_space([[_candidate([_decision(1, True)]),
                          _candidate([_decision(2, False)])]])

    assert _run(space, [{"branch_decisions": [_decision(1, True)]}], writers=[])[0] is True


def test_writer_direction_differing_in_any_candidate_breaks_invariance():
    # the same writer branch true on the reference path and false in another candidate
    space = _path_space([[_candidate([_decision(1, True)]),
                          _candidate([_decision(1, False)])]])

    assert _run(space, [{"branch_decisions": [_decision(1, True)]}], writers=[1])[0] is False


def test_writer_absent_from_a_candidate_breaks_invariance():
    # present (true) on the reference path, absent from the other candidate: a combo
    # through that candidate does not take the branch at all → not invariant
    space = _path_space([[_candidate([_decision(1, True), _decision(3, False)]),
                          _candidate([_decision(3, False)])]])

    assert _run(space, [{"branch_decisions": [_decision(1, True)]}], writers=[1])[0] is False


def test_writer_absent_from_the_reference_path_breaks_invariance():
    # node 2 writes the pointer and is taken in a non-reference candidate, but the
    # reference path does not carry it at all → its value there is "absent", which
    # is not the same as "false" → conservatively not invariant (the old
    # formulation agrees: `ref[2] = None != True`).
    space = _path_space([[_candidate([_decision(1, False)]),
                          _candidate([_decision(1, False), _decision(2, True)])]])

    assert _run(space, [{"branch_decisions": [_decision(1, False)]}], writers=[1, 2])[0] is False


def test_writer_the_reference_carries_is_not_saved_by_being_false():
    # same shape as above with `False` directions: the reference path carries node
    # 2, candidate 0 does not → a combo through candidate 0 does not carry it at
    # all → "absent" is a different value from "false" (no false-negative escape).
    space = _path_space([[_candidate([_decision(1, False)]),
                          _candidate([_decision(1, False), _decision(2, False)])]])

    assert _run(space, [{"branch_decisions": [_decision(1, False), _decision(2, False)]}],
                writers=[1, 2])[0] is False


def test_writer_absent_from_a_dimension_that_never_carries_it_is_fine():
    # node 2 lives only in segment 1's candidates (every candidate carries it), so
    # segment 0's candidates not carrying it is irrelevant — a combo always gets its
    # value from segment 1.
    space = _path_space([[_candidate([_decision(1, False)])],
                         [_candidate([_decision(2, False)])]])

    assert _run(space, [{"branch_decisions": [_decision(1, False), _decision(2, False)]}],
                writers=[1, 2])[0] is True


def test_a_higher_dimension_that_always_carries_the_branch_masks_a_lower_one():
    # node 2 is carried by every candidate of both dimensions; the later dimension
    # always supplies it, so the earlier dimension's True/False never reaches a
    # combo (this is what "the highest dimension wins" buys over a plain per-dimension
    # scan, which would refuse here).
    space = _path_space([[_candidate([_decision(2, True)]), _candidate([_decision(2, False)])],
                         [_candidate([_decision(2, True)])]])

    assert _run(space, [{"branch_decisions": [_decision(2, True)]}], writers=[2])[0] is True


def test_a_partial_higher_dimension_unmasks_the_lower_one():
    # ...but when the later dimension has a candidate that does NOT carry the branch,
    # combos through it fall back to the earlier dimension → its False is seen.
    space = _path_space([[_candidate([_decision(2, True)]), _candidate([_decision(2, False)])],
                         [_candidate([_decision(2, True)]), _candidate([])]])

    assert _run(space, [{"branch_decisions": [_decision(2, True)]}], writers=[2])[0] is False


def test_every_dimension_partial_makes_the_branch_absent_somewhere():
    # every dimension has a candidate without the branch → one combo carries it
    # nowhere, i.e. the branch is absent there
    space = _path_space([[_candidate([_decision(2, True)]), _candidate([])]])

    assert _run(space, [{"branch_decisions": [_decision(2, True)]}], writers=[2])[0] is False


def test_candidates_beyond_num_feasible_are_outside_the_space():
    """Only the first ``num_feasible_candidates`` candidates are enumerable.

    ``iter_combinations`` walks ``range(count)``, so a difference parked in a
    candidate beyond that count is not a combo of the space and must not break the
    generalization — the reference formulation agrees.
    """
    space = {"per_segment": [
        {"candidates": [_candidate([_decision(1, True)]),
                        _candidate([_decision(1, False)])],
         "num_feasible_candidates": 1},
    ]}

    assert _run(space, [{"branch_decisions": [_decision(1, True)]}], writers=[1])[0] is True


# ── exhaustiveness past the old enumeration bound ───────────────────────────

def test_answer_is_not_limited_by_the_combination_enumeration_bound():
    """A differing writer in the LAST combin of a space too large to enumerate.

    60 × 60 × 60 = 216_000 combos exceeds ``combination_enum._ENUM_LIMIT``
    (200_000), so the old cross-product scan saw a prefix of the space — with the
    difference parked in the highest-index candidates it would have reported
    "invariant".  The per-dimension scan reads every candidate, so it sees it.
    """
    from llm_client.fp_analysis.combination_enum import _ENUM_LIMIT

    n = 60
    assert n ** 3 > _ENUM_LIMIT

    def dim_candidates(marked: bool):
        cands = [_candidate([_decision(9, True,)]).copy() for _ in range(n)]
        for c in cands:
            c["branch_decisions"] = [_decision(9, True)]
        if marked:
            # the one candidate that disagrees — reachable only through the last
            # combo of the space, i.e. exactly what a prefix scan cannot see
            cands[-1] = _candidate([_decision(9, False)])
        return cands

    space = _path_space([dim_candidates(False), dim_candidates(False), dim_candidates(True)])
    ref_sel = [{"branch_decisions": [_decision(9, True)]}]

    assert _run(space, ref_sel, writers=[9])[0] is False


def test_duplicate_decision_in_one_candidate_is_conservative():
    """A loop branch recorded twice inside one candidate keeps BOTH values.

    The reference formulation kept the last one; keeping the set is deliberately
    stricter — any value that does not match the reference blocks the
    generalization, which can only ever refuse to conclude FP.
    """
    space = _path_space([[_candidate([_decision(1, True), _decision(1, False)])]])
    ref_sel = [{"branch_decisions": [_decision(1, True)]}]

    assert _run(space, ref_sel, writers=[1])[0] is False


# ── the root cause must be a pointer, not a keyword ─────────────────────────

class _TrigNode:
    def __init__(self, expression, variable=""):
        self.source_line = 1
        self.is_call = False
        self.kind = "return"
        self.expression = expression
        self.variable = variable


class _TrigSdg:
    """An SDG whose trigger node is at the bug line of a fake report."""

    def __init__(self, node):
        self._node = node

    def get_pdg(self, fn):
        if fn != "f":
            return None
        class _P:
            nodes = {"S_1_1": self._node}
            function_name = "f"
        return _P()


class _Parsed:
    def __init__(self, descriptions):
        class _Meta:
            function_name = "f"
            bug_line = 1
        self.metadata = _Meta()
        self.error_start_event = None
        self.path_events = [{"description": d} for d in descriptions]


def test_base_identifier_rejects_keywords():
    from llm_client.fp_analysis.cause_invariance import _base_identifier

    assert _base_identifier("return idx;\n") is None
    assert _base_identifier("nullptr") is None
    assert _base_identifier("sizeof(x)") is None
    # a real receiver still comes through, including through `this->`
    assert _base_identifier("info->ai_addr") == "info"
    assert _base_identifier("this->info_->ai_addr") == "info_"
    assert _base_identifier("*results") == "results"


def test_leak_report_has_no_pointer_root_cause():
    """A leak report's trigger node is the `return`, not a dereference.

    `_base_identifier("return idx;")` used to yield the keyword ``return``, so the
    "root-cause pointer" was a word no branch writes — `cause_invariant` then reported
    invariant and would have generalized a decisive FP for a reason unrelated to the
    report (on a leak report the structural contradiction is about branch directions).
    """
    from llm_client.fp_analysis.cause_invariance import extract_deref_pointer

    sdg = _TrigSdg(_TrigNode("return idx;\n"))
    parsed = _Parsed(["Potential leak of memory pointed to by 'idx'"])

    assert extract_deref_pointer(parsed, sdg) is None


def test_deref_report_still_yields_its_pointer():
    from llm_client.fp_analysis.cause_invariance import extract_deref_pointer

    sdg = _TrigSdg(_TrigNode("info->ai_addr"))
    parsed = _Parsed(["dereference of a null pointer (loaded from variable 'info')"])

    assert extract_deref_pointer(parsed, sdg) == "info"
