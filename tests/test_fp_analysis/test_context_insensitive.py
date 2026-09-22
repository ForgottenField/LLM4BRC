"""Context-insensitive path-space grouping (``annotate_ci_groups``).

The user-facing feature: when the whole-path cross-product is inflated by the
same function's branches re-appearing across segments, collapse them into ONE
enumeration dimension (context-insensitive: their directions are bound together,
全路径一致).  Critically, ONLY *genuine repeated call sites* group — the same
callee invoked at different call sites (``callee_<caller>_<callee>_L<line>``
call-edge nodes).  A single call's OWN loop re-entry (``seg_`` / ``pdg_`` nodes,
context-sensitive) must NOT group: forcing loop iterations identical is unsound
and flips a genuine TP (e.g. GlogFormatter's infinite-loop bug) to FP.

These tests are pure / no LLM.
"""

from llm_client.fp_analysis.path_space import (
    annotate_ci_groups,
    ci_branch_set_of,
    _normalize_call_edge,
)
from llm_client.fp_analysis.combination_enum import (
    _ci_dims,
    build_selections_from_combo,
)


def _seg(branches, ncand=8, fn="fn"):
    bd = [{"node_id": b} for b in branches]
    return {
        "num_feasible_candidates": ncand,
        "candidates": [
            {"branch_decisions": bd, "signature": "L0:T"}
            for _ in range(ncand)
        ],
    }


def _ps(segs):
    """Wrap a list of per-segment branch/count pairs into a path_space dict."""
    return {
        "per_segment": [
            {**_seg(b, nc, fn), "segment_index": i}
            for i, (b, nc, fn) in enumerate(segs)
        ],
    }


class _Seg:
    def __init__(self, idx):
        self.segment_index = idx


# ── normalization ────────────────────────────────────────────────────────────

def test_normalize_call_edge_drops_line():
    assert _normalize_call_edge("callee_foo_to_ascii_L100_B3") == \
        "callee_foo_to_ascii_B3"
    assert _normalize_call_edge("callee_foo_to_ascii_L150_B3") == \
        "callee_foo_to_ascii_B3"
    assert _normalize_call_edge("call_foo_42") == "call_foo_42"


def test_normalize_call_edge_rejects_own_body():
    assert _normalize_call_edge("seg_insertThousandsGroupingUnsafe(...)_B5") is None
    assert _normalize_call_edge("pdg_foo_S_10") is None


# ── genuine multi-call groups (sound: should group) ─────────────────────────

def test_ci_branch_set_multi_call():
    a = {"num_feasible_candidates": 8, "candidates": [
        {"branch_decisions": [
            {"node_id": "callee_foo_to_ascii_L100_B3"},
            {"node_id": "callee_foo_to_ascii_L100_B7"},
        ]},
    ]}
    b = {"num_feasible_candidates": 8, "candidates": [
        {"branch_decisions": [
            {"node_id": "callee_foo_to_ascii_L200_B3"},
            {"node_id": "callee_foo_to_ascii_L200_B7"},
        ]},
    ]}
    # Same callee at different call lines → identical normalized set.
    assert ci_branch_set_of(a) == ci_branch_set_of(b) == frozenset(
        {"callee_foo_to_ascii_B3", "callee_foo_to_ascii_B7"})
    # Different callee → different set.
    c = {"num_feasible_candidates": 8, "candidates": [
        {"branch_decisions": [{"node_id": "callee_foo_to_utf8_L100_B3"}]},
    ]}
    assert ci_branch_set_of(c) != ci_branch_set_of(a)


def test_annotate_groups_multiple_call_sites():
    ps = _ps([
        (["callee_foo_to_ascii_L100_B3", "callee_foo_to_ascii_L100_B7"], 8, "foo"),
        (["callee_foo_to_ascii_L200_B3", "callee_foo_to_ascii_L200_B7"], 8, "foo"),
    ])
    out = annotate_ci_groups(ps)
    assert out["ci_groups"] == [[0, 1]]
    assert out["whole_path_bound"] == 8  # 8*8 → 8 (one collapsed dimension)


# ── loop re-entry (context-sensitive: must NOT group) ───────────────────────

def test_annotate_does_not_group_own_body_loop():
    # insertThousandsGroupingUnsafe is called ONCE; its own loop body (seg_)
    # re-enters across 4 path segments.  These are context-sensitive iterations.
    glog = _ps([
        (["seg_void insertThousandsGroupingUnsafe(...)_B5",
          "seg_void insertThousandsGroupingUnsafe(...)_B8",
          "seg_void insertThousandsGroupingUnsafe(...)_B10"], 8, "insert"),
    ] * 4)
    out = annotate_ci_groups(glog)
    assert out["ci_groups"] == []
    assert out["whole_path_bound"] == 8 ** 4  # stays 4096


def test_annotate_does_not_group_mixed_or_different_callee():
    mixed = _ps([
        (["callee_foo_to_ascii_L100_B3", "pdg_foo_S_10"], 8, "foo"),
        (["callee_foo_to_ascii_L200_B3", "pdg_foo_S_10"], 8, "foo"),
    ])
    assert annotate_ci_groups(mixed)["ci_groups"] == []  # own pdg_ → not a pure call site

    diff = _ps([
        (["callee_foo_to_ascii_L100_B3"], 8, "foo"),
        (["callee_foo_to_utf8_L100_B3"], 8, "foo"),
    ])
    assert annotate_ci_groups(diff)["ci_groups"] == []  # different callee


# ── combo enumeration forces grouped members identical ──────────────────────

def test_grouped_members_share_decisions():
    ps = annotate_ci_groups(_ps([
        (["callee_foo_to_ascii_L100_B3"], 2, "foo"),
        (["callee_foo_to_ascii_L200_B3"], 2, "foo"),
    ]))
    segs = [_Seg(0), _Seg(1)]
    dims = _ci_dims(ps)
    assert len(dims) == 1 and dims[0]["members"] == [0, 1] and dims[0]["count"] == 2

    # combo 0 and 1 each force BOTH members to the same branch_taken.
    for combo in (0, 1):
        sels = build_selections_from_combo(ps, (combo,), segs)
        by_seg = {s["segment_index"]: s for s in sels}
        b0 = by_seg[0]["branch_decisions"]
        b1 = by_seg[1]["branch_decisions"]
        assert b0 == b1  # grouped members bound together (全路径一致)


def test_no_group_enumeration_matches_original():
    # When nothing groups, whole_path_bound and dims are unchanged.
    ps = annotate_ci_groups(_ps([
        (["seg_insertThousandsGroupingUnsafe(...)_B5",
          "seg_insertThousandsGroupingUnsafe(...)_B8"], 8, "insert"),
    ] * 2))
    assert ps["whole_path_bound"] == 64
    assert len(_ci_dims(ps)) == 2  # each loop passage its own dimension
