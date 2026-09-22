"""Tests for the deterministic report self-contradiction check.

The module joins two independently-sourced facts — the *direction* the report
claims for a branch (its own event text) and the *condition text* of that
branch (the CFG/PDG branch tree) — and reports a contradiction when one
condition is asserted both ways on one path with no write in between.

Every test here builds the two inputs directly (a fake segment carrying a
branch tree, and a list of report events), so no report HTML, PDG or CFG cache
is needed.  The guards that exist to keep the check *sound* (and that therefore
must never be relaxed) each get a test that the check stays silent:

    same direction · write between · same line (loop) · recursion ·
    ambiguous line pairing · unreadable source · different function
"""

from __future__ import annotations

import pytest

from llm_client.fp_analysis.cfg_branch_models import CFGBranchNode
from llm_client.fp_analysis import report_consistency as rc

COND = 'h == fourcc("IHNs")'
FN = "faiss::read_index"


# ─── Builders ────────────────────────────────────────────────────────────────


def _node(line: int, condition: str, *, node_id: str | None = None,
          function_name: str = FN, children: list | None = None) -> CFGBranchNode:
    """A branch node keyed by source line; its id carries the PDG column."""
    return CFGBranchNode(
        node_id=node_id or f"S_{line}_13",
        source_line=line,
        function_name=function_name,
        condition_expr=condition,
        children=list(children or []),
    )


class _Seg:
    """Minimal stand-in for ``SegmentInfo`` (only two attrs are read)."""

    def __init__(self, source_file: str, function_file: str | None = None,
                 roots: list | None = None) -> None:
        self.function_file = function_file if function_file is not None else source_file
        self.cfg_branch_tree = _Tree(source_file, roots or [])


class _Tree:
    def __init__(self, source_file: str, roots: list) -> None:
        self.function_name = FN
        self.source_file = source_file
        self.root_branches = roots


def _event(number: int, line: int, description: str = "") -> dict:
    return {"path_number": number, "line_number": line, "description": description}


def _assume(number: int, line: int, direction: bool) -> dict:
    return _event(number, line,
                  f"← Assuming the condition is {str(direction).lower()} →")


@pytest.fixture(autouse=True)
def _clear_source_cache():
    """The source cache is module-global; keep tests independent of order."""
    rc._SOURCE_CACHE.clear()
    yield
    rc._SOURCE_CACHE.clear()


@pytest.fixture
def source(tmp_path):
    """A 10-line function whose only mention of ``h`` is a comparison.

    Line 3 and line 8 carry ``h == fourcc("IHNs")`` — no write to ``h``
    anywhere — which is the real 484-1 shape (``h`` is written once, far
    earlier, at L506).
    """
    text = "\n".join([
        "void read_index(Index*& idx, IOContext& io) {",   # 1
        "  uint32_t h = 0;",                               # 2
        '  if (h == fourcc("IHNs")) {',                    # 3
        "    idx = new IndexHNSWSQ();",                    # 4
        "  }",                                             # 5
        "  // ... dispatch ...",                           # 6
        "  IndexHNSW* idxhnsw = nullptr;",                 # 7
        '  if (h == fourcc("IHNs"))',                      # 8
        "    idxhnsw = new IndexHNSWSQ();",                # 9
        "}",                                               # 10
    ]) + "\n"
    path = tmp_path / "index_read.cpp"
    path.write_text(text, encoding="utf-8")
    return str(path)


# ─── The core case ───────────────────────────────────────────────────────────


def test_opposite_directions_on_one_path_is_a_contradiction(source):
    segments = [_Seg(source, roots=[_node(3, COND), _node(8, COND)])]
    events = [_assume(3, 3, False), _event(4, 4, "Taking false branch"),
              _assume(9, 8, True)]

    contradictions = rc.detect_report_contradictions(
        segments, events, source_root=None)

    assert len(contradictions) == 1
    c = contradictions[0]
    assert c.condition == COND
    assert (c.first.line, c.first.direction, c.first.event_number) == (3, False, 3)
    assert (c.second.line, c.second.direction, c.second.event_number) == (8, True, 9)
    assert "h" in c.variables
    reason = c.reason()
    assert "自相矛盾" in reason and "L3" in reason and "L8" in reason
    # The callee name is not a plausible "what changed" answer, so the
    # human-readable reason reports only `h` while the guard watches both.
    assert c.variables == ["h", "fourcc"]
    assert rc._display_vars(c.condition, c.variables) == ["h"]
    assert "h 在此期间无写入" in reason


def test_claims_pair_report_steps_with_condition_text(source):
    segments = [_Seg(source, roots=[_node(3, COND), _node(8, COND)])]
    events = [_assume(3, 3, False), _assume(9, 8, True)]

    claims = rc.condition_claims(segments, events)

    assert {(c.line, c.direction, c.event_number) for c in claims} == {
        (3, False, 3), (8, True, 9)}
    assert all(c.condition == COND and c.function_name == FN for c in claims)


# ─── Soundness guards: each must keep the check silent ───────────────────────


def test_same_direction_twice_is_not_a_contradiction(source):
    segments = [_Seg(source, roots=[_node(3, COND), _node(8, COND)])]
    events = [_assume(3, 3, True), _assume(9, 8, True)]

    assert rc.detect_report_contradictions(segments, events) == []


def test_write_between_the_claims_is_not_a_contradiction(tmp_path):
    """A write to the condition's variable is exactly the legal explanation."""
    text = "\n".join([
        '  if (h == fourcc("IHNs")) {',   # 1  FALSE
        "    return nullptr;",            # 2
        "  }",                            # 3
        "  h = fourcc(\"IHNs\");",        # 4  ← the write
        '  if (h == fourcc("IHNs"))',     # 5  TRUE
        "    idx = new IndexHNSWSQ();",   # 6
    ]) + "\n"
    path = tmp_path / "index_read.cpp"
    path.write_text(text, encoding="utf-8")

    segments = [_Seg(str(path), roots=[_node(1, COND), _node(5, COND)])]
    events = [_assume(3, 1, False), _assume(9, 5, True)]

    assert rc.detect_report_contradictions(segments, events) == []


def test_same_line_twice_is_a_loop_not_a_contradiction(source):
    """A loop re-entry reads the same line in a fresh iteration."""
    segments = [_Seg(source, roots=[_node(3, COND)])]
    events = [_assume(2, 3, False), _assume(7, 3, True)]

    assert rc.detect_report_contradictions(segments, events) == []


def test_recursive_reentry_between_the_claims_is_not_a_contradiction(source):
    """A fresh frame legitimately holds a different value for the same local."""
    segments = [_Seg(source, roots=[_node(3, COND), _node(8, COND)])]
    events = [_assume(3, 3, False),
              _event(5, 4, "Calling 'read_index'"),
              _assume(9, 8, True)]

    assert rc.detect_report_contradictions(segments, events) == []


def test_any_intervening_call_is_not_a_contradiction(source):
    """An unmodelled call re-creates globals as fresh symbols.

    The two ``Assuming …`` steps would then be about two different symbols, so
    *any* call step between them vetoes the pair (not just a recursive one).
    484-1 has no call step between its two assumptions.
    """
    segments = [_Seg(source, roots=[_node(3, COND), _node(8, COND)])]
    events = [_assume(3, 3, False),
              _event(5, 4, "Calling 'maybe_reinit'"),
              _assume(9, 8, True)]

    assert rc.detect_report_contradictions(segments, events) == []


def test_reported_store_between_the_claims_is_not_a_contradiction(source):
    """The report's own ``stored to`` step outranks a clean source scan."""
    segments = [_Seg(source, roots=[_node(3, COND), _node(8, COND)])]
    events = [_assume(3, 3, False),
              _event(5, 4, "Value stored to 'h' is never read"),
              _assume(9, 8, True)]

    assert rc.detect_report_contradictions(segments, events) == []


def test_different_functions_are_never_paired(source):
    other = "faiss::read_index_header"
    segments = [_Seg(source, roots=[_node(3, COND),
                                    _node(8, COND, function_name=other)])]
    events = [_assume(3, 3, False), _assume(9, 8, True)]

    assert rc.detect_report_contradictions(segments, events) == []


def test_unreadable_source_yields_no_claim(tmp_path):
    """No source ⇒ no knowledge of a write ⇒ no contradiction."""
    segments = [_Seg(str(tmp_path / "missing.cpp"),
                     roots=[_node(3, COND), _node(8, COND)])]
    events = [_assume(3, 3, False), _assume(9, 8, True)]

    assert rc.detect_report_contradictions(segments, events) == []


def test_ambiguous_line_pairing_is_skipped(source):
    """Two conditions on one line but one report step: order would be a guess."""
    segments = [_Seg(source, roots=[
        _node(3, 'h == fourcc("IHNf")', node_id="S_3_13"),
        _node(3, 'h == fourcc("IHNs")', node_id="S_3_40"),
    ])]
    events = [_assume(3, 3, False)]

    assert rc.condition_claims(segments, events) == []


def test_several_conditions_on_one_line_pair_positionally(source):
    """``a || b``: one PDG node per operand, paired by (event, column) order."""
    segments = [_Seg(source, roots=[
        _node(3, 'h == fourcc("IHNf")', node_id="S_3_13"),
        _node(3, 'h == fourcc("IHNs")', node_id="S_3_40"),
    ])]
    events = [_assume(2, 3, False), _assume(3, 3, True)]

    claims = sorted(rc.condition_claims(segments, events), key=lambda c: c.event_number)

    assert [(c.condition, c.direction) for c in claims] == [
        ('h == fourcc("IHNf")', False), ('h == fourcc("IHNs")', True)]


# ─── Non-conditions must never become claims ─────────────────────────────────


def test_callee_sentinel_and_branch_fallback_are_not_conditions(source):
    segments = [_Seg(source, roots=[
        _node(3, "→ read_ivf_header()"),
        _node(8, "branch=True"),
    ])]
    events = [_assume(3, 3, False), _assume(9, 8, True)]

    assert rc.condition_claims(segments, events) == []


def test_condition_variables_drop_string_contents_and_keywords():
    variables = rc._condition_variables('if (n > 0 && sizeof(T) == 4)')

    assert variables == ["n", "T"]


def test_write_detector_ignores_comparisons_but_sees_stores():
    assert rc._writes_var('  if (h == fourcc("IHNs")) {', "h") is False
    assert rc._writes_var("  h = fourcc(\"IHNs\");", "h") is True
    assert rc._writes_var("  h++;", "h") is True
    assert rc._writes_var("  read(&h);", "h") is True
    assert rc._writes_var("  h[0] = 1;", "h") is True
    assert rc._writes_var("  some_other = h;", "h") is False
    assert rc._writes_var("  tags[h] = 1;", "h") is False  # h is only read
