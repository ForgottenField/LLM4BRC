"""Tests for CFGBranchExtractor — segment-level CFG branch tree construction."""

from __future__ import annotations

import pytest
from llm_client.fp_analysis.cfg_branch_extractor import CFGBranchExtractor
from llm_client.fp_analysis.cfg_branch_models import (
    CFGBranchTree,
    Postcondition,
)
from llm_client.fp_analysis.cfg_models import CFGBlock, CFGFunction
from llm_client.fp_analysis.pdg_models import ControlDepEdge, FunctionPDG, PDGNode
from llm_client.fp_analysis.slice_mask import SliceMask


# ─── Fixtures: mock CFG data ─────────────────────────────────────────────────


def _make_cfg_block(
    block_id: str,
    statements: list[str],
    terminator: str | None = None,
    succs: list[str] | None = None,
    label: str = "",
) -> CFGBlock:
    return CFGBlock(
        block_id=block_id,
        statements=statements,
        terminator=terminator,
        successors=succs or [],
        label=label,
    )


@pytest.fixture
def mock_cfg_wslay_event_recv() -> CFGFunction:
    """Mock CFG for wslay_event_recv (segment0: L566-L573)."""
    return CFGFunction(
        function_name="wslay_event_recv",
        blocks={
            "B1": _make_cfg_block(
                "B1",
                ["  566: while(ctx->read_enabled)"],
                terminator="while [B1.1] (ImplicitCastExpr, IntegralToBoolean, int)",
                succs=["B2", "B13"],
            ),
            "B2": _make_cfg_block(
                "B2",
                [
                    "  571: memset(&iocb, 0, sizeof(iocb))",
                    "  572: r = wslay_frame_recv(ctx->frame_ctx, &iocb)",
                ],
                succs=["B3"],
            ),
            "B3": _make_cfg_block(
                "B3",
                [],
                terminator="if [B3.1] (BinaryOperator) r >= 0",
                succs=["B4", "B12"],
            ),
            "B4": _make_cfg_block("B4", [], succs=["B5"]),
            "B12": _make_cfg_block("B12", [], succs=["B13"]),
            "B13": _make_cfg_block("B13", [""], label="EXIT"),
        },
        entry_block_id="B1",
        exit_block_id="B13",
    )


@pytest.fixture
def mock_cfg_wslay_frame_recv() -> CFGFunction:
    """Mock CFG for wslay_frame_recv (callee, depth=1)."""
    return CFGFunction(
        function_name="wslay_frame_recv",
        blocks={
            "wf_B1": _make_cfg_block(
                "wf_B1",
                [],
                terminator="if [wf_B1.1] (ImplicitCastExpr, PointerToBoolean, int)",
                succs=["wf_B2", "wf_B3"],
            ),
            "wf_B2": _make_cfg_block(
                "wf_B2",
                ["[wf_B2.1] (ReturnStmt) return WSLAY_ERR_CALLBACK_FAILURE"],
                succs=["wf_EXIT"],
            ),
            "wf_B3": _make_cfg_block(
                "wf_B3",
                ["[wf_B3.1] (BinaryOperator) r = cb(...)"],
                terminator="if [wf_B3.2] (BinaryOperator) r < 0",
                succs=["wf_B4", "wf_B5"],
            ),
            "wf_B4": _make_cfg_block(
                "wf_B4",
                ["[wf_B4.1] (ReturnStmt) return WSLAY_ERR_CALLBACK_FAILURE"],
                succs=["wf_EXIT"],
            ),
            "wf_B5": _make_cfg_block(
                "wf_B5",
                [],
                terminator="if [wf_B5.1] (BinaryOperator) r == 0",
                succs=["wf_B6", "wf_B7"],
            ),
            "wf_B6": _make_cfg_block(
                "wf_B6",
                ["[wf_B6.1] (ReturnStmt) return WSLAY_ERR_WANT_READ"],
                succs=["wf_EXIT"],
            ),
            "wf_B7": _make_cfg_block(
                "wf_B7",
                ["[wf_B7.1] (ReturnStmt) return nread"],
                succs=["wf_EXIT"],
            ),
            "wf_EXIT": _make_cfg_block("wf_EXIT", [""], label="EXIT"),
        },
        entry_block_id="wf_B1",
        exit_block_id="wf_EXIT",
    )


@pytest.fixture
def mock_cfg_cache(
    mock_cfg_wslay_event_recv: CFGFunction,
    mock_cfg_wslay_frame_recv: CFGFunction,
) -> dict[str, CFGFunction]:
    return {
        "wslay_event_recv": mock_cfg_wslay_event_recv,
        "wslay_frame_recv": mock_cfg_wslay_frame_recv,
    }


@pytest.fixture
def mock_slice_mask() -> SliceMask:
    """SliceMask with all wslay_event_recv blocks retained."""
    return SliceMask(
        pdg_retained_nodes={"wslay_event_recv": {"S_570", "S_571", "S_572", "S_573", "S_574"}},
        cfg_retained_blocks={"wslay_event_recv": {1, 2, 3, 4, 12, 13}},
    )


class MockSDG:
    """Minimal SDG stub for testing.

    ``pdgs`` is the same attribute the real :class:`SDG` exposes, and the three
    accessors the extractor uses (``get_pdg``, ``_resolve_pdg``,
    ``_get_pdg_call_index``) all read *that* dict — so a stub with no PDG data
    takes the CFG fallback, and one with PDG data takes the PDG path.
    """

    def __init__(self):
        self.pdgs: dict[str, FunctionPDG] = {}

    def get_pdg(self, fn_name: str):
        return self.pdgs.get(fn_name)


@pytest.fixture
def mock_sdg() -> MockSDG:
    return MockSDG()


@pytest.fixture
def segment0_postcondition() -> Postcondition:
    return Postcondition(
        function_name="wslay_event_recv",
        source_line=573,
        condition_expr="r >= 0",
        branch_taken=True,
    )


@pytest.fixture
def segment0_info() -> object:
    """Minimal SegmentInfo-like object for extraction testing."""
    from types import SimpleNamespace
    return SimpleNamespace(
        segment_index=0,
        function_name="wslay_event_recv",
        line_start=566,
        line_end=573,
        # Simulate that r >= 0 postcondition variable appears in code
        code_from_entry="",
    )


# ─── Tests ───────────────────────────────────────────────────────────────────


class TestExtractorEmptyCfg:
    """When CFG cache is empty, extractor returns an empty tree."""

    def test_empty_cache_returns_empty_tree(self, segment0_postcondition):
        extractor = CFGBranchExtractor({}, MockSDG(), SliceMask())
        tree = extractor.extract(
            segment_index=0,
            function_name="wslay_event_recv",
            line_start=566,
            line_end=573,
            postcondition=segment0_postcondition,
        )
        assert isinstance(tree, CFGBranchTree)
        assert tree.num_branches == 0


class TestExtractorBasicIf:
    """Basic if statement extraction within one function."""

    def test_extract_simple_if(
        self, mock_cfg_cache, mock_slice_mask, mock_sdg,
        segment0_postcondition,
    ):
        extractor = CFGBranchExtractor(mock_cfg_cache, mock_sdg, mock_slice_mask)
        tree = extractor.extract(
            segment_index=0,
            function_name="wslay_event_recv",
            line_start=566,
            line_end=573,
            postcondition=segment0_postcondition,
        )
        # B1 (while) + B3 (if) + call-node from B2 (callee extraction)
        assert tree.num_branches >= 2
        assert tree.segment_index == 0

    def test_extracted_branches_have_conditions(
        self, mock_cfg_cache, mock_slice_mask, mock_sdg,
        segment0_postcondition,
    ):
        extractor = CFGBranchExtractor(mock_cfg_cache, mock_sdg, mock_slice_mask)
        tree = extractor.extract(
            segment_index=0,
            function_name="wslay_event_recv",
            line_start=566,
            line_end=573,
            postcondition=segment0_postcondition,
        )
        conditions = [n.condition_expr for n in tree.root_branches]
        # Should have at least an "if" condition expression
        assert any(c for c in conditions)


class TestExtractorCalleeExtraction:
    """Cross-procedural expansion on the PDG path (the live mechanism).

    Callee discovery and the callee's own predicates both come from the SDG:
    the segment's PDG must carry an ``is_call`` node (that is what
    ``_find_calls_in_pdg_range`` reads) and the callee's PDG a
    control-dependence source for its predicates.  How far into the callee the
    walk may descend — slice-retention (B) and the recursion/env guards (C) —
    is covered by ``test_path_reachability.TestSliceGating``.
    """

    def test_pdg_path_expands_the_callee_at_depth_one(self):
        callee = FunctionPDG(
            function_name="deep", source_file="/p/deep.cpp",
            nodes={"D_5": PDGNode(id="D_5", kind="binary_op", expression="k > 0",
                                  source_line=5, block_id=1)},
            control_dep_edges=[ControlDepEdge(source_id="D_5", target_id="D_5")],
        )
        caller = FunctionPDG(
            function_name="probe", source_file="/p/probe.cpp",
            nodes={"C_10": PDGNode(id="C_10", kind="call_expr", expression="deep()",
                                   source_line=10, block_id=1, is_call=True,
                                   callee="deep")},
            control_dep_edges=[ControlDepEdge(source_id="C_10", target_id="C_10")],
        )
        sdg = MockSDG()
        sdg.pdgs = {"probe": caller, "deep": callee}
        extractor = CFGBranchExtractor({}, sdg, SliceMask())
        tree = extractor.extract(
            segment_index=0,
            function_name="probe",
            line_start=10,
            line_end=12,
            postcondition=Postcondition("probe", 12, "k > 0", True),
        )

        sentinels = [n for n in tree.root_branches if n.callee_name == "deep"]
        assert len(sentinels) == 1, (
            "the call site in the segment's PDG must yield a callee sentinel"
        )
        assert sentinels[0].condition_expr == "→ deep()"
        assert [c.call_depth for c in sentinels[0].children] == [1], (
            "the callee's own predicates are extracted one level deeper"
        )

    def test_max_depth_limit(
        self, mock_cfg_cache, mock_slice_mask, mock_sdg,
        segment0_postcondition,
    ):
        """Extractor respects max_depth=3 (default)."""
        # This is a structural check: extractor should not crash at any depth
        extractor = CFGBranchExtractor(
            mock_cfg_cache, mock_sdg, mock_slice_mask, max_depth=3,
        )
        tree = extractor.extract(
            segment_index=0,
            function_name="wslay_event_recv",
            line_start=566,
            line_end=573,
            postcondition=segment0_postcondition,
        )
        # Just verify it runs without error at depth limit
        assert tree is not None


class TestLineRangeFiltering:
    """Blocks outside the segment's line range should be excluded."""

    def test_range_outside_segment_excluded(
        self, mock_cfg_cache, mock_slice_mask, mock_sdg,
        segment0_postcondition,
    ):
        """A very narrow line range should exclude most branches."""
        extractor = CFGBranchExtractor(mock_cfg_cache, mock_sdg, mock_slice_mask)
        tree = extractor.extract(
            segment_index=0,
            function_name="wslay_event_recv",
            line_start=999,  # far beyond function range
            line_end=1000,
            postcondition=segment0_postcondition,
        )
        assert tree.num_branches == 0


class TestElementReferenceResolution:
    """Element reference resolution for Clang-native CFG format."""

    @pytest.fixture
    def clang_native_cfg(self) -> CFGFunction:
        """Clang-native CFG dump with element references (no type annotations)."""
        return CFGFunction(
            function_name="wslay_event_recv",
            blocks={
                "B6": _make_cfg_block(
                    "B6",
                    [
                        "[B6.1] (ImplicitCastExpr) ctx->read_enabled",
                        "[B6.2] (ImplicitCastExpr) ctx->imsg",
                        "[B6.11] (IntegerLiteral) 1",
                        "[B6.13] (IntegerLiteral) 0",
                        "[B6.14] [B6.2] == [B6.13]",
                        "[B6.15] [B6.14] != [B6.11]",
                    ],
                    terminator="if [B6.15]",
                    succs=["B7", "B14"],
                ),
            },
            entry_block_id="B6",
        )

    @pytest.fixture
    def clang_cache(self, clang_native_cfg) -> dict:
        return {"wslay_event_recv": clang_native_cfg}

    def test_resolve_direct_variable(self, clang_cache):
        """[B6.1] resolves to ctx->read_enabled (strip type annotation)."""
        extractor = CFGBranchExtractor(clang_cache, MockSDG(), SliceMask())
        cfg = clang_cache["wslay_event_recv"]
        result = extractor._resolve_element_reference("[B6.1]", cfg)
        assert result == "ctx->read_enabled"

    def test_resolve_constant(self, clang_cache):
        """[B6.11] resolves to 1."""
        extractor = CFGBranchExtractor(clang_cache, MockSDG(), SliceMask())
        cfg = clang_cache["wslay_event_recv"]
        result = extractor._resolve_element_reference("[B6.11]", cfg)
        assert result == "1"

    def test_resolve_comparison_one_level(self, clang_cache):
        """[B6.14] resolves to ctx->imsg == 0 (one-level recursion)."""
        extractor = CFGBranchExtractor(clang_cache, MockSDG(), SliceMask())
        cfg = clang_cache["wslay_event_recv"]
        result = extractor._resolve_element_reference("[B6.14]", cfg)
        assert result is not None
        assert "ctx->imsg" in result
        assert "0" in result or "==" in result

    def test_resolve_compound_comparison(self, clang_cache):
        """[B6.15] resolves to (ctx->imsg == 0) != 1 (two-level recursion)."""
        extractor = CFGBranchExtractor(clang_cache, MockSDG(), SliceMask())
        cfg = clang_cache["wslay_event_recv"]
        result = extractor._resolve_element_reference("[B6.15]", cfg)
        assert result is not None
        assert "ctx->imsg" in result

    def test_extract_from_block_sets_resolved_expr(self, clang_cache):
        """Branch node gets resolved_expr when condition is element reference.

        The _extract_from_block should detect element-reference conditions
        and populate both condition_expr (raw) and resolved_expr (human-readable).
        """
        extractor = CFGBranchExtractor(clang_cache, MockSDG(), SliceMask())
        cfg = clang_cache["wslay_event_recv"]
        # B6 is a branch block (terminator="if [B6.15]")
        node = extractor._extract_from_block(
            cfg=cfg, block_id="B6", fn_name="wslay_event_recv",
            depth=0, parent_call=None,
        )
        assert node is not None
        # condition_expr should still be element-reference form
        # resolved_expr should be human-readable
        assert node.resolved_expr != "", "resolved_expr should not be empty"
        assert "ctx->imsg" in node.resolved_expr or "ctx->read_enabled" in node.resolved_expr


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Cross-block element reference resolution
# ═══════════════════════════════════════════════════════════════════════


class TestCrossBlockResolution:
    """Cross-block element reference resolution (short-circuit eval).

    Clang CFG distributes compound conditions across blocks::

        B5: [B5.1] a
            T: if [B5.1] || ([B8.6] && [B7.6])
        B8: [B8.6] b
        B7: [B7.6] c

    The old code only searched the CURRENT block (B5) for ``[B8.6]``
    and failed.  The new code builds a function-wide index.
    """

    @pytest.fixture
    def cross_block_cfg(self) -> CFGFunction:
        """CFG with short-circuit evaluation producing cross-block refs."""
        return CFGFunction(
            function_name="wslay_event_recv",
            blocks={
                "B5": _make_cfg_block(
                    "B5",
                    ["[B5.1] (ImplicitCastExpr) ctx->read_enabled"],
                    terminator="if [B5.1] || ([B8.6] && [B7.6])",
                    succs=["B9", "B13"],
                ),
                "B7": _make_cfg_block(
                    "B7",
                    ["[B7.4] (ImplicitCastExpr) ctx->error",
                     "[B7.5] (IntegerLiteral) 0",
                     "[B7.6] (BinaryOperator) [B7.4] != [B7.5]"],
                    succs=["B9", "B14"],
                ),
                "B8": _make_cfg_block(
                    "B8",
                    ["[B8.6] (ImplicitCastExpr) ctx->imsg"],
                    terminator="if [B8.6] && [B7.6]",
                    succs=["B7", "B14"],
                ),
                "B9": _make_cfg_block("B9", [], succs=["B13"]),
                "B13": _make_cfg_block("B13", [""], label="EXIT"),
                "B14": _make_cfg_block("B14", [""], label="EXIT"),
            },
            entry_block_id="B5",
        )

    @pytest.fixture
    def cc_cache(self, cross_block_cfg) -> dict:
        return {"wslay_event_recv": cross_block_cfg}

    def test_global_index_built(self, cc_cache):
        """_build_cfg_index creates index covering all blocks."""
        extractor = CFGBranchExtractor(cc_cache, MockSDG(), SliceMask())
        cfg = cc_cache["wslay_event_recv"]
        extractor._build_cfg_index(cfg)
        assert hasattr(extractor, "_cfg_index")
        # Should have entries for B5, B7, B8
        assert "B5" in extractor._cfg_index
        assert "B7" in extractor._cfg_index
        assert "B8" in extractor._cfg_index

    def test_resolve_same_block_ref(self, cc_cache):
        """[B5.1] resolved from B5's own statements."""
        extractor = CFGBranchExtractor(cc_cache, MockSDG(), SliceMask())
        cfg = cc_cache["wslay_event_recv"]
        extractor._build_cfg_index(cfg)
        result = extractor._resolve_single_element("B5", 1, cfg, 0)
        assert result is not None
        assert "ctx->read_enabled" in result

    def test_resolve_cross_block_ref(self, cc_cache):
        """[B8.6] in B5's terminator resolves from B8's statements."""
        extractor = CFGBranchExtractor(cc_cache, MockSDG(), SliceMask())
        cfg = cc_cache["wslay_event_recv"]
        extractor._build_cfg_index(cfg)
        result = extractor._resolve_single_element("B8", 6, cfg, 0)
        assert result is not None
        assert "ctx->imsg" in result

    def test_resolve_compound_cross_block(self, cc_cache):
        """[B5.1] || ([B8.6] && [B7.6]) fully resolves."""
        extractor = CFGBranchExtractor(cc_cache, MockSDG(), SliceMask())
        cfg = cc_cache["wslay_event_recv"]
        result = extractor._resolve_element_reference(
            "[B5.1] || ([B8.6] && [B7.6])", cfg,
        )
        assert result is not None
        assert "ctx->read_enabled" in result
        assert "ctx->imsg" in result
        assert "ctx->error" in result or "!=" in result

    def test_resolve_b7_term_cross_ref(self, cc_cache):
        """Terminator in B8 references B7.6 — resolves cross-block."""
        extractor = CFGBranchExtractor(cc_cache, MockSDG(), SliceMask())
        cfg = cc_cache["wslay_event_recv"]
        result = extractor._resolve_element_reference("[B8.6] && [B7.6]", cfg)
        assert result is not None
        assert "ctx->imsg" in result
        assert "ctx->error" in result or "!=" in result

    def test_not_found_returns_original(self, cc_cache):
        """[B99.99] nonexistent → returns original text."""
        extractor = CFGBranchExtractor(cc_cache, MockSDG(), SliceMask())
        cfg = cc_cache["wslay_event_recv"]
        result = extractor._resolve_element_reference("[B99.99]", cfg)
        assert result == "[B99.99]"

    def test_branch_extraction_with_cross_block(self, cc_cache):
        """_extract_from_block on B5 populates resolved_expr for cross-block ref."""
        extractor = CFGBranchExtractor(cc_cache, MockSDG(), SliceMask())
        cfg = cc_cache["wslay_event_recv"]
        node = extractor._extract_from_block(
            cfg=cfg, block_id="B5", fn_name="wslay_event_recv",
            depth=0, parent_call=None,
        )
        assert node is not None
        assert node.resolved_expr != "", f"resolved_expr should not be empty, got '{node.resolved_expr}'"
        assert "ctx->read_enabled" in node.resolved_expr


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: CallGraph-enhanced callee name resolution
# ═══════════════════════════════════════════════════════════════════════


class TestCallGraphResolution:
    """CallGraph-based callee resolution for cross-procedural extraction.

    Tests ``_resolve_cfg_by_short_name`` with CG data.  When CG is
    available, short names should be resolved to qualified names via
    ``get_callees(caller)`` or ``find_nodes_matching()``.
    """

    @pytest.fixture
    def cg_with_calls(self) -> object:
        """CallGraph with cross-file calls for disambiguation testing."""
        from llm_client.csa_analysis.callgraph_models import CallGraph
        return CallGraph.from_dict({
            "functions": {
                "wslay_event_queue_msg_ex": {
                    "file": "wslay/wslay_event.c", "line": 338,
                },
                "wslay_queue_push": {
                    "file": "wslay/wslay_event.c", "line": 120,
                },
                "other::wslay_queue_push": {
                    "file": "wslay/wslay_other.c", "line": 50,
                },
                "wslay_frame_recv": {
                    "file": "wslay/wslay_frame.c", "line": 200,
                },
            },
            "calls": [
                {"caller": "wslay_event_queue_msg_ex",
                 "callee": "wslay_queue_push",
                 "file": "wslay/wslay_event.c", "line": 355},
                {"caller": "wslay_event_queue_msg_ex",
                 "callee": "wslay_frame_recv",
                 "file": "wslay/wslay_event.c", "line": 350},
            ],
        })

    @pytest.fixture
    def sdg_with_cg(self, cg_with_calls) -> object:
        """SDG stub with callgraph attribute."""
        sdg = MockSDG()
        sdg.callgraph = cg_with_calls
        return sdg

    @pytest.fixture
    def cfg_cache_two_callees(self) -> dict:
        """CFG cache with two wslay_queue_push entries (disambiguation target)."""
        from llm_client.fp_analysis.cfg_models import CFGFunction, CFGBlock
        _empty: dict[str, CFGBlock] = {}
        return {
            "wslay_event_queue_msg_ex": CFGFunction(
                function_name="wslay_event_queue_msg_ex",
                source_file="wslay/wslay_event.c",
                blocks=_empty,
            ),
            "wslay_queue_push": CFGFunction(
                function_name="wslay_queue_push",
                source_file="wslay/wslay_event.c",
                blocks=_empty,
            ),
            "other::wslay_queue_push": CFGFunction(
                function_name="other::wslay_queue_push",
                source_file="wslay/wslay_other.c",
                blocks=_empty,
            ),
            "wslay_frame_recv": CFGFunction(
                function_name="wslay_frame_recv",
                source_file="wslay/wslay_frame.c",
                blocks=_empty,
            ),
        }

    # ── Resolve via CG ─────────────────────────────────────────────────

    def test_resolve_via_callgraph_matches_callee(self, sdg_with_cg):
        """get_callees(caller) finds the right qualified callee."""
        ext = CFGBranchExtractor({}, sdg_with_cg, SliceMask())
        result = ext._resolve_via_callgraph(
            "wslay_queue_push",
            caller_fn="wslay_event_queue_msg_ex",
        )
        assert result == "wslay_queue_push"
        # Should NOT return the other::wslay_queue_push since it's not
        # in the caller's callee list.

    def test_resolve_via_callgraph_fallback_to_fuzzy(self, sdg_with_cg):
        """Without caller, falls back to fuzzy match."""
        ext = CFGBranchExtractor({}, sdg_with_cg, SliceMask())
        result = ext._resolve_via_callgraph("wslay_queue_push")
        # Fuzzy match may return either; should just not be None
        assert result is not None
        assert "wslay_queue_push" in result

    def test_resolve_via_callgraph_no_cg(self):
        """No CG → returns None."""
        sdg = MockSDG()
        sdg.callgraph = None
        ext = CFGBranchExtractor({}, sdg, SliceMask())
        result = ext._resolve_via_callgraph("wslay_queue_push")
        assert result is None

    # ── Integration with _resolve_cfg_by_short_name ────────────────────

    def test_cfg_resolved_via_cg(self, cfg_cache_two_callees, sdg_with_cg):
        """_resolve_cfg_by_short_name uses CG to find correct callee CFG."""
        ext = CFGBranchExtractor(cfg_cache_two_callees, sdg_with_cg, SliceMask())
        cfg = ext._resolve_cfg_by_short_name(
            "wslay_queue_push",
            caller_fn="wslay_event_queue_msg_ex",
        )
        assert cfg is not None
        assert cfg.function_name == "wslay_queue_push"
        assert cfg.source_file == "wslay/wslay_event.c"

    def test_cfg_resolved_fallback_no_cg(self, cfg_cache_two_callees):
        """No CG → falls back to text matching (existing behavior)."""
        sdg = MockSDG()
        sdg.callgraph = None
        ext = CFGBranchExtractor(cfg_cache_two_callees, sdg, SliceMask())
        cfg = ext._resolve_cfg_by_short_name("wslay_queue_push")
        assert cfg is not None  # should find via text match

    def test_cfg_not_in_cache(self, sdg_with_cg):
        """Callee has CG entry but no CFG → returns None."""
        ext = CFGBranchExtractor({}, sdg_with_cg, SliceMask())
        cfg = ext._resolve_cfg_by_short_name(
            "wslay_frame_recv",
            caller_fn="wslay_event_queue_msg_ex",
        )
        assert cfg is None

    # ── End-to-end: extraction triggers CG resolution ──────────────────
