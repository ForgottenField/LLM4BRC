"""Tests for CFGBranchExtractor — segment-level CFG branch tree construction."""

from __future__ import annotations

import pytest
from llm_client.fp_analysis.cfg_branch_extractor import CFGBranchExtractor
from llm_client.fp_analysis.cfg_branch_models import (
    BranchTag,
    CFGBranchNode,
    CFGBranchTree,
    Postcondition,
)
from llm_client.fp_analysis.cfg_models import CFGBlock, CFGFunction
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
    """Minimal SDG stub for testing."""

    def __init__(self):
        self._pdgs: dict[str, object] = {}

    def get_pdg(self, fn_name: str):
        return self._pdgs.get(fn_name)


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


class TestExtractorFullExpansion:
    """⛔ RED: Full expansion — all callees extracted regardless of return var."""

    def test_callee_extracted_when_return_unrelated(
        self, mock_cfg_cache, mock_slice_mask, mock_sdg,
    ):
        """Callee IS extracted even when return var doesn't match postcondition.

        Uses line range covering B2 (L571-L572) which contains the
        wslay_frame_recv call. Full expansion ensures the callee is
        extracted regardless of postcondition variable mismatch: the
        postcondition is 'msg_length > 0' but the return var is 'r'.
        """
        unrelated_pc = Postcondition(
            function_name="wslay_event_recv",
            source_line=575,
            condition_expr="msg_length > 0",
            branch_taken=True,
        )
        extractor = CFGBranchExtractor(mock_cfg_cache, mock_sdg, mock_slice_mask)
        tree = extractor.extract(
            segment_index=2,
            function_name="wslay_event_recv",
            line_start=570,  # Covers B2 (L571-L572) where wslay_frame_recv is called
            line_end=575,
            postcondition=unrelated_pc,
        )
        # Full expansion: callee should be extracted even though 'r' != 'msg_length'
        callee_nodes = _count_by_depth(tree, min_depth=1)
        assert callee_nodes > 0, (
            "Full expansion: callee should be extracted even when return var "
            "doesn't match postcondition variable"
        )

    def test_callee_extracted_void_function(self, mock_cfg_cache, mock_slice_mask, mock_sdg):
        """Void function calls (no return var) are also expanded."""
        unrelated_pc = Postcondition(
            function_name="wslay_event_recv",
            source_line=600,
            condition_expr="msg_length > 0",
            branch_taken=True,
        )
        # Create a mock CFG that includes a void function call block
        mock_cfg_cache["wslay_event_recv"].blocks["B10"] = _make_cfg_block(
            "B10",
            ["  580: log_debug(ctx, \"processing msg\")"],
            succs=["B11"],
        )
        mock_cfg_cache["wslay_event_recv"].blocks["B11"] = _make_cfg_block(
            "B11", [], succs=["B12"],
        )
        # Add log_debug to cfg_cache
        mock_cfg_cache["log_debug"] = CFGFunction(
            function_name="log_debug",
            blocks={
                "ld_B1": _make_cfg_block(
                    "ld_B1", [],
                    terminator="if [ld_B1.1] (BinaryOperator) debug_enabled > 0",
                    succs=["ld_B2", "ld_B3"],
                ),
                "ld_B2": _make_cfg_block("ld_B2", ["printf(...)"]),
                "ld_B3": _make_cfg_block("ld_B3", [], label="EXIT"),
            },
            entry_block_id="ld_B1",
        )
        # Also add B10 to mask
        mock_cfg_mask = SliceMask(
            pdg_retained_nodes={"wslay_event_recv": {"S_570", "S_571", "S_572", "S_573", "S_580"}},
            cfg_retained_blocks={"wslay_event_recv": {1, 2, 3, 10, 11, 12, 13}},
        )
        extractor = CFGBranchExtractor(mock_cfg_cache, mock_sdg, mock_cfg_mask)
        tree = extractor.extract(
            segment_index=3,
            function_name="wslay_event_recv",
            line_start=575,
            line_end=580,
            postcondition=unrelated_pc,
        )
        # Even void function log_debug (no return var) should be expanded
        assert tree.num_branches > 0


class TestExtractorCalleeExtraction:
    """Cross-procedural callee CFG extraction."""

    def test_callee_extracted_when_return_matches_postcondition(
        self, mock_cfg_cache, mock_slice_mask, mock_sdg,
        segment0_postcondition,
    ):
        """wslay_frame_recv should be extracted because 'r' is postcondition var."""
        extractor = CFGBranchExtractor(mock_cfg_cache, mock_sdg, mock_slice_mask)
        tree = extractor.extract(
            segment_index=0,
            function_name="wslay_event_recv",
            line_start=566,
            line_end=573,
            postcondition=segment0_postcondition,
        )
        # Count nodes with call_depth > 0 (callee branches)
        callee_nodes = _count_by_depth(tree, min_depth=1)
        assert callee_nodes > 0, (
            "Expected callee CFG nodes (depth>=1) to be extracted "
            "when return value matches postcondition variable"
        )

    def test_callee_extracted_when_return_unrelated(
        self, mock_cfg_cache, mock_slice_mask, mock_sdg,
    ):
        """🟢 GREEN: Callee IS extracted even when return var doesn't match.

        Uses line range covering B2 (L571-L572) where wslay_frame_recv is called.
        """
        unrelated_pc = Postcondition(
            function_name="wslay_event_recv",
            source_line=575,
            condition_expr="msg_length > 0",
            branch_taken=True,
        )
        extractor = CFGBranchExtractor(mock_cfg_cache, mock_sdg, mock_slice_mask)
        tree = extractor.extract(
            segment_index=2,
            function_name="wslay_event_recv",
            line_start=570,
            line_end=575,
            postcondition=unrelated_pc,
        )
        callee_nodes = _count_by_depth(tree, min_depth=1)
        assert callee_nodes > 0, (
            "Full expansion: callee should be extracted even when return var "
            "doesn't match postcondition variable"
        )

    def test_callee_depth_increments(
        self, mock_cfg_cache, mock_slice_mask, mock_sdg,
        segment0_postcondition,
    ):
        """Callee branches should have call_depth=1."""
        extractor = CFGBranchExtractor(mock_cfg_cache, mock_sdg, mock_slice_mask)
        tree = extractor.extract(
            segment_index=0,
            function_name="wslay_event_recv",
            line_start=566,
            line_end=573,
            postcondition=segment0_postcondition,
        )
        # Check that at least one callee node has depth=1
        has_depth1 = _has_depth(tree, 1)
        assert has_depth1, "Expected at least one branch with call_depth=1"

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
        pc = Postcondition("wslay_event_recv", 100, "r >= 0", True)
        cfg = clang_cache["wslay_event_recv"]
        # B6 is a branch block (terminator="if [B6.15]")
        node = extractor._extract_from_block(
            cfg=cfg, block_id="B6", fn_name="wslay_event_recv",
            depth=0, parent_call=None, postcondition=pc,
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
        pc = Postcondition("wslay_event_recv", 100, "r >= 0", True)
        cfg = cc_cache["wslay_event_recv"]
        node = extractor._extract_from_block(
            cfg=cfg, block_id="B5", fn_name="wslay_event_recv",
            depth=0, parent_call=None, postcondition=pc,
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

    @pytest.fixture
    def cfg_with_call_in_block(self) -> dict:
        """CFG with a block containing a function call statement."""
        from llm_client.fp_analysis.cfg_models import CFGFunction, CFGBlock
        return {
            "caller_fn": CFGFunction(
                function_name="caller_fn",
                source_file="test.c",
                blocks={
                    "B1": CFGBlock(
                        block_id="B1",
                        statements=[
                            "L10: r = callee_fn(arg1, arg2)",
                        ],
                        successors=["B2"],
                    ),
                    "B2": CFGBlock(
                        block_id="B2",
                        statements=[],
                        terminator="if [B2.1] (BinaryOperator) r >= 0",
                        successors=["B3", "B4"],
                    ),
                    "B3": CFGBlock(block_id="B3", statements=[]),
                    "B4": CFGBlock(block_id="B4", statements=[""], label="EXIT"),
                },
                entry_block_id="B1",
            ),
            "callee_fn": CFGFunction(
                function_name="callee_fn",
                source_file="test.c",
                blocks={
                    "C1": CFGBlock(
                        block_id="C1",
                        statements=[],
                        terminator="if [C1.1] (BinaryOperator) x == 0",
                        succs=["C2", "C3"],
                    ),
                    "C2": CFGBlock(block_id="C2", statements=[]),
                    "C3": CFGBlock(block_id="C3", statements=[""], label="EXIT"),
                },
                entry_block_id="C1",
            ),
        }

    @pytest.fixture
    def sdg_with_call_cg(self) -> object:
        """SDG with CG that resolves caller_fn → callee_fn."""
        from llm_client.csa_analysis.callgraph_models import CallGraph
        cg = CallGraph.from_dict({
            "functions": {
                "caller_fn": {"file": "test.c", "line": 1},
                "callee_fn": {"file": "test.c", "line": 20},
            },
            "calls": [
                {"caller": "caller_fn", "callee": "callee_fn",
                 "file": "test.c", "line": 10},
            ],
        })
        sdg = MockSDG()
        sdg.callgraph = cg
        return sdg

    def test_callee_extracted_via_cg(
        self, cfg_with_call_in_block, sdg_with_call_cg,
    ):
        """_extract_callees_from_block resolves callee CFG via CG."""
        ext = CFGBranchExtractor(
            cfg_with_call_in_block, sdg_with_call_cg, SliceMask(),
        )
        pc = Postcondition("caller_fn", 11, "r >= 0", True)
        cfg = cfg_with_call_in_block["caller_fn"]
        callees = ext._extract_callees_from_block(
            cfg=cfg, block_id="B1", fn_name="caller_fn",
            depth=0, postcondition=pc,
        )
        # Should find the callee branch inside callee_fn
        assert len(callees) > 0
        assert callees[0].callee_name == "callee_fn"
        assert len(callees[0].children) > 0  # callee's internal branches


def _count_by_depth(tree: CFGBranchTree, min_depth: int) -> int:
    """Count branch nodes with call_depth >= min_depth."""
    count = 0

    def _walk(nodes):
        nonlocal count
        for n in nodes:
            if n.call_depth >= min_depth:
                count += 1
            _walk(n.children)

    _walk(tree.root_branches)
    return count


def _has_depth(tree: CFGBranchTree, target_depth: int) -> bool:
    """Check if any node has the exact call_depth."""
    found = False

    def _walk(nodes):
        nonlocal found
        for n in nodes:
            if n.call_depth == target_depth:
                found = True
            _walk(n.children)

    _walk(tree.root_branches)
    return found
