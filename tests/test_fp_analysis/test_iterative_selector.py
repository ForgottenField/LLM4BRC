"""Tests for Phase 2: IterativePathSelector + actual PDG pruning.

Phase 2 merges PDG backward slicing with segment-by-segment path selection
and upgrades ``prune_with_anchor_state`` from a no-op to actual PDG node
pruning using established state constraints.
"""

from __future__ import annotations

import pytest

from llm_client.fp_analysis.pdg_models import (
    ControlDepEdge,
    FunctionPDG,
    PDGNode,
    SDG,
    SliceResult,
)
from llm_client.fp_analysis.slice_mask import SliceMask
from llm_client.fp_analysis.iterative_selector import IterativePathSelector


# ===================================================================
# Fixtures — minimal SDG with control-dependence edges
# ===================================================================


@pytest.fixture
def simple_sdg():
    """One function with a binary_op predicate ``x <= 0`` and two children.

    PDG structure::

        S_10_3  (binary_op "x <= 0")   ← predicate, controls children
           ├── S_11_5  (decl "y = 1")       ← control-dependent on pred
           └── S_12_1  (call_expr "log()")  ← control-dependent on pred
        S_20_1  (decl "z = 2")               ← NOT dependent
    """
    pred = PDGNode(
        id="S_10_3", kind="binary_op",
        expression="x <= 0", variable="",
        source_line=10, block_id=2,
    )
    child1 = PDGNode(
        id="S_11_5", kind="decl", variable="y",
        expression="y = 1", source_line=11, block_id=3,
    )
    child2 = PDGNode(
        id="S_12_1", kind="call_expr", variable="",
        expression="log()", source_line=12, block_id=3,
    )
    other = PDGNode(
        id="S_20_1", kind="decl", variable="z",
        expression="z = 2", source_line=20, block_id=5,
    )

    pdg = FunctionPDG(
        function_name="test_fn",
        source_file="/fake/test.cc",
        nodes={
            "S_10_3": pred,
            "S_11_5": child1,
            "S_12_1": child2,
            "S_20_1": other,
        },
        control_dep_edges=[
            ControlDepEdge(source_id="S_10_3", target_id="S_11_5", kind="if_branch"),
            ControlDepEdge(source_id="S_10_3", target_id="S_12_1", kind="if_branch"),
        ],
    )
    sdg = SDG(pdgs={"test_fn": pdg})
    sdg.build_file_index()
    return sdg


@pytest.fixture
def null_check_sdg():
    """One function with a null-check predicate ``ptr == NULL``.

    PDG structure::

        S_15_3  (binary_op "ptr == NULL")   ← predicate
           ├── S_16_5  (decl "x = 0")            ← if-branch (ptr is NULL)
           └── S_18_1  (call_expr "use(ptr)")    ← else-branch (ptr not NULL)
        S_25_1  (return)                          ← not dependent
    """
    pred = PDGNode(
        id="S_15_3", kind="binary_op",
        expression="ptr == NULL", variable="",
        source_line=15, block_id=2,
    )
    null_child = PDGNode(
        id="S_16_5", kind="decl", variable="x",
        expression="x = 0", source_line=16, block_id=3,
    )
    use_child = PDGNode(
        id="S_18_1", kind="call_expr", variable="",
        expression="use(ptr)", source_line=18, block_id=4,
    )
    ret = PDGNode(
        id="S_25_1", kind="return", variable="",
        expression="return x", source_line=25, block_id=6,
    )

    pdg = FunctionPDG(
        function_name="null_check_fn",
        source_file="/fake/test.cc",
        nodes={
            "S_15_3": pred,
            "S_16_5": null_child,
            "S_18_1": use_child,
            "S_25_1": ret,
        },
        control_dep_edges=[
            ControlDepEdge(source_id="S_15_3", target_id="S_16_5", kind="if_branch"),
            ControlDepEdge(source_id="S_15_3", target_id="S_18_1", kind="if_branch"),
        ],
    )
    sdg = SDG(pdgs={"null_check_fn": pdg})
    sdg.build_file_index()
    return sdg


# ===================================================================
# Phase 2: SliceMask.prune_with_anchor_state with SDG
# ===================================================================


class TestPhase2PrunePredicate:
    """Prune contradictory predicates using established_state + sdg."""

    def test_removes_contradictory_predicate(self, simple_sdg):
        """Predicate ``x <= 0`` with TRUE branch + state ``x > 0`` → pruned."""
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("test_fn", 10): True},
            established_state={"x": "> 0"},
            sdg=simple_sdg,
        )
        # Predicate removed from retained → added to removed
        assert "S_10_3" not in result.pdg_retained_nodes["test_fn"]
        assert "S_10_3" in result.pdg_removed_nodes["test_fn"]

    def test_removes_predicate_branch_flag(self, simple_sdg):
        """Pruning also moves the node out of retained_branches."""
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("test_fn", 10): True},
            established_state={"x": "> 0"},
            sdg=simple_sdg,
        )
        assert "S_10_3" not in result.pdg_retained_branches["test_fn"]
        assert "S_10_3" in result.pdg_removed_branches["test_fn"]

    def test_cascades_to_control_dependents(self, simple_sdg):
        """Children of contradictory predicate are also pruned."""
        mask = SliceMask(
            pdg_retained_nodes={
                "test_fn": {"S_10_3", "S_11_5", "S_12_1", "S_20_1"},
            },
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 3, 5}},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("test_fn", 10): True},
            established_state={"x": "> 0"},
            sdg=simple_sdg,
        )
        assert "S_11_5" not in result.pdg_retained_nodes["test_fn"]
        assert "S_12_1" not in result.pdg_retained_nodes["test_fn"]
        assert "S_11_5" in result.pdg_removed_nodes["test_fn"]
        assert "S_12_1" in result.pdg_removed_nodes["test_fn"]

    def test_cascade_updates_cfg_blocks(self, simple_sdg):
        """CFG blocks that lose all retained nodes are moved to removed."""
        mask = SliceMask(
            pdg_retained_nodes={
                "test_fn": {"S_10_3", "S_11_5", "S_12_1", "S_20_1"},
            },
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 3, 5}},
            cfg_removed_blocks={"test_fn": set()},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("test_fn", 10): True},
            established_state={"x": "> 0"},
            sdg=simple_sdg,
        )
        # Block 3 had only S_11_5 and S_12_1 → both pruned → block 3 removed
        assert 3 not in result.cfg_retained_blocks["test_fn"]
        assert 3 in result.cfg_removed_blocks["test_fn"]
        # Block 2 had S_10_3 (predicate) → pruned → block 2 removed
        assert 2 not in result.cfg_retained_blocks["test_fn"]
        assert 2 in result.cfg_removed_blocks["test_fn"]
        # Block 5 had S_20_1 → NOT pruned → block 5 stays
        assert 5 in result.cfg_retained_blocks["test_fn"]

    def test_consistent_predicate_survives(self, simple_sdg):
        """Predicate with TRUE branch + supporting state → not pruned."""
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("test_fn", 10): True},
            established_state={"x": "<= 0"},
            sdg=simple_sdg,
        )
        assert "S_10_3" in result.pdg_retained_nodes["test_fn"]
        assert "S_10_3" not in result.pdg_removed_nodes.get("test_fn", set())

    def test_false_branch_with_contradictory_state(self, simple_sdg):
        """FALSE branch of ``x <= 0`` means ``x > 0`` — state ``x > 0`` is consistent."""
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        # FALSE branch of x <= 0 → x > 0, state says x > 0 → consistent
        result = mask.prune_with_anchor_state(
            anchor_state={("test_fn", 10): False},
            established_state={"x": "> 0"},
            sdg=simple_sdg,
        )
        assert "S_10_3" in result.pdg_retained_nodes["test_fn"]

    def test_unrelated_variable_unaffected(self, simple_sdg):
        """State for unrelated variable does not trigger pruning."""
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("test_fn", 10): True},
            established_state={"unrelated": "> 0"},
            sdg=simple_sdg,
        )
        assert "S_10_3" in result.pdg_retained_nodes["test_fn"]

    def test_multiple_anchors_all_contradictory(self, simple_sdg):
        """Multiple anchor events each contradictory → all predicates pruned.

        Sets up a second predicate at line 15 (not in simple_sdg's PDG,
        so only the line-10 anchor triggers pruning).
        """
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        # One anchor at a real line, one at a non-existent line (ignored)
        result = mask.prune_with_anchor_state(
            anchor_state={
                ("test_fn", 10): True,
                ("test_fn", 99): False,  # line 99 doesn't exist → skip
            },
            established_state={"x": "> 0"},
            sdg=simple_sdg,
        )
        assert "S_10_3" not in result.pdg_retained_nodes["test_fn"]

    def test_anchor_line_missing_from_pdg_no_crash(self, simple_sdg):
        """Anchor line that doesn't match any PDG node → graceful skip."""
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("test_fn", 99): True},  # line 99 has no PDG node
            established_state={"x": "> 0"},
            sdg=simple_sdg,
        )
        assert "S_10_3" in result.pdg_retained_nodes["test_fn"]


class TestPhase2NullCheckPruning:
    """Pruning with null-check predicates (ptr == NULL / ptr != NULL)."""

    def test_true_null_with_nonnull_state_contradictory(self, null_check_sdg):
        """TRUE branch of ``ptr == NULL`` + state ``allocated`` → contradictory."""
        mask = SliceMask(
            pdg_retained_nodes={
                "null_check_fn": {"S_15_3", "S_16_5", "S_18_1", "S_25_1"},
            },
            pdg_retained_branches={"null_check_fn": {"S_15_3"}},
            cfg_retained_blocks={"null_check_fn": {2, 3, 4, 6}},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("null_check_fn", 15): True},  # TRUE branch: ptr IS null
            established_state={"ptr": "non-null allocated pointer"},
            sdg=null_check_sdg,
        )
        assert "S_15_3" not in result.pdg_retained_nodes["null_check_fn"]
        assert "S_16_5" not in result.pdg_retained_nodes["null_check_fn"]

    def test_false_null_with_nonnull_state_consistent(self, null_check_sdg):
        """FALSE branch of ``ptr == NULL`` → ptr NOT null + state non-null → consistent."""
        mask = SliceMask(
            pdg_retained_nodes={
                "null_check_fn": {"S_15_3", "S_16_5", "S_18_1", "S_25_1"},
            },
            pdg_retained_branches={"null_check_fn": {"S_15_3"}},
            cfg_retained_blocks={"null_check_fn": {2, 3, 4, 6}},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("null_check_fn", 15): False},  # FALSE: ptr NOT null
            established_state={"ptr": "non-null allocated pointer"},
            sdg=null_check_sdg,
        )
        # Consistent → nothing pruned
        assert "S_15_3" in result.pdg_retained_nodes["null_check_fn"]
        assert "S_18_1" in result.pdg_retained_nodes["null_check_fn"]


class TestPhase2MetadataFallback:
    """Without sdg, prune_with_anchor_state falls back to Phase 1 metadata-only."""

    def test_no_sdg_does_not_prune(self, simple_sdg):
        """Without sdg arg, nodes are preserved (Phase 1 compat)."""
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("test_fn", 10): True},
            established_state={"x": "> 0"},
            # no sdg → Phase 1 mode
        )
        assert result.pruning_metadata is not None
        assert "S_10_3" in result.pdg_retained_nodes["test_fn"]
        assert "S_20_1" in result.pdg_retained_nodes["test_fn"]

    def test_no_sdg_survives_serialization(self):
        """Phase 1 metadata-only roundtrip unchanged."""
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2}},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("test_fn", 10): True},
            established_state={"x": "> 0"},
        )
        data = result.to_dict()
        restored = SliceMask.from_dict(data)
        assert restored.pruning_metadata is not None
        assert restored.pruning_metadata["anchor_state"] == {("test_fn", 10): True}
        assert "S_10_3" in restored.pdg_retained_nodes["test_fn"]


# ===================================================================
# IterativePathSelector — unified slicing + segment selection
# ===================================================================


class TestIterativePathSelector:
    """IterativePathSelector initialization and basic flow."""

    def test_init_stores_sdg_and_source_root(self, simple_sdg):
        """Selector stores SDG and source_root from constructor."""
        selector = IterativePathSelector(
            sdg=simple_sdg,
            source_root="/fake/root",
        )
        assert selector._sdg is simple_sdg
        assert selector._source_root == "/fake/root"
        assert selector._config is not None

    def test_init_with_config(self, simple_sdg):
        """Selector accepts optional config overrides."""
        selector = IterativePathSelector(
            sdg=simple_sdg,
            source_root="/fake/root",
            max_depth=5,
        )
        assert selector._config.max_depth == 5

    def test_build_mask_returns_none_without_slice(self, simple_sdg):
        """_build_mask returns None when no slice result is available."""
        selector = IterativePathSelector(
            sdg=simple_sdg,
            source_root="/fake/root",
        )
        mask = selector._build_mask(None)
        assert mask is None

    def test_build_mask_from_slice_result(self, simple_sdg):
        """_build_mask creates a SliceMask from a SliceResult."""
        selector = IterativePathSelector(
            sdg=simple_sdg,
            source_root="/fake/root",
        )
        # Build a minimal SliceResult
        slice_result = SliceResult(
            trigger_function="test_fn",
            trigger_node_id="S_20_1",
            nodes=[],  # empty nodes list — mask will be empty too
            involved_functions={"test_fn"},
        )
        mask = selector._build_mask(slice_result)
        assert mask is not None
        assert isinstance(mask, SliceMask)

    def test_apply_state_pruning_returns_pruned_mask(self, simple_sdg):
        """_apply_state_pruning prunes contradictory predicates."""
        selector = IterativePathSelector(
            sdg=simple_sdg,
            source_root="/fake/root",
        )
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        pruned = selector._apply_state_pruning(
            mask=mask,
            anchor_state={("test_fn", 10): True},
            established_state={"x": "> 0"},
        )
        assert "S_10_3" not in pruned.pdg_retained_nodes["test_fn"]
        assert "S_10_3" in pruned.pdg_removed_nodes["test_fn"]

    def test_apply_state_pruning_consistent(self, simple_sdg):
        """_apply_state_pruning preserves consistent predicates."""
        selector = IterativePathSelector(
            sdg=simple_sdg,
            source_root="/fake/root",
        )
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        pruned = selector._apply_state_pruning(
            mask=mask,
            anchor_state={("test_fn", 10): True},
            established_state={"x": "<= 0"},
        )
        assert "S_10_3" in pruned.pdg_retained_nodes["test_fn"]

    def test_anchor_event_not_in_mask_skipped(self, simple_sdg):
        """Anchor events with no matching PDG node skip gracefully."""
        selector = IterativePathSelector(
            sdg=simple_sdg,
            source_root="/fake/root",
        )
        mask = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        # Line 99 has no PDG node in simple_sdg → should skip without error
        pruned = selector._apply_state_pruning(
            mask=mask,
            anchor_state={("test_fn", 99): True},
            established_state={"x": "> 0"},
        )
        assert pruned is not None
        assert "S_10_3" in pruned.pdg_retained_nodes["test_fn"]

    def test_pruning_immutable_original(self, simple_sdg):
        """_apply_state_pruning returns a new mask without mutating the original."""
        selector = IterativePathSelector(
            sdg=simple_sdg,
            source_root="/fake/root",
        )
        original = SliceMask(
            pdg_retained_nodes={"test_fn": {"S_10_3", "S_20_1"}},
            pdg_retained_branches={"test_fn": {"S_10_3"}},
            cfg_retained_blocks={"test_fn": {2, 5}},
        )
        _ = selector._apply_state_pruning(
            mask=original,
            anchor_state={("test_fn", 10): True},
            established_state={"x": "> 0"},
        )
        # Original unchanged
        assert "S_10_3" in original.pdg_retained_nodes["test_fn"]
