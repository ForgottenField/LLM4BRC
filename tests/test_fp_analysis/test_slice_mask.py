"""Tests for SliceMask — anchor-driven pruning and state management."""

from __future__ import annotations

from llm_client.fp_analysis.slice_mask import SliceMask


class TestSliceMaskPruneWithAnchorState:
    """SliceMask.prune_with_anchor_state — Phase 1 metadata-only interface."""

    def test_returns_new_mask_on_prune(self):
        """prune_with_anchor_state returns a new SliceMask (immutable pattern)."""
        mask = SliceMask()
        result = mask.prune_with_anchor_state(
            anchor_state={("f", 10): True},
            established_state={},
        )
        assert isinstance(result, SliceMask)
        assert result is not mask  # different object

    def test_metadata_attached_on_prune(self):
        """The new mask carries pruning_metadata for prompt injection."""
        result = SliceMask().prune_with_anchor_state(
            anchor_state={("f", 42): False},
            established_state={"x": "> 0"},
        )
        assert hasattr(result, "pruning_metadata")
        meta = result.pruning_metadata
        assert "anchor_state" in meta
        assert "established_state" in meta
        assert meta["anchor_state"] == {("f", 42): False}
        assert meta["established_state"] == {"x": "> 0"}

    def test_empty_state_produces_empty_metadata(self):
        """prune_with empty state produces a mask with empty pruning_metadata."""
        result = SliceMask().prune_with_anchor_state(
            anchor_state={},
            established_state={},
        )
        assert result.pruning_metadata["anchor_state"] == {}
        assert result.pruning_metadata["established_state"] == {}

    def test_original_mask_unchanged(self):
        """Original SliceMask is not mutated by prune_with_anchor_state."""
        original = SliceMask()
        _ = original.prune_with_anchor_state(
            anchor_state={("f", 10): True},
            established_state={"x": "1"},
        )
        # original should have no pruning_metadata
        assert not hasattr(original, "pruning_metadata") or \
            original.pruning_metadata is None

    def test_preserves_existing_slice_data(self):
        """Existing slice data survives the prune operation."""
        mask = SliceMask(
            pdg_retained_nodes={"f": {"S_10_5", "S_20_3"}},
            pdg_retained_branches={"f": {"S_10_5"}},
        )
        result = mask.prune_with_anchor_state(
            anchor_state={("f", 10): True},
            established_state={"x": "> 0"},
        )
        # Retained nodes unchanged in Phase 1
        assert result.pdg_retained_nodes == mask.pdg_retained_nodes
        assert result.pdg_retained_branches == mask.pdg_retained_branches

    def test_serialization_roundtrip_with_metadata(self):
        """pruning_metadata survives to_dict/from_dict roundtrip."""
        mask = SliceMask(
            pdg_retained_nodes={"f": {"S_10_5"}},
            pdg_retained_branches={"f": {"S_10_5"}},
        )
        pruned = mask.prune_with_anchor_state(
            anchor_state={("f", 42): False, ("g", 10): True},
            established_state={"x": "> 0", "y": "NULL"},
        )
        data = pruned.to_dict()
        restored = SliceMask.from_dict(data)
        assert restored.pruning_metadata is not None
        assert restored.pruning_metadata["anchor_state"] == {
            ("f", 42): False, ("g", 10): True,
        }
        assert restored.pruning_metadata["established_state"] == {
            "x": "> 0", "y": "NULL",
        }
        # Original slice data survives
        assert restored.pdg_retained_nodes == {"f": {"S_10_5"}}

    def test_serialization_roundtrip_without_metadata(self):
        """to_dict/from_dict work when pruning_metadata is None."""
        mask = SliceMask(pdg_retained_nodes={"f": {"S_10_5"}})
        data = mask.to_dict()
        assert "pruning_metadata" not in data
        restored = SliceMask.from_dict(data)
        assert restored.pruning_metadata is None
        assert restored.pdg_retained_nodes == {"f": {"S_10_5"}}
