"""Iterative Path Selector — merges PDG slicing with segment-by-segment selection.

Phase 2 of the feasibility-driven pipeline.  The ``IterativePathSelector``
unifies what were previously two separate phases (backward slicing and
segment path selection) into a single iterative process:

1. Run PDG backward slicing from the bug trigger → ``SliceMask``.
2. For each execution segment, use the established state from the preceding
   segment to prune PDG nodes that contradict known variable values.
3. The pruned ``SliceMask`` is passed to the next segment, progressively
   narrowing the feasible execution space.
4. After all segments, the final (maximally pruned) mask and the selected
   segment decisions are returned.

The pruning in Step 2 uses ``SliceMask.prune_with_anchor_state(sdg=…)``,
which in Phase 2 actually removes contradictory PDG predicate nodes and
their control-dependent children from the mask (down from Phase 1's
metadata-only no-op).
"""

from __future__ import annotations

import logging
from typing import Optional

from llm_client.fp_analysis.pdg_models import SDG, SliceResult
from llm_client.fp_analysis.slice_mask import SliceMask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class IterativeSelectorConfig:
    """Configuration for :class:`IterativePathSelector`.

    Attributes:
        max_depth: Maximum recursion depth for the backward slicer.
    """

    def __init__(self, max_depth: int = 10) -> None:
        self.max_depth = max_depth


# ---------------------------------------------------------------------------
# Iterative Path Selector
# ---------------------------------------------------------------------------


class IterativePathSelector:
    """Merge PDG backward slicing with segment-by-segment path selection.

    The selector drives the iterative pruning loop::

        SliceResult → SliceMask
            │
            ▼
        Segment 0 → prune_with_anchor_state(state={}) → SliceMask₁
            │
            ▼
        Segment 1 → prune_with_anchor_state(state=state₀) → SliceMask₂
            │
            ▼
        Segment 2 → prune_with_anchor_state(state=state₁) → SliceMask₃
            │
            ▼
        Final SliceMask + segment selections

    Args:
        sdg: The System Dependence Graph (loaded PDGs + call graph).
        source_root: Root directory of the target C++ project.
        max_depth: Maximum recursion depth for backward slicing (default 10).
    """

    def __init__(
        self,
        sdg: SDG,
        source_root: str,
        max_depth: int = 10,
    ) -> None:
        self._sdg = sdg
        self._source_root = source_root
        self._config = IterativeSelectorConfig(max_depth=max_depth)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_path(
        self,
        slice_result: SliceResult | None,
        anchor_states: list[dict[tuple[str, int], bool]],
        established_states: list[dict[str, str]],
    ) -> SliceMask | None:
        """Iteratively prune the slice mask through a sequence of anchor states.

        Each element of *anchor_states* and *established_states* corresponds
        to one execution segment.  The mask is pruned segment by segment,
        returning the final (most pruned) mask.

        Args:
            slice_result: The initial ``SliceResult`` from backward slicing.
            anchor_states: One ``{(fn, line): branch_taken}`` dict per segment.
            established_states: One ``{var: constraint}`` dict per segment
                (may be empty for the first segment).

        Returns:
            The final pruned ``SliceMask``, or ``None`` if no slice result
            is available.
        """
        mask = self._build_mask(slice_result)
        if mask is None:
            return None

        for seg_idx, (anchors, state) in enumerate(
            zip(anchor_states, established_states)
        ):
            if not anchors:
                logger.debug("Segment %d: no anchor events — skipping pruning.", seg_idx)
                continue
            if not state:
                logger.debug("Segment %d: no established state — skipping pruning.", seg_idx)
                continue

            mask = mask.prune_with_anchor_state(
                anchor_state=anchors,
                established_state=state,
                sdg=self._sdg,
            )

            pruned_count = len(
                mask.pruning_metadata.get("anchor_state", {})
            ) if mask.pruning_metadata else 0
            logger.debug(
                "Segment %d: %d anchor(s) evaluated, mask pruned.",
                seg_idx, pruned_count,
            )

        return mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_mask(self, slice_result: SliceResult | None) -> SliceMask | None:
        """Build a ``SliceMask`` from a ``SliceResult`` (or return ``None``)."""
        if slice_result is None:
            return None
        return SliceMask.from_slice_result(slice_result, self._sdg)

    def _apply_state_pruning(
        self,
        mask: SliceMask,
        anchor_state: dict[tuple[str, int], bool],
        established_state: dict[str, str],
    ) -> SliceMask:
        """Apply Phase 2 state-based pruning and return the pruned mask.

        Args:
            mask: The ``SliceMask`` to prune.
            anchor_state: ``{(function_name, line): branch_taken}``.
            established_state: ``{variable: constraint}``.

        Returns:
            A new ``SliceMask`` with contradictory branches pruned.
        """
        return mask.prune_with_anchor_state(
            anchor_state=anchor_state,
            established_state=established_state,
            sdg=self._sdg,
        )
