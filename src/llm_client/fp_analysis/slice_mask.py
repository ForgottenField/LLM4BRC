"""Slice mask data structure — tracks which PDG/CFG nodes were retained vs removed by slicing.

The ``SliceMask`` records, for each function involved in a backward slice:

- Which PDG nodes survived the slice (retained) vs were cut (removed)
- Which control-dependence predicates (branches) were retained vs removed
- Which Clang CFG blocks have at least one retained node

Downstream consumers (constraint completion, feasibility checks, path selection)
can query the mask to know which parts of the PDG/CFG are still "live" after
slicing, and which branches/statements were eliminated.

Usage::

    mask = SliceMask.from_slice_result(slice_result, sdg)

    # Query helpers
    mask.is_node_retained("MyClass::method", "S_42_5")   → True/False
    mask.is_branch_retained("MyClass::method", "S_40_3") → True/False
    mask.is_cfg_block_retained("MyClass::method", 3)      → True/False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from llm_client.fp_analysis.pdg_models import SDG, SliceResult


@dataclass
class SliceMask:
    """Records which PDG nodes and CFG blocks were retained vs removed by backward slicing.

    For each function involved in the slice:

    * ``pdg_retained_nodes``  — PDG node IDs that survived the slice.
    * ``pdg_removed_nodes``   — PDG node IDs that were cut.
    * ``pdg_retained_branches``  — Control-dependence predicate node IDs retained.
    * ``pdg_removed_branches``   — Control-dependence predicate node IDs removed.
    * ``cfg_retained_blocks``    — CFG block IDs that have at least one retained node.
    * ``cfg_removed_blocks``     — CFG block IDs with *no* retained nodes.
    """

    pdg_retained_nodes: dict[str, set[str]] = field(default_factory=dict)
    pdg_removed_nodes: dict[str, set[str]] = field(default_factory=dict)
    pdg_retained_branches: dict[str, set[str]] = field(default_factory=dict)
    pdg_removed_branches: dict[str, set[str]] = field(default_factory=dict)
    cfg_retained_blocks: dict[str, set[int]] = field(default_factory=dict)
    cfg_removed_blocks: dict[str, set[int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_slice_result(
        cls,
        slice_result: SliceResult,
        sdg: SDG,
    ) -> SliceMask:
        """Build a ``SliceMask`` from a ``SliceResult`` and the original ``SDG``.

        Compares every ``SliceNode`` against the full PDG of each involved
        function, then derives the retained/removed sets for PDG nodes,
        control-dependence predicates, and CFG blocks.

        Functions that have *no* nodes in the slice are not included.
        """
        # ── Collect retained node IDs per function ──
        retained: dict[str, set[str]] = {}
        for sn in slice_result.nodes:
            retained.setdefault(sn.function_name, set()).add(sn.node.id)

        removed: dict[str, set[str]] = {}
        retained_branches: dict[str, set[str]] = {}
        removed_branches: dict[str, set[str]] = {}
        cfg_retained: dict[str, set[int]] = {}
        cfg_removed: dict[str, set[int]] = {}

        for fn_name in slice_result.involved_functions:
            pdg = sdg.get_pdg(fn_name)
            if pdg is None:
                continue

            all_nodes = set(pdg.nodes.keys())
            retained_set = retained.get(fn_name, set())
            removed_set = all_nodes - retained_set
            removed[fn_name] = removed_set

            # ── Branch tracking: control-dep source IDs ──
            for cde in pdg.control_dep_edges:
                if cde.source_id in retained_set:
                    retained_branches.setdefault(fn_name, set()).add(cde.source_id)
                else:
                    removed_branches.setdefault(fn_name, set()).add(cde.source_id)

            # ── CFG block tracking ──
            all_blocks: set[int] = set()
            retained_blocks: set[int] = set()
            for nid, node in pdg.nodes.items():
                block_id = node.block_id
                all_blocks.add(block_id)
                if nid in retained_set:
                    retained_blocks.add(block_id)
            cfg_retained[fn_name] = retained_blocks
            cfg_removed[fn_name] = all_blocks - retained_blocks

        return cls(
            pdg_retained_nodes=retained,
            pdg_removed_nodes=removed,
            pdg_retained_branches=retained_branches,
            pdg_removed_branches=removed_branches,
            cfg_retained_blocks=cfg_retained,
            cfg_removed_blocks=cfg_removed,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_node_retained(self, function_name: str, node_id: str) -> bool:
        """Check if a PDG node was retained in the slice."""
        return node_id in self.pdg_retained_nodes.get(function_name, set())

    def is_branch_retained(self, function_name: str, predicate_id: str) -> bool:
        """Check if a control-dependence predicate was retained."""
        return predicate_id in self.pdg_retained_branches.get(function_name, set())

    def is_cfg_block_retained(self, function_name: str, block_id: int) -> bool:
        """Check if a CFG block has any retained nodes."""
        return block_id in self.cfg_retained_blocks.get(function_name, set())

    # ------------------------------------------------------------------
    # Serialization (sets → lists for JSON compatibility)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (sets converted to sorted lists)."""
        result: dict = {
            "pdg_retained_nodes": {
                fn: sorted(nids) for fn, nids in self.pdg_retained_nodes.items()
            },
            "pdg_removed_nodes": {
                fn: sorted(nids) for fn, nids in self.pdg_removed_nodes.items()
            },
            "pdg_retained_branches": {
                fn: sorted(pids) for fn, pids in self.pdg_retained_branches.items()
            },
            "pdg_removed_branches": {
                fn: sorted(pids) for fn, pids in self.pdg_removed_branches.items()
            },
            "cfg_retained_blocks": {
                fn: sorted(bids) for fn, bids in self.cfg_retained_blocks.items()
            },
            "cfg_removed_blocks": {
                fn: sorted(bids) for fn, bids in self.cfg_removed_blocks.items()
            },
        }
        # Phase 1: serialise pruning_metadata (tuple keys → "fn:line" strings)
        if self.pruning_metadata is not None:
            meta = self.pruning_metadata
            serialised_anchor: dict[str, bool] = {}
            for (fn, line), val in meta.get("anchor_state", {}).items():
                serialised_anchor[f"{fn}:{line}"] = val
            result["pruning_metadata"] = {
                "anchor_state": serialised_anchor,
                "established_state": meta.get("established_state", {}),
            }
        return result

    # ------------------------------------------------------------------
    # Phase 1: Anchor-driven pruning (metadata-only for prompt injection)
    # ------------------------------------------------------------------

    pruning_metadata: Optional[dict] = None
    """Metadata for anchor-driven pruning.

    In Phase 1 this is a no-op: the metadata is passed to the LLM prompt
    as additional context.  In Phase 2, when *sdg* is provided to
    :meth:`prune_with_anchor_state`, contradictory PDG predicate nodes are
    actually removed from the mask along with their control-dependent children.
    """

    def prune_with_anchor_state(
        self,
        anchor_state: dict[tuple[str, int], bool],
        established_state: dict[str, str],
        sdg: object | None = None,
    ) -> SliceMask:
        """Return a new ``SliceMask`` with anchor-state pruning applied.

        **Phase 1** (``sdg is None``): metadata-only.  The returned mask
        carries ``pruning_metadata`` for LLM prompt injection.  No PDG nodes
        are pruned — existing slice data is preserved verbatim.

        **Phase 2** (``sdg`` provided): actual PDG node pruning.  For each
        anchor event, the PDG predicate node at that line is checked against
        the established state.  If the predicate's branch decision contradicts
        the constraint, the predicate AND its control-dependent children are
        moved from ``pdg_retained_nodes`` to ``pdg_removed_nodes``.

        Args:
            anchor_state: ``{(function_name, line): branch_taken}`` — the
                branch decisions from the CSA report's anchor events.
            established_state: ``{variable: constraint}`` — the confirmed
                state from previous segments.
            sdg: Optional ``SDG`` for Phase 2 PDG node lookup.  When
                ``None``, falls back to Phase 1 metadata-only behaviour.

        Returns:
            A new ``SliceMask`` (immutable).  The original is not mutated.
        """
        if sdg is None:
            # ── Phase 1: metadata-only ──
            return SliceMask(
                pdg_retained_nodes=self.pdg_retained_nodes.copy(),
                pdg_removed_nodes=self.pdg_removed_nodes.copy(),
                pdg_retained_branches=self.pdg_retained_branches.copy(),
                pdg_removed_branches=self.pdg_removed_branches.copy(),
                cfg_retained_blocks=self.cfg_retained_blocks.copy(),
                cfg_removed_blocks=self.cfg_removed_blocks.copy(),
                pruning_metadata={
                    "anchor_state": anchor_state,
                    "established_state": established_state,
                },
            )

        # ── Phase 2: actual PDG pruning ──
        # Deep-copy all sets so mutation does not affect the original.
        retained: dict[str, set[str]] = {
            fn: set(s) for fn, s in self.pdg_retained_nodes.items()
        }
        removed: dict[str, set[str]] = {
            fn: set(s) for fn, s in self.pdg_removed_nodes.items()
        }
        retained_branches: dict[str, set[str]] = {
            fn: set(s) for fn, s in self.pdg_retained_branches.items()
        }
        removed_branches: dict[str, set[str]] = {
            fn: set(s) for fn, s in self.pdg_removed_branches.items()
        }
        cfg_retained: dict[str, set[int]] = {
            fn: set(s) for fn, s in self.cfg_retained_blocks.items()
        }
        cfg_removed: dict[str, set[int]] = {
            fn: set(s) for fn, s in self.cfg_removed_blocks.items()
        }

        _prune_nodes(
            sdg=sdg,
            anchor_state=anchor_state,
            established_state=established_state,
            retained=retained,
            removed=removed,
            retained_branches=retained_branches,
            removed_branches=removed_branches,
            cfg_retained=cfg_retained,
            cfg_removed=cfg_removed,
        )

        return SliceMask(
            pdg_retained_nodes=retained,
            pdg_removed_nodes=removed,
            pdg_retained_branches=retained_branches,
            pdg_removed_branches=removed_branches,
            cfg_retained_blocks=cfg_retained,
            cfg_removed_blocks=cfg_removed,
            pruning_metadata={
                "anchor_state": anchor_state,
                "established_state": established_state,
            },
        )

    # ------------------------------------------------------------------
    # Phase 2: pruning helper — imported lazily to avoid circular deps
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> SliceMask:
        """Reconstruct from a dict previously returned by :meth:`to_dict`."""
        # Deserialise pruning_metadata ("fn:line" strings → tuple keys)
        raw_meta = data.get("pruning_metadata")
        pruning_metadata: dict | None = None
        if raw_meta is not None:
            deserialised_anchor: dict[tuple[str, int], bool] = {}
            for key, val in raw_meta.get("anchor_state", {}).items():
                if ":" in key:
                    fn, line_str = key.rsplit(":", 1)
                    deserialised_anchor[(fn, int(line_str))] = val
            pruning_metadata = {
                "anchor_state": deserialised_anchor,
                "established_state": raw_meta.get("established_state", {}),
            }

        return cls(
            pdg_retained_nodes={
                fn: set(nids) for fn, nids in data.get("pdg_retained_nodes", {}).items()
            },
            pdg_removed_nodes={
                fn: set(nids) for fn, nids in data.get("pdg_removed_nodes", {}).items()
            },
            pdg_retained_branches={
                fn: set(pids) for fn, pids in data.get("pdg_retained_branches", {}).items()
            },
            pdg_removed_branches={
                fn: set(pids) for fn, pids in data.get("pdg_removed_branches", {}).items()
            },
            cfg_retained_blocks={
                fn: set(bids) for fn, bids in data.get("cfg_retained_blocks", {}).items()
            },
            cfg_removed_blocks={
                fn: set(bids) for fn, bids in data.get("cfg_removed_blocks", {}).items()
            },
            pruning_metadata=pruning_metadata,
        )


# ---------------------------------------------------------------------------
# Phase 2: prune PDG nodes that contradict established state
# ---------------------------------------------------------------------------

# Predicate node kinds from the C++ PDG builder.
_CONTROL_PREDICATE_KINDS: frozenset[str] = frozenset({
    "if_cond", "while_cond", "for_cond", "binary_op",
})


def _prune_nodes(
    sdg: object,
    anchor_state: dict[tuple[str, int], bool],
    established_state: dict[str, str],
    retained: dict[str, set[str]],
    removed: dict[str, set[str]],
    retained_branches: dict[str, set[str]],
    removed_branches: dict[str, set[str]],
    cfg_retained: dict[str, set[int]],
    cfg_removed: dict[str, set[int]],
) -> None:
    """Inline-prune PDG nodes that contradict established state.

    Mutates *retained*, *removed*, *retained_branches*, *removed_branches*,
    *cfg_retained*, *cfg_removed* **in place** for each anchor event where
    the predicate's branch decision contradicts the established state.

    When a predicate is pruned, its direct control-dependent children
    (from ``FunctionPDG.control_dep_edges``) are also pruned, and CFG
    blocks that lose all their retained nodes are moved to *cfg_removed*.
    """

    for (fn_name, line), branch_taken in anchor_state.items():
        pdg = sdg.get_pdg(fn_name)  # type: ignore[union-attr]
        if pdg is None:
            continue

        # ── Find the predicate PDG node at this line ──
        nid_pred: str | None = None
        expr: str = ""
        for nid, node in pdg.nodes.items():
            if node.source_line != line:
                continue
            if node.kind not in _CONTROL_PREDICATE_KINDS:
                continue
            nid_pred = nid
            expr = node.expression or ""
            break

        if nid_pred is None:
            continue  # no predicate at this anchor line

        # ── Extract variable and check against established state ──
        # Lazy import to avoid circular dependency (path_analyzer → slice_mask)
        from llm_client.fp_analysis.path_analyzer import (
            _evaluate_constraint_against_branch,
            _find_variable_in_condition,
        )

        var = _find_variable_in_condition(expr)
        if var is None or var not in established_state:
            continue

        constraint = established_state[var]
        result, _msg = _evaluate_constraint_against_branch(
            anchor_branch=branch_taken,
            condition=expr,
            constraint_str=constraint,
        )

        if result != "contradictory":
            continue

        # ── Prune the predicate node ──
        if fn_name in retained:
            retained[fn_name].discard(nid_pred)
            removed.setdefault(fn_name, set()).add(nid_pred)
        if fn_name in retained_branches:
            retained_branches[fn_name].discard(nid_pred)
            removed_branches.setdefault(fn_name, set()).add(nid_pred)

        # ── Prune control-dependent children ──
        for cde in pdg.control_dep_edges:
            if cde.source_id == nid_pred:
                child_id = cde.target_id
                if fn_name in retained:
                    retained[fn_name].discard(child_id)
                    removed.setdefault(fn_name, set()).add(child_id)

    # ── Rebuild CFG block tracking after pruning ──
    for fn_name in list(retained.keys()):
        fn_pdg = sdg.get_pdg(fn_name)  # type: ignore[union-attr]
        if fn_pdg is None:
            continue

        # Compute which blocks still have retained nodes
        remaining_blocks: set[int] = set()
        for nid in retained.get(fn_name, set()):
            node = fn_pdg.nodes.get(nid)
            if node and node.block_id is not None:
                remaining_blocks.add(node.block_id)

        # Move blocks that lost all nodes from retained → removed
        old_blocks = set(cfg_retained.get(fn_name, set()))
        for bid in old_blocks:  # copy for safe iteration
            if bid not in remaining_blocks:
                cfg_retained[fn_name].discard(bid)
                cfg_removed.setdefault(fn_name, set()).add(bid)
