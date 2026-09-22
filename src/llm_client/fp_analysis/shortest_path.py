"""ShortestPathComputer + CalleePathCache — deterministic shortest-path computation
on CFG branch trees.

Phase 1 of the shortest-path selection + conflict-driven learning design.
Replaces the current "LLM blindly picks from all branches" approach with
deterministic metric-based candidate ranking (S → C → V).

Usage::

    callee_cache = CalleePathCache(sdg)
    pc = ShortestPathComputer(sdg, callee_cache)
    candidates = pc.compute_topk(segment.cfg_branch_tree, k=3)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from llm_client.fp_analysis.cfg_branch_models import (
    CFGBranchNode,
    CFGBranchTree,
)

logger = logging.getLogger(__name__)

# PDG node kinds that should NOT count toward statement count.
_AUXILIARY_KINDS: frozenset[str] = frozenset({
    "recovery", "param", "synthetic_param",
})


# ─── PathMetrics ────────────────────────────────────────────────────────────────


@dataclass
class PathMetrics:
    """Metrics for ranking path candidates.

    Ordering priority: **S ↑ → C ↑ → V ↑**  (ascending — smaller is better).

    Attributes:
        S: Statement count — total C/C++ statements on the path (excluding
           recovery, param, synthetic_param nodes).
        C: Callee count — number of distinct callee function calls on the path.
        V: Variable count — number of distinct variables in branch conditions.
    """

    S: int = 0
    C: int = 0
    V: int = 0

    def __lt__(self, other: PathMetrics) -> bool:
        """Order by: S↑ → C↑ → V↑ (ascending)."""
        return (self.S, self.C, self.V) < (other.S, other.C, other.V)

    def __add__(self, other: PathMetrics) -> PathMetrics:
        return PathMetrics(
            S=self.S + other.S,
            C=self.C + other.C,
            V=self.V + other.V,
        )


# ─── BranchDecision (re-used lightweight version for path selection) ────────────


@dataclass
class PathBranchDecision:
    """A single branch decision on a computed path.

    Unlike :class:`cfg_branch_models.BranchDecision` (which covers CFG block
    successors), this records the branch NODE identity and which side was taken.

    Attributes:
        node_id: The ``CFGBranchNode.node_id``.
        branch_taken: ``True`` if the taken (true) side was selected.
        condition_expr: Human-readable condition text from the branch node.
        source_line: Source line of the branch.
    """

    node_id: str
    branch_taken: bool = True
    condition_expr: str = ""
    source_line: int = 0


# ─── CalleePathCache ────────────────────────────────────────────────────────────


@dataclass
class CalleeReturnPath:
    """A single return path through a callee function.

    Attributes:
        return_line: Source line of the return statement.
        statement_count: Number of non-auxiliary PDG nodes on this path.
        return_expression: The return expression text (e.g. ``"-1"``, ``"r"``).
    """

    return_line: int
    statement_count: int = 0
    return_expression: str = ""


class CalleePathCache:
    """Pre-computes shortest paths from function entry to each return statement.

    Uses BFS on the PDG (backward from return nodes to entry nodes) to find
    the minimum number of statements on any path to each return.

    When a callee's PDG is unavailable, returns conservative defaults.
    """

    _EXCLUDED_KINDS: frozenset[str] = _AUXILIARY_KINDS

    def __init__(self, sdg: object) -> None:
        """Initialise with an SDG for PDG lookups.

        Args:
            sdg: An :class:`SDG` providing ``get_pdg(fn_name)``.
        """
        self._sdg = sdg
        self._cache: dict[str, list[CalleeReturnPath]] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def get_min_stmts(self, callee_name: str) -> int:
        """Return the minimum statement count from entry to any return.

        If *callee_name* has not been precomputed or has no paths, returns 1
        (conservative default).
        """
        paths = self._cache.get(callee_name)
        if not paths:
            logger.debug("No cached paths for callee '%s' — using default=1", callee_name)
            return 1
        return min(p.statement_count for p in paths)

    def get_paths(self, callee_name: str) -> list[CalleeReturnPath]:
        """Return all cached return paths for *callee_name*."""
        return list(self._cache.get(callee_name, []))

    def precompute(self, callee_name: str) -> None:
        """Pre-compute shortest return paths for *callee_name*.

        If the PDG cannot be resolved, the function name stays uncached
        and :meth:`get_min_stmts` falls back to the conservative default.
        """
        if callee_name in self._cache:
            return  # already computed

        pdg = self._get_pdg(callee_name)
        if pdg is None or not pdg.nodes:
            logger.debug("No PDG for callee '%s' — skipping precompute", callee_name)
            return  # will use conservative default

        paths = self._compute_return_paths(pdg, callee_name)
        self._cache[callee_name] = paths

    # ── Internal ────────────────────────────────────────────────────────────

    def _get_pdg(self, fn_name: str) -> object | None:
        """Resolve a function name to its PDG via the SDG."""
        sdg = self._sdg
        if sdg is None:
            return None
        if hasattr(sdg, "get_pdg"):
            return sdg.get_pdg(fn_name)
        return None

    def _compute_return_paths(
        self, pdg: object, _callee_name: str,
    ) -> list[CalleeReturnPath]:
        """BFS from each return node backward to entry nodes via PDG edges.

        For each return node, runs BFS backward along data_dep + control_dep
        edges to any entry (param) node.  Counts non-auxiliary nodes on the
        shortest path from entry to this return.
        """
        from llm_client.fp_analysis.pdg_models import PDGNode

        nodes: dict[str, PDGNode] = getattr(pdg, "nodes", {})
        if not nodes:
            return []

        # Identify entry nodes: param/synthetic_param nodes with source_line == 0
        entry_ids: set[str] = set()
        return_nodes: list[tuple[str, PDGNode]] = []
        for nid, node in nodes.items():
            if node.kind == "return":
                return_nodes.append((nid, node))
            if node.kind in ("param", "synthetic_param") and node.source_line == 0:
                entry_ids.add(nid)

        if not return_nodes or not entry_ids:
            return []

        # Build backward adjacency: node → list of predecessors
        # (nodes that depend on this node via data_dep or control_dep)
        predecessors: dict[str, list[str]] = {}
        for nid in nodes:
            predecessors.setdefault(nid, [])

        # Data dependence edges: def → use.  Backward: use → def
        for due in getattr(pdg, "def_use_edges", []):
            pred = getattr(due, "def_node_id", None)
            succ = getattr(due, "use_node_id", None)
            if pred and succ and pred in nodes and succ in nodes:
                predecessors.setdefault(succ, []).append(pred)

        # Control dependence edges: source → target.  Backward: target → source
        for cde in getattr(pdg, "control_dep_edges", []):
            src = getattr(cde, "source_id", None)
            tgt = getattr(cde, "target_id", None)
            if src and tgt and src in nodes and tgt in nodes:
                predecessors.setdefault(tgt, []).append(src)

        paths: list[CalleeReturnPath] = []
        for ret_id, ret_node in return_nodes:
            # BFS backward from return to any entry node
            stmt_count = self._bfs_min_stmts(
                start_id=ret_id,
                target_ids=entry_ids,
                nodes=nodes,
                predecessors=predecessors,
            )
            paths.append(CalleeReturnPath(
                return_line=ret_node.source_line,
                statement_count=stmt_count,
                return_expression=ret_node.expression or "",
            ))

        return paths

    @staticmethod
    def _bfs_min_stmts(
        start_id: str,
        target_ids: set[str],
        nodes: dict,
        predecessors: dict[str, list[str]],
    ) -> int:
        """BFS from *start_id* backward to any node in *target_ids*.

        Returns the minimum number of non-auxiliary nodes on the path
        (including *start_id* if not auxiliary, excluding target nodes).
        """
        if start_id in target_ids:
            return 0

        # Count the start node itself (if non-auxiliary)
        start_node = nodes.get(start_id)
        start_count = 0 if (start_node and start_node.kind in _AUXILIARY_KINDS) else 1

        # BFS queue: (node_id, depth_from_start)
        queue: deque[tuple[str, int]] = deque()
        queue.append((start_id, 0))
        visited: set[str] = set()

        while queue:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # Check if we reached a target (entry param node)
            if current in target_ids:
                return start_count + depth

            preds = predecessors.get(current, [])
            for pred in preds:
                if pred in visited:
                    continue
                node = nodes.get(pred)
                if node is None:
                    continue
                # Only count non-auxiliary nodes
                is_aux = node.kind in _AUXILIARY_KINDS
                new_depth = depth + (0 if is_aux else 1)
                queue.append((pred, new_depth))

        # No path found — return 1 as fallback
        return 1


# ─── ShortestPathComputer ───────────────────────────────────────────────────────


class ShortestPathComputer:
    """Enumerate and rank path candidates by shortest-path metric.

    Performs DFS on a :class:`CFGBranchTree`, respecting ConflictChecker tags:
    contradictory branches are pruned, consistent branches are auto-confirmed,
    and insufficient_info branches are explored both ways.

    Each complete path is ranked by ``(S, C, V)`` ascending.
    """

    def __init__(self, sdg: object, callee_cache: CalleePathCache) -> None:
        """Initialise with an SDG and pre-computed callee cache.

        Args:
            sdg: For block statement estimation via PDG queries.
            callee_cache: Pre-computed callee shortest paths.
        """
        self._sdg = sdg
        self._callee_cache = callee_cache

    # Maximum number of complete paths to enumerate (prevents combinatorial
    # explosion for segments with many insufficient_info branches).
    _MAX_CANDIDATES: int = 200

    # Threshold above which we use greedy instead of full enumeration.
    _GREEDY_THRESHOLD: int = 15

    # ── Public API ──────────────────────────────────────────────────────────

    def compute_topk(
        self,
        tree: CFGBranchTree,
        k: int = 3,
        postcondition=None,
    ) -> list[tuple[list[PathBranchDecision], PathMetrics]]:
        """Enumerate and rank the top-k shortest path candidates.

        When *postcondition* is provided, candidates whose branch decisions
        violate the required branch direction at *postcondition.source_line*
        are pruned **before** ranking, so the top-k are always constraint-
        respecting paths.

        Args:
            tree: The segment's CFG branch tree (with ConflictChecker tags).
            k: Maximum number of candidates to return.
            postcondition: Optional :class:`Postcondition` — if set, only
                candidates matching its branch direction are retained.

        Returns:
            List of ``(branch_decisions, metrics)`` sorted by S↑ → C↑ → V↑.
            Returns at least one candidate even if all branches are pruned.
        """
        if not tree.root_branches:
            return [([], PathMetrics())]

        # For large branch sets, use greedy to avoid 2^N explosion
        if len(tree.root_branches) > self._GREEDY_THRESHOLD:
            return self._compute_greedy(tree.root_branches, k, postcondition)

        # ── Pre-process: demote "consistent" branches that violate the
        #     postcondition to "insufficient_info", so the DFS explores
        #     both directions and the filter has enough candidates. ──
        if postcondition is not None:
            pc_line = getattr(postcondition, "source_line", 0)
            pc_taken = getattr(postcondition, "branch_taken", None)
            if pc_line and pc_taken is not None:
                self._demote_conflicting_consistent(
                    tree.root_branches, pc_line, pc_taken,
                )

        candidates: list[tuple[list[PathBranchDecision], PathMetrics]] = []

        # Detect mutually-exclusive branch groups (same variable == different
        # constants) once for the whole segment tree, to prune infeasible
        # candidates after enumeration.
        exclusive_groups = self._extract_equality_exclusive_groups(
            tree.root_branches,
        )

        self._dfs_collect(
            nodes=tree.root_branches,
            idx=0,
            path_so_far=[],
            metrics_so_far=PathMetrics(),
            candidates=candidates,
        )

        if not candidates:
            return [([], PathMetrics())]

        # Filter by postcondition before ranking, so the top-k are
        # always constraint-respecting and not all the same direction.
        if postcondition is not None:
            pc_line = getattr(postcondition, "source_line", 0)
            pc_taken = getattr(postcondition, "branch_taken", None)
            if pc_line and pc_taken is not None:
                filtered: list = []
                for cand in candidates:
                    decisions, _metrics = cand
                    violation = any(
                        getattr(d, "source_line", 0) == pc_line
                        and getattr(d, "branch_taken", True) != pc_taken
                        for d in decisions
                    )
                    if not violation:
                        filtered.append(cand)
                if filtered:
                    logger.info(
                        "compute_topk: postcondition filter pruned "
                        "%d/%d candidates (L%d must be %s)",
                        len(candidates) - len(filtered), len(candidates),
                        pc_line,
                        "TRUE" if pc_taken else "FALSE",
                    )
                    candidates = filtered
                else:
                    logger.warning(
                        "compute_topk: postcondition filter pruned ALL "
                        "%d candidates (L%d must be %s) — all fallback",
                        len(candidates), pc_line,
                        "TRUE" if pc_taken else "FALSE",
                    )
                # If ALL candidates are pruned, keep the original list
                # (best-effort — the LLM will need to handle it).

        # Prune mutually-exclusive branch combinations so the candidates are a
        # feasibility-filtered set, not just statement-minimal.  The top-k must
        # never force two equality branches (same var, different consts) true.
        if exclusive_groups:
            kept, pruned = self._filter_mutually_exclusive(
                candidates, exclusive_groups,
            )
            if pruned:
                logger.info(
                    "compute_topk: mutual-exclusion pruned %d/%d candidates "
                    "(%d exclusive group(s))",
                    pruned, len(candidates), len(exclusive_groups),
                )
                if kept:
                    candidates = kept
                else:
                    logger.warning(
                        "compute_topk: mutual-exclusion pruned ALL candidates "
                        "(%d exclusive group(s)) — all fallback",
                        len(exclusive_groups),
                    )

        candidates.sort(key=lambda x: (x[1].S, x[1].C, x[1].V))
        return candidates[:k]

    # ── Greedy fallback (large branch sets) ─────────────────────────────────

    def _compute_greedy(
        self,
        root_branches: list[CFGBranchNode],
        k: int,
        postcondition=None,
    ) -> list[tuple[list[PathBranchDecision], PathMetrics]]:
        """Greedy path computation for segments with many root branches.

        For each branch, picks the single shortest sub-path direction.
        Then generates *k* variants by flipping one branch at a time
        and taking the *k* shortest results.

        When *postcondition* is provided, candidates whose branch decisions
        violate the required branch direction are pruned before ranking
        (mirrors the filter in :meth:`compute_topk`).
        """
        exclusive_groups = self._extract_equality_exclusive_groups(root_branches)

        # Phase 1: per-branch shortest sub-path
        per_branch: list[tuple[list[PathBranchDecision], PathMetrics]] = []
        for node in root_branches:
            shortest = self._shortest_sub_path(node)
            per_branch.append(shortest)

        # Base candidate: union of all per-branch shortest decisions
        base_decisions: list[PathBranchDecision] = []
        base_metrics = PathMetrics()
        for decs, mets in per_branch:
            base_decisions.extend(decs)
            base_metrics = PathMetrics(
                S=base_metrics.S + mets.S,
                C=base_metrics.C + mets.C,
                V=base_metrics.V + mets.V,
            )

        candidates = [(base_decisions, base_metrics)]

        # Phase 2: flip one branch at a time to generate variants
        max_flips = min(k * 3, len(root_branches))
        for i in range(max_flips):
            alt_decisions: list[PathBranchDecision] = []
            alt_metrics = PathMetrics()
            for j, node in enumerate(root_branches):
                if j == i:
                    flipped = self._flip_branch(node, per_branch[j][0])
                    alt_decisions.extend(flipped[0])
                    alt_metrics = PathMetrics(
                        S=alt_metrics.S + flipped[1].S,
                        C=alt_metrics.C + flipped[1].C,
                        V=alt_metrics.V + flipped[1].V,
                    )
                else:
                    alt_decisions.extend(per_branch[j][0])
                    alt_metrics = PathMetrics(
                        S=alt_metrics.S + per_branch[j][1].S,
                        C=alt_metrics.C + per_branch[j][1].C,
                        V=alt_metrics.V + per_branch[j][1].V,
                    )
            # Only add if different from base
            if alt_decisions != base_decisions:
                candidates.append((alt_decisions, alt_metrics))

        # Prune candidates that violate the postcondition's required branch
        # direction, so the greedy top-k always respect the constraint.
        if postcondition is not None:
            pc_line = getattr(postcondition, "source_line", 0)
            pc_taken = getattr(postcondition, "branch_taken", None)
            if pc_line and pc_taken is not None:
                filtered: list = []
                for cand in candidates:
                    decisions, _metrics = cand
                    violation = any(
                        getattr(d, "source_line", 0) == pc_line
                        and getattr(d, "branch_taken", True) != pc_taken
                        for d in decisions
                    )
                    if not violation:
                        filtered.append(cand)
                if filtered:
                    logger.info(
                        "compute_topk: postcondition filter pruned "
                        "%d/%d greedy candidates (L%d must be %s)",
                        len(candidates) - len(filtered), len(candidates),
                        pc_line,
                        "TRUE" if pc_taken else "FALSE",
                    )
                    candidates = filtered
                else:
                    logger.warning(
                        "compute_topk: postcondition filter pruned ALL "
                        "%d greedy candidates (L%d must be %s) — all fallback",
                        len(candidates), pc_line,
                        "TRUE" if pc_taken else "FALSE",
                    )

        # Prune mutually-exclusive branch combinations (same var == different
        # consts cannot both be true) so greedy top-k respect exclusivity too.
        if exclusive_groups:
            kept, pruned = self._filter_mutually_exclusive(
                candidates, exclusive_groups,
            )
            if pruned:
                logger.info(
                    "compute_topk: mutual-exclusion pruned %d/%d greedy "
                    "candidates (%d exclusive group(s))",
                    pruned, len(candidates), len(exclusive_groups),
                )
                if kept:
                    candidates = kept

        candidates.sort(key=lambda x: (x[1].S, x[1].C, x[1].V))
        logger.debug(
            "Greedy: %d root branches → %d candidates (top-%d returned).",
            len(root_branches), len(candidates), k,
        )
        return candidates[:k]

    def _shortest_sub_path(
        self, node: CFGBranchNode,
    ) -> tuple[list[PathBranchDecision], PathMetrics]:
        """Return the single shortest sub-path through *node*."""
        tag = node.tag
        verdict = tag.verdict if tag else "insufficient_info"
        stmts_here = self._estimate_stmts(node)
        if node.callee_name:
            stmts_here += self._callee_cache.get_min_stmts(node.callee_name)

        if verdict == "contradictory":
            return ([], PathMetrics())

        sub_paths = self._enumerate_node_paths(node, verdict, stmts_here)
        if not sub_paths:
            return ([], PathMetrics())

        # Pick shortest by (S, C, V)
        sub_paths.sort(key=lambda x: (x[1].S, x[1].C, x[1].V))
        return sub_paths[0]

    def _flip_branch(
        self, node: CFGBranchNode, original: list[PathBranchDecision],
    ) -> tuple[list[PathBranchDecision], PathMetrics]:
        """Return sub-path with the opposite direction for *node*."""
        tag = node.tag
        verdict = tag.verdict if tag else "insufficient_info"
        if verdict == "consistent":
            # Consistent → can't flip
            return self._shortest_sub_path(node)

        stmts_here = self._estimate_stmts(node)
        if node.callee_name:
            stmts_here += self._callee_cache.get_min_stmts(node.callee_name)

        sub_paths = self._enumerate_node_paths(node, verdict, stmts_here)
        if len(sub_paths) < 2:
            return sub_paths[0] if sub_paths else ([], PathMetrics())

        # Sort and pick the one with opposite direction
        sub_paths.sort(key=lambda x: (x[1].S, x[1].C, x[1].V))
        for decs, mets in sub_paths:
            if decs and decs[0].branch_taken != (original[0].branch_taken if original else True):
                return (decs, mets)
        # Fallback: return the first (shortest)
        return sub_paths[0]

    # ── DFS collection ──────────────────────────────────────────────────────

    def _dfs_collect(
        self,
        nodes: list[CFGBranchNode],
        idx: int,
        path_so_far: list[PathBranchDecision],
        metrics_so_far: PathMetrics,
        candidates: list[tuple[list[PathBranchDecision], PathMetrics]],
    ) -> None:
        """DFS over the branch tree, collecting complete paths.

        At each branch node, prunes contradictory branches, auto-confirms
        consistent branches, and explores both sides for insufficient_info.

        Children are traversed inline — sub-paths through children are
        enumerated, then siblings are visited afterward.

        Stops early when ``len(candidates) >= self._MAX_CANDIDATES`` to
        prevent combinatorial explosion.
        """
        # Safety cap — stop enumerating
        if len(candidates) >= self._MAX_CANDIDATES:
            return

        if idx >= len(nodes):
            candidates.append((list(path_so_far), PathMetrics(
                S=metrics_so_far.S,
                C=metrics_so_far.C,
                V=metrics_so_far.V,
            )))
            return

        node = nodes[idx]
        tag = node.tag
        verdict = tag.verdict if tag else "insufficient_info"

        # Estimate statement count for this node's blocks
        stmts_here = self._estimate_stmts(node)

        # Cross-procedure: add callee shortest-path statement count
        if node.callee_name:
            stmts_here += self._callee_cache.get_min_stmts(node.callee_name)

        if verdict == "contradictory":
            # Prune — skip this branch node entirely, continue to siblings
            self._dfs_collect(
                nodes, idx + 1,
                path_so_far, metrics_so_far,
                candidates,
            )
            return

        # Enumerate all sub-paths through this node (including children)
        sub_paths = self._enumerate_node_paths(node, verdict, stmts_here)

        for sub_decisions, sub_metrics in sub_paths:
            if len(candidates) >= self._MAX_CANDIDATES:
                return
            self._dfs_collect(
                nodes, idx + 1,
                path_so_far + sub_decisions,
                PathMetrics(
                    S=metrics_so_far.S + sub_metrics.S,
                    C=metrics_so_far.C + sub_metrics.C,
                    V=metrics_so_far.V + sub_metrics.V,
                ),
                candidates,
            )

    def _enumerate_node_paths(
        self,
        node: CFGBranchNode,
        verdict: str,
        stmts_here: int,
    ) -> list[tuple[list[PathBranchDecision], PathMetrics]]:
        """Enumerate all sub-paths through a branch node, including children.

        For consistent branches, only one direction is explored.
        For insufficient_info branches, both True and False are explored.
        Children are recursively enumerated via ``_dfs_collect``.
        """
        branch_vars = self._extract_variables(node)
        base_metrics = PathMetrics(
            S=stmts_here,
            C=(1 if node.callee_name else 0),
            V=len(branch_vars),
        )

        # Enumerate paths through children (if any)
        if node.children:
            child_candidates: list[tuple[list[PathBranchDecision], PathMetrics]] = []
            self._dfs_collect(
                node.children, 0,
                [], PathMetrics(),
                child_candidates,
            )
            child_paths = child_candidates
        else:
            child_paths = [([], PathMetrics())]

        # Determine which branch directions to explore
        if verdict == "consistent":
            directions = [self._resolve_consistent_direction(node)]
        else:
            directions = [True, False]

        results: list[tuple[list[PathBranchDecision], PathMetrics]] = []
        for direction in directions:
            decision = PathBranchDecision(
                node_id=node.node_id,
                branch_taken=direction,
                condition_expr=node.condition_expr,
                source_line=node.source_line,
            )
            for child_decisions, child_metrics in child_paths:
                results.append(
                    ([decision] + child_decisions,
                     PathMetrics(
                         S=base_metrics.S + child_metrics.S,
                         C=base_metrics.C + child_metrics.C,
                         V=base_metrics.V + child_metrics.V,
                     ))
                )
        return results

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_consistent_direction(node: CFGBranchNode) -> bool:
        """Determine which branch direction is consistent with the postcondition.

        For consistent-tagged branches, we use the node's ``is_taken_branch``
        as the resolved direction (consistent = follow the confirmed branch).
        """
        if node.tag and node.tag.verdict == "consistent":
            return node.is_taken_branch
        return node.is_taken_branch  # fallback

    @classmethod
    def _demote_conflicting_consistent(
        cls,
        roots: list[CFGBranchNode],
        pc_line: int,
        pc_taken: bool,
    ) -> None:
        """Demote "consistent" branches whose resolved direction conflicts
        with the postcondition to "insufficient_info", so the DFS explores
        both directions and the postcondition filter has enough candidates.

        Operates recursively on *roots* and their children in-place.
        """
        for node in roots:
            if (
                node.tag
                and node.tag.verdict == "consistent"
                and node.source_line == pc_line
                and node.is_taken_branch != pc_taken
            ):
                from llm_client.fp_analysis.cfg_branch_models import (
                    BranchTag,
                )
                old_layer = node.tag.layer if node.tag else 0
                node.tag = BranchTag(
                    branch_id=node.node_id,
                    verdict="insufficient_info",
                    layer=old_layer,
                    reason="demoted: consistent direction conflicts "
                           "with segment postcondition",
                )
                logger.debug(
                    "Demoted consistent→insufficient_info for L%d: "
                    "resolved=%s conflicts with postcondition=%s",
                    pc_line,
                    "TRUE" if node.is_taken_branch else "FALSE",
                    "TRUE" if pc_taken else "FALSE",
                )
            if node.children:
                cls._demote_conflicting_consistent(
                    node.children, pc_line, pc_taken,
                )

    @staticmethod
    def _extract_variables(node: CFGBranchNode) -> set[str]:
        """Extract distinct variable names from branch condition expressions."""
        import re
        vars_found: set[str] = set()
        for expr in (node.condition_expr, node.resolved_expr):
            if not expr:
                continue
            # Simple identifier extraction
            for match in re.finditer(r'\b([a-zA-Z_]\w*)\b', expr):
                word = match.group(1)
                # Filter out C++ keywords and common operators
                if word.lower() in {'if', 'else', 'for', 'while', 'return',
                                     'true', 'false', 'null', 'nullptr',
                                     'sizeof', 'int', 'char', 'void', 'bool',
                                     'auto', 'const', 'static', 'struct', 'class'}:
                    continue
                vars_found.add(word)
        return vars_found

    def _estimate_stmts(self, node: CFGBranchNode) -> int:
        """Estimate the number of C/C++ statements in this branch node's blocks.

        Queries the PDG to count non-auxiliary nodes in the blocks referenced
        by *node.blocks_taken* and *node.blocks_not_taken*.

        Falls back to 1 (conservative minimum) when PDG is unavailable.
        """
        fn_pdg = self._get_pdg(node.function_name)
        if fn_pdg is None:
            return 1

        target_blocks: set[str] = set()
        if node.blocks_taken:
            target_blocks.update(node.blocks_taken)
        if node.blocks_not_taken:
            target_blocks.update(node.blocks_not_taken)

        if not target_blocks:
            return 1

        count = 0
        pdg_nodes = getattr(fn_pdg, "nodes", {})
        for nid, pnode in pdg_nodes.items():
            block_id = getattr(pnode, 'block_id', None)
            if block_id is not None and str(block_id) in target_blocks:
                kind = getattr(pnode, 'kind', '')
                if kind not in _AUXILIARY_KINDS:
                    count += 1

        return max(count, 1)  # at least 1 for the branch itself

    def _get_pdg(self, fn_name: str) -> object | None:
        """Look up a function's PDG via the SDG."""
        sdg = self._sdg
        if sdg is None:
            return None
        if hasattr(sdg, "get_pdg"):
            return sdg.get_pdg(fn_name)
        return None

    # ── Mutual-exclusion pruning ────────────────────────────────────────────

    @staticmethod
    def _extract_equality_exclusive_groups(
        root_nodes: list[CFGBranchNode],
    ) -> list[list[CFGBranchNode]]:
        """Detect mutually-exclusive equality branch groups.

        Two branches are mutually exclusive when, within the same function,
        they compare the **same variable** to **different constants** via
        ``==`` (e.g. ``address->sa_family == 2`` and ``address->sa_family
        == 10``) — a variable can only equal one constant at a time.  A
        candidate that takes two of these as ``true`` is infeasible.

        Returns a list of groups; each group is the list of branch nodes
        (across ≥2 distinct constants for the same variable) that are
        mutually exclusive.  Branches whose condition is not a simple
        variable/``==``/numeric-constant form are ignored (under-prune is
        the safe direction).
        """
        # Collect all branch nodes in the tree.
        nodes: list[CFGBranchNode] = []
        stack = list(root_nodes or [])
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(getattr(node, "children", None) or [])

        by_var: dict[tuple, dict] = {}  # (fn, lhs) -> {const: node}
        for node in nodes:
            expr = (getattr(node, "condition_expr", "") or "").strip()
            if not expr:
                expr = (getattr(node, "resolved_expr", "") or "").strip()
            lhs, sep, rhs = expr.partition("==")
            if not sep:
                continue
            lhs = lhs.strip()
            rhs = rhs.strip()
            if not lhs or not rhs:
                continue
            # lhs must be a simple variable / member access (exclude calls,
            # string literals, etc. — e.g. ``strcmp(a,b)==0`` has a comma/paren).
            if any(ch in lhs for ch in ('"', ",", "(", ")")):
                continue
            # rhs must be a numeric constant.
            try:
                const = int(rhs, 0)
            except ValueError:
                continue
            key = (getattr(node, "function_name", None) or "", lhs)
            by_var.setdefault(key, {})[const] = node

        groups: list[list[CFGBranchNode]] = []
        for var_group in by_var.values():
            if len(var_group) >= 2:
                groups.append(list(var_group.values()))
        return groups

    @staticmethod
    def _filter_mutually_exclusive(
        candidates: list[tuple[list[PathBranchDecision], PathMetrics]],
        groups: list[list[CFGBranchNode]],
    ) -> tuple[list, int]:
        """Drop candidates that take ≥2 true branches in one exclusive group.

        Returns ``(kept, pruned)``.  Sound: a variable can only equal one
        constant, so forcing two distinct equality branches true is infeasible.
        A candidate with 0 or 1 true member per group is retained (the variable
        may legitimately take some other value).
        """
        if not groups:
            return candidates, 0

        # node_id → group index
        node_group: dict[str, int] = {}
        for gi, group in enumerate(groups):
            for node in group:
                node_group[getattr(node, "node_id", None)] = gi

        kept: list = []
        pruned = 0
        for cand in candidates:
            decisions, _metrics = cand
            counts: dict[int, int] = {}
            conflict = False
            for d in decisions:
                gid = node_group.get(getattr(d, "node_id", None))
                if gid is None or not getattr(d, "branch_taken", True):
                    continue
                counts[gid] = counts.get(gid, 0) + 1
                if counts[gid] >= 2:
                    conflict = True
                    break
            if conflict:
                pruned += 1
            else:
                kept.append(cand)
        return kept, pruned
