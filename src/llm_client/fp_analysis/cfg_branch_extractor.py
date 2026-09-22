"""CFG Branch Extraction (Step 3) — segment-level branch tree construction.

Given a segment, the PDG (primary) or CFG cache (fallback), and the SliceMask
from PDG backward slicing, this module extracts all branch nodes within the
segment's source line range, including cross-procedural callee branches up to
a configurable depth.

Two extraction strategies:
- **PDG-based (primary)**: Extracts ``binary_op`` / ``unary_op`` nodes that
  are sources of ``control_dep_edges``.  Covers ALL 24K+ functions in the
  PDG, including callees from different source files.
- **CFG-based (fallback)**: Used when PDG is unavailable for a function.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from llm_client.fp_analysis.cfg_cache_set import _method_component
from llm_client.fp_analysis.cfg_branch_models import (
    CFGBranchNode,
    CFGBranchTree,
    Postcondition,
)
from llm_client.fp_analysis.cfg_models import CFGBlock, CFGFunction
from llm_client.fp_analysis.slice_mask import SliceMask

logger = logging.getLogger(__name__)

# Regex patterns for CFG dump terminator parsing.
# CFG dump format: "if [B3.1] (BinaryOperator) r >= 0"
# We extract everything after the parenthesized type annotation.
_TERMINATOR_IF_RE = re.compile(
    r"\bif\b"
)
_TERMINATOR_WHILE_RE = re.compile(
    r"\bwhile\b"
)
_TERMINATOR_FOR_RE = re.compile(
    r"\bfor\b"
)
# Extracts the trailing condition expression after "[...] (Type) "
_COND_AFTER_TYPE_RE = re.compile(
    r"\)\s*(.+)$"
)

# Regex for extracting line numbers from statement/terminator text
_LINE_NUMBER_RE = re.compile(r"(?:^|\s)(\d+):\s+")

# Path reachability (A) reads ``node.block_id`` off the PDG and ``B<n>`` ids off
# the CFG cache.  Those two come from *different* Clang runs and therefore do
# not always agree (see ``_alignment_score``), so A verifies the agreement
# before it prunes anything: at least ``_MIN_ALIGNED_BLOCKS`` blocks must be
# comparable and at least ``_MIN_ALIGNMENT`` of them must agree.
_MIN_ALIGNED_BLOCKS = 5
_MIN_ALIGNMENT = 0.7


def _norm_expr(text: str) -> str:
    """Whitespace-free form of an expression, for cross-dump comparison."""
    return re.sub(r"\s+", "", text or "")


class CFGBranchExtractor:
    """Extract a branch tree for one segment from PDG (primary) or CFG (fallback).

    The extractor works within the constraints of the PDG backward slice.
    Cross-procedural extraction uses the PDG call graph to recursively
    extract branches from callee functions.

    Args:
        cfg_cache: ``{function_name: CFGFunction}`` — pre-loaded CFG data
            (used as fallback when PDG is unavailable for a function).
        sdg: System Dependence Graph for PDG queries.
        slice_mask: Post-slice retained/removed node tracking.
        max_depth: Maximum callee extraction depth (default 3).
        path_annotations: ``{(qualified_fn, line): branch_taken}`` describing
            which direction each predicate line took on the *reported* CSA path
            (:func:`build_dense_path_annotations`).  Enables path-reachability
            scoping: a segment's line range routinely spans the then-body of a
            branch the report took as false, and expanding those calls is the
            single largest source of phantom branches.  ``None`` disables the
            filtering entirely.
        shallow_callee_depth: Callee expansions at or below this depth happen
            regardless of slice retention, so a segment's own call sites keep
            their first-level branch conditions visible to conflict checking.
            Deeper callees must be slice-retained to be expanded at all.
        source_root: Project root; a callee whose PDG lives outside it (a
            system header such as ``bits/stl_vector.h``) is never expanded
            beyond its own predicates.
    """

    def __init__(
        self,
        cfg_cache: dict[str, CFGFunction],
        sdg: object,
        slice_mask: SliceMask,
        max_depth: int = 3,
        path_annotations: dict | None = None,
        shallow_callee_depth: int = 1,
        source_root: str = "",
    ) -> None:
        self._cfg_cache = cfg_cache
        self._sdg = sdg
        self._slice_mask = slice_mask
        self._max_depth = max_depth
        self._shallow_callee_depth = shallow_callee_depth
        self._source_root = source_root
        # {(qualified_fn, line): branch_taken} → {qualified_fn: {line: taken}}
        self._decisions_by_fn: dict[str, dict[int, bool]] = {}
        for key, taken in (path_annotations or {}).items():
            if isinstance(key, tuple) and len(key) == 2:
                self._decisions_by_fn.setdefault(str(key[0]), {})[int(key[1])] = bool(taken)
        # On-path source lines per function: ``{fn_key: set(line)}``; a ``None``
        # value means "could not be proven — do not filter this function".
        self._on_path_lines_cache: dict[str, frozenset[int] | None] = {}
        # Callees currently being expanded on this branch of the recursion, so a
        # mutual-recursion cycle (read_index ↔ read_ivf_header) terminates by
        # itself instead of relying on ``max_depth``.
        self._expansion_stack: tuple[str, ...] = ()

        # ── PDG-based caches ──
        # Callee branch cache: {callee_pdg_fn_name: list[CFGBranchNode]}
        # Avoids re-extracting the same callee's branches from PDG.
        # Keyed by (function, depth, nested_ok, expansion stack) — gating makes
        # the answer context-sensitive, so the function name alone is not enough.
        self._callee_pdg_cache: dict[tuple, list[CFGBranchNode]] = {}
        # Branch source nodes per PDG: {fn_name: set(node_id)}
        # A "branch source" is a binary_op/unary_op node that is a source
        # of a control_dep_edge.  Built lazily in _get_branch_sources().
        self._branch_sources_cache: dict[str, set[str]] = {}
        # Source lines per PDG: {fn_name: set(source_line)} — lets a segment's
        # (file, line range) verify/replace a name-only PDG resolution.
        self._pdg_lines_cache: dict[str, set[int]] = {}

        # ── CFG-based caches (fallback) ──
        # Phase 2: CFG global index for cross-block element reference resolution.
        self._cfg_index: dict[str, dict[int, str]] = {}
        self._cfg_array_order: dict[str, list[int]] = {}
        # PDG-based block → source line mapping
        self._block_source_lines: dict[str, dict[int, set[int]]] = {}
        # Callee branch cache (CFG): {callee_cfg_key: list[CFGBranchNode]}
        self._callee_branch_cache: dict[str, list[CFGBranchNode]] = {}
        # PDG call index cache: {sdg_fn_name: {block_id: [(callee, source_line)]}}
        self._pdg_call_index_cache: dict[str, dict[int, list[tuple[str, int]]]] = {}

    # ─── Public API ───────────────────────────────────────────────────────────

    def extract(
        self,
        segment_index: int,
        function_name: str,
        line_start: int,
        line_end: int,
        postcondition: Postcondition,
        segmented_callees: set[str] | None = None,
        function_file: str = "",
    ) -> CFGBranchTree:
        """Extract the branch tree for one segment.

        Two-phase extraction:

        1. **Intra-procedural** — extract branch nodes from PDG (or CFG fallback)
           within the segment's ``[line_start, line_end]``.
        2. **Cross-procedural** — for each function call site within the segment
           (detected via PDG ``is_call`` nodes), recursively extract branches from
           the callee's PDG.  Results are cached per callee.

        PDG-based extraction is the primary strategy; CFG is only used as
        fallback when a function has no PDG data.

        Callee branches are wrapped in sentinel nodes with ``callee_name`` set,
        distinguishing them from intra-procedural branches (``call_depth == 0``).

        Args:
            segment_index: Zero-based segment index.
            function_name: Function containing the segment code.
            line_start / line_end: Source line range of the segment.
            postcondition: Postcondition for branch filtering.
            segmented_callees: Set of function names that have their own
                explicit segments in the CSA trace.  When a callee called
                within this segment appears in this set, its internal
                branches are NOT recursively expanded — they belong to
                the callee's own segment(s) and will be extracted there.
            function_file: Source file the segment lives in.  Together with
                the line range it disambiguates same-named functions, which a
                bare ``function_name`` alone cannot (see
                :meth:`resolve_segment_pdg`).
        """
        tree = CFGBranchTree(
            segment_index=segment_index,
            function_name=function_name,
            line_start=line_start,
            line_end=line_end,
            postcondition=postcondition,
        )

        # ── Primary path: PDG-based extraction ──
        # Only when the PDG actually says something about this range.  A PDG
        # that resolves but carries no branch/call node in
        # ``[line_start, line_end]`` is no better than no PDG at all — and it
        # is worse than the CFG, which is range-aware and whose block
        # terminators still hold the macro-expanded conditions the PDG builder
        # did not turn into control-dependence edges (``clone_IndexIVF``'s
        # ``TRYCLONE`` dynamic_casts are exactly that case: the PDG resolves,
        # but with zero CD edges, while the CFG has the four real branches).
        pdg = self.resolve_segment_pdg(
            function_name, function_file, line_start, line_end)
        if pdg is not None and self._pdg_yields(pdg, line_start, line_end):
            self._extract_via_pdg(
                tree=tree,
                pdg=pdg,
                fn_name=pdg.function_name,
                line_start=line_start,
                line_end=line_end,
                depth=0,
                segmented_callees=segmented_callees,
                segment_index=segment_index,
            )
            return tree

        # ── Fallback: CFG-based extraction (existing logic) ──
        fn_cfg = self._cfg_cache.get(function_name)
        if fn_cfg is None:
            fn_cfg = self._resolve_cfg_by_short_name(function_name)
        if fn_cfg is None:
            logger.debug(
                "CFG not found for %s (segment %d)", function_name, segment_index,
            )
            return tree

        retained_blocks = self._get_retained_cfg_blocks(function_name)
        filtered = self._filter_blocks_by_range(
            fn_cfg, retained_blocks, line_start, line_end,
        )

        self._build_cfg_index(fn_cfg)

        cfg_fn_key = fn_cfg.function_name
        pdg_call_index = self._get_pdg_call_index(cfg_fn_key)
        caller_short = fn_cfg.function_name.split("(")[0].strip().split()[-1] if "(" in fn_cfg.function_name else fn_cfg.function_name

        # Phase 1: intra-procedural
        seen_calls: set[tuple[str, int]] = set()
        for block_id in filtered:
            branch = self._extract_from_block(
                cfg=fn_cfg, block_id=block_id, fn_name=function_name,
                depth=0, parent_call=None,
            )
            if branch is not None:
                # Exclude boundary line when there is a previous segment.
                if segment_index > 0 and branch.source_line == line_start:
                    continue
                tree.add_branch(branch)

            bid_num = self._parse_block_id(block_id)
            for callee_name, call_line in pdg_call_index.get(bid_num, []):
                seen_calls.add((callee_name, call_line))

        # Phase 2: cross-procedural (CFG)
        for callee_name, call_line in sorted(seen_calls, key=lambda x: x[1]):
            # Skip recursive expansion if this callee has its own segment(s)
            if segmented_callees and callee_name in segmented_callees:
                continue
            # Exclude boundary line when there is a previous segment.
            if segment_index > 0 and call_line == line_start:
                continue
            callee_cfg = self._resolve_cfg_by_short_name(
                callee_name, caller_fn=caller_short,
                caller_file=fn_cfg.source_file or "", call_line=call_line,
            )
            if callee_cfg is None:
                continue
            callee_branches = self._extract_callee_branches(
                callee_cfg=callee_cfg, callee_name=callee_name, depth=1,
            )
            if callee_branches:
                sentinel = CFGBranchNode(
                    node_id=f"callee_{function_name}_{callee_name}_L{call_line}",
                    source_line=call_line,
                    function_name=function_name,
                    condition_expr=f"→ {callee_name}()",
                    resolved_expr="",
                    call_depth=0,
                    callee_name=callee_name,
                )
                for child in callee_branches:
                    sentinel.children.append(child)
                tree.add_branch(sentinel)

        return tree

    # ─── PDG-based branch extraction (primary path) ────────────────────────────
    #
    # PDG nodes of kind "binary_op" / "unary_op" that appear as *sources* of
    # control_dep_edges represent if/while/for branch conditions.  We extract
    # them directly from PDG, which covers ALL 24K+ functions without needing
    # per-file CFG compilation.
    #
    # Only _BRANCH_CONDITION_KINDS nodes are included — `assign`, `decl`, and
    # `call_expr` nodes that are control-dep sources (by virtue of being
    # positionally at the top of a branch block) are ignored since they don't
    # represent actual branch conditions.
    # ───────────────────────────────────────────────────────────────────────────

    _BRANCH_CONDITION_KINDS = frozenset({"binary_op", "unary_op", "call_expr"})

    def _resolve_pdg(self, fn_name: str) -> "FunctionPDG | None":
        """Resolve a function name to its PDG via the SDG.

        Handles short names (``"wslay_frame_recv"``), full signatures, and
        partial matches through :meth:`SDG.get_pdg`.
        """
        sdg = self._sdg
        if sdg is None:
            return None
        pdgs = getattr(sdg, "pdgs", {})
        if not pdgs:
            return None

        # Extract the unqualified short name first (e.g. "setFromLocalAddr"
        # from "void SocketAddress::setFromLocalAddr(const ...)").  Template
        # argument lists and the return type's pointer/reference marker must be
        # removed as well, otherwise a CFG key like
        # "template<> inline void heap_push<faiss::CMax<float, long>>(...)"
        # yields the nonsense short name "long>>" and the substring levels below
        # match an arbitrary unrelated function.
        short = _method_component(fn_name)

        # Prefer SDG.get_pdg on the SHORT name first — it resolves via the
        # exact last `::` component, so "setFromLocalAddr" unambiguously maps
        # to "folly::SocketAddress::setFromLocalAddr" and NOT to the
        # substring-colliding "folly::SocketAddress::setFromLocalAddress".
        # get_pdg(fn_name) (full signature) usually misses, which is why the
        # plain substring levels below used to hijack the wrong overload.
        for cand in (short, fn_name):
            if not cand:
                continue
            pdg = sdg.get_pdg(cand)
            if pdg is not None:
                return pdg

        # Fallback: fuzzy match — fn_name may be a CFG signature like
        # "int wslay_event_recv(...)" while PDG keys use short names.
        #
        # Resolution order (most → least specific):
        #   1. Exact short-name match (e.g. "wslay_event_queue_fragmented_msg_ex")
        #   2. Short name is a substring of a PDG key (key is more qualified)
        #   3. PDG key is a substring of short name (e.g. suffix "_ex" variants
        #      where "queue_fragmented_msg" ⊂ "queue_fragmented_msg_ex")
        # Level 1: exact short-name match
        if short in pdgs:
            return pdgs[short]

        # Level 2: short ⊂ key (PDG key is more qualified)
        for key in pdgs:
            if short in key:
                return pdgs[key]

        # Level 3: key ⊂ short (suffix variants — least specific)
        for key in pdgs:
            if key in short:
                return pdgs[key]

        return None

    # ─── Segment-aware PDG resolution (file + line evidence) ─────────────────

    def _pdg_lines(self, pdg) -> set[int]:
        """Source lines this PDG has nodes on (cached per PDG)."""
        cached = self._pdg_lines_cache.get(pdg.function_name)
        if cached is None:
            cached = {n.source_line for n in pdg.nodes.values()
                      if n.source_line > 0}
            self._pdg_lines_cache[pdg.function_name] = cached
        return cached

    def _coverage(self, pdg, line_start: int, line_end: int) -> int:
        """How many of the segment's lines this PDG actually has nodes on."""
        if pdg is None:
            return 0
        return sum(1 for ln in self._pdg_lines(pdg)
                   if line_start <= ln <= line_end)

    def _best_file_local_pdg(
        self, function_name: str, function_file: str,
        line_start: int, line_end: int,
    ):
        """PDG of the function in *function_file* covering the segment's lines.

        Restricted to the segment's own file, which is what rules out the
        libstdc++/other-TU functions that share the bare method name (``search``
        → ``std::search``, ``clear`` → ``std::_Hashtable::clear``).
        """
        sdg = self._sdg
        if sdg is None or not function_file:
            return None
        try:
            file_index = sdg.build_file_index()
        except AttributeError:
            return None
        candidates = list(file_index.get(str(function_file), ()))
        if not candidates:
            # Tolerate path spelling differences (build-time vs local root).
            want = Path(function_file).name
            candidates = [pdg for sf, pdgs in file_index.items()
                          if Path(sf).name == want for pdg in pdgs]
        short = _method_component(function_name) if function_name else ""
        best, best_key = None, None
        for pdg in candidates:
            coverage = self._coverage(pdg, line_start, line_end)
            if not coverage:
                continue
            # Prefer a candidate whose method name matches the segment's name,
            # then the one with the most nodes inside the range; the name
            # breaks ties so the choice does not depend on dict order.
            key = (1 if short and _method_component(pdg.function_name) == short
                   else 0, coverage)
            if best_key is None or key > best_key or (
                    key == best_key and pdg.function_name < best.function_name):
                best, best_key = pdg, key
        return best

    def _pdg_yields(self, pdg, line_start: int, line_end: int) -> bool:
        """Does *pdg* have a branch condition or a call site in this range?

        Used to decide whether the PDG path is worth taking at all; see
        :meth:`extract`.
        """
        for node_id in self._get_branch_sources(pdg):
            node = pdg.nodes.get(node_id)
            if node is not None and line_start <= node.source_line <= line_end:
                return True
        return bool(self._find_calls_in_pdg_range(pdg, line_start, line_end))

    def resolve_segment_pdg(
        self, function_name: str, function_file: str,
        line_start: int, line_end: int,
    ):
        """Resolve the PDG of the function a segment really lives in.

        A segment only carries a bare method name (``search``, ``read_index``,
        ``heap_push``, or — for events inside a macro expansion — the macro's
        own name, or nothing at all).  Resolving that name alone is ambiguous on
        any real project, so the name-based answer is *verified against the
        segment's own (file, line range)*:

        * accepted only when the PDG has nodes inside the range, and
        * otherwise replaced by the best covering PDG from the same file.

        Returns ``None`` only when the name itself resolves to nothing; a PDG
        that no line evidence supports is still returned as a last resort (the
        PDG may simply lack source-file metadata), leaving it to :meth:`extract`
        to decide whether it is worth using.
        """
        named = self._resolve_pdg(function_name) if function_name else None
        if named is not None and self._coverage(named, line_start, line_end) > 0:
            return named
        local = self._best_file_local_pdg(
            function_name, function_file, line_start, line_end)
        if local is not None:
            if named is not None:
                logger.warning(
                    "segment L%d-L%d in %s: name %r resolved to %s, which has "
                    "no nodes in that range — using %s instead",
                    line_start, line_end, Path(function_file).name,
                    function_name, named.function_name, local.function_name,
                )
            return local
        return named

    def _extract_via_pdg(
        self,
        tree: CFGBranchTree,
        pdg: "FunctionPDG",
        fn_name: str,
        line_start: int,
        line_end: int,
        depth: int,
        segmented_callees: set[str] | None = None,
        segment_index: int = 0,
    ) -> None:
        """Core PDG-based extraction: intra-procedural branches + cross-procedural calls.

        Args:
            tree: The branch tree to populate.
            pdg: The function's PDG.
            fn_name: Short function name (for display).
            line_start / line_end: Source line range to scope extraction.
            depth: Current call depth (0 = main function).
            segmented_callees: Set of function names that have their own
                segments.  Callees in this set are not recursively expanded
                — their branches belong to their own segment(s).
            segment_index: Zero-based segment index.  When > 0, branches
                and call sites at *line_start* are excluded — they belong
                to the previous segment's boundary analysis.
        """
        # ── Phase 1: intra-procedural branch conditions ──
        branch_sources = self._get_branch_sources(pdg)
        seen_calls: list[tuple[str, int]] = []  # (callee_name, call_line)
        on_path = (
            self._on_path_lines_for_segment(pdg.function_name, line_end)
            if depth == 0 else None
        )

        for node_id in sorted(branch_sources):
            node = pdg.nodes.get(node_id)
            if node is None:
                continue
            # Filter by segment line range
            if not (line_start <= node.source_line <= line_end):
                continue
            # Exclude boundary line when there is a previous segment:
            # branches/calls at line_start belong to the previous segment.
            if segment_index > 0 and depth == 0 and node.source_line == line_start:
                continue
            # (A) the reported path never reaches this line: the predicate is
            # dead code for this bug, whatever it says.
            if on_path is not None and node.source_line not in on_path:
                continue

            branch_node = CFGBranchNode(
                node_id=f"pdg_{fn_name}_{node_id}",
                source_line=node.source_line,
                function_name=fn_name,
                condition_expr=node.expression,
                resolved_expr=node.expression,
                call_depth=depth,
            )
            tree.add_branch(branch_node)

        # ── Phase 2: cross-procedural calls ──
        calls = self._find_calls_in_pdg_range(pdg, line_start, line_end)
        for callee_name, call_line in calls:
            # Exclude boundary line when there is a previous segment.
            if segment_index > 0 and depth == 0 and call_line == line_start:
                continue
            # (A) the call sits in a branch the reported path did not take —
            # expanding it is what put 89.7% of segment 21's nodes under
            # ``read_ivf_header`` at L825, inside the skipped ``IwSh`` body.
            if on_path is not None and call_line not in on_path:
                continue
            # Skip recursive expansion if this callee has its own segment(s).
            # The callee's internal branches are handled by its own segments.
            if segmented_callees and callee_name in segmented_callees:
                continue
            callee_pdg = self._resolve_pdg(callee_name)
            if callee_pdg is None:
                logger.debug(
                    "Callee '%s' (called at L%d) not in PDG — skipping",
                    callee_name, call_line,
                )
                continue

            self._add_callee_subtree(
                tree=tree,
                callee_pdg=callee_pdg,
                callee_name=callee_name,
                call_line=call_line,
                parent_fn=fn_name,
                depth=depth,
            )

    def _get_branch_sources(self, pdg: "FunctionPDG") -> set[str]:
        """Return the set of PDG node IDs that are branch condition predicates.

        A node is a "branch source" if:
        - Its ``kind`` is ``binary_op``, ``unary_op``, or ``call_expr``
        - It appears as a ``source_id`` in a ``control_dep_edge``

        When a control_dep source has a non-qualifying kind (``decl``,
        ``event``, etc.), the PDG builder assigned the dependency to the
        first node inside the controlled block rather than to the condition
        expression.  We resolve these by finding the nearest qualifying
        node in the same function.

        Results are cached per PDG function name.
        """
        fn_name = pdg.function_name
        cached = self._branch_sources_cache.get(fn_name)
        if cached is not None:
            return cached

        # Build set of node IDs that are sources of control_dep_edges
        ctrl_source_ids: set[str] = set()
        for cde in pdg.control_dep_edges:
            ctrl_source_ids.add(cde.source_id)

        # Filter to only branch-condition kinds, resolving non-qualifying
        # sources to their nearest qualifying condition expression.
        result: set[str] = set()
        for nid in ctrl_source_ids:
            node = pdg.nodes.get(nid)
            if node is None:
                continue
            if node.kind in self._BRANCH_CONDITION_KINDS:
                result.add(nid)
            else:
                resolved = self._resolve_branch_condition_node(pdg, node)
                if resolved is not None:
                    result.add(resolved)

        self._branch_sources_cache[fn_name] = result
        return result

    def _resolve_branch_condition_node(
        self,
        pdg: "FunctionPDG",
        source_node: "PDGNode",
    ) -> str | None:
        """Find the actual branch condition for a non-qualifying ctrl source.

        When the PDG uses a ``decl`` (e.g. ``int r;`` inside an if-body)
        or ``event`` (e.g. ``ctx->write_enabled`` as the first operand of
        short-circuit ``&&``) as a control_dep source, walk outward to
        find the nearest ``binary_op`` / ``unary_op`` / ``call_expr``
        whose expression represents the actual branch condition.

        Args:
            pdg: The function's PDG.
            source_node: The non-qualifying control_dep source node.

        Returns:
            The node ID of the nearest qualifying condition expression,
            or ``None`` if no suitable node is found within range.
        """
        # Collect candidate nodes: same function, qualifying kind,
        # source_line within ±5 of the source node.
        candidates: list[tuple[int, int, int, "PDGNode"]] = []
        for n in pdg.nodes.values():
            if n.id == source_node.id:
                continue
            if n.kind not in self._BRANCH_CONDITION_KINDS:
                continue
            dist = abs(n.source_line - source_node.source_line)
            if dist > 5:
                continue
            # -len(expr): prefer longer expression (more comprehensive)
            candidates.append((dist, len(n.expression or ""), n.source_line, n))

        if not candidates:
            return None

        # Sort: closest line first, then longest expression, then later line
        candidates.sort(key=lambda x: (x[0], -x[1], -x[2]))
        return candidates[0][3].id

    @staticmethod
    def _find_calls_in_pdg_range(
        pdg: "FunctionPDG",
        line_start: int,
        line_end: int,
    ) -> list[tuple[str, int]]:
        """Find ``(callee_name, source_line)`` for all *is_call* nodes
        in *pdg* whose ``source_line`` falls within ``[line_start, line_end]``.
        """
        result: list[tuple[str, int]] = []
        seen: set[tuple[str, int]] = set()
        for node in pdg.nodes.values():
            if not node.is_call:
                continue
            if not (line_start <= node.source_line <= line_end):
                continue
            entry = (node.callee, node.source_line)
            if entry not in seen:
                seen.add(entry)
                result.append(entry)
        return result

    def _build_callee_node(
        self,
        callee_pdg: "FunctionPDG",
        callee_name: str,
        call_line: int,
        parent_fn: str,
        depth: int,
    ) -> CFGBranchNode | None:
        """The call-edge node for *callee_pdg* called at *call_line*, or ``None``.

        Returns a sentinel whose children are the callee's extracted branches —
        or, when :meth:`_expansion_mode` refuses to go deeper, a *leaf* sentinel
        with no children (and ``None`` when the callee had nothing to say even
        before gating, so gating never adds nodes).
        """
        mode = self._expansion_mode(callee_pdg, depth + 1)
        sentinel = CFGBranchNode(
            node_id=f"callee_{parent_fn}_{callee_name}_L{call_line}",
            source_line=call_line,
            function_name=parent_fn,
            condition_expr=f"→ {callee_name}()",
            resolved_expr="",
            call_depth=depth,
            callee_name=callee_name,
        )

        if mode == "none":
            return sentinel if self._callee_has_content(callee_pdg) else None

        self._expansion_stack = self._expansion_stack + (callee_pdg.function_name,)
        try:
            children = self._extract_callee_pdg_branches(
                callee_pdg=callee_pdg,
                callee_name=callee_pdg.function_name,
                depth=depth + 1,
                nested_ok=(mode == "full"),
            )
        finally:
            self._expansion_stack = self._expansion_stack[:-1]

        if not children:
            return None
        sentinel.children.extend(children)
        return sentinel

    def _add_callee_subtree(
        self,
        tree: CFGBranchTree,
        callee_pdg: "FunctionPDG",
        callee_name: str,
        call_line: int,
        parent_fn: str,
        depth: int,
    ) -> None:
        """Attach the call-edge node for a callee found by the segment scan."""
        node = self._build_callee_node(
            callee_pdg=callee_pdg,
            callee_name=callee_name,
            call_line=call_line,
            parent_fn=parent_fn,
            depth=depth,
        )
        if node is not None:
            tree.add_branch(node)

    def _extract_callee_pdg_branches(
        self,
        callee_pdg: "FunctionPDG",
        callee_name: str,
        depth: int,
        nested_ok: bool = True,
    ) -> list[CFGBranchNode]:
        """Extract ALL branch nodes from *callee_pdg*, including nested calls.

        Results are cached by ``(function, depth, nested_ok, expansion stack)``
        — the answer now depends on all four (gating is depth- and
        context-sensitive), where the old cache keyed on the function alone.

        Args:
            callee_pdg: The callee's PDG.
            callee_name: Short callee name.
            depth: Current call depth (1 = direct callee).
            nested_ok: ``False`` extracts the callee's own predicates only
                (``"shallow"`` mode — see :meth:`_expansion_mode`).

        Returns:
            Flat list of ``CFGBranchNode`` with ``call_depth >= depth``.
        """
        cache_key = (callee_pdg.function_name, depth, nested_ok, self._expansion_stack)
        cached = self._callee_pdg_cache.get(cache_key)
        if cached is not None:
            return cached

        all_branches: list[CFGBranchNode] = []
        on_path = self._on_path_lines(callee_pdg.function_name)

        # ── Step 1: extract all branch conditions from the callee ──
        branch_sources = self._get_branch_sources(callee_pdg)
        for node_id in sorted(branch_sources):
            node = callee_pdg.nodes.get(node_id)
            if node is None:
                continue
            # (A) same reachability rule inside the callee's own CFG.
            if on_path is not None and node.source_line not in on_path:
                continue
            branch_node = CFGBranchNode(
                node_id=f"pdg_{callee_name}_{node_id}",
                source_line=node.source_line,
                function_name=callee_name,
                condition_expr=node.expression,
                resolved_expr=node.expression,
                call_depth=depth,
            )
            all_branches.append(branch_node)

        # ── Step 2: recursively extract nested callee calls ──
        if nested_ok and depth < self._max_depth:
            # Find the call sites the reported path can actually reach — the
            # old unscoped ``0, 999999`` scan pulled in calls from branches the
            # path never took (e.g. 7 of ``read_index``'s 8 ``read_ivf_header``
            # call sites, behind an else-if chain that can only take one).
            nested_calls = [
                (name, line) for name, line
                in self._find_calls_in_pdg_range(callee_pdg, 0, 999999)
                if on_path is None or line in on_path
            ]
            for nested_callee_name, nested_line in nested_calls:
                if nested_callee_name == callee_name:
                    continue  # guard against recursion

                nested_pdg = self._resolve_pdg(nested_callee_name)
                if nested_pdg is None:
                    continue

                nested_node = self._build_callee_node(
                    callee_pdg=nested_pdg,
                    callee_name=nested_callee_name,
                    call_line=nested_line,
                    parent_fn=callee_name,
                    depth=depth,
                )
                if nested_node is None:
                    continue

                # Attach the nested callee's nodes as children to the branch
                # node in the same block, or keep the call-edge node.
                attached = False
                # Find PDG node for this call to get its block_id
                call_node = next(
                    (n for n in callee_pdg.nodes.values()
                     if n.is_call and n.callee == nested_callee_name
                     and n.source_line == nested_line),
                    None,
                )
                call_block_id = call_node.block_id if call_node else -1

                if call_block_id > 0:
                    for branch in all_branches:
                        # Match by block_id: a branch in the same block as the call
                        bnode = callee_pdg.nodes.get(
                            branch.node_id.replace(f"pdg_{callee_name}_", ""),
                        )
                        if bnode is not None and bnode.block_id == call_block_id:
                            branch.children.extend(nested_node.children)
                            attached = True
                            break

                if not attached:
                    all_branches.append(nested_node)

        self._callee_pdg_cache[cache_key] = all_branches
        return all_branches

    # ─── Block-level extraction (CFG fallback) ─────────────────────────────────

    def _extract_from_block(
        self,
        cfg: CFGFunction,
        block_id: str,
        fn_name: str,
        depth: int = 0,
        parent_call: str | None = None,
    ) -> CFGBranchNode | None:
        """Extract branch information from a single CFG block.

        This is a **pure intra-procedural** extractor: it only reads the
        block's terminator.  Cross-procedural callee extraction is handled
        separately by :meth:`_extract_callee_branches` (invoked from
        :meth:`extract` Phase 2).

        Returns ``None`` if the block has no branch terminator (e.g. a
        straight-line block with only assignments).
        """
        block = cfg.blocks.get(block_id)
        if block is None or block.terminator is None:
            return None

        branch_info = self._parse_terminator(block.terminator)
        if branch_info is None:
            return None

        cond, line_no = branch_info

        # When the condition contains Clang element references (e.g. "[B6.18]"),
        # first try to extract a comparison expression from the block's statements.
        # If that fails, keep the full condition (which may include element refs)
        # so the resolver can still attempt to resolve them.
        if cond.startswith("if [") or cond.startswith("while [") or cond.startswith("for [") or cond.startswith("["):
            stmt_cond = self._extract_condition_from_block(block)
            if stmt_cond:
                cond = stmt_cond
            line_no = self._extract_line_from_statements(block)

        # Resolve element references via CFG data-flow chain following
        resolved_expr = ""
        if "[" in cond and "]" in cond:
            resolved = self._resolve_condition_expression(cond, block_id, cfg)
            if resolved and resolved != cond:
                resolved_expr = resolved

        node = CFGBranchNode(
            node_id=f"seg{self._node_id(fn_name, block_id)}",
            source_line=line_no,
            function_name=fn_name,
            condition_expr=cond,
            resolved_expr=resolved_expr,
            call_depth=depth,
            parent_call_site=parent_call,
            # Both sides are "relevant" — the ConflictChecker will decide
            blocks_taken=block.successors or [],
            blocks_not_taken=[],
        )

        return node

    # ─── Terminator parsing ───────────────────────────────────────────────────

    def _parse_terminator(
        self, text: str,
    ) -> tuple[str, int] | None:
        """Parse a CFG terminator string into (condition_expression, line_number).

        Handles formats like:
        - ``"if [B3.1] (BinaryOperator) r >= 0"``  → ``("r >= 0", 0)``
        - ``"while [B1.1] (ImplicitCastExpr, IntegralToBoolean, int)"``
          → ``("int", 0)``  (the cast target type)
        - ``"for [B5.3] ..."``

        When the condition can't be parsed from the type-annotated form,
        falls back to the raw text after the keyword.

        Returns ``None`` for non-branch terminators (e.g. simple ``return``).
        """
        if not text:
            return None

        text_stripped = text.strip()
        is_branch = False
        for keyword in ("if", "while", "for"):
            if text_stripped.startswith(keyword):
                is_branch = True
                break
        if not is_branch:
            return None

        line_no = _extract_line_number(text)

        # Try to extract condition after the last ")"
        m = _COND_AFTER_TYPE_RE.search(text_stripped)
        if m:
            cond = m.group(1).strip()
            if cond:
                return (cond, line_no)

        # Fallback: return the raw text after the keyword
        rest = text_stripped
        for kw in ("if", "while", "for"):
            if text_stripped.startswith(kw):
                rest = text_stripped[len(kw):].strip()
                break

        # Clang-native format: rest starts with an element reference like
        # "[B9.5] || ([B8.6] && [B7.6])".  Keep the full expression so the
        # element reference resolver can handle it later.
        if rest.startswith("["):
            return (rest, line_no)

        # Remove [...]
        rest = re.sub(r'\[[^\]]*\]', '', rest).strip()
        # Remove (...)
        rest = re.sub(r'\([^)]*\)', '', rest).strip()
        if rest:
            return (rest, line_no)

        return (text_stripped, line_no)

    # ─── Condition extraction from Clang-native CFG statements ──────────────

    # ─── Element reference resolution ──────────────────────────────────────────

    _ELEMENT_REF_RE = re.compile(r'\[(\w+)\.(\d+)\]')
    _TYPE_ANNOT_RE = re.compile(r'^\s*\([^)]*\)\s+')

    # ─── Global CFG index (Phase 2) ─────────────────────────────────────────

    def _build_cfg_index(self, cfg: CFGFunction) -> None:
        """Build ``{block_id: {stmt_index: stmt_text}}`` for the entire CFG.

        This index enables cross-block element reference resolution: when a
        terminator in block B5 references ``[B8.6]``, the resolver can find
        that element in B8's statement list via this index.

        The index covers ALL blocks in the function, not just the currently
        filtered range.  Only elements whose prefix block_id matches their
        containing block are indexed (``[B5.1]`` in block B5).
        """
        index: dict[str, dict[int, str]] = {}
        array_order: dict[str, list[int]] = {}
        for bid, block in cfg.blocks.items():
            stmts: dict[int, str] = {}
            order: list[int] = []
            for stmt in block.statements:
                m = re.match(r'^\s*\[(\w+)\.(\d+)\]\s*(.*)', stmt)
                if m and m.group(1) == bid:
                    elem_idx = int(m.group(2))
                    stmts[elem_idx] = m.group(3).strip()
                    order.append(elem_idx)
            if stmts:
                index[bid] = stmts
                array_order[bid] = order
        self._cfg_index = index
        self._cfg_array_order = array_order

    def _resolve_element_reference(
        self,
        ref: str,
        cfg: CFGFunction,
        depth: int = 0,
    ) -> str | None:
        """Resolve a Clang element reference (e.g. ``[B6.15]``) to human-readable form.

        Walks the element's defining statement in the CFG and recursively
        resolves sub-references.

        Args:
            ref: Element reference like ``[B6.15]`` or a mixed expression
                 like ``[B6.14] != [B6.11]``.
            cfg: The :class:`CFGFunction` containing the block.
            depth: Recursion depth guard (max 5).

        Returns:
            Human-readable expression string, or the original text if
            resolution fails.
        """
        if depth > 5:
            return ref  # guard against deep recursion

        # If the whole string is a single element reference, resolve it
        m = self._ELEMENT_REF_RE.fullmatch(ref.strip())
        if m:
            resolved = self._resolve_single_element(m.group(1), int(m.group(2)), cfg, depth)
            return resolved or ref

        # Mixed expression: replace each [Bx.y] occurrence recursively
        parts = self._ELEMENT_REF_RE.split(ref)
        if len(parts) > 1:
            resolved_parts = []
            for i, part in enumerate(parts):
                if i % 3 == 0:
                    # Text between references
                    resolved_parts.append(part)
                elif i % 3 == 1:
                    # Block ID (e.g. "B6")
                    block_id = part
                else:
                    # Statement index (e.g. 15)
                    stmt_idx = int(part)
                    sub_ref = f"[{block_id}.{stmt_idx}]"
                    resolved = self._resolve_single_element(block_id, stmt_idx, cfg, depth + 1)
                    resolved_parts.append(resolved or sub_ref)
            return " ".join(resolved_parts).strip()

        # No element references found — return as-is
        stripped = self._TYPE_ANNOT_RE.sub("", ref).strip()
        return stripped or ref

    def _resolve_single_element(
        self,
        block_id: str,
        stmt_index: int,
        cfg: CFGFunction,
        depth: int,
    ) -> str | None:
        """Resolve a single ``[Bx.y]`` element to its source-level expression.

        Resolution order:
        1. **Global CFG index** — ``self._cfg_index[block_id][stmt_index]``.
           This handles cross-block references (a terminator in B5 referencing
           ``[B8.6]``) and sparse indices (not every number 1..N appears in
           a block's statement list).
        2. **Prefix match** — fallback: scan the block's statements for the
           ``[Bx.y]`` prefix (current behavior).
        3. Return ``None`` if not found in either.

        Clang CFG element indices are inherently sparse and the global index
        covers ALL blocks in the function, not just the current block.
        """
        # Phase 1: global CFG index (cross-block + sparse index support)
        stmt: str | None = None
        idx = self._cfg_index.get(block_id)
        if idx is not None and stmt_index in idx:
            stmt = idx[stmt_index]

        # Phase 2: fallback — prefix match in the block's statements
        if stmt is None:
            block = cfg.blocks.get(block_id)
            if block is None:
                return None
            prefix = f"[{block_id}.{stmt_index}]"
            for s in block.statements:
                if s.strip().startswith(prefix):
                    stmt = s.strip()
                    # Strip the leading "[Bx.y] " prefix
                    stmt = self._ELEMENT_REF_RE.sub("", stmt, count=1).strip()
                    break

        if stmt is None:
            return None

        # Strip type annotation like "(ImplicitCastExpr) ctx->x"
        stmt = self._TYPE_ANNOT_RE.sub("", stmt).strip()

        if not stmt:
            return None

        # Recursively resolve any sub-references in the statement
        sub_m = self._ELEMENT_REF_RE.search(stmt)
        if sub_m:
            # The statement itself contains sub-references
            return self._resolve_element_reference(stmt, cfg, depth + 1)

        # Simple leaf expression (variable name, constant, etc.)
        return stmt

    # ─── CFG data-flow chain following (Phase 2 enhancement) ───────────────────
    #
    # Clang CFG decomposes expressions into sequential computation steps stored
    # as statements in ARRAY position order.  Each element depends on the value
    # produced by the preceding element.  For example:
    #
    #   array[0] 'ctx'                         ← leaf variable
    #   array[1] [B7.1] (ImplicitCastExpr,..)  ← cast → pass-through
    #   array[2] [B7.2] ->error                 ← member access on [B7.1]'s result
    #   array[3] [B7.3] (ImplicitCastExpr,..)  ← cast → pass-through
    #   array[4] '0'                           ← leaf literal
    #   array[5] [B7.4] != [B7.5]             ← binary: (prev) != [B7.5]
    #
    # The chain reconstructs: ctx → ->error → ctx->error → != 0 = "ctx->error != 0"
    #
    # Key insight: types of CFG elements
    # 1. CAST:  text starts with "(Type, ...)" — semantic pass-through
    # 2. MEMBER ACCESS:  text starts with "->" or "." — implicit dep on prev
    # 3. BINARY OP:  text like "!= [Bx.y]" or "[Bx.y] >= [Bx.z]" — use prev + ref
    # 4. LEAF:  identifier ("ctx", "iocb") or literal ("0")
    # 5. UNRESOLVED REF:  element number refers to implicit/last value

    def _resolve_condition_expression(
        self, cond_text: str, block_id: str, cfg: CFGFunction,
    ) -> str | None:
        """Resolve a Clang CFG condition expression to human-readable form.

        Handles Clang's implicit data flow: in ``[B7.4] != [B7.5]``,
        ``[B7.4]`` is the comparison *node* (its LHS is the PREVIOUS element
        in the data flow chain); ``[B7.5]`` is the explicit RHS operand.

        Returns the human-readable expression (e.g. ``"ctx->error != 0"``)
        or the original text if resolution fails.
        """
        if not cond_text:
            return None

        stripped = cond_text.strip()

        # Pattern 1: binary comparison with single explicit RHS ref
        #   "!= [B7.5]"  →  resolve(prev) + " != " + resolve(B7.5)
        m1 = re.match(
            r'^\s*(==|!=|>=|<=|>|<)\s+\[(\w+)\.(\d+)\]\s*$', stripped,
        )
        if m1:
            op = m1.group(1)
            rhs_bid, rhs_idx = m1.group(2), int(m1.group(3))
            lhs = self._chain_to_var(block_id, None, cfg)
            rhs = self._chain_to_var(rhs_bid, rhs_idx, cfg)
            return f"{lhs or '?'} {op} {rhs or '?'}"

        # Pattern 2: full comparison with two explicit refs
        #   "[B6.14] != [B6.11]" or "[B7.4] != [B7.5]"
        #
        # In Clang CFG, [Bx.y] MAY be the comparison node itself (its text
        # defines the comparison with an IMPLICIT LHS operand from the
        # preceding element).  It MAY also be a regular element reference
        # to a nested sub-expression.
        #
        # Heuristic: resolve [Bx.y] via chain-following.  If the result
        # starts with the comparison OPERATOR, then [Bx.y] IS the
        # comparison node (implicit LHS).  Otherwise it's a regular ref.
        m2 = re.match(
            r'^\[(\w+)\.(\d+)\]\s*(==|!=|>=|<=|>|<)\s*\[(\w+)\.(\d+)\]\s*$',
            stripped,
        )
        if m2:
            op = m2.group(3)
            lhs_bid, lhs_idx = m2.group(1), int(m2.group(2))
            rhs_bid, rhs_idx = m2.group(4), int(m2.group(5))
            lhs_raw = self._chain_to_var(lhs_bid, lhs_idx, cfg)
            rhs = self._chain_to_var(rhs_bid, rhs_idx, cfg)
            if lhs_raw and op in lhs_raw:
                # Self-referential: [Bx.y] IS the comparison node.
                # LHS comes from the preceding element in data flow.
                prev = self._find_prev_element_array(lhs_bid, lhs_idx, cfg)
                lhs = (
                    self._chain_to_var(lhs_bid, prev, cfg)
                    if prev is not None else None
                )
            else:
                # Regular reference: [Bx.y] is a sub-expression value
                lhs = lhs_raw
            return f"{lhs or '?'} {op} {rhs or '?'}"

        # Pattern 3: single element reference
        #   "[B7.6]"  →  chain-to-var
        m3 = self._ELEMENT_REF_RE.fullmatch(stripped)
        if m3:
            return self._chain_to_var(m3.group(1), int(m3.group(2)), cfg)

        # Pattern 4: compound expression with multiple refs (short-circuit)
        #   "[B9.5] || ([B8.6] && [B7.6])"
        #   → resolve each ref independently
        if "[" in stripped and "]" in stripped:
            result = stripped
            for m in reversed(list(self._ELEMENT_REF_RE.finditer(stripped))):
                resolved = self._chain_to_var(
                    m.group(1), int(m.group(2)), cfg,
                )
                if resolved:
                    result = result[:m.start()] + resolved + result[m.end():]
            if result != stripped:
                return result

        # Fallback: return original
        return stripped

    def _chain_to_var(
        self,
        block_id: str,
        stmt_idx: int | None,
        cfg: CFGFunction,
        depth: int = 0,
    ) -> str | None:
        """Follow the Clang CFG data flow chain to reconstruct a variable expression.

        When *stmt_idx* is an element number, starts from that element and walks
        backward through the chain, skipping casts and collecting member accesses.
        When *stmt_idx* is ``None``, finds the LAST value-producing element in the
        block (used for the implicit LHS of a comparison).

        Example chain for ``ctx->error``:
          B7.3(cast) → B7.2(->error) → B7.1(cast) → ctx
          Result: "ctx->error"
        """
        if depth > 10:
            return None

        index = self._cfg_index.get(block_id, {})
        order = self._cfg_array_order.get(block_id, [])

        if stmt_idx is not None:
            # ── resolve a specific element ──
            stmt_text = index.get(stmt_idx)
            if stmt_text is None:
                # Try prefix fallback in block statements
                block = cfg.blocks.get(block_id)
                if block:
                    prefix = f"[{block_id}.{stmt_idx}]"
                    for s in block.statements:
                        if s.strip().startswith(prefix):
                            stmt_text = re.sub(
                                r'^\s*\[\w+\.\d+\]\s*', '', s.strip(),
                            )
                            break
            if stmt_text is None:
                # Element not in index — try to find an unnumbered value
                # at the position following the previous indexed element
                return self._find_unresolved_value(block_id, stmt_idx, cfg)

            stmt_text = stmt_text.strip()
            if not stmt_text:
                return None

            # Strip leading type annotation (cast nodes)
            if stmt_text.startswith('('):
                type_m = re.match(r'^\([^)]*\)\s*', stmt_text)
                if type_m:
                    rest = stmt_text[type_m.end():].strip()
                    if not rest:
                        # Pure CAST — pass through to previous element
                        prev, intermediate_unnumbered = (
                            self._find_input_element(
                                block_id, stmt_idx, cfg,
                            )
                        )
                        if intermediate_unnumbered is not None:
                            # Unnumbered statement between prev and current
                            # starts a new chain branch — use it directly
                            return intermediate_unnumbered
                        if prev is not None:
                            return self._chain_to_var(
                                block_id, prev, cfg, depth + 1,
                            )
                        # No preceding input — scan backward for root
                        return self._find_preceding_unnumbered(
                            block_id, stmt_idx, cfg,
                        )
                    stmt_text = rest

            # Check for explicit sub-references in the statement
            sub_refs = list(self._ELEMENT_REF_RE.finditer(stmt_text))

            if not sub_refs:
                # ── Leaf or member access suffix ──
                if stmt_text.startswith('->') or stmt_text.startswith('.'):
                    # Member access with IMPLICIT base (prev element in chain)
                    prev, intermediate_unnumbered = (
                        self._find_input_element(
                            block_id, stmt_idx, cfg,
                        )
                    )
                    if intermediate_unnumbered is not None:
                        return intermediate_unnumbered + stmt_text
                    if prev is not None:
                        base = self._chain_to_var(
                            block_id, prev, cfg, depth + 1,
                        )
                        if base:
                            return base + stmt_text
                    # No preceding input — scan backward for root
                    base = self._find_preceding_unnumbered(
                        block_id, stmt_idx, cfg,
                    )
                    if base:
                        return base + stmt_text
                    return stmt_text
                # Plain leaf (identifier, literal, etc.)
                return stmt_text

            # ── Has sub-references — resolve them within the expression ──
            result = stmt_text
            for m in reversed(sub_refs):
                resolved = self._chain_to_var(
                    m.group(1), int(m.group(2)), cfg, depth + 1,
                )
                if resolved:
                    result = result[:m.start()] + resolved + result[m.end():]
            return result if result != stmt_text and result.strip() else None

        # ── stmt_idx is None → find the LAST value producer in this block ──
        if order:
            last = order[-1]
            return self._chain_to_var(block_id, last, cfg, depth + 1)

        # Fallback: build order on-the-fly by scanning block statements
        block = cfg.blocks.get(block_id)
        if block and block.statements:
            # Find the last indexed element in array order
            last_elem = None
            for s in block.statements:
                m = re.match(r'^\s*\[(\w+)\.(\d+)\]\s*', s)
                if m and m.group(1) == block_id:
                    last_elem = int(m.group(2))
            if last_elem is not None:
                return self._chain_to_var(block_id, last_elem, cfg, depth + 1)

            # No indexed elements — scan unnumbered statements for last identifier
            for s in reversed(block.statements):
                s_stripped = s.strip()
                if (
                    s_stripped
                    and not s_stripped.startswith('(')
                    and not self._ELEMENT_REF_RE.match(s_stripped)
                ):
                    return s_stripped
        return None

    def _find_prev_element_array(
        self, block_id: str, stmt_idx: int, cfg: CFGFunction,
    ) -> int | None:
        """Find the element number of the preceding value in the data flow.

        Uses ARRAY position order, NOT numeric element order, because Clang CFG
        places elements in array order to represent data flow dependencies.

        For element [B50.4] at array[8], the previous elements are
        array[7] (B50.7), array[6] (B50.6), ... and B50.7 is returned as
        the most recent value producer.
        """
        order = self._cfg_array_order.get(block_id)
        if order and stmt_idx in order:
            pos = order.index(stmt_idx)
            if pos > 0:
                return order[pos - 1]

        # Fallback: build order on-the-fly by scanning block statements
        # (needed when _build_cfg_index was not called externally)
        block = cfg.blocks.get(block_id)
        if block is None:
            return None
        elem_order: list[tuple[int, int]] = []
        for arr_i, s in enumerate(block.statements):
            m = re.match(r'^\s*\[(\w+)\.(\d+)\]\s*', s)
            if m and m.group(1) == block_id:
                elem_order.append((arr_i, int(m.group(2))))

        for i in range(1, len(elem_order)):
            if elem_order[i][1] == stmt_idx:
                return elem_order[i - 1][1]

        return None

    def _find_input_element(
        self, block_id: str, stmt_idx: int, cfg: CFGFunction,
    ) -> tuple[int | None, str | None]:
        """Find the data flow input for *stmt_idx*.

        Returns ``(prev_indexed_element, nearest_unnumbered_or_None)``.

        When an unnumbered statement (like a bare ``'ctx'``) exists between
        the previous indexed element and the current element, it represents
        a NEW chain branch (independent data flow root).  The caller should
        use the unnumbered statement directly in that case.

        For example in block B50 (two independent chains)::

          arr[3] [B50.3] cast       ← end of chain 1 (ctx->ipayloadoff)
          arr[4] 'ctx'              ← ROOT of chain 2 (unnumbered!)
          arr[5] [B50.5] cast       ← start of chain 2

          _find_input_element("B50", 5, cfg):
            → _find_prev_element_array returns 3 (B50.3)
            → unnumbered 'ctx' at arr[4] between arr[3] and arr[5]
            → returns (3, "ctx")  — caller uses "ctx" as the new chain root
        """
        # 1. Find the preceding indexed element
        prev = self._find_prev_element_array(block_id, stmt_idx, cfg)
        if prev is None:
            return (None, None)

        # 2. Build array position maps
        block = cfg.blocks.get(block_id)
        if block is None:
            return (prev, None)

        elem_at_arr: dict[int, int] = {}  # array_pos → elem_idx
        arr_at_elem: dict[int, int] = {}  # elem_idx → array_pos
        for arr_i, s in enumerate(block.statements):
            m = re.match(r'^\s*\[(\w+)\.(\d+)\]\s*', s)
            if m and m.group(1) == block_id:
                eid = int(m.group(2))
                elem_at_arr[arr_i] = eid
                arr_at_elem[eid] = arr_i

        # 3. Check for unnumbered statements between prev's position
        #    and current element's position
        cur_arr = arr_at_elem.get(stmt_idx)
        prev_arr = arr_at_elem.get(prev)
        if cur_arr is not None and prev_arr is not None:
            # Scan backward from cur_arr - 1 to prev_arr + 1
            for i in range(cur_arr - 1, prev_arr, -1):
                s = block.statements[i].strip()
                if (
                    s
                    and not re.match(r'^\s*\[\w+\.\d+\]', s)
                    and not s.startswith('(')
                    and not s.startswith('&')
                ):
                    return (prev, s)

        return (prev, None)

    def _find_preceding_unnumbered(
        self, block_id: str, stmt_idx: int, cfg: CFGFunction,
    ) -> str | None:
        """Find the nearest unnumbered statement preceding *stmt_idx* in array order.

        Unnumbered statements (like ``'ctx'`` at array[0] in B7) are the
        ROOT of the Clang CFG data flow chain — they introduce the actual
        variable names and literals that subsequent elements reference.
        """
        block = cfg.blocks.get(block_id)
        if not block or not block.statements:
            return None

        # Build array position → element number map
        elem_at_arr: dict[int, int] = {}
        for arr_i, s in enumerate(block.statements):
            m = re.match(r'^\s*\[(\w+)\.(\d+)\]\s*', s)
            if m and m.group(1) == block_id:
                elem_at_arr[arr_i] = int(m.group(2))

        # Find the array position of stmt_idx
        target_arr = None
        for arr_i, eid in elem_at_arr.items():
            if eid == stmt_idx:
                target_arr = arr_i
                break

        if target_arr is None:
            return None

        # Scan backward for the first unnumbered statement
        for i in range(target_arr - 1, -1, -1):
            s = block.statements[i].strip()
            if (
                s
                and not re.match(r'^\s*\[\w+\.\d+\]', s)
                and not s.startswith('(')
                and not s.startswith('&')
            ):
                return s
        return None

    def _find_unresolved_value(
        self, block_id: str, stmt_idx: int, cfg: CFGFunction,
    ) -> str | None:
        """Find the value for a Clang element reference not in the index.

        Unresolved references like ``[B7.5]`` or ``[B50.8]`` typically point
        to the nearest unnumbered leaf expression (literal, identifier) that
        precedes the comparison node in the statement array.
        """
        block = cfg.blocks.get(block_id)
        if not block or not block.statements:
            return None

        order = self._cfg_array_order.get(block_id, [])

        # If this element has an ARRAY order position, scan backward
        # from that position to find the first unnumbered leaf.
        if stmt_idx in order:
            elem_pos = order.index(stmt_idx)
            # Count how many indexed statements are before this one
            # so we can find the corresponding array position
            arr_positions = {}
            for arr_i, s in enumerate(block.statements):
                m = re.match(r'^\s*\[(\w+)\.(\d+)\]\s*', s)
                if m and m.group(1) == block_id:
                    arr_positions[int(m.group(2))] = arr_i

            if stmt_idx in arr_positions:
                arr_pos = arr_positions[stmt_idx]
                # Scan backward from arr_pos - 1
                for i in range(arr_pos - 1, -1, -1):
                    s = block.statements[i].strip()
                    # Skip indexed statements
                    if re.match(r'^\s*\[\w+\.\d+\]', s):
                        continue
                    # Skip casts and function names at start
                    if s.startswith('(') or s.startswith('&'):
                        continue
                    if s and not s.startswith('['):
                        return s

        # Fallback: scan entire block backward for the last literal/identifier
        for s in reversed(block.statements):
            s_stripped = s.strip()
            if (
                s_stripped
                and not s_stripped.startswith('[')
                and not s_stripped.startswith('(')
            ):
                return s_stripped
        return None

    @staticmethod
    def _extract_condition_from_block(block: CFGBlock) -> str | None:
        """Extract the comparison condition from a block's statements.

        The Clang-native CFG dump puts the condition in a statement like::

            [B7.6] [B7.4] != [B7.5]

        This method scans the block statements for the last statement
        containing a comparison operator and extracts a readable form.
        """
        for stmt in reversed(block.statements):
            raw = stmt.strip()
            # Compound condition (text that contains || or &&)
            if '||' in raw or '&&' in raw:
                return raw
            # Member access hint: "[B15.2]->callbacks" or "ctx->read_enabled"
            # Must be checked BEFORE the comparison regex to avoid misinterpreting
            # ``->`` as identifier ``ctx-`` followed by comparison ``> read_enabled``.
            if '->' in raw:
                continue  # Not a branch condition by itself
            # Simple comparison: "[B7.4] != [B7.5]" or "[B31.8] >= [B31.10]"
            # Match without stripping element references.
            m = re.search(
                r'(\[?\w[\w>.\-]*\]?)\s*(==|!=|>=|<=|>|<)\s*(\[?\w[\w>.\-]*\]?)',
                raw,
            )
            if m:
                return m.group(0)
        return None

    @staticmethod
    def _extract_line_from_statements(block: CFGBlock) -> int:
        """Extract the first source line number from a block's statements.

        Clang CFG statements sometimes include source locations as trailing
        comments like ``[B7.2]->error   (wslay_event.c:573)``.
        """
        for stmt in block.statements:
            m = re.search(r'\(([^:]+):(\d+)\)', stmt)
            if m:
                return int(m.group(2))
        return 0

    # ─── Call detection ───────────────────────────────────────────────────────

    def _find_calls_in_block(self, block: CFGBlock) -> list[dict]:
        """Find function call expressions in a block's statements.

        Returns a list of dicts with keys ``line``, ``expr``, ``callee``.
        """
        calls: list[dict] = []
        for stmt in block.statements:
            parsed = self._parse_call_statement(stmt)
            if parsed:
                calls.append(parsed)
        return calls

    def _parse_call_statement(self, stmt: str) -> dict | None:
        """Parse a single statement for a function call pattern.

        Matches patterns like:
        - ``"572: r = wslay_frame_recv(ctx->frame_ctx, &iocb)"``
        - ``"[B2.2] (CallExpr) memset(...)"``

        Returns a dict with ``line``, ``expr``, ``callee``, and ``ret_var``,
        or ``None`` if the statement is not a function call.

        The method does NOT rely on ``CallExpr`` markers because source-code
        statements (from ``source_between``) use plain C++ syntax without
        AST annotations.
        """
        line_no = _extract_line_number(stmt)

        # Try to extract assignment: "variable = func_name(" pattern
        assign_m = re.search(r'(\w+)\s*=\s*(\w+)\s*\(', stmt)
        if assign_m:
            ret_var = assign_m.group(1)
            callee = assign_m.group(2)
            # Filter out common keywords/constants
            if callee.lower() in {"sizeof", "true", "false", "null", "nullptr",
                                   "if", "while", "for", "return", "int", "char",
                                   "auto", "struct", "class"}:
                return None
            return {
                "line": line_no,
                "expr": stmt.strip(),
                "callee": callee,
                "ret_var": ret_var,
            }

        # Fallback: find any function-call pattern "func("
        call_m = re.search(r'(?:^|\s)([a-z_]\w*)\s*\(', stmt, re.IGNORECASE)
        if call_m:
            callee = call_m.group(1)
            if callee.lower() in {"if", "while", "for", "switch", "return", "sizeof"}:
                return None
            return {
                "line": line_no,
                "expr": stmt.strip(),
                "callee": callee,
                "ret_var": None,
            }

        return None

    # ─── Adaptive depth control ───────────────────────────────────────────────

    def _should_extract_callee(
        self,
        call_site: dict,
        depth: int,
        postcondition: Postcondition,
    ) -> bool:
        """Decide whether to recursively extract a callee's CFG branches.

        Full expansion: ALL callees with CFG data are extracted regardless of
        return value relevance to the postcondition.  The pruning phase (Step 4,
        ConflictChecker) determines which branches are relevant.

        Only the depth cap limits extraction.
        """
        if depth >= self._max_depth:
            return False

        # Full expansion: always extract when within depth and CFG exists
        # (The caller is responsible for checking CFG cache availability.)
        return True

    # ─── Callee extraction (for non-branch blocks) ──────────────────────────

    def _extract_callees_from_block(
        self,
        cfg: CFGFunction,
        block_id: str,
        fn_name: str,
        depth: int,
        postcondition: Postcondition,
    ) -> list[CFGBranchNode]:
        """Scan a block for callee calls and extract their CFG subtrees.

        This handles *non-branch* blocks — straight-line blocks containing
        function calls like ``r = wslay_frame_recv(...)``.  Branch extraction
        (for blocks with if/while/for terminators) is handled by
        :meth:`_extract_from_block`.
        """
        block = cfg.blocks.get(block_id)
        if block is None:
            return []

        call_sites = self._find_calls_in_block(block)
        result: list[CFGBranchNode] = []
        for call_site in call_sites:
            if not self._should_extract_callee(
                call_site=call_site,
                depth=depth,
                postcondition=postcondition,
            ):
                continue

            callee_cfg = self._resolve_cfg_by_short_name(
                call_site["callee"],
                caller_fn=fn_name,
                caller_file=cfg.source_file or "",
                call_line=call_site.get("line", 0),
            )
            if callee_cfg is None:
                continue

            # Create a wrapper node for the call site
            call_node = CFGBranchNode(
                node_id=f"call_{fn_name}_{block_id}",
                source_line=call_site.get("line", 0),
                function_name=fn_name,
                condition_expr=f"call {call_site['callee']}",
                call_depth=depth,
                callee_name=call_site["callee"],
            )

            # Extract callee's branches
            callee_blocks = self._get_entry_blocks(callee_cfg)
            for cb_id in callee_blocks:
                child = self._extract_from_block(
                    cfg=callee_cfg,
                    block_id=cb_id,
                    fn_name=call_site["callee"],
                    depth=depth + 1,
                    parent_call=f"L{call_site.get('line', 0)} {call_site['callee']}",
                    postcondition=postcondition,
                )
                if child is not None:
                    call_node.children.append(child)

            if call_node.children:
                result.append(call_node)

        return result

    # ─── SliceMask helpers ────────────────────────────────────────────────────

    def _get_retained_cfg_blocks(self, fn_name: str) -> set[int]:
        """Get retained CFG block IDs from the SliceMask with name resolution."""
        blocks = self._slice_mask.cfg_retained_blocks.get(fn_name)
        if blocks is not None:
            return blocks
        # Try partial match against stored keys
        for key, bset in self._slice_mask.cfg_retained_blocks.items():
            if fn_name in key or key in fn_name:
                return bset
        return set()

    def _resolve_cfg_by_short_name(
        self,
        short_name: str,
        caller_fn: str = "",
        caller_file: str = "",
        call_line: int = 0,
    ):
        """Resolve a short function name to a CFG entry.

        Resolution order:
        1. **CallGraph** — if SDG has a call graph, use it to find the
           qualified name via *caller_fn* or fuzzy matching.
        2. **Text match** — fallback: scan ``_cfg_cache`` keys for the
           short name (existing behavior).

        Args:
            short_name: Short function name (e.g. ``"wslay_queue_push"``).
            caller_fn: Calling function's short name (for CG disambiguation).
            caller_file: Calling function's source file (for CG disambiguation).
            call_line: Call site line number (for CG disambiguation).
        """
        # Phase 1: CallGraph resolution
        if self._sdg is not None:
            cg = getattr(self._sdg, "callgraph", None)
            if cg is not None:
                qualified = self._resolve_via_callgraph(
                    short_name, caller_fn, caller_file, call_line,
                )
                if qualified is not None and qualified in self._cfg_cache:
                    return self._cfg_cache[qualified]

        # Phase 2: fallback — text matching (existing behavior)
        for key, cfg in self._cfg_cache.items():
            if short_name in key or key.startswith(short_name):
                return cfg
            without_ret = key.split(" ", 1)[-1] if " " in key else key
            if without_ret.startswith(short_name):
                return cfg
        return None

    # ─── CallGraph helpers (Phase 3) ───────────────────────────────────────

    def _resolve_via_callgraph(
        self,
        short_name: str,
        caller_fn: str = "",
        caller_file: str = "",
        call_line: int = 0,
    ) -> str | None:
        """Resolve a short name to a qualified name via the CallGraph.

        Three strategies, tried in order:
        1. If *caller_fn* is known, get its callees from the CG and find
           the one matching *short_name*.
        2. If *caller_file* is known, use ``find_nodes_matching`` with
           file+line proximity.
        3. Pure fuzzy name match via ``find_nodes_matching``.

        Returns the qualified function name (CG node ID), or ``None``.
        """
        cg = self._sdg.callgraph
        if cg is None:
            return None

        # Strategy 1: via caller's callee list
        if caller_fn:
            caller_q = self._resolve_caller_qualified(caller_fn)
            if caller_q is not None:
                for callee in cg.get_callees(caller_q):
                    if callee == short_name or callee.endswith("::" + short_name):
                        return callee

        # Strategy 2: file + line proximity
        if caller_file:
            nodes = cg.find_nodes_matching(short_name, caller_file, call_line)
            if nodes:
                return nodes[0].id

        # Strategy 3: pure fuzzy match
        nodes = cg.find_nodes_matching(short_name)
        if nodes:
            return nodes[0].id

        return None

    def _resolve_caller_qualified(self, caller_fn_short: str) -> str | None:
        """Resolve a caller's short name to its CG qualified name."""
        cg = self._sdg.callgraph
        if cg is None:
            return None
        node = cg.find_node_by_unqualified_name(caller_fn_short)
        if node:
            return node.id
        return None

    # ─── PDG-based call index and cross-procedural extraction ─────────────────
    #
    # The CFG cache stores Clang-native element references (e.g. ``[B6.1]``
    # ``(ImplicitCastExpr, ...)``) without parsed function call information.
    # The PDG (built from Clang AST) records ``is_call`` / ``callee`` /
    # ``block_id`` / ``source_line`` for every call expression, providing a
    # reliable cross-procedural call index.
    #
    # ────────────────────────────────────────────────────────────────────────

    def _get_pdg_call_index(
        self, fn_name_or_cfg_key: str,
    ) -> dict[int, list[tuple[str, int]]]:
        """Build ``{block_id: [(callee, source_line)]}`` from the PDG.

        Used by :meth:`extract` Phase 2 to discover function call sites within
        a segment's filtered blocks.  Results are cached per SDG function name
        so repeated lookups (across segments of the same function) are instant.
        """
        sdg = self._sdg
        if sdg is None:
            return {}
        pdgs = getattr(sdg, "pdgs", {})
        if not pdgs:
            return {}

        # Resolve CFG key → SDG key
        sdg_key = self._resolve_sdg_key(pdgs, fn_name_or_cfg_key)
        if sdg_key is None:
            return {}

        # Check cache
        cached = self._pdg_call_index_cache.get(sdg_key)
        if cached is not None:
            return cached

        pdg = pdgs.get(sdg_key)
        if pdg is None:
            return {}

        call_index: dict[int, list[tuple[str, int]]] = {}
        for node in pdg.nodes.values():
            if node.is_call and node.block_id > 0:
                bid = node.block_id
                if bid not in call_index:
                    call_index[bid] = []
                # Deduplicate by callee+line (same call may appear in multiple
                # PDG nodes due to expression decomposition)
                entry = (node.callee, node.source_line)
                if entry not in call_index[bid]:
                    call_index[bid].append(entry)

        self._pdg_call_index_cache[sdg_key] = call_index
        return call_index

    def _extract_callee_branches(
        self,
        callee_cfg: CFGFunction,
        callee_name: str,
        depth: int,
    ) -> list[CFGBranchNode]:
        """Extract ALL branch nodes from *callee_cfg*, including nested calls.

        Args:
            callee_cfg: The callee's CFG function data.
            callee_name: Short callee name (e.g. ``"wslay_event_queue_close_wrapper"``).
            depth: Current call depth (1 = direct callee, 2 = callee of callee, …).

        Returns:
            Flat list of ``CFGBranchNode`` — each branch from the callee's CFG
            (and nested callees) with ``call_depth >= depth``.

        Caching: results are keyed by ``callee_cfg.function_name`` (the full
        signature key in ``cfg_cache``).  If the same function is called from
        multiple sites, the second and subsequent calls return the cached list
        instantly.
        """
        cfg_key = callee_cfg.function_name
        cached = self._callee_branch_cache.get(cfg_key)
        if cached is not None:
            return cached

        all_branches: list[CFGBranchNode] = []

        # Save the parent function's CFG element index so it can be restored
        # after we process this callee.  Without this, resolving conditions
        # in nested callees would overwrite the index needed by the outer
        # callee's own blocks.
        _saved_index = self._cfg_index
        _saved_array = self._cfg_array_order

        # Build CFG element index for condition resolution
        self._build_cfg_index(callee_cfg)

        # ── Step 1: extract intra-procedural branches from the callee ──
        for bid_str, block in callee_cfg.blocks.items():
            if not self._is_branch_terminator(block.terminator):
                continue
            branch = self._extract_from_block(
                cfg=callee_cfg,
                block_id=bid_str,
                fn_name=callee_name,
                depth=depth,
            )
            if branch is not None:
                all_branches.append(branch)

        # ── Step 2: recursively extract nested callee calls ──
        if depth < self._max_depth:
            callee_call_index = self._get_pdg_call_index(cfg_key)
            for bid_str, block in callee_cfg.blocks.items():
                bid_num = self._parse_block_id(bid_str)
                for nested_callee, nested_line in callee_call_index.get(bid_num, []):
                    # Guard against recursion
                    if nested_callee == callee_name:
                        continue
                    nested_cfg = self._resolve_cfg_by_short_name(
                        nested_callee, caller_fn=callee_name,
                    )
                    if nested_cfg is None:
                        continue

                    nested_branches = self._extract_callee_branches(
                        callee_cfg=nested_cfg,
                        callee_name=nested_callee,
                        depth=depth + 1,
                    )
                    if not nested_branches:
                        continue

                    # Find the callee's branch node that's in the call-containing
                    # block so we can attach nested branches as its children.
                    # If no branch node is in this block, create a sentinel.
                    attached = False
                    for branch in all_branches:
                        # Match by block ID — check if this branch came from
                        # the same block that contains the nested call.
                        # Use endswith to avoid "_B1" matching "_B10".
                        if branch.node_id.endswith(f"_{bid_str}"):
                            for child in nested_branches:
                                branch.children.append(child)
                            attached = True
                            break

                    if not attached:
                        # Create a sentinel — the call is in a non-branch block
                        sentinel = CFGBranchNode(
                            node_id=f"callee_{callee_name}_{nested_callee}_L{nested_line}",
                            source_line=nested_line,
                            function_name=callee_name,
                            condition_expr=f"→ {nested_callee}()",
                            call_depth=depth,
                            callee_name=nested_callee,
                        )
                        for child in nested_branches:
                            sentinel.children.append(child)
                        all_branches.append(sentinel)

        # Restore the parent's CFG element index
        self._cfg_index = _saved_index
        self._cfg_array_order = _saved_array

        self._callee_branch_cache[cfg_key] = all_branches
        return all_branches

    # ─── PDG-based block source line resolution ──────────────────────────────
    #
    # The CFG cache may store blocks in Clang-native format (element references
    # like ``[B6.1]``) without source line numbers.  When this happens,
    # _filter_blocks_by_range cannot use line numbers from CFG statements or
    # terminators to scope blocks to a segment.
    #
    # Instead, we use the PDG (built from Clang AST) which records both
    # source_line and block_id for each node.  By aggregating PDG nodes by
    # block_id we recover the source line range for each CFG block and can
    # correctly filter by segment line range.
    #
    # ────────────────────────────────────────────────────────────────────────

    # ─── Path reachability (A) ────────────────────────────────────────────────

    def _decisions_for(self, fn_key: str) -> dict[int, bool]:
        """CSA branch decisions for *fn_key*, resolving by exact then short name."""
        if not self._decisions_by_fn:
            return {}
        exact = self._decisions_by_fn.get(fn_key)
        if exact:
            return exact
        short = _method_component(fn_key)
        merged: dict[int, bool] = {}
        for name, decisions in self._decisions_by_fn.items():
            if _method_component(name) == short:
                merged.update(decisions)
        return merged

    def _on_path_lines(self, fn_key: str) -> frozenset[int] | None:
        """Source lines the reported path can reach inside *fn_key*.

        Walks the function's CFG from its entry block, pruning at every branch
        block that has a CSA decision: successor ``[0]`` is the *true* arm and
        ``[1]`` the *false* arm (verified against Clang's CFG dump — e.g. the
        ternary ``is_map2 ? new IndexIDMap2() : new IndexIDMap()`` yields that
        order).  Blocks with no decision, or whose predicate line carries
        conflicting directions, keep all successors.

        Returns ``None`` when the result cannot be trusted — no CFG, no
        decisions, or a self-check failure (the exit block is unreachable, which
        would mean the successor order is not what we assume).  Callers must
        treat ``None`` as "filter nothing".
        """
        if fn_key in self._on_path_lines_cache:
            return self._on_path_lines_cache[fn_key]

        result: frozenset[int] | None = None
        decisions = self._decisions_for(fn_key)
        cfg = self._resolve_cfg_for_reachability(fn_key, decisions)
        if decisions and cfg is not None:
            score, compared = self._alignment_score(fn_key, cfg)
            if compared >= _MIN_ALIGNED_BLOCKS and score < _MIN_ALIGNMENT:
                logger.warning(
                    "path-reachability disabled for %s: the PDG's block ids and "
                    "%s's B<n> ids disagree (%.0f%% of %d comparable blocks "
                    "match) — not filtering",
                    fn_key, cfg.function_name, 100 * score, compared,
                )
                cfg = None
        if decisions and cfg is not None:
            # Keyed by the PDG's own name: the CFG cache key carries the return
            # type, which the SDG lookup does not resolve.
            block_lines = self._get_block_source_lines(fn_key) or \
                self._get_block_source_lines(cfg.function_name)
            if block_lines and cfg.entry_block_id in cfg.blocks:
                reached: set[str] = set()
                stack = [cfg.entry_block_id]
                while stack:
                    bid = stack.pop()
                    if bid in reached:
                        continue
                    reached.add(bid)
                    block = cfg.blocks.get(bid)
                    if block is None:
                        continue
                    successors = [s for s in (block.successors or [])
                                  if s in cfg.blocks]
                    if len(successors) < 2:
                        stack.extend(successors)
                        continue
                    taken = self._block_decision(bid, block_lines, decisions)
                    if taken is None:
                        stack.extend(successors)
                    else:
                        stack.append(successors[0] if taken else successors[1])

                if cfg.exit_block_id not in reached:
                    logger.warning(
                        "path-reachability self-check failed for %s: exit block "
                        "%s unreachable under CSA decisions — not filtering",
                        cfg.function_name, cfg.exit_block_id,
                    )
                else:
                    lines: set[int] = set()
                    for bid in reached:
                        bid_num = self._parse_block_id(bid)
                        lines.update(block_lines.get(bid_num, ()))
                    result = frozenset(lines)

        self._on_path_lines_cache[fn_key] = result
        return result

    def _alignment_score(self, fn_key: str, cfg) -> tuple[float, int]:
        """How well *cfg*'s block numbering agrees with the PDG's.

        The walk in :meth:`_on_path_lines` pairs each CFG block ``B<n>`` with the
        PDG nodes whose ``block_id`` is ``n``.  That pairing only holds when both
        sides come from the same CFG construction — and they do not: the cache is
        dumped by ``clang -cc1 -analyze -analyzer-checker=debug.DumpCFG``, which
        adds implicit/temporary-dtor and ``do ... while`` scope blocks, while the
        PDG comes from PDGBuilder's ``CFG::buildCFG`` with default
        ``BuildOptions``.  Every inserted block shifts all later ids, so the two
        diverging id-by-id is silent corruption rather than an error.

        Measured on the faiss cache, the shift is only 1–2 blocks for most
        functions — but ``read_index`` carries 414 extra cache blocks, the PDG
        puts ``fourcc("IxFI")`` (L507) in block 1161 where the cache has the
        L755 ``snprintf`` body, and the real ladder is at ``B1295``.

        The check is content-based and cheap: Clang emits a source expression as
        one statement per subexpression, so the PDG's ``fourcc("IxFI")`` shows up
        in the block's statements as ``fourcc`` / ``"IxFI"``.  A block *agrees*
        when some statement of it is a substring of an expression the PDG
        recorded for the same id.  Returns ``(ratio, comparable_blocks)``; a
        ratio of 1.0 with 0 comparable blocks means "nothing to check", which
        the caller treats as no evidence either way.
        """
        pdg = self._pdg_for(fn_key)
        if pdg is None:
            return 1.0, 0
        by_block: dict[int, list[str]] = {}
        for node in pdg.nodes.values():
            expr = _norm_expr(node.expression)
            if len(expr) >= 4:
                by_block.setdefault(node.block_id, []).append(expr)

        agreed = compared = 0
        for block_id, block in cfg.blocks.items():
            bid = self._parse_block_id(block_id)
            statements = [_norm_expr(s) for s in block.statements]
            statements = [s for s in statements if len(s) >= 4]
            exprs = by_block.get(bid) or []
            if not statements or not exprs:
                continue  # nothing to compare on one side
            compared += 1
            if any(s in e or e in s for s in statements for e in exprs):
                agreed += 1
        return (agreed / compared if compared else 1.0), compared

    def _pdg_for(self, fn_key: str):
        """The SDG PDG for *fn_key*, or ``None``."""
        sdg = self._sdg
        pdgs = getattr(sdg, "pdgs", None)
        if not pdgs:
            return None
        sdg_key = self._resolve_sdg_key(pdgs, fn_key)
        return pdgs[sdg_key] if sdg_key else None

    def _resolve_cfg_for_reachability(self, fn_key: str, decisions: dict):
        """The CFG to walk for *fn_key* — or ``None`` when it cannot be trusted.

        CFG cache keys carry the return type (``faiss::Index *read_index(...)``),
        so the qualified name misses and the lookup has to fall back to the bare
        method component — which is ambiguous across overloads.  The Clang block
        numbers make the answer checkable: the PDG records ``node.block_id`` with
        the *same* numbering as the CFG's ``B<n>`` ids, so a candidate CFG that
        does not contain the PDG's blocks is a different function and must not be
        walked.  Walking the wrong overload would silently prune reachable code.
        """
        if not decisions:
            return None

        short = _method_component(fn_key)
        pdg_blocks = set(self._get_block_source_lines(fn_key))
        # Clang numbers the PDG's ``block_id`` and the CFG's ``B<n>`` alike, so
        # the PDG decides which same-named candidate is the right overload:
        # ``read_index`` must not be answered with ``read_index_header``'s CFG.
        candidates, seen = [], set()
        direct = self._cfg_cache.get(fn_key)
        if direct is not None:
            candidates.append(direct)
            seen.add(id(direct))
        for key, cfg in self._cfg_cache.items():
            if id(cfg) in seen or _method_component(key) != short:
                continue
            candidates.append(cfg)
            seen.add(id(cfg))
        if not candidates:
            fallback = self._resolve_cfg_by_short_name(short)
            if fallback is not None:
                candidates.append(fallback)

        for cfg in candidates:
            if not pdg_blocks:
                return cfg
            if pdg_blocks <= {self._parse_block_id(b) for b in cfg.blocks}:
                return cfg

        logger.warning(
            "path-reachability: no CFG of %s (%s) matches the PDG's block "
            "numbering — not filtering", fn_key,
            ", ".join(c.function_name for c in candidates) or "none found",
        )
        return None

    @staticmethod
    def _block_decision(
        block_id: str,
        block_lines: dict[int, set[int]],
        decisions: dict[int, bool],
    ) -> bool | None:
        """The direction a branch block took, or ``None`` when undecidable.

        A line may map to several blocks: Clang splits short-circuit ``a || b``
        tests into one block per operand.  When the decisions disagree (only
        possible if one operand's line was reported both ways) there is no way
        to tell which successor we are looking at, so the caller keeps both.
        """
        bid_num = CFGBranchExtractor._parse_block_id(block_id)
        line_decisions = {taken for line, taken in decisions.items()
                          if line in block_lines.get(bid_num, ())}
        if len(line_decisions) != 1:
            return None
        return line_decisions.pop()

    def _on_path_lines_for_segment(
        self, fn_key: str, line_end: int,
    ) -> frozenset[int] | None:
        """On-path lines for a segment, with the segment-level self-check.

        The segment ends where the report says it does, so ``line_end`` must be
        reachable — if it is not, the reachability walk disagrees with the
        report and pruning would be silently wrong.  ``None`` disables A for
        this segment.
        """
        on_path = self._on_path_lines(fn_key)
        if on_path is None:
            return None
        if line_end and line_end not in on_path:
            logger.warning(
                "path-reachability self-check failed for %s: segment end L%d is "
                "not reachable under CSA decisions — not filtering this segment",
                fn_key, line_end,
            )
            return None
        return on_path

    def _is_path_reachable(self, fn_key: str, line: int) -> bool:
        """Whether *line* is on the reported path inside *fn_key* (permissive)."""
        on_path = self._on_path_lines(fn_key)
        return on_path is None or line in on_path

    # ─── Relevance / structural guards (B, C) ─────────────────────────────────

    def _slice_retains(self, fn_name: str) -> bool:
        """Whether the PDG slice kept any node of *fn_name*."""
        mask = self._slice_mask
        retained = getattr(mask, "pdg_retained_nodes", None) or {}
        if not retained:
            return False
        if retained.get(fn_name):
            return True
        short = _method_component(fn_name)
        for name, nodes in retained.items():
            if nodes and _method_component(name) == short:
                return True
        return False

    def _expansion_mode(self, callee_pdg: "FunctionPDG", depth: int) -> str:
        """How far into *callee_pdg* the extraction may descend at *depth*.

        ``"full"``   — own predicates plus nested calls (the old behaviour).
        ``"shallow"``— own predicates only (C-ii: environment code).
        ``"none"``   — nothing but a leaf sentinel (B slice gating, C-i recursion).

        A ``"none"`` callee still yields a sentinel carrying ``callee_name``,
        the ``→ name()`` expression and the ``callee_…_L<line>`` id — the three
        things ``shortest_path`` metrics, ``ReturnValueMapper``, the CSA
        postcondition matcher, ``branch_relevance`` and ``path_space`` read.
        """
        name = callee_pdg.function_name

        # (C-i) never re-enter a callee already on this branch of the recursion.
        if name in self._expansion_stack:
            return "none"

        # (C-ii) environment code (libstdc++ headers, ...): its own predicates
        # can still contradict a bug constraint, but its internals never can —
        # and walking them was 68% of the 484-2 tree.
        file = getattr(callee_pdg, "source_file", "") or ""
        if self._source_root and file and not self._is_project_file(file):
            return "shallow"

        # (B) beyond the shallow depth, only slice-retained callees are expanded.
        if depth > self._shallow_callee_depth and not self._slice_retains(name):
            return "none"
        return "full"

    def _callee_has_content(self, callee_pdg: "FunctionPDG") -> bool:
        """Whether a non-expanded callee would have produced any branch node.

        Keeps the "a callee with nothing to say adds no node at all" rule of the
        un-gated extractor: without it, every trivial leaf call would suddenly
        grow a sentinel, changing the trees the downstream consumers see.
        """
        if self._get_branch_sources(callee_pdg):
            return True
        return bool(self._find_calls_in_pdg_range(callee_pdg, 0, 999999))

    def _is_project_file(self, file: str) -> bool:
        try:
            Path(file).resolve().relative_to(Path(self._source_root).resolve())
            return True
        except (ValueError, OSError):
            return False

    def _get_block_source_lines(self, fn_name_or_key: str) -> dict[int, set[int]]:
        """Build ``{block_id: set(source_line)}`` from PDG for *fn_name_or_key*.

        Matches *fn_name_or_key* against SDG function names using:
        1. Exact match
        2. Short name appears within SDG key (or vice versa)

        The mapping is cached in ``self._block_source_lines`` so subsequent
        lookups for the same function are instant.
        """
        # Check cache first
        if fn_name_or_key in self._block_source_lines:
            return self._block_source_lines[fn_name_or_key]

        # Resolve to an SDG key
        sdg = self._sdg
        if sdg is None:
            return {}
        pdgs = getattr(sdg, "pdgs", {})
        if not pdgs:
            return {}

        sdg_key = self._resolve_sdg_key(pdgs, fn_name_or_key)
        if sdg_key is None:
            return {}

        # Build block_id → set(source_line) from PDG nodes
        pdg = pdgs[sdg_key]
        block_lines: dict[int, set[int]] = {}
        for node in pdg.nodes.values():
            bid = node.block_id
            sl = node.source_line
            if bid > 0 and sl > 0:
                if bid not in block_lines:
                    block_lines[bid] = set()
                block_lines[bid].add(sl)

        self._block_source_lines[fn_name_or_key] = block_lines
        # Also cache under the resolved SDG key for future direct hits
        self._block_source_lines[sdg_key] = block_lines
        return block_lines

    @staticmethod
    def _resolve_sdg_key(
        pdgs: dict, fn_name_or_key: str,
    ) -> str | None:
        """Find the SDG key that matches *fn_name_or_key*."""
        # Exact match
        if fn_name_or_key in pdgs:
            return fn_name_or_key

        # Extract short function name from CFG key like
        # "int wslay_event_recv(...)" → "wslay_event_recv"
        short = fn_name_or_key.split("(")[0].strip().split()[-1] if "(" in fn_name_or_key else fn_name_or_key

        # Try short name match
        for key in pdgs:
            if short in key or key in short:
                return key

        return None

    def _filter_blocks_by_range(
        self,
        cfg: CFGFunction,
        retained_blocks: set[int],
        line_start: int,
        line_end: int,
    ) -> list[str]:
        """Filter CFG blocks by line range, keeping only in-range retained blocks.

        Two strategies, tried in order:

        1. **CFG line numbers** — if any block in the function has a line number
           prefix (``\\d+:``), use the CFG statement/terminator line numbers to
           determine which blocks fall within ``[line_start, line_end]``.

        2. **PDG source line mapping** — if the CFG has no line numbers (Clang-native
           format), use the PDG's ``block_id → source_line`` mapping to scope blocks.
           Only branch-terminator blocks whose PDG source lines intersect the segment's
           range are included.  Non-branch blocks (needed for callee extraction) are
           included conservatively when they have no PDG mapping.
        """
        # When the SliceMask has no retained blocks for this function
        # it means no PDG slicing was applied → allow all blocks.
        allow_all = len(retained_blocks) == 0

        function_has_line_numbers = self._function_has_any_line_numbers(cfg)
        if not function_has_line_numbers:
            # ── Strategy 2: PDG-based source line mapping ──
            block_lines = self._get_block_source_lines(cfg.function_name)
            result: list[str] = []
            for bid_str, block in cfg.blocks.items():
                bid = self._parse_block_id(bid_str)
                if not allow_all and bid not in retained_blocks:
                    continue

                lines = block_lines.get(bid)
                if lines:
                    # Block has PDG source lines: include only if any line
                    # falls within the segment's range
                    if any(line_start <= ln <= line_end for ln in lines):
                        result.append(bid_str)
                else:
                    # Block has no PDG source line info.  Non-branch blocks are
                    # included conservatively (they may contain callee calls).
                    # Branch-terminator blocks without source info are excluded
                    # because we cannot determine their segment affiliation.
                    if not self._is_branch_terminator(block.terminator):
                        result.append(bid_str)
                    elif block_lines:
                        # There IS PDG data for this function but this specific
                        # branch block has no line info → exclude
                        logger.debug(
                            "Excluding branch block %s (no PDG source line, "
                            "segment L%d-L%d)", bid_str, line_start, line_end,
                        )
                    else:
                        # No PDG data at all for this function → fall back to
                        # including ALL retained branch blocks on the assumption
                        # that the caller will filter them later
                        result.append(bid_str)
            return result

        # ── Strategy 1: CFG-native line numbers ──
        explicit_in_range: set[str] = set()
        for bid_str, block in cfg.blocks.items():
            bid = self._parse_block_id(bid_str)
            if not allow_all and bid not in retained_blocks:
                continue
            if self._block_in_range(block, line_start, line_end):
                explicit_in_range.add(bid_str)
            if self._terminator_in_range(block, line_start, line_end):
                explicit_in_range.add(bid_str)

        at_least_one_in_range = len(explicit_in_range) > 0

        result = list(explicit_in_range)
        for bid_str, block in cfg.blocks.items():
            bid = self._parse_block_id(bid_str)
            if (not allow_all and bid not in retained_blocks) or bid_str in explicit_in_range:
                continue
            if at_least_one_in_range and self._is_branch_terminator(block.terminator):
                result.append(bid_str)

        return result

    def _is_branch_terminator(self, terminator: str | None) -> bool:
        """Check if a terminator represents an if/while/for branch."""
        if not terminator:
            return False
        ts = terminator.strip()
        return any(ts.startswith(kw) for kw in ("if", "while", "for"))

    def _function_has_any_line_numbers(self, cfg: CFGFunction) -> bool:
        """Check if any block in the function has line numbers in its content."""
        for block in cfg.blocks.values():
            for stmt in block.statements:
                if _extract_line_number(stmt) > 0:
                    return True
            if block.terminator and _extract_line_number(block.terminator) > 0:
                return True
        return False

    def _block_in_range(self, block: CFGBlock, ls: int, le: int) -> bool:
        """Check if any statement in the block has a line number in range."""
        for stmt in block.statements:
            ln = _extract_line_number(stmt)
            if ln and ls <= ln <= le:
                return True
        return False

    def _terminator_in_range(self, block: CFGBlock, ls: int, le: int) -> bool:
        """Check if the terminator has a line number in range."""
        if block.terminator:
            ln = _extract_line_number(block.terminator)
            if ln and ls <= ln <= le:
                return True
        return False

    # ─── Misc helpers ─────────────────────────────────────────────────────────

    def _get_entry_blocks(self, cfg: CFGFunction) -> list[str]:
        """Get the entry blocks of a CFG function (starting points for traversal)."""
        if cfg.entry_block_id:
            return [cfg.entry_block_id]
        # Fallback: first block by key order
        keys = sorted(cfg.blocks.keys())
        return [keys[0]] if keys else []

    @staticmethod
    def _parse_block_id(bid_str: str) -> int:
        """Extract numeric ID from a block ID string (e.g. 'B3' → 3)."""
        m = re.search(r"(\d+)", bid_str)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _callee_names_match(pdg_callee: str, boundary_callee: str) -> bool:
        """Check if two function names refer to the same callee.

        Handles differences like ``"wslay_event_queue_fragmented_msg_ex"``
        vs ``"wslay_event_queue_fragmented_msg_ex(int, ...)"``.
        """
        if pdg_callee == boundary_callee:
            return True
        short_a = pdg_callee.split("(")[0].strip()
        short_b = boundary_callee.split("(")[0].strip()
        return short_a == short_b

    @staticmethod
    def _node_id(fn_name: str, block_id: str) -> str:
        """Generate a stable node ID from function name and block ID."""
        return f"_{fn_name}_{block_id}"


# ─── Shared helpers ──────────────────────────────────────────────────────────


def _extract_line_number(text: str) -> int:
    """Extract the first line number from text, or return 0.

    Matches patterns like:
    - ``"L572: r = wslay_frame_recv(...)"``
    - ``"  572: r = wslay_frame_recv(...)"`` (from source-snippet format)
    - ``"if [B3.1] (BinaryOperator) r >= 0"`` (CFG dump, no explicit line)
    """
    m = _LINE_NUMBER_RE.search(text)
    if m:
        return int(m.group(1))
    return 0
