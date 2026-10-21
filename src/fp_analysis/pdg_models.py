"""Data models for Program Dependence Graph (PDG), System Dependence Graph (SDG),
and backward slicing results.

These models serve as the data contract between the C++ PDG builder tool and
the Python-based backward slicing engine.

A PDG captures two kinds of dependencies *within a single function*:
- **Data dependence**: statement S₂ is data-dependent on S₁ if S₁ defines a
  variable that S₂ uses, and there is a def-clear path from S₁ to S₂.
- **Control dependence**: statement S₂ is control-dependent on a predicate P
  (if/while/for condition) if one execution of P leads to S₂ and another does
  not — that is, P determines whether S₂ executes.

An SDG links per-function PDGs with call-graph edges, enabling cross-function
backward slicing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

#: Version of the ``pdg_<project>.json`` artifact this code understands.  It
#: MUST equal ``Metadata["version"]`` in
#: ``src/llm_client/csa_analysis/pdg/PDGBuilder.cpp`` — that constant is the
#: builder's own statement of which graph shape it emits, and the two are
#: compared by ``tools/build_project_deps.py`` (before and after a build) and by
#: ``run_path_selection._resolve_paths`` (before the pipeline touches a report).
#:
#: Why this exists: ``pdg_protobuf.json`` sat at 1.0 for weeks while the source
#: declared 1.2, so the protobuf corpus was analysed with a graph that lacked
#: the macro-branch annotation and the postdominator-derived control
#: dependencies — the stage-6 gate then compared condition texts the old
#: builder had attributed to macro bookkeeping nodes instead of the branch
#: predicates.  A stale artifact parses fine; only its declared version says so.
#:
#: 1.3: node ``expression`` is stored in full (no 117-char cap).  The cap cut
#: exactly the machine-generated texts — template instantiations, macro
#: expansions — so conditions that differ beyond char 117 became byte-identical
#: and stage 6 paired them.  Consumers that need shorter text must cap when they
#: render, not when they store.
PDG_ARTIFACT_VERSION = "1.3"

# ---------------------------------------------------------------------------
# PDG components (intra-procedural)
# ---------------------------------------------------------------------------


@dataclass
class PDGNode:
    """A single statement-level node in a function's Program Dependence Graph.

    Each statement or expression in the function body that has a side effect
    or defines/uses a variable becomes a PDG node.  The unique identifier
    ``id`` follows the convention ``S_<line>_<col>``, with a disambiguation
    suffix when multiple statements share the same line and column.

    Attributes:
        id: Unique node identifier within the function (e.g. ``"S_22_5"``).
        kind: Node kind — ``"decl"``, ``"assign"``, ``"call_expr"``,
            ``"if_cond"``, ``"while_cond"``, ``"for_cond"``, ``"return"``,
            ``"binary_op"``, ``"unary_op"``, ``"declref"``, ``"member_expr"``,
            ``"array_subscript"``.
        variable: The variable *defined* by this node (empty if none).
        expression: Human-readable textual representation of the expression.
        type: C++ type string of the defined variable or expression result.
        source_line: 1-based source line number.
        source_col: 1-based source column number.
        block_id: Index of the containing Clang CFG basic block.
        is_call: True if this node represents a function call expression.
        callee: Qualified callee name (only when ``is_call`` is True).
        actual_params: Variable names passed as actual arguments (only when
            ``is_call`` is True). Positional order matches the call site.
        is_macro: True when the node's text was produced by a macro expansion
            (``SourceLocation::isMacroID``). ``source_line``/``source_col``
            still hold the *expansion* (call-site) coordinates; the fields
            below record where the macro body lives.
        macro_name: Immediate macro name (``"READANDCHECK"``), empty when the
            node is not macro-derived.
        macro_def_file: File the expanded token is spelled in — i.e. the macro
            definition header (``io_macros.h``), not the caller.
        macro_def_line: Line inside ``macro_def_file``.
    """

    id: str
    kind: str = "event"
    variable: str = ""
    expression: str = ""
    type: str = ""
    source_line: int = 0
    source_col: int = 0
    block_id: int = 0
    is_call: bool = False
    callee: str = ""
    actual_params: list[str] = field(default_factory=list)
    is_macro: bool = False
    macro_name: str = ""
    macro_def_file: str = ""
    macro_def_line: int = 0


@dataclass
class DefUseEdge:
    """A data-dependence edge from a definition to a use.

    Attributes:
        def_node_id: Node ID of the statement that defines ``variable``.
        use_node_id: Node ID of the statement that uses ``variable``.
        variable: The variable name flowing from def to use.
    """

    def_node_id: str
    use_node_id: str
    variable: str


@dataclass
class ControlDepEdge:
    """A control-dependence edge from a predicate to a dependent statement.

    Attributes:
        source_id: Node ID of the control predicate (if/while/for condition).
        target_id: Node ID of the statement whose execution is controlled.
        kind: The kind of control dependence — ``"if_branch"``, ``"loop_body"``,
            ``"loop_exit"``, ``"switch_case"``.
        is_macro_branch: True when the predicate (``source_id``) is text
            produced by a macro expansion — i.e. this branch exists only
            because a macro like ``READANDCHECK`` was expanded at the source
            line. Lets macro-implied branches be counted separately from
            hand-written ones.
    """

    source_id: str
    target_id: str
    kind: str = "if_branch"
    is_macro_branch: bool = False


@dataclass
class FunctionSummary:
    """Summary of a function's data-flow interface for cross-function slicing.

    This summary enables the backward slicer to propagate dependencies across
    function-call boundaries *without* recursively analysing the callee's body.

    Attributes:
        input_variables: Variables whose values are read before being written
            inside the function.  Includes parameters read, globals read, and
            heap memory read via pointer parameters.
        output_variables: Variables whose values are written and survive past
            the function's exit.  Includes return value, pointer/reference
            parameters written through, and globals written.
        input_output_dep: Pairs recording which outputs depend (transitively
            via data-flow) on which inputs.
    """

    input_variables: list[str] = field(default_factory=list)
    output_variables: list[str] = field(default_factory=list)
    input_output_dep: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Function-level PDG
# ---------------------------------------------------------------------------


@dataclass
class FunctionPDG:
    """Program Dependence Graph for a single function.

    Combines the function's PDG nodes with data-dependence and
    control-dependence edges, plus pre-computed indices for efficient
    backward traversal during slicing.

    Attributes:
        function_name: Qualified function name (e.g. ``"MyClass::method"``).
        source_file: Absolute or project-relative source file path.
        nodes: All PDG nodes in the function, keyed by node ID.
        def_use_edges: Data-dependence edges.
        control_dep_edges: Control-dependence edges.
        summary: Function data-flow summary for cross-function propagation.
    """

    function_name: str
    source_file: str
    nodes: dict[str, PDGNode] = field(default_factory=dict)
    def_use_edges: list[DefUseEdge] = field(default_factory=list)
    control_dep_edges: list[ControlDepEdge] = field(default_factory=list)
    summary: FunctionSummary = field(default_factory=FunctionSummary)

    # ---- Indices (built in __post_init__) ----

    defs_by_variable: dict[str, list[str]] = field(init=False, repr=False)
    uses_by_variable: dict[str, list[str]] = field(init=False, repr=False)
    data_preds_by_node: dict[str, list[str]] = field(init=False, repr=False)
    data_succs_by_node: dict[str, list[str]] = field(init=False, repr=False)
    ctrl_preds_by_node: dict[str, list[str]] = field(init=False, repr=False)
    ctrl_succs_by_node: dict[str, list[str]] = field(init=False, repr=False)

    def __post_init__(self):
        """Build reverse and forward indices for fast graph traversal."""
        # --- defs_by_variable ---
        d: dict[str, list[str]] = {}
        for node_id, node in self.nodes.items():
            if node.variable:
                d.setdefault(node.variable, []).append(node_id)
        self.defs_by_variable = d

        # --- uses_by_variable ---
        u: dict[str, list[str]] = {}
        for due in self.def_use_edges:
            u.setdefault(due.variable, []).append(due.use_node_id)
        self.uses_by_variable = u

        # --- Data dependence indices ---
        dp: dict[str, list[str]] = {}  # predecessor (def -> use)
        ds: dict[str, list[str]] = {}  # successor (use -> def)
        for due in self.def_use_edges:
            dp.setdefault(due.use_node_id, []).append(due.def_node_id)
            ds.setdefault(due.def_node_id, []).append(due.use_node_id)
        self.data_preds_by_node = dp
        self.data_succs_by_node = ds

        # --- Control dependence indices ---
        cp: dict[str, list[str]] = {}  # target -> [source predicate]
        cs: dict[str, list[str]] = {}  # source predicate -> [target]
        for cde in self.control_dep_edges:
            cp.setdefault(cde.target_id, []).append(cde.source_id)
            cs.setdefault(cde.source_id, []).append(cde.target_id)
        self.ctrl_preds_by_node = cp
        self.ctrl_succs_by_node = cs

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_definitions_of(self, variable: str) -> list[str]:
        """Return node IDs that define *variable*."""
        return self.defs_by_variable.get(variable, [])

    def get_uses_of(self, variable: str) -> list[str]:
        """Return node IDs that use *variable*."""
        return self.uses_by_variable.get(variable, [])

    def backward_dep_nodes(self, node_id: str) -> set[str]:
        """Return the *direct* backward dependencies of *node_id*.

        Includes both:
        - Data-dependence predecessors (defs of variables used by this node)
        - Control-dependence predecessors (predicates that control this node)

        Returns an empty set if *node_id* is not found in any indices.
        """
        deps: set[str] = set()
        deps.update(self.data_preds_by_node.get(node_id, []))
        deps.update(self.ctrl_preds_by_node.get(node_id, []))
        return deps

    def forward_dep_nodes(self, node_id: str) -> set[str]:
        """Return the *direct* forward dependents of *node_id*."""
        deps: set[str] = set()
        deps.update(self.data_succs_by_node.get(node_id, []))
        deps.update(self.ctrl_succs_by_node.get(node_id, []))
        return deps

    @property
    def is_empty(self) -> bool:
        """True if this PDG has no nodes."""
        return len(self.nodes) == 0


# ---------------------------------------------------------------------------
# SDG — System Dependence Graph (linked per-function PDGs + call graph)
# ---------------------------------------------------------------------------


@dataclass
class SDG:
    """System Dependence Graph — links per-function PDGs with a call graph.

    The SDG enables cross-function backward slicing: when the slicer reaches
    a call node, it can recurse into the callee's PDG.

    Attributes:
        pdgs: Per-function PDGs keyed by qualified function name.
        callgraph: The project's ``CallGraph``.
        file_index: ``{source_file → [FunctionPDG]}`` — lazily built on
            first call to :meth:`build_file_index`.  Drastically speeds up
            ``find_pdg_nodes_for_event`` by avoiding a full scan of all
            24K+ functions for each event.
        line_index: ``{(source_file, line) → [(fn_name, node_id)]}`` —
            lazily built on first call to :meth:`build_line_index`.
            Enables O(1) line-based node lookup.
    """

    pdgs: dict[str, FunctionPDG] = field(default_factory=dict)
    callgraph: object = None  # CallGraph, imported lazily to avoid circular deps
    file_index: dict[str, list[FunctionPDG]] | None = field(
        default=None, init=False, repr=False,
    )
    line_index: dict[tuple[str, int], list[tuple[str, str]]] | None = field(
        default=None, init=False, repr=False,
    )

    def build_file_index(self) -> dict[str, list[FunctionPDG]]:
        """Build ``{source_file → [FunctionPDG]}`` for fast file-based lookup.

        Maps each unique source file path to the list of FunctionPDG objects
        from that file.  The index is cached after first build.

        Memory: ~400 KB for 24K functions across 760 unique files.
        """
        if self.file_index is not None:
            return self.file_index
        idx: dict[str, list[FunctionPDG]] = {}
        for pdg in self.pdgs.values():
            sf = pdg.source_file
            if sf:
                idx.setdefault(sf, []).append(pdg)
        self.file_index = idx
        return idx

    def build_line_index(
        self,
    ) -> dict[tuple[str, int], list[tuple[str, str]]]:
        """Build ``{(source_file, line) → [(fn_name, node_id)]}``.

        Enables O(1) lookup of PDG nodes by file+line during event matching.
        The index is cached after first build.

        Memory: ~18 MB for 134K nodes across 24K functions (see
        :meth:`estimate_memory` for details).  Only build when the
        line-based lookup performance is critical.
        """
        if self.line_index is not None:
            return self.line_index
        idx: dict[tuple[str, int], list[tuple[str, str]]] = {}
        for fn_name, pdg in self.pdgs.items():
            sf = pdg.source_file
            if not sf:
                continue
            for nid, node in pdg.nodes.items():
                if node.source_line > 0:
                    key = (sf, node.source_line)
                    idx.setdefault(key, []).append((fn_name, nid))
        self.line_index = idx
        return idx

    def file_index_memory(self) -> dict:
        """Return memory estimate for the file index."""
        if self.file_index is None:
            return {"built": False}
        n_files = len(self.file_index)
        total_fn_refs = sum(len(v) for v in self.file_index.values())
        return {
            "built": True,
            "unique_files": n_files,
            "function_references": total_fn_refs,
            "estimated_kb": round(
                (n_files * 232 + total_fn_refs * 8 + n_files * 64) / 1024, 1,
            ),
        }

    def line_index_memory(self) -> dict:
        """Return memory estimate for the line index."""
        if self.line_index is None:
            return {"built": False}
        n_keys = len(self.line_index)
        total_refs = sum(len(v) for v in self.line_index.values())
        key_bytes = n_keys * 56
        dict_bytes = n_keys * 232
        val_tuple_bytes = total_refs * 56
        list_overhead = n_keys * 64 + total_refs * 8
        total = key_bytes + dict_bytes + val_tuple_bytes + list_overhead
        return {
            "built": True,
            "unique_keys": n_keys,
            "total_references": total_refs,
            "estimated_kb": round(total / 1024, 1),
        }

    def get_pdg(self, function_name: str) -> Optional[FunctionPDG]:
        """Look up a PDG by qualified function name.

        Tries (in order):
        1. Exact qualified name match
        2. Suffix match (``ends_with("::" + name)``)
        3. Unqualified name match (splits on ``::``)

        Returns ``None`` if no match is found.
        """
        # Exact match
        if function_name in self.pdgs:
            return self.pdgs[function_name]

        # Suffix match (qualified caller stores ``ClassName::method``,
        # but the target may be just ``method``)
        for qname, pdg in self.pdgs.items():
            if qname.endswith("::" + function_name):
                return pdg

        # Unqualified match (last component)
        for qname, pdg in self.pdgs.items():
            parts = qname.split("::")
            if parts and parts[-1] == function_name:
                return pdg

        return None

    def get_callee_pdg(
        self, caller: str, callee: str
    ) -> Optional[FunctionPDG]:
        """Look up the PDG of a callee function.

        Optionally resolves through the call graph to find the actual
        qualified callee name if the caller is known.
        """
        # Direct lookup first
        pdg = self.get_pdg(callee)
        if pdg is not None:
            return pdg

        # Try resolving via call graph
        if self.callgraph is not None:
            try:
                callees = self.callgraph.get_callees(caller)
            except Exception:
                callees = []

            for c_name in callees:
                if callee in c_name or c_name.endswith("::" + callee):
                    return self.pdgs.get(c_name)

        return None

    @property
    def function_names(self) -> list[str]:
        """Return all qualified function names in the SDG."""
        return list(self.pdgs.keys())


# ---------------------------------------------------------------------------
# Slicing result types
# ---------------------------------------------------------------------------


@dataclass
class SliceNode:
    """A single node in the computed backward slice.

    Attributes:
        function_name: Qualified function name this node belongs to.
        source_file: Source file path for this node.
        node: The underlying PDGNode.
        reason: Why this node was included — ``"trigger"``, ``"data_dep"``,
            ``"control_dep"``, ``"call_context"``, ``"entry_point"``.
    """

    function_name: str
    source_file: str
    node: PDGNode
    reason: str = "data_dep"


@dataclass
class SliceResult:
    """Result of a backward-slicing operation from a bug trigger point.

    Attributes:
        trigger_function: The function containing the slicing criterion.
        trigger_node_id: Node ID of the slicing criterion.
        nodes: All nodes in the slice, ordered by insertion (backward chain).
        input_variables: Variables used within the slice but not defined by
            any slice node — these form the *frontier* inputs that the
            reduced program must receive from outside.
        involved_functions: Names of all functions that have nodes in the
            slice (including the trigger function).
        involves_globals: True if any slice node reads or writes a global
            or static variable.
        truncated: True if the slice was truncated due to depth limit.
    """

    trigger_function: str
    trigger_node_id: str
    nodes: list[SliceNode] = field(default_factory=list)
    input_variables: set[str] = field(default_factory=set)
    involved_functions: set[str] = field(default_factory=set)
    involves_globals: bool = False
    truncated: bool = False
