"""Data models for call graph representation and entry point discovery.

A call graph is a directed graph where nodes are functions and edges are
caller→callee relationships.  The graph is built from all translation units
in a project and then used to find *entry points* — functions that can reach
a given bug function through the call chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CallGraphNode:
    """A function definition in the call graph."""

    id: str  # qualified function name, e.g. "folly::detail::insertThousandsGroupingUnsafe"
    file: str  # source file path
    line: int
    linkage: str = "external"  # external | internal | none | other
    is_main: bool = False
    return_type: str = "void"
    param_types: list[str] = field(default_factory=list)
    is_method: bool = False  # True for class/struct methods
    class_name: str = ""  # class name if is_method


@dataclass
class CallGraphEdge:
    """A call from one function to another."""

    caller: str  # qualified name of the caller
    callee: str  # qualified name of the callee
    file: str = ""  # source file of the call site
    line: int = 0
    is_indirect: bool = False  # True for function pointer / virtual calls


@dataclass
class CallGraph:
    """Complete call graph with forward and reverse indices."""

    functions: dict[str, CallGraphNode] = field(default_factory=dict)
    calls_by_caller: dict[str, list[CallGraphEdge]] = field(default_factory=dict)
    calls_by_callee: dict[str, list[CallGraphEdge]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> CallGraph:
        """Load a call graph from the JSON produced by ``CallGraphBuilder``."""
        import json

        raw = json.loads(Path(path).read_text())

        cg = cls()
        for f in raw.get("functions", []):
            node = CallGraphNode(
                id=f["id"],
                file=f.get("file", ""),
                line=f.get("line", 0),
                linkage=f.get("linkage", "external"),
                is_main=f.get("is_main", False),
                return_type=f.get("return_type", "void"),
                param_types=list(f.get("param_types", [])),
                is_method=f.get("is_method", False),
                class_name=f.get("class_name", ""),
            )
            cg.functions[node.id] = node
            cg.calls_by_caller.setdefault(node.id, [])
            cg.calls_by_callee.setdefault(node.id, [])

        for c in raw.get("calls", []):
            edge = CallGraphEdge(
                caller=c["caller"],
                callee=c["callee"],
                file=c.get("file", ""),
                line=c.get("line", 0),
                is_indirect=c.get("is_indirect", False),
            )
            cg.calls_by_caller.setdefault(edge.caller, []).append(edge)
            cg.calls_by_callee.setdefault(edge.callee, []).append(edge)

        return cg

    @classmethod
    def from_dict(cls, data: dict) -> CallGraph:
        """Build a call graph from a dict (useful for tests)."""
        cg = cls()
        for fid, fdata in data.get("functions", {}).items():
            node = CallGraphNode(
                id=fid,
                file=fdata.get("file", ""),
                line=fdata.get("line", 0),
                linkage=fdata.get("linkage", "external"),
                is_main=fdata.get("is_main", False),
                return_type=fdata.get("return_type", "void"),
                param_types=list(fdata.get("param_types", [])),
                is_method=fdata.get("is_method", False),
                class_name=fdata.get("class_name", ""),
            )
            cg.functions[node.id] = node
            cg.calls_by_caller.setdefault(node.id, [])
            cg.calls_by_callee.setdefault(node.id, [])

        for c in data.get("calls", []):
            edge = CallGraphEdge(
                caller=c["caller"],
                callee=c["callee"],
                file=c.get("file", ""),
                line=c.get("line", 0),
                is_indirect=c.get("is_indirect", False),
            )
            cg.calls_by_caller.setdefault(edge.caller, []).append(edge)
            cg.calls_by_callee.setdefault(edge.callee, []).append(edge)

        return cg

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_callers(self, func_name: str) -> list[str]:
        """Return the qualified names of all functions that call *func_name*."""
        return [e.caller for e in self.calls_by_callee.get(func_name, [])]

    def get_callees(self, func_name: str) -> list[str]:
        """Return the qualified names of all functions called by *func_name*."""
        return [e.callee for e in self.calls_by_caller.get(func_name, [])]

    def find_node_by_qualified_name(self, name: str) -> Optional[CallGraphNode]:
        """Find a node by exact qualified name match."""
        return self.functions.get(name)

    def find_node_by_file_line(self, file_path: str, line: int, tolerance: int = 5) -> Optional[CallGraphNode]:
        """Find a node by source file and line (exact or within tolerance)."""
        for node in self.functions.values():
            if (node.file == file_path or node.file.endswith(file_path)) and abs(node.line - line) <= tolerance:
                return node
        return None

    def find_node_by_unqualified_name(self, name: str) -> Optional[CallGraphNode]:
        """Find a node whose qualified name ends with ``::name`` or equals *name*."""
        for node in self.functions.values():
            if node.id == name or node.id.endswith("::" + name):
                return node
        return None

    def find_nodes_matching(self, function_name: str, file_path: str = "", line: int = 0) -> list[CallGraphNode]:
        """Fuzzy match a bug function to call graph nodes.

        Tries (in order):
          1. Exact qualified name match
          2. File + line proximity
          3. Unqualified name suffix match
        """
        # Exact match first
        node = self.find_node_by_qualified_name(function_name)
        if node:
            return [node]

        # File + line proximity
        if file_path:
            node = self.find_node_by_file_line(file_path, line)
            if node:
                return [node]

        # Unqualified name suffix
        node = self.find_node_by_unqualified_name(function_name)
        if node:
            return [node]

        return []


@dataclass
class EntryPoint:
    """An entry point function that can reach the bug function."""

    function_name: str  # qualified name of the entry point
    node: CallGraphNode  # the node in the call graph
    score: int = 0  # higher = better entry point
    distance: int = 0  # call chain length from entry point to bug function


@dataclass
class EntryPointResult:
    """Result of entry point discovery for a single bug."""

    bug_function: str
    bug_file: str
    bug_line: int
    entry_points: list[EntryPoint] = field(default_factory=list)
    total_ancestors: int = 0
    error: Optional[str] = None
