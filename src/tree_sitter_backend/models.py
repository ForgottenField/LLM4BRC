"""Data model for the lightweight Tree-sitter analysis backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    function: str
    kind: str
    file: str
    line: int
    end_line: int
    text: str
    variables: tuple[str, ...] = ()
    block_id: int = 0
    defines: tuple[str, ...] = ()
    uses: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    kind: str
    confidence: str = "syntactic"
    variable: str = ""


@dataclass
class FunctionGraph:
    function: str
    file: str
    start_line: int
    end_line: int
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    basic_blocks: list[BasicBlock] = field(default_factory=list)
    cfg_edges: list[CFGEdge] = field(default_factory=list)
    entry_block: int = 0
    exit_block: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BasicBlock:
    block_id: int
    kind: str
    start_line: int
    end_line: int
    node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CFGEdge:
    source_block: int
    target_block: int
    kind: str
    label: str = ""


@dataclass(frozen=True)
class FunctionInfo:
    function_id: str
    name: str
    qualified_name: str
    file: str
    start_line: int
    end_line: int
    parameters: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CallRef:
    caller: str
    callee: str
    file: str
    line: int
    argument_count: int = 0
    resolved_function_id: str = ""
    confidence: str = "syntactic"
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CodeIndex:
    """Index for a bounded set of source files."""

    root: Path
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    functions: dict[str, FunctionInfo] = field(default_factory=dict)
    graphs: dict[str, FunctionGraph] = field(default_factory=dict)
    calls: list[CallRef] = field(default_factory=list)
    includes: dict[str, list[str]] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    compile_commands: str = ""
    preprocessor_mode: str = "raw"

    def merge(self, other: CodeIndex) -> None:
        """Merge another bounded index into this one."""
        self.files.update(other.files)
        self.functions.update(other.functions)
        self.graphs.update(other.graphs)
        seen_calls = {
            (call.caller, call.file, call.line, call.callee)
            for call in self.calls
        }
        self.calls.extend(
            call for call in other.calls
            if (call.caller, call.file, call.line, call.callee) not in seen_calls
        )
        self.includes.update(other.includes)
        self.errors.extend(other.errors)
        if other.compile_commands:
            self.compile_commands = other.compile_commands
        if other.preprocessor_mode != "raw":
            self.preprocessor_mode = other.preprocessor_mode

    def function_for(self, file: str, line: int) -> FunctionInfo | None:
        matches = [
            fn for fn in self.functions.values()
            if fn.file == file and fn.start_line <= line <= fn.end_line
        ]
        if not matches:
            return None
        return min(matches, key=lambda fn: fn.end_line - fn.start_line)

    def focus_graphs(self, function_names: set[str] | None = None) -> list[FunctionGraph]:
        names = {name.split("::")[-1] for name in (function_names or set()) if name}
        graphs = list(self.graphs.values())
        if not names:
            return graphs
        return [
            graph for graph in graphs
            if graph.function in names
            or graph.function.split("::")[-1] in names
        ]

    def render_evidence(
        self,
        function_names: set[str] | None = None,
        max_functions: int = 6,
    ) -> str:
        graphs = self.focus_graphs(function_names)
        if not graphs:
            return ""
        lines = [
            "## Tree-sitter AST / PDG-Lite Evidence",
            f"Preprocessor mode: {self.preprocessor_mode}. This evidence is "
            "syntax-derived; calls may remain incomplete around templates, "
            "overloads, and external APIs.",
        ]
        for graph in graphs[:max_functions]:
            lines.append(
                f"- Function `{graph.function}` at {Path(graph.file).name}:"
                f"{graph.start_line}-{graph.end_line}"
            )
            control_nodes = [n for n in graph.nodes if n.kind == "control"]
            call_edges = [e for e in graph.edges if e.kind == "call"]
            data_edges = [e for e in graph.edges if e.kind == "data"]
            for node in control_nodes[:8]:
                lines.append(
                    f"  - branch L{node.line}: {node.text[:180]}"
                )
            for edge in call_edges[:10]:
                source = next((n for n in graph.nodes if n.node_id == edge.source), None)
                if source:
                    lines.append(
                        f"  - call L{source.line}: {source.text[:180]} "
                        f"({edge.confidence})"
                    )
            graph_calls = [
                call for call in self.calls
                if call.caller == graph.function
                and graph.start_line <= call.line <= graph.end_line
            ]
            rendered_call_lines = {
                line.rsplit("L", 1)[-1].split(":", 1)[0]
                for line in lines if line.lstrip().startswith("- call L")
            }
            for call in graph_calls[:10]:
                if str(call.line) in rendered_call_lines:
                    continue
                lines.append(
                    f"  - call L{call.line}: "
                    f"{call.text or call.callee + '()'} "
                    f"({call.confidence})"
                )
            if data_edges:
                lines.append(
                    f"  - reaching-definition dependencies: {len(data_edges)} "
                    f"(CFG-based)"
                )
            lines.append(
                f"  - CFG: {len(graph.basic_blocks)} blocks, "
                f"{len(graph.cfg_edges)} edges; "
                f"graph nodes: {len(graph.nodes)}, dependencies: "
                f"{len(graph.edges)}"
            )
        if len(graphs) > max_functions:
            lines.append(f"- … {len(graphs) - max_functions} more function graph(s)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "tree-sitter",
            "version": "1",
            "precision": "syntax",
            "root": str(self.root),
            "files": self.files,
            "functions": [fn.to_dict() for fn in self.functions.values()],
            "calls": [call.to_dict() for call in self.calls],
            "includes": self.includes,
            "errors": self.errors,
            "compile_commands": self.compile_commands,
            "preprocessor_mode": self.preprocessor_mode,
        }
