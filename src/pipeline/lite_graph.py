"""Application adapter for the independent Tree-sitter analysis backend."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fp_analysis.pdg_models import (
    ControlDepEdge,
    DefUseEdge,
    FunctionPDG,
    PDGNode,
    SDG,
)
from pipeline.layout import get_logger
from tree_sitter_backend import (
    CodeIndex,
    build_index_isolated,
    default_compile_commands,
)

logger = get_logger()


@dataclass(frozen=True)
class LiteGraphEvidence:
    enabled: bool
    prompt_text: str
    summary: dict[str, Any]
    error: str = ""
    index: CodeIndex | None = field(default=None, repr=False)
    sdg: SDG | None = field(default=None, repr=False)

    @property
    def available(self) -> bool:
        return bool(self.sdg and self.sdg.pdgs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "tree-sitter",
            "enabled": self.enabled,
            "precision": "syntax",
            **self.summary,
            "error": self.error or None,
            "prompt_evidence": self.prompt_text or None,
        }


def build_lite_graph(
    loaded,
    source_root: str | Path,
    *,
    enabled: bool = True,
    max_files: int = 32,
    timeout: float = 30.0,
    compile_commands: str | Path | None = None,
) -> LiteGraphEvidence:
    """Build a bounded syntax graph in an isolated worker.

    The pipeline consumes the returned approximate SDG as its fast-path graph.
    It remains explicitly lower-confidence than the Clang/LibTooling artifact,
    which callers can choose as a fallback when syntax evidence is unavailable.
    """
    if not enabled:
        return LiteGraphEvidence(False, "", {"reason": "disabled"})

    root = Path(source_root).resolve()
    compile_db_path = (
        Path(compile_commands).expanduser().resolve()
        if compile_commands is not None
        else default_compile_commands(root)
    )
    try:
        finding = loaded.finding
        source_files: list[Path] = []
        unresolved = 0
        for reference in loaded.code.references(finding):
            if not reference.resolution.local:
                unresolved += 1
                continue
            source_files.append(reference.resolution.local)

        source_files = list(dict.fromkeys(source_files))[:max_files]
        if not source_files:
            return LiteGraphEvidence(
                True, "", {"source_files": 0},
                error="no report source files resolved",
            )
        index, worker_error = build_index_isolated(
            source_files,
            root,
            compile_commands=compile_db_path,
            timeout=timeout,
        )
        failures: list[str] = []
        if index is None:
            # A native parser failure affects only one report file.  Retry the
            # bounded file set one at a time so one pathological TU cannot
            # discard evidence from every other source file in the report.
            index = CodeIndex(root=root)
            per_file_timeout = max(2.0, min(timeout, 15.0))
            for source_file in source_files:
                partial, error = build_index_isolated(
                    [source_file],
                    root,
                    compile_commands=compile_db_path,
                    timeout=per_file_timeout,
                )
                if partial is None:
                    failures.append(f"{source_file.name}: {error}")
                    continue
                index.merge(partial)
            if not index.files:
                raise RuntimeError(worker_error or "; ".join(failures))
            logger.warning(
                "Tree-sitter isolated recovery skipped %d of %d file(s): %s",
                len(failures), len(source_files), "; ".join(failures),
            )
        preprocess_errors = [
            error for error in index.errors
            if error.get("kind") == "preprocess"
        ]
        if preprocess_errors:
            detail = "; ".join(
                f"{Path(error.get('file', '')).name}: {error.get('error', '')}"
                for error in preprocess_errors[:4]
            )
            raise RuntimeError(
                f"preprocessing failed for {len(preprocess_errors)} source "
                f"file(s): {detail}"
            )
        if not index.files:
            raise RuntimeError("Tree-sitter parsed no source files")
        sdg = _build_lite_sdg(index)
        focus = {
            finding.function,
            *(step.function for step in finding.steps if step.function),
            *(step.callee for step in finding.steps if step.callee),
        }
        text = index.render_evidence(focus)
    except Exception as exc:  # noqa: BLE001 - advisory backend must not block exact PDG
        logger.warning("Tree-sitter graph unavailable; using exact graph only: %s", exc)
        return LiteGraphEvidence(
            True, "", {"source_files": 0},
            error=str(exc),
        )

    summary = {
        "source_files": len(source_files),
        "parsed_files": len(index.files),
        "parsed_functions": len(index.functions),
        "call_sites": len(index.calls),
        "parse_error_files": sum(1 for info in index.files.values() if info["has_error"]),
        "unresolved_references": unresolved,
        "includes": sum(len(paths) for paths in index.includes.values()),
        "graph_functions": len(sdg.pdgs),
        "worker_failures": failures,
        "compile_commands": str(compile_db_path),
        "preprocessor_mode": index.preprocessor_mode,
    }
    return LiteGraphEvidence(True, text, summary, index=index, sdg=sdg)


def _build_lite_sdg(index: CodeIndex) -> SDG:
    """Adapt the independent syntax graph to the pipeline's PDG contract."""
    pdgs: dict[str, FunctionPDG] = {}
    calls_by_function: dict[str, list] = {}
    for call in index.calls:
        calls_by_function.setdefault(call.caller, []).append(call)

    for graph in index.graphs.values():
        function_name = graph.function
        if function_name in pdgs:
            function_name = f"{function_name}#L{graph.start_line}"
        nodes: dict[str, PDGNode] = {}
        for graph_node in graph.nodes:
            node_kind = _pdg_node_kind(graph_node)
            nodes[graph_node.node_id] = PDGNode(
                id=graph_node.node_id,
                kind=node_kind,
                variable=graph_node.variables[0] if graph_node.variables else "",
                expression=graph_node.text,
                source_line=graph_node.line,
                source_col=0,
                block_id=graph_node.block_id,
            )

        data_edges = [
            DefUseEdge(edge.source, edge.target, edge.variable)
            for edge in graph.edges
            if edge.kind == "data" and edge.variable
            and edge.source in nodes and edge.target in nodes
        ]
        control_edges = [
            ControlDepEdge(edge.source, edge.target)
            for edge in graph.edges
            if edge.kind == "control"
            and edge.source in nodes and edge.target in nodes
        ]

        for call in calls_by_function.get(graph.function, []):
            if not (graph.start_line <= call.line <= graph.end_line):
                continue
            candidates = [
                node for node in nodes.values()
                if node.source_line == call.line
            ]
            if candidates:
                target = candidates[-1]
            else:
                node_id = (
                    "ts_call_"
                    + hashlib.sha1(
                        f"{graph.file}:{call.line}:{call.callee}".encode()
                    ).hexdigest()[:16]
                )
                target = PDGNode(
                    id=node_id,
                    kind="call_expr",
                    expression=call.callee,
                    source_line=call.line,
                )
                nodes[node_id] = target
            target.is_call = True
            target.callee = call.callee

        pdgs[function_name] = FunctionPDG(
            function_name=function_name,
            source_file=graph.file,
            nodes=nodes,
            def_use_edges=data_edges,
            control_dep_edges=control_edges,
        )
    return SDG(pdgs=pdgs)


def _pdg_node_kind(node) -> str:
    if node.kind == "control":
        return "binary_op"
    text = node.text.lstrip()
    if text.startswith("return"):
        return "return"
    if text.startswith(("case ", "default:")):
        return "call_expr"
    if "=" in text:
        return "assign"
    if "(" in text:
        return "call_expr"
    return "decl"
