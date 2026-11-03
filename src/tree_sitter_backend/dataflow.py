"""Statement-level definitions, uses, and reaching-definition analysis."""

from __future__ import annotations

import re
from collections import defaultdict

from tree_sitter_backend.models import FunctionGraph, GraphEdge
from tree_sitter_backend.syntax import SyntaxNode

_TYPE_NODES = {
    "primitive_type", "type_identifier", "sized_type_specifier",
    "namespace_identifier", "auto", "decltype",
}
_COMPOUND_ASSIGNMENT = re.compile(r"(?:<<|>>|[-+*/%&|^])=")


def split_def_use(
    node: SyntaxNode,
    text_of,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return direct writes and reads of one statement or condition."""
    if node.type == "update_expression":
        target = node.children[0] if node.children else None
        return (
            tuple(_target_names(target, text_of)),
            tuple(_expression_reads(target, text_of)),
        )
    if node.type == "assignment_expression":
        lhs = node.children[0] if node.children else None
        rhs = node.children[1] if len(node.children) > 1 else None
        defines = _target_names(lhs, text_of)
        uses: list[str] = []
        if _COMPOUND_ASSIGNMENT.search(text_of(node)):
            _extend_unique(uses, _expression_reads(lhs, text_of))
        _extend_unique(uses, _expression_reads(rhs, text_of))
        return tuple(defines), tuple(uses)

    defines: list[str] = []
    uses: list[str] = []

    for child in node.children:
        if child.type == "assignment_expression":
            lhs = child.children[0] if child.children else None
            rhs = child.children[1] if len(child.children) > 1 else None
            _extend_unique(defines, _target_names(lhs, text_of))
            if _COMPOUND_ASSIGNMENT.search(text_of(child)):
                _extend_unique(uses, _expression_reads(lhs, text_of))
            _extend_unique(uses, _expression_reads(rhs, text_of))
        elif child.type == "update_expression":
            target = child.children[0] if child.children else None
            _extend_unique(defines, _target_names(target, text_of))
            _extend_unique(uses, _expression_reads(target, text_of))
        elif child.type == "init_declarator":
            declarator = child.children[0] if child.children else None
            initializer = child.children[1] if len(child.children) > 1 else None
            _extend_unique(defines, _target_names(declarator, text_of))
            _extend_unique(uses, _expression_reads(initializer, text_of))
        elif child.type == "declaration":
            _extend_unique(defines, _declaration_definitions(child, text_of))
            _extend_unique(uses, _declaration_reads(child, text_of))
        else:
            _extend_unique(uses, _expression_reads(child, text_of))

    if not node.children:
        _extend_unique(uses, _expression_reads(node, text_of))
    return tuple(defines), tuple(uses)


def compute_reaching_definitions(graph: FunctionGraph) -> None:
    """Compute standard block-level IN/OUT reaching definitions.

    The implementation follows the CFG-based intent of
    ``csa_analysis/pdg/PDGBuilder.cpp`` while keeping a set of definitions for
    every variable.  A single global ``variable -> last definition`` map cannot
    represent mutually exclusive branch definitions correctly.
    """
    blocks = {block.block_id: block for block in graph.basic_blocks}
    nodes = {node.node_id: node for node in graph.nodes}
    predecessors: dict[int, set[int]] = {block_id: set() for block_id in blocks}
    for edge in graph.cfg_edges:
        if edge.source_block in predecessors and edge.target_block in predecessors:
            predecessors[edge.target_block].add(edge.source_block)

    defs_by_variable: dict[str, set[str]] = defaultdict(set)
    gen: dict[int, dict[str, set[str]]] = {}
    for block_id, block in blocks.items():
        current: dict[str, set[str]] = {}
        for node_id in block.node_ids:
            node = nodes.get(node_id)
            if node is None:
                continue
            for variable in node.defines:
                defs_by_variable[variable].add(node_id)
                current[variable] = {node_id}
        gen[block_id] = current

    all_defs = {var: set(defs) for var, defs in defs_by_variable.items()}
    out_sets: dict[int, dict[str, set[str]]] = {
        block_id: {} for block_id in blocks
    }
    changed = True
    iterations = 0
    while changed and iterations < max(2, len(blocks) * 4):
        changed = False
        iterations += 1
        for block_id in sorted(blocks):
            incoming: dict[str, set[str]] = {}
            for pred in predecessors[block_id]:
                for variable, definitions in out_sets[pred].items():
                    incoming.setdefault(variable, set()).update(definitions)
            outgoing = {
                variable: set(definitions)
                for variable, definitions in incoming.items()
            }
            for variable, definitions in gen[block_id].items():
                killed = all_defs[variable] - set(definitions)
                remaining = outgoing.get(variable, set()) - killed
                outgoing[variable] = remaining | set(definitions)
            if outgoing != out_sets[block_id]:
                out_sets[block_id] = outgoing
                changed = True

    existing = {
        (edge.source, edge.target, edge.variable)
        for edge in graph.edges if edge.kind == "data"
    }
    for block_id in sorted(blocks):
        incoming: dict[str, set[str]] = {}
        for pred in predecessors[block_id]:
            for variable, definitions in out_sets[pred].items():
                incoming.setdefault(variable, set()).update(definitions)
        for node_id in blocks[block_id].node_ids:
            node = nodes.get(node_id)
            if node is None:
                continue
            for variable in node.uses:
                for definition in sorted(incoming.get(variable, ())):
                    key = (definition, node_id, variable)
                    if definition != node_id and key not in existing:
                        graph.edges.append(
                            GraphEdge(
                                definition,
                                node_id,
                                "data",
                                "reaching-def",
                                variable,
                            )
                        )
                        existing.add(key)
            for variable in node.defines:
                incoming[variable] = {node_id}


def _target_names(node: SyntaxNode | None, text_of) -> list[str]:
    if node is None:
        return []
    if node.type == "identifier":
        return [text_of(node)]
    if node.type in {"field_expression", "subscript_expression"}:
        base = node.children[0] if node.children else None
        suffix = ""
        if node.type == "field_expression" and len(node.children) > 1:
            suffix = "." + text_of(node.children[-1])
        return [f"{name}{suffix}" for name in _target_names(base, text_of)]
    if node.type == "pointer_expression":
        target = node.children[0] if node.children else None
        return [f"*{name}" for name in _target_names(target, text_of)]
    if node.type in {"init_declarator", "pointer_declarator", "reference_declarator",
                     "array_declarator", "parenthesized_declarator"}:
        for child in node.children:
            names = _target_names(child, text_of)
            if names:
                return names
    return []


def _declaration_definitions(node: SyntaxNode, text_of) -> list[str]:
    result: list[str] = []
    for child in node.children:
        if child.type in _TYPE_NODES or child.type.endswith("_type"):
            continue
        _extend_unique(result, _target_names(child, text_of))
    return result


def _declaration_reads(node: SyntaxNode, text_of) -> list[str]:
    result: list[str] = []
    for child in node.children:
        if child.type in _TYPE_NODES or child.type.endswith("_type"):
            continue
        if child.type == "init_declarator":
            if len(child.children) > 1:
                _extend_unique(result, _expression_reads(child.children[1], text_of))
        elif child.type.endswith("declarator"):
            continue
        else:
            _extend_unique(result, _expression_reads(child, text_of))
    return result


def _expression_reads(node: SyntaxNode | None, text_of) -> list[str]:
    if node is None:
        return []
    if node.type == "identifier":
        return [text_of(node)]
    if node.type in _TYPE_NODES or node.type == "field_identifier":
        return []
    if node.type == "call_expression":
        result: list[str] = []
        target = node.children[0] if node.children else None
        if target is not None and target.type == "field_expression":
            base = target.children[0] if target.children else None
            _extend_unique(result, _expression_reads(base, text_of))
        for child in node.children[1:]:
            _extend_unique(result, _expression_reads(child, text_of))
        return result
    if node.type == "field_expression":
        return _target_names(node, text_of)
    if node.type == "subscript_expression":
        result: list[str] = []
        for child in node.children:
            _extend_unique(result, _expression_reads(child, text_of))
        return result
    if node.type == "qualified_identifier":
        return []
    result = []
    for child in node.children:
        _extend_unique(result, _expression_reads(child, text_of))
    return result


def _extend_unique(target: list[str], values) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)
