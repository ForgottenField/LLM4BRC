"""Structured C/C++ control-flow graph construction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from tree_sitter_backend.dataflow import (
    _expression_reads,
    _target_names,
    compute_reaching_definitions,
    split_def_use,
)
from tree_sitter_backend.models import (
    BasicBlock,
    CFGEdge,
    FunctionGraph,
    GraphEdge,
    GraphNode,
)
from tree_sitter_backend.syntax import SyntaxNode

NodeFactory = Callable[..., GraphNode]

_CONDITION_TYPES = {
    "if_statement", "while_statement", "do_statement", "for_statement",
    "for_range_loop", "switch_statement",
}
_TERMINATORS = {
    "return_statement", "throw_statement", "co_return_statement",
}
_STATEMENT_TYPES = {
    "declaration", "expression_statement", "return_statement",
    "throw_statement", "co_return_statement", "break_statement",
    "continue_statement", "goto_statement", "delete_expression",
    "using_declaration", "type_definition", "static_assert_declaration",
    "update_expression", "assignment_expression", "init_declarator",
}


@dataclass
class _Block:
    block_id: int
    kind: str = "normal"
    start_line: int = 0
    end_line: int = 0
    node_ids: list[str] = field(default_factory=list)


class CFGBuilder:
    """Translate a structured syntax tree into basic blocks."""

    def __init__(
        self,
        graph: FunctionGraph,
        text_of,
        make_node: NodeFactory,
    ) -> None:
        self.graph = graph
        self.text_of = text_of
        self.make_node = make_node
        self._blocks: list[_Block] = [
            _Block(0, "entry"),
            _Block(1, "exit"),
        ]
        self._edges: set[CFGEdge] = set()
        self._current = 0
        self._reachable = True
        self._break_targets: list[int] = []
        self._continue_targets: list[int] = []
        self._predicates_by_block: dict[int, str] = {}

    def build(self, body: SyntaxNode, entry_node: GraphNode) -> None:
        self._blocks[0].node_ids.append(entry_node.node_id)
        self.graph.nodes.append(entry_node)
        self._visit(body)
        if self._reachable:
            self._edge(self._current, 1, "fallthrough")

        self.graph.basic_blocks = [
            BasicBlock(
                block_id=block.block_id,
                kind=block.kind,
                start_line=block.start_line,
                end_line=block.end_line,
                node_ids=tuple(block.node_ids),
            )
            for block in self._blocks
        ]
        self.graph.cfg_edges = sorted(
            self._edges,
            key=lambda edge: (
                edge.source_block, edge.target_block, edge.kind, edge.label,
            ),
        )
        self._add_control_dependencies()
        compute_reaching_definitions(self.graph)

    def _visit(self, node: SyntaxNode | None) -> None:
        if node is None:
            return
        if node.type in {"compound_statement", "declaration_list"}:
            for child in node.children:
                self._visit(child)
            return
        if node.type in _CONDITION_TYPES:
            handler = getattr(self, f"_visit_{node.type}")
            handler(node)
            return
        if node.type in _TERMINATORS:
            self._statement(node)
            self._terminate("return")
            return
        if node.type == "break_statement":
            self._statement(node)
            if self._break_targets:
                self._edge(self._current, self._break_targets[-1], "break")
            self._unreachable()
            return
        if node.type == "continue_statement":
            self._statement(node)
            if self._continue_targets:
                self._edge(self._current, self._continue_targets[-1], "continue")
            self._unreachable()
            return
        if node.type in _STATEMENT_TYPES or node.type.endswith("_statement"):
            self._statement(node)
            return
        if node.type == "else_clause":
            for child in node.children:
                self._visit(child)
            return
        if node.type == "try_statement":
            self._visit_try(node)
            return
        if node.type == "case_statement":
            self._visit_case(node)
            return
        if node.type == "labeled_statement":
            for child in node.children[1:]:
                self._visit(child)
            return
        if node.type in {"preproc_ifdef", "preproc_if", "preproc_else", "preproc_elif"}:
            # Configured runs should not retain conditional directives.  A raw
            # parse keeps all alternatives rather than pretending one branch is
            # the build configuration.
            for child in node.children:
                if child.type != "identifier":
                    self._visit(child)
            return
        if node.type == "function_definition":
            return
        if node.type in {
            "comment", "preproc_include", "preproc_def", "preproc_function_def",
            "preproc_call", "preproc_arg",
        }:
            return

    def _visit_if_statement(self, node: SyntaxNode) -> None:
        condition = _first_child(node, {"condition_clause"})
        consequence = _control_body(node, condition)
        alternative = _first_child(node, {"else_clause"})
        self._control(node, condition)
        then_block = self._new_block("branch_true")
        else_block = self._new_block("branch_false")
        join_block = self._new_block("join")
        self._edge(self._current, then_block, "true")
        self._edge(self._current, else_block, "false")

        self._current = then_block
        self._reachable = True
        self._visit(consequence)
        if self._reachable:
            self._edge(self._current, join_block, "join")

        self._current = else_block
        self._reachable = True
        if alternative is not None:
            self._visit(alternative)
        if self._reachable:
            self._edge(self._current, join_block, "join")

        join_reachable = any(
            edge.target_block == join_block for edge in self._edges
        )
        self._current = join_block if join_reachable else -1
        self._reachable = join_reachable

    def _visit_while_statement(self, node: SyntaxNode) -> None:
        condition = _first_child(node, {"condition_clause"})
        body = _control_body(node, condition)
        incoming_block = self._current
        condition_block = self._new_block("loop_condition")
        self._edge(incoming_block, condition_block, "loop_entry")
        self._current = condition_block
        predicate = self._control(node, condition)
        body_block = self._new_block("loop_body")
        exit_block = self._new_block("loop_exit")
        self._edge(condition_block, body_block, "true", "loop")
        self._edge(condition_block, exit_block, "false", "loop")
        self._break_targets.append(exit_block)
        self._continue_targets.append(condition_block)
        self._current = body_block
        self._reachable = True
        self._visit(body)
        self._continue_targets.pop()
        self._break_targets.pop()
        if self._reachable:
            self._edge(self._current, condition_block, "loop_back")
        self._current = exit_block
        self._reachable = True
        self._predicates_by_block[condition_block] = predicate

    def _visit_do_statement(self, node: SyntaxNode) -> None:
        condition = _first_child(node, {"condition_clause"})
        body = _control_body(node, None)
        body_block = self._new_block("loop_body")
        condition_block = self._new_block("loop_condition")
        exit_block = self._new_block("loop_exit")
        self._edge(self._current, body_block, "loop_entry")
        self._break_targets.append(exit_block)
        self._continue_targets.append(condition_block)
        self._current = body_block
        self._reachable = True
        self._visit(body)
        self._continue_targets.pop()
        self._break_targets.pop()
        if self._reachable:
            self._edge(self._current, condition_block, "loop_test")
        self._current = condition_block
        self._reachable = True
        predicate = self._control(node, condition)
        self._edge(condition_block, body_block, "true", "loop")
        self._edge(condition_block, exit_block, "false", "loop")
        self._current = exit_block
        self._reachable = True
        self._predicates_by_block[condition_block] = predicate

    def _visit_for_statement(self, node: SyntaxNode) -> None:
        children = list(node.children)
        body = children[-1] if children else None
        update = next(
            (child for child in children if child.type == "update_expression"),
            None,
        )
        remaining = [
            child for child in children
            if child is not body and child is not update
        ]
        condition = next(
            (child for child in remaining if child.type == "condition_clause"),
            None,
        )
        initializers: list[SyntaxNode] = []
        if condition is None:
            if remaining and remaining[0].type in {
                "declaration", "assignment_expression", "init_declarator",
            }:
                initializers = [remaining.pop(0)]
            condition = remaining[0] if remaining else None
        self._build_for_like(node, condition, body, update, initializers)

    def _visit_for_range_loop(self, node: SyntaxNode) -> None:
        children = list(node.children)
        body = children[-1] if children else None
        header = [child for child in children if child is not body]
        defines: list[str] = []
        uses: list[str] = []
        iterable = header[-1] if header else None
        declarators = header[:-1] if iterable is not None else header
        for child in declarators:
            if child.type.endswith("declarator"):
                for name in _target_names(child, self.text_of):
                    if name not in defines:
                        defines.append(name)
            elif child.type == "identifier":
                name = self.text_of(child).strip()
                if name and name not in defines:
                    defines.append(name)
        if iterable is not None:
            for name in _expression_reads(iterable, self.text_of):
                if name not in uses:
                    uses.append(name)
        declaration_text = " ".join(
            " ".join(self.text_of(child).split()) for child in declarators
        )
        iterable_text = self.text_of(iterable).strip() if iterable is not None else ""
        text = (
            f"{declaration_text} : {iterable_text}"
            if iterable_text else declaration_text
        )
        self._build_for_like(
            node,
            None,
            body,
            None,
            [],
            control_text=f"for ({text})",
            control_defines=tuple(defines),
            control_uses=tuple(uses),
        )

    def _build_for_like(
        self,
        node: SyntaxNode,
        condition: SyntaxNode | None,
        body: SyntaxNode | None,
        update: SyntaxNode | None,
        initializers: list[SyntaxNode],
        *,
        control_text: str | None = None,
        control_defines: tuple[str, ...] | None = None,
        control_uses: tuple[str, ...] | None = None,
    ) -> None:
        for child in initializers:
            self._visit(child)
        incoming_block = self._current
        condition_block = self._new_block("loop_condition")
        self._edge(incoming_block, condition_block, "loop_entry")
        self._current = condition_block
        if condition is None and control_text is None:
            control_text = "for (;;)"
            control_defines = ()
            control_uses = ()
        predicate = self._control(
            node,
            condition,
            text_override=control_text,
            defines=control_defines,
            uses=control_uses,
        )
        body_block = self._new_block("loop_body")
        exit_block = self._new_block("loop_exit")
        update_block = self._new_block("loop_update") if update is not None else None
        self._edge(condition_block, body_block, "true", "loop")
        self._edge(condition_block, exit_block, "false", "loop")
        self._break_targets.append(exit_block)
        self._continue_targets.append(update_block or condition_block)
        self._current = body_block
        self._reachable = True
        self._visit(body)
        self._continue_targets.pop()
        self._break_targets.pop()
        if self._reachable:
            self._edge(
                self._current,
                update_block if update_block is not None else condition_block,
                "loop_update" if update_block is not None else "loop_back",
            )
        if update_block is not None:
            self._current = update_block
            self._reachable = True
            self._visit(update)
            if self._reachable:
                self._edge(update_block, condition_block, "loop_back")
        self._current = exit_block
        self._reachable = True
        self._predicates_by_block[condition_block] = predicate

    def _visit_switch_statement(self, node: SyntaxNode) -> None:
        condition = _first_child(node, {"condition_clause"})
        body = _control_body(node, condition)
        predicate = self._control(node, condition)
        dispatch_block = self._current
        cases = [
            child for child in (body.children if body is not None else ())
            if child.type == "case_statement"
        ]
        exit_block = self._new_block("switch_exit")
        if not cases:
            self._current = self._new_block("switch_body")
            self._reachable = True
            self._edge(dispatch_block, self._current, "switch_body")
            self._visit(body)
            if self._reachable:
                self._edge(self._current, exit_block, "join")
            self._current = exit_block
            self._predicates_by_block[dispatch_block] = predicate
            return

        case_blocks = [self._new_block("switch_case") for _ in cases]
        has_default = False
        self._break_targets.append(exit_block)
        for index, (case, case_block) in enumerate(zip(cases, case_blocks)):
            text = " ".join(self.text_of(case).split())[:240]
            is_default = text.startswith("default")
            has_default |= is_default
            case_node = self.make_node(
                case, "case", case_block, text_override=text,
            )
            self.graph.nodes.append(case_node)
            self._append_node(case_block, case_node)
            self._edge(
                dispatch_block, case_block, "switch_case",
                "default" if text.startswith("default") else "case",
            )
            self._current = case_block
            self._reachable = True
            body_start = 0 if is_default else 1
            for child in case.children[body_start:]:
                self._visit(child)
            if self._reachable and index + 1 < len(case_blocks):
                self._edge(
                    self._current, case_blocks[index + 1],
                    "switch_fallthrough",
                )
            elif self._reachable:
                self._edge(self._current, exit_block, "switch_fallthrough")
        self._break_targets.pop()
        if not has_default:
            self._edge(dispatch_block, exit_block, "switch_default")
        self._current = exit_block
        self._reachable = True
        self._predicates_by_block[dispatch_block] = predicate

    def _visit_try(self, node: SyntaxNode) -> None:
        dispatch_block = self._ensure_current()
        body = _control_body(node, None)
        handlers = [
            child for child in node.children if child.type == "catch_clause"
        ]
        join_block = self._new_block("try_join")
        body_block = self._new_block("try_body")
        self._edge(dispatch_block, body_block, "try")
        for handler in handlers:
            handler_block = self._new_block("catch")
            self._edge(dispatch_block, handler_block, "exception")
            self._current = handler_block
            self._reachable = True
            self._visit(handler)
            if self._reachable:
                self._edge(self._current, join_block, "join")
        self._current = body_block
        self._reachable = True
        self._visit(body)
        if self._reachable:
            self._edge(self._current, join_block, "join")
        self._current = join_block
        self._reachable = True

    def _visit_case(self, node: SyntaxNode) -> None:
        for child in node.children[1:]:
            self._visit(child)

    def _statement(self, node: SyntaxNode) -> None:
        defines, uses = split_def_use(node, self.text_of)
        graph_node = self.make_node(
            node,
            "statement",
            self._ensure_current(),
            defines=defines,
            uses=uses,
        )
        self.graph.nodes.append(graph_node)
        self._append_node(self._current, graph_node)

    def _control(
        self,
        node: SyntaxNode,
        condition: SyntaxNode | None,
        *,
        text_override: str | None = None,
        defines: tuple[str, ...] | None = None,
        uses: tuple[str, ...] | None = None,
    ) -> str:
        source = condition or node
        if defines is None or uses is None:
            inferred_defines, inferred_uses = split_def_use(
                source, self.text_of,
            )
            defines = inferred_defines if defines is None else defines
            uses = inferred_uses if uses is None else uses
        graph_node = self.make_node(
            node,
            "control",
            self._ensure_current(),
            text_override=(
                text_override
                if text_override is not None
                else _condition_text(source, self.text_of)
            ),
            defines=defines,
            uses=uses,
        )
        self.graph.nodes.append(graph_node)
        self._append_node(self._current, graph_node)
        return graph_node.node_id

    def _append_node(self, block_id: int, node: GraphNode) -> None:
        block = self._blocks[block_id]
        block.node_ids.append(node.node_id)
        if not block.start_line or node.line < block.start_line:
            block.start_line = node.line
        block.end_line = max(block.end_line, node.end_line or node.line)

    def _ensure_current(self) -> int:
        if self._current < 0:
            self._current = self._new_block("unreachable")
            self._reachable = False
        return self._current

    def _new_block(self, kind: str) -> int:
        block_id = len(self._blocks)
        self._blocks.append(_Block(block_id, kind))
        return block_id

    def _edge(
        self,
        source: int,
        target: int,
        kind: str,
        label: str = "",
    ) -> None:
        if source < 0 or target < 0:
            return
        self._edges.add(CFGEdge(source, target, kind, label))

    def _terminate(self, kind: str) -> None:
        self._edge(self._current, 1, kind)
        self._unreachable()

    def _unreachable(self) -> None:
        self._current = -1
        self._reachable = False

    def _add_control_dependencies(self) -> None:
        block_ids = {block.block_id for block in self._blocks}
        successors: dict[int, set[int]] = defaultdict(set)
        for edge in self._edges:
            if edge.target_block in block_ids:
                successors[edge.source_block].add(edge.target_block)

        all_blocks = set(block_ids)
        postdominators: dict[int, set[int]] = {}
        for block_id in block_ids:
            postdominators[block_id] = (
                {1} if block_id == 1 else set(all_blocks)
            )
        changed = True
        iterations = 0
        while changed and iterations < max(2, len(block_ids) * 4):
            changed = False
            iterations += 1
            for block_id in sorted(block_ids - {1}):
                succs = successors.get(block_id, set())
                common = set(all_blocks)
                for successor in succs:
                    common &= postdominators.get(successor, set())
                common.add(block_id)
                if not succs:
                    common = {block_id}
                if common != postdominators[block_id]:
                    postdominators[block_id] = common
                    changed = True

        nodes_by_block: dict[int, list[str]] = defaultdict(list)
        for node in self.graph.nodes:
            nodes_by_block[node.block_id].append(node.node_id)
        existing = {
            (edge.source, edge.target)
            for edge in self.graph.edges if edge.kind == "control"
        }
        for block_id, succs in successors.items():
            if len(succs) < 2:
                continue
            predicate = self._predicates_by_block.get(block_id)
            if predicate is None:
                for node_id in reversed(nodes_by_block.get(block_id, [])):
                    node = _find_node(self.graph.nodes, node_id)
                    if node is not None and node.kind == "control":
                        predicate = node_id
                        break
            if predicate is None:
                continue
            for successor in succs:
                for controlled in block_ids:
                    if controlled == block_id:
                        continue
                    if (
                        controlled in postdominators.get(successor, set())
                        and controlled not in postdominators.get(block_id, set())
                    ):
                        for node_id in nodes_by_block.get(controlled, []):
                            if node_id == predicate:
                                continue
                            key = (predicate, node_id)
                            if key not in existing:
                                self.graph.edges.append(
                                    GraphEdge(
                                        predicate, node_id, "control",
                                        "syntactic",
                                    )
                                )
                                existing.add(key)


def build_cfg(
    graph: FunctionGraph,
    body: SyntaxNode,
    entry_node: GraphNode,
    text_of,
    make_node: NodeFactory,
) -> None:
    CFGBuilder(graph, text_of, make_node).build(body, entry_node)


def _first_child(node: SyntaxNode, kinds: set[str]) -> SyntaxNode | None:
    return next((child for child in node.children if child.type in kinds), None)


def _control_body(
    node: SyntaxNode, condition: SyntaxNode | None,
) -> SyntaxNode | None:
    for child in node.children:
        if child.type == "compound_statement":
            return child
    seen_condition = condition is None
    for child in node.children:
        if child is condition:
            seen_condition = True
            continue
        if not seen_condition:
            continue
        if child.type in {
            "condition_clause", "else_clause", "catch_clause",
            "update_expression",
        }:
            continue
        return child
    return None


def _condition_text(node: SyntaxNode, text_of) -> str:
    if node.type == "condition_clause" and node.children:
        body = node.children[0]
        while body.type == "parenthesized_expression" and len(body.children) == 1:
            body = body.children[0]
        return " ".join(text_of(body).split())[:240]
    return " ".join(text_of(node).split())[:240]


def _find_node(nodes: list[GraphNode], node_id: str) -> GraphNode | None:
    return next((node for node in nodes if node.node_id == node_id), None)
