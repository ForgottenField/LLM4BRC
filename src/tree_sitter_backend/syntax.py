"""Detached syntax-tree model used by the Tree-sitter backend."""

from __future__ import annotations

from dataclasses import dataclass, field

from tree_sitter import Node


@dataclass
class SyntaxNode:
    """A Tree-sitter node detached from native parser-owned state."""

    type: str
    start_byte: int
    end_byte: int
    start_row: int
    end_row: int
    has_error: bool = False
    children: tuple[SyntaxNode, ...] = field(default_factory=tuple)


def snapshot_tree(root: Node) -> SyntaxNode:
    """Copy the named syntax tree into plain Python objects.

    Tree-sitter nodes borrow parser state and are cheap, but retaining large
    numbers of wrappers across recursive analysis can expose native lifetime
    bugs on long generated C++ translation units.  The graph builder only needs
    coordinates and child structure, so it works on this detached snapshot.
    """
    detached = _snapshot_node(root)
    cursor = root.walk()
    named_ancestors: list[tuple[int, SyntaxNode]] = [(0, detached)]

    def add_named() -> None:
        node = cursor.node
        if not node.is_named:
            return
        depth = cursor.depth
        while len(named_ancestors) > 1 and named_ancestors[-1][0] >= depth:
            named_ancestors.pop()
        parent = named_ancestors[-1][1]
        child = _snapshot_node(node)
        parent.children = (*parent.children, child)
        named_ancestors.append((depth, child))

    while True:
        if cursor.goto_first_child():
            add_named()
            continue

        while True:
            if cursor.goto_next_sibling():
                add_named()
                break
            if not cursor.goto_parent():
                return detached


def _snapshot_node(node: Node) -> SyntaxNode:
    return SyntaxNode(
        type=node.type,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        # Row coordinates from the native cursor are intentionally ignored.
        # Source lines are derived from byte offsets by the backend.
        start_row=0,
        end_row=0,
        has_error=node.has_error,
    )
