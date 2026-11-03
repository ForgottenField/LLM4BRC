"""Lightweight C/C++ syntax and program-dependence analysis."""

from tree_sitter_backend.backend import TreeSitterBackend
from tree_sitter_backend.compile_db import (
    COMPILE_COMMANDS_NAME,
    CompileDatabase,
    CompileDatabaseError,
    default_compile_commands,
)
from tree_sitter_backend.models import (
    BasicBlock,
    CallRef,
    CFGEdge,
    CodeIndex,
    FunctionGraph,
    FunctionInfo,
    GraphEdge,
    GraphNode,
)
from tree_sitter_backend.syntax import SyntaxNode, snapshot_tree
from tree_sitter_backend.worker import build_index_isolated

__all__ = [
    "CallRef",
    "BasicBlock",
    "CFGEdge",
    "CodeIndex",
    "COMPILE_COMMANDS_NAME",
    "CompileDatabase",
    "CompileDatabaseError",
    "FunctionGraph",
    "FunctionInfo",
    "GraphEdge",
    "GraphNode",
    "TreeSitterBackend",
    "build_index_isolated",
    "default_compile_commands",
    "SyntaxNode",
    "snapshot_tree",
]
