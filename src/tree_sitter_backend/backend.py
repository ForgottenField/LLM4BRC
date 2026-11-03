"""Fast configured C/C++ AST and PDG-Lite construction with Tree-sitter."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from tree_sitter import Language, Node, Parser
import tree_sitter_c
import tree_sitter_cpp

from tree_sitter_backend.cfg import build_cfg
from tree_sitter_backend.compile_db import (
    CompileDatabase,
    SourceMap,
    SourcePosition,
    raw_source_map,
)
from tree_sitter_backend.models import (
    CallRef,
    CodeIndex,
    FunctionGraph,
    FunctionInfo,
    GraphNode,
)
from tree_sitter_backend.syntax import SyntaxNode, snapshot_tree

_SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl",
}
_CAST_NAMES = {
    "reinterpret_cast", "static_cast", "const_cast", "dynamic_cast",
    "bit_cast", "duration_cast", "time_point_cast",
}


class TreeSitterBackend:
    """Parse a bounded file set, optionally through a compile database."""

    def __init__(self) -> None:
        # Parsers borrow their Language object.  Keep both alive to avoid
        # native lifetime corruption on large translation units.
        self._languages = {
            "c": Language(tree_sitter_c.language()),
            "cpp": Language(tree_sitter_cpp.language()),
        }
        self._parsers = {
            "c": Parser(self._languages["c"]),
            "cpp": Parser(self._languages["cpp"]),
        }
        self._requested_path = Path()
        self._allowed_files: set[str] = set()
        self._source_map: SourceMap | None = None
        self._line_starts: tuple[int, ...] = ()
        self._parse_errors: list[dict[str, str]] = []

    def build_index(
        self,
        source_files: Iterable[str | Path],
        source_root: str | Path,
        *,
        compile_commands: str | Path | None = None,
    ) -> CodeIndex:
        """Build a bounded index.

        When ``compile_commands`` is supplied, each requested source is
        preprocessed with its exact compile-database entry before parsing.
        """
        root = Path(source_root).resolve()
        database = (
            CompileDatabase.load(compile_commands)
            if compile_commands is not None
            else None
        )
        index = CodeIndex(
            root=root,
            compile_commands=str(database.path) if database is not None else "",
            preprocessor_mode="configured" if database is not None else "raw",
        )
        unique_files = dict.fromkeys(
            Path(path).resolve() for path in source_files
        )
        for source_file in unique_files:
            if source_file.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            try:
                if database is not None:
                    prepared = database.preprocess(source_file)
                    text = prepared.text
                    source_map = prepared.source_map
                    compiler = prepared.compiler
                    language = prepared.language
                else:
                    text = source_file.read_text(encoding="utf-8", errors="replace")
                    source_map = raw_source_map(source_file, text)
                    compiler = ""
                    language = (
                        "c" if source_file.suffix.lower() == ".c" else "cpp"
                    )
            except (OSError, RuntimeError) as exc:
                index.errors.append({
                    "file": str(source_file),
                    "kind": "preprocess",
                    "error": str(exc),
                })
                continue
            self._parse_source(
                index,
                source_file,
                text,
                source_map=source_map,
                compiler=compiler,
                language=language,
            )
        self._resolve_calls(index)
        return index

    def _parse_source(
        self,
        index: CodeIndex,
        path: Path,
        text: str,
        *,
        source_map: SourceMap,
        compiler: str,
        language: str,
    ) -> None:
        self._requested_path = path
        self._allowed_files = {str(path)}
        self._source_map = source_map
        self._parse_errors = []
        self._line_starts = source_map.line_starts
        data = text.encode("utf-8", errors="replace")
        tree = self._parsers[language].parse(data)
        self._collect_diagnostics(tree.root_node, data)
        root = snapshot_tree(tree.root_node)
        metadata = {
            "language": language,
            "bytes": len(data),
            "preprocessed_bytes": len(data),
            "preprocessor_mode": index.preprocessor_mode,
            "compiler": compiler,
            "has_error": bool(self._parse_errors),
            "node_count": self._count_nodes(root),
            "parse_errors": list(self._parse_errors),
        }
        index.files[str(path)] = metadata
        includes = [
            self._node_text(node.children[0] if node.children else None, data)
            for node in self._walk(root)
            if node.type == "preproc_include"
        ]
        index.includes[str(path)] = [
            value.strip().strip('<>"') for value in includes if value.strip()
        ]
        self._collect_scopes(index, root, data, path, ())
        index.errors.extend(self._parse_errors)

    def _collect_diagnostics(self, root: Node, data: bytes) -> None:
        cursor = root.walk()
        while True:
            node = cursor.node
            if node.type == "ERROR" or node.is_missing:
                position = self._position(node.start_byte)
                if position is not None and position.file in self._allowed_files:
                    snippet = " ".join(
                        data[node.start_byte:node.end_byte]
                        .decode("utf-8", errors="replace")
                        .split()
                    )[:160]
                    self._parse_errors.append({
                        "file": position.file,
                        "line": str(position.line),
                        "column": str(position.column),
                        "kind": "missing" if node.is_missing else "error",
                        "node": node.type,
                        "text": snippet,
                    })
            if cursor.goto_first_child():
                continue
            while not cursor.goto_next_sibling():
                if not cursor.goto_parent():
                    return

    def _collect_scopes(
        self,
        index: CodeIndex,
        root: SyntaxNode,
        data: bytes,
        path: Path,
        initial_scope: tuple[str, ...],
    ) -> None:
        stack: list[tuple[SyntaxNode, tuple[str, ...], bool]] = [
            (root, initial_scope, True)
        ]
        while stack:
            node, scope, is_root = stack.pop()
            position = self._position(node.start_byte)
            if (
                not is_root
                and position is not None
                and position.file not in self._allowed_files
            ):
                continue
            children = list(reversed(node.children))
            for child in children:
                if child.type == "namespace_definition":
                    name = self._first_named_text(
                        child, {"identifier", "namespace_identifier"}, data,
                    )
                    body = self._first_child(
                        child, {"declaration_list", "compound_statement"},
                    )
                    stack.append((
                        body or child,
                        scope + ((name,) if name else ()),
                        False,
                    ))
                elif child.type in {"class_specifier", "struct_specifier"}:
                    name = self._first_named_text(
                        child, {"type_identifier"}, data,
                    )
                    body = self._first_child(child, {"field_declaration_list"})
                    stack.append((
                        body or child,
                        scope + ((name,) if name else ()),
                        False,
                    ))
                elif child.type == "function_definition":
                    self._add_function(index, child, data, path, scope)
                else:
                    stack.append((child, scope, False))

    def _add_function(
        self,
        index: CodeIndex,
        node: SyntaxNode,
        data: bytes,
        path: Path,
        scope: tuple[str, ...],
    ) -> None:
        start = self._position(node.start_byte)
        if start is None or start.file not in self._allowed_files:
            return
        declarator = self._first_child(
            node,
            {
                "function_declarator", "pointer_declarator",
                "reference_declarator",
            },
        )
        name = self._declarator_name(declarator, data)
        if not name:
            return
        qualified = (
            name if "::" in name
            else "::".join((*scope, name)) if scope else name
        )
        end = self._position(max(node.start_byte, node.end_byte - 1)) or start
        function_id = self._node_id("fn", path, node)
        params_node = (
            self._find_descendant(declarator, "parameter_list")
            if declarator is not None else None
        )
        params = tuple(self._parameter_names(params_node, data))
        index.functions[function_id] = FunctionInfo(
            function_id=function_id,
            name=name.split("::")[-1],
            qualified_name=qualified,
            file=start.file,
            start_line=start.line,
            end_line=end.line,
            parameters=params,
        )

        body = self._first_child(node, {"compound_statement"}) or node
        graph = FunctionGraph(
            function=qualified,
            file=start.file,
            start_line=start.line,
            end_line=end.line,
        )
        entry = GraphNode(
            node_id=self._node_id("entry", path, node),
            function=qualified,
            kind="entry",
            file=start.file,
            line=start.line,
            end_line=start.line,
            text="function entry",
            variables=params,
            block_id=0,
            defines=params,
        )

        def make_node(
            syntax: SyntaxNode,
            kind: str,
            block_id: int,
            *,
            text_override: str | None = None,
            defines: tuple[str, ...] = (),
            uses: tuple[str, ...] = (),
        ) -> GraphNode:
            position = self._position(syntax.start_byte) or start
            end_position = (
                self._position(max(syntax.start_byte, syntax.end_byte - 1))
                or position
            )
            text = (
                text_override
                if text_override is not None
                else self._node_text(syntax, data)
            )
            return GraphNode(
                node_id=self._node_id(kind, path, syntax),
                function=qualified,
                kind=kind,
                file=position.file,
                line=position.line,
                end_line=end_position.line,
                text=" ".join(text.split())[:240],
                variables=tuple(dict.fromkeys((*defines, *uses))),
                block_id=block_id,
                defines=defines,
                uses=uses,
            )

        build_cfg(
            graph,
            body,
            entry,
            text_of=lambda syntax: self._node_text(syntax, data),
            make_node=make_node,
        )
        index.graphs[function_id] = graph
        index.calls.extend(
            self._collect_calls(
                body,
                caller=qualified,
                default_file=start.file,
                data=data,
            )
        )

    def _collect_calls(
        self,
        node: SyntaxNode,
        *,
        caller: str,
        default_file: str,
        data: bytes,
    ) -> list[CallRef]:
        calls: list[CallRef] = []
        for child in self._walk(node):
            if child.type != "call_expression" or not child.children:
                continue
            target = child.children[0]
            callee = self._node_text(target, data).strip()
            if not callee or _is_language_cast(callee):
                continue
            args = next(
                (item for item in child.children if item.type == "argument_list"),
                None,
            )
            position = self._position(child.start_byte)
            calls.append(CallRef(
                caller=caller,
                callee=callee,
                file=position.file if position is not None else default_file,
                line=position.line if position is not None else 0,
                argument_count=len(args.children) if args else 0,
                text=" ".join(self._node_text(child, data).split())[:180],
            ))
        return calls

    @staticmethod
    def _resolve_calls(index: CodeIndex) -> None:
        by_short: dict[str, list[FunctionInfo]] = {}
        for function in index.functions.values():
            by_short.setdefault(function.name, []).append(function)
        resolved: list[CallRef] = []
        for call in index.calls:
            short = re.split(r"::|->|\.", call.callee.split("<", 1)[0])[-1].strip()
            matches = by_short.get(short, [])
            resolved.append(CallRef(
                caller=call.caller,
                callee=call.callee,
                file=call.file,
                line=call.line,
                argument_count=call.argument_count,
                resolved_function_id=(
                    matches[0].function_id if len(matches) == 1 else ""
                ),
                confidence="syntactic" if len(matches) == 1 else "candidate",
                text=call.text,
            ))
        index.calls = resolved

    def _position(self, byte_offset: int) -> SourcePosition | None:
        if self._source_map is None:
            return None
        return self._source_map.locate(byte_offset)

    @staticmethod
    def _walk(node: SyntaxNode):
        stack = [node]
        while stack:
            current = stack.pop()
            yield current
            stack.extend(reversed(current.children))

    @classmethod
    def _parameter_names(
        cls, params: SyntaxNode | None, data: bytes,
    ) -> list[str]:
        if params is None:
            return []
        names: list[str] = []
        for param in params.children:
            if param.type not in {"parameter_declaration", "optional_parameter_declaration"}:
                continue
            declarator = next(
                (
                    child for child in reversed(param.children)
                    if child.type == "identifier"
                    or child.type.endswith("declarator")
                ),
                None,
            )
            name = cls._declarator_name(declarator, data)
            if name and not name.startswith("__"):
                names.append(name)
        return list(dict.fromkeys(names))

    @classmethod
    def _declarator_name(cls, node: SyntaxNode | None, data: bytes) -> str:
        if node is None:
            return ""
        if node.type in {
            "identifier", "field_identifier", "qualified_identifier",
            "operator_name", "destructor_name",
        }:
            return cls._node_text(node, data)
        for child in node.children:
            if child.type.endswith("declarator") or child.type in {
                "identifier", "field_identifier", "qualified_identifier",
                "operator_name", "destructor_name",
            }:
                found = cls._declarator_name(child, data)
                if found:
                    return found
        return ""

    @staticmethod
    def _find_descendant(
        node: SyntaxNode | None, kind: str,
    ) -> SyntaxNode | None:
        if node is None:
            return None
        stack = list(reversed(node.children))
        while stack:
            child = stack.pop()
            if child.type == kind:
                return child
            stack.extend(reversed(child.children))
        return None

    @staticmethod
    def _first_child(
        node: SyntaxNode, kinds: set[str],
    ) -> SyntaxNode | None:
        return next((child for child in node.children if child.type in kinds), None)

    @staticmethod
    def _first_named_text(
        node: SyntaxNode, kinds: set[str], data: bytes,
    ) -> str:
        target = TreeSitterBackend._first_child(node, kinds)
        return TreeSitterBackend._node_text(target, data)

    @staticmethod
    def _node_text(node: SyntaxNode | None, data: bytes) -> str:
        if node is None:
            return ""
        return data[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    @staticmethod
    def _count_nodes(root: SyntaxNode) -> int:
        stack = [root]
        count = 0
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children)
        return count

    @staticmethod
    def _node_id(kind: str, path: Path, node: SyntaxNode) -> str:
        raw = f"{path}:{node.start_byte}:{node.end_byte}:{kind}".encode()
        return "ts_" + hashlib.sha1(raw).hexdigest()[:16]


def _is_language_cast(callee: str) -> bool:
    base = callee.split("<", 1)[0].strip()
    return base in _CAST_NAMES
