"""Call graph traversal and entry point discovery.

Given a call graph and a bug function, finds all project entry points
(functions without callers, ``main()``, public APIs, test functions)
that can reach the bug function through the call chain.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

from llm_client.csa_analysis.callgraph_models import (
    CallGraph,
    CallGraphNode,
    EntryPoint,
    EntryPointResult,
)


def _looks_like_test_file(file_path: str) -> bool:
    """Heuristic: does *file_path* look like a test file?"""
    lower = file_path.lower()
    return any(p in lower for p in ["test", "mock", "benchmark", "example", "demo", "samples"])


def _is_eligible_entry_point(node: CallGraphNode, callgraph: CallGraph) -> bool:
    """A function qualifies as an entry point if it is callable
    from outside the project with reasonable symbolic inputs.

    Rules:
      - ``main()`` is always eligible.
      - Functions with **no callers** are potential entry points
        (but we skip class methods).
      - Functions with **external linkage** and no callers are public APIs.
      - Functions in test/example files are eligible.
    """
    # main() is always eligible
    if node.is_main:
        return True

    # Skip class methods (require object construction)
    if node.is_method:
        return False

    callers = callgraph.get_callers(node.id)
    no_callers = len(callers) == 0

    # No callers → project root call
    if no_callers or _looks_like_test_file(node.file):
        return True

    # External linkage with no callers → public API
    if node.linkage == "external" and no_callers:
        return True

    return False


def _score_entry_point(node: CallGraphNode, distance: int) -> int:
    """Score an entry point — higher is better.

    Bonus points:
      - 1000  ``main()``
      -  200  free function (not a method)
      -   50  per simple parameter type (int, size_t, bool, char*, void)
      -  100  zero parameters
      -   50  external linkage
      -   80  in a test file
      -   30  not the bug function itself (avoids trivial self-harness)
      -  -10  per hop of distance (prefer shorter call chains)
    """
    score = 0

    if node.is_main:
        score += 1000
    if not node.is_method:
        score += 200

    simple_types = {"int", "size_t", "bool", "char *", "void *",
                    "unsigned int", "long", "unsigned long",
                    "unsigned long long", "long long", "double",
                    "float", "unsigned char", "signed char",
                    "const char *", "const void *", "const std::string &",
                    "std::string", "std::string_view", "FILE *"}
    for pt in node.param_types:
        if pt in simple_types:
            score += 50

    if not node.param_types:
        score += 100
    if node.linkage == "external":
        score += 50
    if _looks_like_test_file(node.file):
        score += 80

    # Penalise distance (prefer shorter paths)
    score -= distance * 10

    return max(score, 1)


def find_entry_points(
    callgraph: CallGraph,
    bug_function: str,
    bug_file: str = "",
    bug_line: int = 0,
    max_depth: int = 100,
) -> EntryPointResult:
    """Find all eligible entry points that can reach the bug function.

    Args:
        callgraph: The project call graph.
        bug_function: The function name from the bug finding.
        bug_file: Source file of the bug (for fuzzy matching).
        bug_line: Line number of the bug (for fuzzy matching).
        max_depth: Maximum reverse traversal depth.

    Returns:
        An ``EntryPointResult`` with ranked entry points.
    """
    # Step 1: Find the bug function node in the call graph
    nodes = callgraph.find_nodes_matching(bug_function, bug_file, bug_line)
    if not nodes:
        return EntryPointResult(
            bug_function=bug_function,
            bug_file=bug_file,
            bug_line=bug_line,
            error=f"bug function '{bug_function}' not found in call graph",
        )

    result = EntryPointResult(
        bug_function=bug_function,
        bug_file=bug_file,
        bug_line=bug_line,
    )

    # Step 2: Reverse BFS to find all ancestors
    ancestors: set[str] = set()
    queue = deque()
    distance_map: dict[str, int] = {}

    for node in nodes:
        queue.append((node.id, 0))

    while queue:
        func_name, depth = queue.popleft()
        if depth > max_depth:
            continue
        if func_name in ancestors:
            continue
        ancestors.add(func_name)
        distance_map[func_name] = depth

        for caller in callgraph.get_callers(func_name):
            if caller not in ancestors:
                queue.append((caller, depth + 1))

    result.total_ancestors = len(ancestors)

    # Step 3: Collect eligible entry points
    entry_points: list[EntryPoint] = []
    for ancestor_id in ancestors:
        node = callgraph.functions.get(ancestor_id)
        if not node:
            continue
        if _is_eligible_entry_point(node, callgraph):
            distance = distance_map.get(ancestor_id, 0)
            score = _score_entry_point(node, distance)
            entry_points.append(EntryPoint(
                function_name=ancestor_id,
                node=node,
                score=score,
                distance=distance,
            ))

    # Step 4: Sort by score descending
    entry_points.sort(key=lambda ep: ep.score, reverse=True)
    result.entry_points = entry_points
    return result


def select_best_entry_points(
    result: EntryPointResult,
    max_count: int = 3,
    min_score: int = 1,
) -> list[EntryPoint]:
    """Select the top-ranked entry points from the discovery result.

    Filters by minimum score and returns at most *max_count* entries.
    """
    return [ep for ep in result.entry_points if ep.score >= min_score][:max_count]


def build_synthetic_callgraph() -> CallGraph:
    """Build a small synthetic call graph for testing.

    Structure::

        main  ──→  readConfig  ──→  parseLine
                   processData ──→  transform
                        │
                        └────────→  (bug) validateBuffer
                   writeOutput

        test_validateBuffer  ──→  validateBuffer  (test file)
        benchmark_transform  ──→  transform  (test file)

    This lets us test entry point discovery without running the C++ tool.
    """
    data = {
        "functions": {
            "main": {
                "file": "/src/main.cpp", "line": 5, "linkage": "external",
                "is_main": True, "return_type": "int", "param_types": ["int", "char **"],
                "is_method": False, "class_name": "",
            },
            "readConfig": {
                "file": "/src/config.cpp", "line": 12, "linkage": "external",
                "is_main": False, "return_type": "void", "param_types": ["const char *"],
                "is_method": False, "class_name": "",
            },
            "parseLine": {
                "file": "/src/config.cpp", "line": 45, "linkage": "internal",
                "is_main": False, "return_type": "bool", "param_types": ["const char *", "size_t"],
                "is_method": False, "class_name": "",
            },
            "processData": {
                "file": "/src/processor.cpp", "line": 22, "linkage": "external",
                "is_main": False, "return_type": "int", "param_types": ["int", "int"],
                "is_method": False, "class_name": "",
            },
            "transform": {
                "file": "/src/transform.cpp", "line": 8, "linkage": "internal",
                "is_main": False, "return_type": "int", "param_types": ["int"],
                "is_method": False, "class_name": "",
            },
            "validateBuffer": {
                "file": "/src/buffer.cpp", "line": 15, "linkage": "internal",
                "is_main": False, "return_type": "bool",
                "param_types": ["const char *", "size_t"],
                "is_method": False, "class_name": "",
            },
            "writeOutput": {
                "file": "/src/output.cpp", "line": 30, "linkage": "external",
                "is_main": False, "return_type": "void", "param_types": ["const char *"],
                "is_method": False, "class_name": "",
            },
            "test_validateBuffer": {
                "file": "/tests/test_buffer.cpp", "line": 5, "linkage": "external",
                "is_main": False, "return_type": "void", "param_types": [],
                "is_method": False, "class_name": "",
            },
            "benchmark_transform": {
                "file": "/tests/bench_transform.cpp", "line": 3, "linkage": "external",
                "is_main": False, "return_type": "void", "param_types": [],
                "is_method": False, "class_name": "",
            },
        },
        "calls": [
            {"caller": "main", "callee": "readConfig", "file": "/src/main.cpp", "line": 10},
            {"caller": "main", "callee": "processData", "file": "/src/main.cpp", "line": 11},
            {"caller": "main", "callee": "writeOutput", "file": "/src/main.cpp", "line": 12},
            {"caller": "readConfig", "callee": "parseLine", "file": "/src/config.cpp", "line": 20},
            {"caller": "processData", "callee": "transform", "file": "/src/processor.cpp", "line": 30},
            {"caller": "processData", "callee": "validateBuffer", "file": "/src/processor.cpp", "line": 35},
            {"caller": "test_validateBuffer", "callee": "validateBuffer", "file": "/tests/test_buffer.cpp", "line": 10},
            {"caller": "benchmark_transform", "callee": "transform", "file": "/tests/bench_transform.cpp", "line": 8},
        ],
    }
    return CallGraph.from_dict(data)
