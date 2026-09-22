"""Load PDG JSON produced by the C++ PDG builder into Python data models.

Usage::

    >>> from llm_client.fp_analysis.pdg_loader import load_pdg, build_sdg
    >>> pdgs = load_pdg("path/to/pdg.json")
    >>> from llm_client.csa_analysis.callgraph_models import CallGraph
    >>> cg = CallGraph.from_json("path/to/callgraph.json")
    >>> sdg = build_sdg(pdgs, cg)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from llm_client.fp_analysis.pdg_augment import augment_pdg_sdg
from llm_client.fp_analysis.pdg_models import (
    ControlDepEdge,
    DefUseEdge,
    FunctionPDG,
    FunctionSummary,
    PDGNode,
    SDG,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDG loader
# ---------------------------------------------------------------------------


def load_pdg(json_path: str | Path) -> dict[str, FunctionPDG]:
    """Load PDG JSON from **json_path** into ``{function_name: FunctionPDG}``.

    The JSON is expected to match the output format of ``PDGBuilder.cpp``:

    .. code:: json

        {
          "functions": [
            {
              "function_name": "foo",
              "source_file": "/src/foo.cc",
              "nodes": [
                {"id": "S_10_3", "kind": "decl", "variable": "x", ...}
              ],
              "def_use_edges": [...],
              "control_dep_edges": [...],
              "input_variables": ["x"],
              "output_variables": ["result"]
            }
          ]
        }

    Returns an empty dict if the file cannot be read or parsed.
    """
    path = Path(json_path)
    if not path.exists():
        logger.warning("PDG file not found: %s", path)
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load PDG JSON from %s: %s", path, exc)
        return {}

    functions: dict[str, FunctionPDG] = {}
    for func_data in raw.get("functions", []):
        function_name = func_data.get("function_name", "")
        if not function_name:
            continue

        # Build nodes dict
        nodes: dict[str, PDGNode] = {}
        for n in func_data.get("nodes", []):
            node = PDGNode(
                id=n.get("id", ""),
                kind=n.get("kind", "event"),
                variable=n.get("variable", ""),
                expression=n.get("expression", ""),
                type=n.get("type", ""),
                source_line=n.get("source_line", 0),
                source_col=n.get("source_col", 0),
                block_id=n.get("block_id", 0),
                is_call=n.get("is_call", False),
                callee=n.get("callee", ""),
                actual_params=list(n.get("actual_params", [])),
                is_macro=bool(n.get("is_macro", False)),
                macro_name=n.get("macro_name", ""),
                macro_def_file=n.get("macro_def_file", ""),
                macro_def_line=n.get("macro_def_line", 0),
            )
            nodes[node.id] = node

        # Build edges
        def_use_edges = [
            DefUseEdge(
                def_node_id=e.get("def_node_id", ""),
                use_node_id=e.get("use_node_id", ""),
                variable=e.get("variable", ""),
            )
            for e in func_data.get("def_use_edges", [])
        ]

        control_dep_edges = [
            ControlDepEdge(
                source_id=e.get("source_id", ""),
                target_id=e.get("target_id", ""),
                kind=e.get("kind", "if_branch"),
                is_macro_branch=bool(e.get("is_macro_branch", False)),
            )
            for e in func_data.get("control_dep_edges", [])
        ]

        # Build summary
        summary = FunctionSummary(
            input_variables=list(func_data.get("input_variables", [])),
            output_variables=list(func_data.get("output_variables", [])),
            input_output_dep=list(func_data.get("input_output_dep", [])),
        )

        pdg = FunctionPDG(
            function_name=function_name,
            source_file=func_data.get("source_file", ""),
            nodes=nodes,
            def_use_edges=def_use_edges,
            control_dep_edges=control_dep_edges,
            summary=summary,
        )

        functions[function_name] = pdg

    logger.info(
        "Loaded PDG data: %d functions, %d total nodes",
        len(functions),
        sum(len(f.nodes) for f in functions.values()),
    )
    return functions


# ---------------------------------------------------------------------------
# SDG construction
# ---------------------------------------------------------------------------


def build_sdg(
    pdgs: dict[str, FunctionPDG],
    callgraph: object,
    augment: bool = True,
) -> SDG:
    """Build a System Dependence Graph from per-function PDGs and call graph.

    The SDG wraps the PDG dict together with the call graph, providing
    cross-function lookup methods.  No additional linking is required —
    the ``SDG.get_callee_pdg()`` method uses the call graph to resolve
    caller→callee name variations.

    When *augment* is True (default), applies PDG augmentation passes
    (member field tracking, opaque call effects, union alias analysis)
    that synthesise def-use edges the C++ PDG builder cannot derive
    from Clang CFG reaching-definitions alone.

    Args:
        pdgs: Per-function PDGs (from :func:`load_pdg`).
        callgraph: A ``CallGraph`` instance (from
            ``llm_client.csa_analysis.callgraph_models``).
        augment: Apply augmentation passes (member field tracking,
            opaque call effects, union aliases).  Default True.

    Returns:
        An :class:`SDG` instance (augmented when *augment* is True).
    """
    sdg = SDG(pdgs=pdgs, callgraph=callgraph)
    if augment:
        augment_pdg_sdg(sdg)
    return sdg


# ---------------------------------------------------------------------------
# Convenience: load both PDG and call graph from file paths
# ---------------------------------------------------------------------------


def load_sdg_from_files(
    pdg_path: str | Path,
    callgraph_path: str | Path,
    augment: bool = True,
) -> Optional[SDG]:
    """Load a complete SDG from PDG and call graph JSON files.

    This is a convenience wrapper that:
    1. Loads the call graph via ``CallGraph.from_json()``
    2. Loads PDG data via :func:`load_pdg`
    3. Applies augmentation (default) and returns the linked :class:`SDG`

    Returns ``None`` if either file cannot be loaded.
    """
    from llm_client.csa_analysis.callgraph_models import CallGraph

    cg_path = Path(callgraph_path)
    if not cg_path.exists():
        logger.warning("Call graph file not found: %s", cg_path)
        return None

    try:
        callgraph = CallGraph.from_json(str(cg_path))
    except Exception as exc:
        logger.warning("Failed to load call graph from %s: %s", cg_path, exc)
        return None

    pdgs = load_pdg(pdg_path)
    if not pdgs:
        return None

    return build_sdg(pdgs, callgraph, augment=augment)
