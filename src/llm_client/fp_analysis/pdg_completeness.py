"""PDG completeness validation and LLM-assisted patching.

Strategy 5 (Completeness Metrics)
----------------------------------
Validates PDG quality by computing per-function completeness scores,
detecting recovery nodes, missing call coverage, and sparse def-use edges.
Provides actionable reports for functions that are "unfit for slicing."

Strategy 4 (LLM PDG Patcher)
-----------------------------
For functions whose PDG is incomplete, uses an LLM to infer missing
call_expr nodes and def-use edges directly from the function's source
code, then applies the patches to the SDG in-memory.

Typical usage::

    # Validate completeness
    report = PDGCompletenessReport(sdg)
    report.print_summary()
    weak_fns = report.unfit_functions()

    # Patch incomplete functions
    patcher = LLMPDGPatchProvider(sdg, source_root="project/aria2")
    patcher.patch_function("aria2::OptionParser::parseArg")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llm_client.base import LLMConfig, LLMRequest
from llm_client.factory import ProviderFactory
from llm_client.fp_analysis.pdg_models import SDG, FunctionPDG, PDGNode

logger = logging.getLogger(__name__)

# =============================================================================
# Strategy 5: Completeness validation
# =============================================================================


@dataclass
class FunctionCompleteness:
    """Completeness assessment for a single function's PDG.

    Attributes:
        function_name: Qualified function name.
        source_file: Source file path.
        total_nodes: Total PDG nodes in the function.
        recovery_nodes: Number of recovery-expr nodes.
        def_use_edges: Number of def-use edges.
        ctrl_dep_edges: Number of control-dependence edges.
        score: 0.0 (unusable) to 1.0 (pristine).
        flags: Human-readable issue descriptions.
    """

    function_name: str = ""
    source_file: str = ""
    total_nodes: int = 0
    recovery_nodes: int = 0
    def_use_edges: int = 0
    ctrl_dep_edges: int = 0
    score: float = 0.0
    flags: list[str] = field(default_factory=list)

    @property
    def fit_for_slicing(self) -> bool:
        """Whether this PDG is reliable enough for backward slicing.

        Acceptable if:
        1. Score >= 0.3, AND
        2. Has at least some data-flow information (def-use edges > 0 OR
           the function is very small, <= 3 nodes)
        """
        if self.score >= 0.3 and (self.def_use_edges > 0 or self.total_nodes <= 3):
            return True
        return False


class PDGCompletenessReport:
    """Compute and report PDG completeness across all functions in an SDG.

    Processes completeness metadata embedded in the PDG JSON (written by the
    enhanced PDGBuilder.cpp), or computes it on-the-fly for loaded PDGs.
    """

    def __init__(self, sdg: SDG, raw_json_path: str | None = None):
        self.sdg = sdg
        self._raw_meta: dict[str, dict] = {}
        self._assessments: dict[str, FunctionCompleteness] = {}

        # Load raw JSON metadata if available
        if raw_json_path:
            self._load_raw_meta(raw_json_path)

        # Build assessments
        self._assess()

    def _load_raw_meta(self, path: str) -> None:
        """Load completeness metadata from raw PDG JSON."""
        try:
            data = json.loads(Path(path).read_text())
            for f in data.get("functions", []):
                fn = f.get("function_name", "")
                comp = f.get("completeness", {})
                if comp:
                    self._raw_meta[fn] = comp
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Could not load raw PDG metadata: %s", exc)

    def _assess(self) -> None:
        """Compute completeness assessments for all functions."""
        for fn_name, pdg in self.sdg.pdgs.items():
            ca = self._compute_single(fn_name, pdg)
            self._assessments[fn_name] = ca

    def _compute_single(self, fn_name: str, pdg: FunctionPDG) -> FunctionCompleteness:
        """Compute completeness for a single function."""
        ca = FunctionCompleteness(
            function_name=fn_name,
            source_file=pdg.source_file,
            total_nodes=len(pdg.nodes),
            def_use_edges=len(pdg.def_use_edges),
            ctrl_dep_edges=len(pdg.control_dep_edges),
        )

        # Count recovery nodes
        for node in pdg.nodes.values():
            if node.kind == "recovery":
                ca.recovery_nodes += 1

        # Use raw JSON completeness if available
        if fn_name in self._raw_meta:
            raw = self._raw_meta[fn_name]
            ca.recovery_nodes = raw.get("recovery_node_count", ca.recovery_nodes)
            ca.score = raw.get("completeness_score", 0.0)
        else:
            # Compute score on the fly
            node_score = (
                1.0 - ca.recovery_nodes / max(ca.total_nodes, 1)
                if ca.total_nodes > 0
                else 0.0
            )
            expected_due = max(ca.total_nodes // 4, 1)
            due_score = min(1.0, ca.def_use_edges / expected_due) if ca.total_nodes > 0 else 0.0
            ca.score = round(node_score * 0.6 + due_score * 0.4, 2)

        # Build flags
        if ca.recovery_nodes > 0:
            ca.flags.append(f"{ca.recovery_nodes} recovery-expr node(s)")
        if ca.def_use_edges == 0 and ca.total_nodes > 3:
            ca.flags.append("No def-use edges (data-flow missing)")
        if ca.ctrl_dep_edges == 0 and ca.total_nodes > 3:
            ca.flags.append("No control-dep edges (control-flow missing)")
        if ca.score < 0.3:
            ca.flags.append("Completeness score too low for slicing")

        return ca

    def unfit_functions(
        self, min_nodes: int = 5
    ) -> list[FunctionCompleteness]:
        """Return functions that are NOT fit for slicing.

        Args:
            min_nodes: Only flag functions with at least this many nodes
                (tiny functions are trivially complete enough).

        Returns:
            List of completeness assessments for unfit functions.
        """
        return [
            ca
            for ca in self._assessments.values()
            if ca.total_nodes >= min_nodes and not ca.fit_for_slicing
        ]

    def function_report(self, fn_name: str) -> Optional[FunctionCompleteness]:
        """Get completeness report for a specific function."""
        return self._assessments.get(fn_name)

    def print_summary(
        self, top_n: int = 10, min_nodes: int = 5
    ) -> None:
        """Print a summary of PDG completeness."""
        unfit = self.unfit_functions(min_nodes=min_nodes)
        all_fns = len(self._assessments)
        total_recovery = sum(ca.recovery_nodes for ca in self._assessments.values())
        total_no_due = sum(
            1 for ca in self._assessments.values()
            if ca.def_use_edges == 0 and ca.total_nodes >= min_nodes
        )

        print(f"PDG Completeness Report")
        print(f"{'='*60}")
        print(f"  Total functions: {all_fns}")
        print(f"  Unfit for slicing: {len(unfit)} ({len(unfit)/all_fns*100:.0f}%)")
        print(f"  Functions with recovery-expr: {total_recovery}")
        print(f"  Functions with zero def-use: {total_no_due}")
        print()

        if unfit:
            print(f"  Top {top_n} worst functions:")
            unfit_sorted = sorted(unfit, key=lambda x: x.score)[:top_n]
            for ca in unfit_sorted:
                flags = "; ".join(ca.flags[:2])
                print(f"    {ca.function_name}")
                print(f"      score={ca.score:.2f} nodes={ca.total_nodes} "
                      f"due={ca.def_use_edges} cde={ca.ctrl_dep_edges} "
                      f"recovery={ca.recovery_nodes}")
                if flags:
                    print(f"      ⚠ {flags}")


# =============================================================================
# Strategy 4: LLM PDG Patcher
# =============================================================================


_PDG_PATCH_PROMPT = """You are analyzing a C++ function whose Program Dependence Graph (PDG) is incomplete.
The PDG is missing call nodes and data-flow edges because of template instantiation failures or missing type information.

## The function source code
```cpp
{source_code}
```

## The current (incomplete) PDG nodes
The following nodes were successfully extracted from the Clang AST:

{existing_nodes}

## Task
Analyze the source code and identify ANY function calls or data-flow relationships
that are MISSING from the PDG. Return your findings as a JSON patch.

For each function call that should exist:
{{
  "patches": [
    {{
      "type": "call_expr",
      "line": <source line number>,
      "callee": "<function name>",
      "actual_params": ["<param1>", "<param2>", ...],
      "variable": "<assigned variable if any>",
      "expression": "<full call expression text>"
    }}
  ],
  "def_use_patches": [
    {{
      "type": "def_use",
      "def_variable": "<variable name being defined>",
      "use_variable": "<variable name being used>",
      "def_line": <line where the variable is defined>,
      "use_line": <line where the variable is used>
    }}
  ]
}}

Be precise: only include calls and data-flow that you can clearly see in the source code.
Do NOT include nodes that already exist in the PDG above.
Focus on calls that are clearly function invocations: putOptions(...), getopt_long(...), etc.
"""


class LLMPDGPatchProvider:
    """Use an LLM to infer missing PDG nodes and edges for incomplete functions.

    This is Strategy 4 — the last-resort compensation for functions whose PDG
    could not be fully built by Clang (due to recovery-expr, template failures,
    missing headers, etc.).

    The LLM reads the function's source code and the existing (incomplete) PDG,
    then suggests missing call_expr nodes and def-use edges. These patches are
    applied to the SDG in-memory.

    Args:
        sdg: The SDG to patch (modified in-place).
        source_root: Project root directory for resolving source files.
        llm_config: Optional LLM config. If not provided, patches are logged
            but not applied (dry-run mode for testing).
    """

    def __init__(
        self,
        sdg: SDG,
        source_root: str | Path | None = None,
        llm_config: LLMConfig | None = None,
    ):
        self.sdg = sdg
        self.source_root = Path(source_root) if source_root else None
        self._llm_config = llm_config
        self._provider = None
        if llm_config:
            self._provider = ProviderFactory.create(llm_config)

    def find_incomplete(self) -> list[str]:
        """Return names of functions with low completeness that could be patched.

        Filters for functions with:
        - Score < 0.5 (moderate imperfection) and at least 5 nodes
        - Has some code that the LLM could analyze
        """
        report = PDGCompletenessReport(self.sdg)
        unfit = report.unfit_functions(min_nodes=5)
        # Also include moderate imperfections (score < 0.5 but still > 0)
        moderate = [
            fn.function_name
            for fn in unfit
            if fn.score < 0.5
        ]
        # And functions with recovery nodes
        recovery_fn = [
            fn.function_name
            for fn in report._assessments.values()
            if fn.recovery_nodes > 0 and fn.total_nodes >= 5
        ]
        return list(set(moderate + recovery_fn))

    def patch_function(self, fn_name: str) -> bool:
        """Attempt to patch a single function's PDG using the LLM.

        Args:
            fn_name: Qualified function name.

        Returns:
            True if patches were applied, False otherwise.
        """
        pdg = self.sdg.get_pdg(fn_name)
        if pdg is None:
            logger.warning("Function '%s' not found in SDG.", fn_name)
            return False

        source_code = self._read_function_source(pdg)
        if not source_code:
            logger.warning("Could not read source for '%s'.", fn_name)
            return False

        # Format existing nodes
        existing_lines = []
        for nid, node in pdg.nodes.items():
            existing_lines.append(
                f"  {nid:20s} | L{node.source_line:<4} | {node.kind:12s} | "
                f"is_call={int(node.is_call)} | callee={node.callee or '':20s} | "
                f"var={node.variable or '':12s} | {node.expression[:60] if node.expression else ''}"
            )
        existing_str = "\n".join(existing_lines)

        if self._provider is None:
            # Dry-run mode
            logger.info("[DRY RUN] Would patch '%s' (nodes=%d, due=%d)",
                        fn_name, len(pdg.nodes), len(pdg.def_use_edges))
            return False

        # Build LLM prompt
        prompt = _PDG_PATCH_PROMPT.format(
            source_code=source_code,
            existing_nodes=existing_str,
        )

        # Request structured output via JSON template
        request = LLMRequest(
            messages=[
                {"role": "system", "content": "You are a C++ static analysis expert. Return ONLY valid JSON matching the requested schema."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
            model=self._llm_config.default_model if self._llm_config else "deepseek-chat",
        )

        try:
            response = self._provider.send(request)
            raw = response.content
        except Exception as exc:
            logger.error("LLM request failed for '%s': %s", fn_name, exc)
            return False

        # Parse JSON from response
        patches = self._parse_patches(raw)
        if not patches:
            logger.info("No patches suggested by LLM for '%s'.", fn_name)
            return False

        # Apply patches
        return self._apply_patches(fn_name, pdg, patches)

    def _read_function_source(self, pdg: FunctionPDG) -> str:
        """Read the source code of a function from the filesystem."""
        source_file = Path(pdg.source_file)
        if not source_file.exists() and self.source_root:
            # Try resolving relative to source root
            candidate = self.source_root / pdg.source_file
            if candidate.exists():
                source_file = candidate

        if not source_file.exists():
            return ""

        try:
            lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return ""

        # Find the function: look for its name in the source
        fn_name_short = pdg.function_name.split("::")[-1]
        start_line = 0
        brace_depth = 0
        fn_lines = []

        for i, line in enumerate(lines, 1):
            if fn_name_short in line and i <= start_line + 5:
                # Found the declaration; look for opening brace
                pass

        # Simpler approach: find by line number range from PDG nodes
        node_lines = {n.source_line for n in pdg.nodes.values() if n.source_line > 0}
        if not node_lines:
            return ""

        min_line = max(min(node_lines) - 2, 1)
        max_line = min(max(node_lines) + 5, len(lines))

        return "\n".join(lines[min_line - 1:max_line])

    def _parse_patches(self, raw: str) -> list[dict]:
        """Parse LLM response into patch specifications."""
        import re

        # Try to find JSON in the response
        json_match = re.search(r"\{[\s\S]*\"patches\"[\s\S]*\}", raw)
        if not json_match:
            return []

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        patches = data.get("patches", [])
        if not isinstance(patches, list):
            return []

        return patches

    def _apply_patches(
        self, fn_name: str, pdg: FunctionPDG, patches: list[dict]
    ) -> bool:
        """Apply LLM-suggested patches to the PDG."""

        applied = 0
        used_ids: set[str] = set(pdg.nodes.keys())

        for patch in patches:
            ptype = patch.get("type", "")

            if ptype == "call_expr":
                # Create a new PDGNode for the missing call
                line = patch.get("line", 0)
                callee = patch.get("callee", "")
                if not callee or line == 0:
                    continue

                # Check if a node for this call already exists
                existing = [
                    nid for nid, n in pdg.nodes.items()
                    if n.source_line == line and n.is_call and callee in (n.callee or "")
                ]
                if existing:
                    continue

                # Generate a unique node ID
                nid = f"P_{line}_{callee[:8]}"
                if nid in used_ids:
                    for i in range(1, 100):
                        cid = f"{nid}_{i}"
                        if cid not in used_ids:
                            nid = cid
                            break
                used_ids.add(nid)

                new_node = PDGNode(
                    id=nid,
                    kind="call_expr",
                    variable=patch.get("variable", ""),
                    expression=patch.get("expression", ""),
                    source_line=line,
                    source_col=1,
                    is_call=True,
                    callee=callee,
                    actual_params=patch.get("actual_params", []),
                )
                pdg.nodes[nid] = new_node
                applied += 1
                logger.debug("  Patched call_expr '%s' at L%s", callee, line)

            elif ptype == "def_use":
                # Add a def-use edge suggestion
                def_var = patch.get("def_variable", "")
                use_var = patch.get("use_variable", "")
                def_line = patch.get("def_line", 0)
                use_line = patch.get("use_line", 0)

                if not def_var or not use_var:
                    continue

                # Find the def and use node IDs
                def_nid = None
                use_nid = None
                for nid, n in pdg.nodes.items():
                    if n.source_line == def_line and n.variable == def_var:
                        def_nid = nid
                    if n.source_line == use_line and use_var in (n.expression or ""):
                        use_nid = nid

                if def_nid and use_nid:
                    # Check if edge already exists
                    existing = [
                        e for e in pdg.def_use_edges
                        if e.def_node_id == def_nid and e.use_node_id == use_nid
                    ]
                    if not existing:
                        from llm_client.fp_analysis.pdg_models import DefUseEdge
                        pdg.def_use_edges.append(DefUseEdge(
                            def_node_id=def_nid,
                            use_node_id=use_nid,
                            variable=def_var,
                        ))
                        applied += 1
                        logger.debug("  Patched def-use '%s' L%s→L%s", def_var, def_line, use_line)

        if applied > 0:
            logger.info("Applied %d patches to '%s'.", applied, fn_name)
            # Rebuild indices
            pdg.__post_init__()

        return applied > 0

    def patch_all(self, fn_names: list[str] | None = None) -> int:
        """Patch multiple functions.

        Args:
            fn_names: Specific functions to patch. If None, uses
                :meth:`find_incomplete` to auto-detect.

        Returns:
            Number of functions successfully patched.
        """
        if fn_names is None:
            fn_names = self.find_incomplete()

        patched = 0
        for fn_name in fn_names:
            if self.patch_function(fn_name):
                patched += 1

        return patched
