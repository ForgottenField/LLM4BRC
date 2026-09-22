"""Post-slice context control: decide when sliced code is compact enough.

After PDG backward slicing we get reduced code.  This module checks whether
that code is small enough for the LLM to analyse precisely, or whether we
need to further compress it with on-demand function summarization.

Flow::

    SliceResult + reduced_code
        │
        ▼
    ContextController.check()
        │
        ├── call_depth ≤ max_call_depth AND
        │   code_chars ≤ max_code_chars
        │   → USE SLICED CODE DIRECTLY
        │
        └── threshold exceeded
            → ON-DEMAND SUMMARIZATION for deep/external functions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from llm_client.base import LLMProvider
from llm_client.fp_analysis.function_analyzer import FunctionAnalyzer
from llm_client.fp_analysis.function_summary import FunctionSummary
from llm_client.fp_analysis.pdg_models import SDG, SliceResult
from llm_client.fp_analysis.summary_cache import SummaryCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ContextControlConfig:
    """Thresholds for context control.

    Attributes:
        max_call_depth: Maximum allowed function call depth in the slice.
            Default 3 (A→B→C).  Deeper chains trigger summarization.
        max_code_chars: Maximum allowed characters in reduced code before
            summarization is triggered.  Default 12 000 (≈3 000 tokens,
            comfortable for precise LLM analysis).
    """
    max_call_depth: int = 3
    max_code_chars: int = 12_000


# ---------------------------------------------------------------------------
# Context info
# ---------------------------------------------------------------------------


@dataclass
class SliceContextInfo:
    """Computed information about the post-slice context.

    Attributes:
        code_chars: Total character count of the reduced code.
        max_call_depth: Deepest function call chain in the slice.
        function_count: Number of distinct functions in the slice.
        library_function_count: Number of external-library functions.
        exceeds_threshold: True if any configured threshold is exceeded.
    """
    code_chars: int = 0
    max_call_depth: int = 0
    function_count: int = 0
    library_function_count: int = 0
    exceeds_threshold: bool = False

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (
            f"Context: {self.code_chars} chars, "
            f"depth={self.max_call_depth}, "
            f"{self.function_count} functions"
            f"{' ⚠ EXCEEDS THRESHOLD' if self.exceeds_threshold else ''}"
        )


# ---------------------------------------------------------------------------
# Context Controller
# ---------------------------------------------------------------------------


class ContextController:
    """Decides whether sliced code is compact enough for LLM analysis.

    If thresholds are exceeded, triggers on-demand summarization for the
    deepest call chains or external-library functions.
    """

    def __init__(
        self,
        config: ContextControlConfig | None = None,
        llm_provider: LLMProvider | None = None,
        summary_cache: SummaryCache | None = None,
        project_name: str = "unknown",
    ) -> None:
        self._config = config or ContextControlConfig()
        self._function_analyzer: FunctionAnalyzer | None = None
        self._project_name = project_name

        if llm_provider:
            self._function_analyzer = FunctionAnalyzer(
                llm_provider=llm_provider,
                cache=summary_cache or SummaryCache(),
                project_name=project_name,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        slice_result: SliceResult | None,
        reduced_code: str,
        sdg: SDG | None = None,
    ) -> SliceContextInfo:
        """Evaluate post-slice context against thresholds.

        Args:
            slice_result: Result from backward slicing (may be None).
            reduced_code: Reduced C++ code from ``code_reducer``.
            sdg: System dependence graph (used for call-depth computation).

        Returns:
            ``SliceContextInfo`` with computed metrics.
        """
        info = self._compute_info(slice_result, reduced_code, sdg)
        info.exceeds_threshold = (
            info.max_call_depth > self._config.max_call_depth
            or info.code_chars > self._config.max_code_chars
        )
        logger.info("Context check: %s", info.summary())
        return info

    def compress(
        self,
        slice_result: SliceResult | None,
        reduced_code: str,
        sdg: SDG | None = None,
        bug_type: str = "",
        source_root: str = "",
    ) -> CompressedContext:
        """If thresholds are exceeded, produce a compressed version.

        The compression strategy:

        1. For external-library functions (no source in project):
           Generate a ``FunctionSummary`` (best-guess from name/signature).

        2. For deeply nested project-internal functions:
           Generate a ``FunctionSummary`` from the function body.

        3. For the top-level trigger function:
           Keep the reduced code as-is.

        Returns a ``CompressedContext`` with both the original and compressed
        versions so the caller can choose.
        """
        # --- Phase 1: Check thresholds ---
        info = self.check(slice_result, reduced_code, sdg)

        if not info.exceeds_threshold or self._function_analyzer is None:
            return CompressedContext(
                original_code=reduced_code,
                compressed_code=reduced_code,
                summaries=[],
                was_compressed=False,
                info=info,
            )

        # --- Phase 2: Identify functions to summarize ---
        candidates = self._find_summary_candidates(
            slice_result, sdg, source_root,
        )
        logger.info(
            "Compressing context: %d candidate function(s) for summarization",
            len(candidates),
        )

        summaries: list[FunctionSummary] = []
        for fn_name in candidates:
            summary = self._function_analyzer.summarize_function(
                function_name=fn_name,
                source_root=source_root,
                bug_type=bug_type,
            )
            if summary:
                summaries.append(summary)

        # --- Phase 3: Build compressed code ---
        compressed = self._build_compressed_code(
            reduced_code, summaries, info,
        )

        return CompressedContext(
            original_code=reduced_code,
            compressed_code=compressed,
            summaries=summaries,
            was_compressed=True,
            info=info,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_info(
        slice_result: SliceResult | None,
        reduced_code: str,
        sdg: SDG | None = None,
    ) -> SliceContextInfo:
        """Compute context metrics from slice result and reduced code."""
        code_chars = len(reduced_code)
        function_count = 0
        max_call_depth = 0
        library_function_count = 0

        if slice_result is not None:
            function_count = len(slice_result.involved_functions)

            # Compute call depth from SDG call graph
            if sdg is not None:
                max_call_depth = _compute_max_call_depth(
                    slice_result, sdg,
                )

            # Estimate library functions (those not in PDG)
            if function_count > 0 and sdg is not None:
                for fn in slice_result.involved_functions:
                    if sdg.get_pdg(fn) is None:
                        library_function_count += 1

        return SliceContextInfo(
            code_chars=code_chars,
            max_call_depth=max_call_depth,
            function_count=function_count,
            library_function_count=library_function_count,
        )

    @staticmethod
    def _find_summary_candidates(
        slice_result: SliceResult | None,
        sdg: SDG | None = None,
        source_root: str = "",
    ) -> list[str]:
        """Identify functions that should be summarized.

        Priority (sorted):
        1. Library functions (no PDG → best guess from name).
        2. Deeply nested internal functions (in the deepest call chain).
        3. Static helpers called only once.

        Returns a list of function names, ordered by summarization priority.
        """
        if slice_result is None:
            return []

        candidates: list[tuple[str, int]] = []  # (name, priority)

        involved = slice_result.involved_functions

        for fn in involved:
            # Skip the trigger function itself
            if fn == slice_result.trigger_function:
                continue

            # Library functions (no PDG) → high priority
            if sdg is not None and sdg.get_pdg(fn) is None:
                candidates.append((fn, 0))  # priority 0 = highest
            else:
                # Project-internal functions get lower priority
                candidates.append((fn, 1))

        # Sort by priority, then by name for determinism
        candidates.sort(key=lambda x: (x[1], x[0]))

        # Return at most 10 summaries (to avoid excessive LLM calls)
        return [c[0] for c in candidates[:10]]

    @staticmethod
    def _build_compressed_code(
        original_code: str,
        summaries: list[FunctionSummary],
        info: SliceContextInfo,
    ) -> str:
        """Build a compressed version of the code with inline placeholders.

        When summarization is applied, we keep the top-level trigger function's
        reduced code but replace deeply-nested function bodies with concise
        summary references.
        """
        if not summaries:
            return original_code

        lines = [
            "// ================================================================",
            f"// SLICED CODE (compressed — {info.code_chars} chars exceeds threshold)",
            f"// Function summaries for {len(summaries)} opaque/deep functions:",
            "// ================================================================",
            "",
        ]

        # Add summaries as comments at the top
        for s in summaries:
            summary_text = s.format_for_llm()
            for line in summary_text.split("\n"):
                lines.append(f"// {line}")
            lines.append("")

        lines.append("// --- Reduced code (trigger function) ---")
        lines.append("")

        # Add the original reduced code
        lines.append(original_code)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compressed context result
# ---------------------------------------------------------------------------


@dataclass
class CompressedContext:
    """Result of context compression.

    Attributes:
        original_code: The original reduced code from slicing.
        compressed_code: The compressed version (same as original if no
            compression was needed).
        summaries: Function summaries generated during compression
            (empty if no compression was needed).
        was_compressed: Whether compression was actually applied.
        info: Context metrics.
    """
    original_code: str = ""
    compressed_code: str = ""
    summaries: list[FunctionSummary] = field(default_factory=list)
    was_compressed: bool = False
    info: SliceContextInfo = field(default_factory=SliceContextInfo)


# ---------------------------------------------------------------------------
# Call depth computation (standalone utility)
# ---------------------------------------------------------------------------


def _compute_max_call_depth(
    slice_result: SliceResult, sdg: SDG,
) -> int:
    """Compute the maximum function call depth in the slice.

    Traces call→callee edges through the SDG starting from the trigger
    function.  Uses a simple DFS with cycle detection.
    """
    if not slice_result.involved_functions:
        return 0

    trigger_fn = slice_result.trigger_function

    # Build adjacency list from call graph
    call_graph: dict[str, list[str]] = {}
    for fn in slice_result.involved_functions:
        pdg = sdg.get_pdg(fn)
        if pdg is None:
            continue
        callees: list[str] = []
        for node in pdg.nodes.values():
            if node.is_call and node.callee:
                if node.callee in slice_result.involved_functions:
                    callees.append(node.callee)
        if callees:
            call_graph[fn] = callees

    # DFS for max depth
    visited: set[str] = set()
    max_depth = [0]

    def dfs(node: str, depth: int) -> None:
        max_depth[0] = max(max_depth[0], depth)
        if node in visited:
            return
        visited.add(node)
        for callee in call_graph.get(node, []):
            dfs(callee, depth + 1)
        visited.discard(node)

    dfs(trigger_fn, 0)
    return max_depth[0]
