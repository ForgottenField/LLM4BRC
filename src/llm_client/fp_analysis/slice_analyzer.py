"""Integration adapter: connects PDG-based backward slicing with FPAnalyzer.

This module bridges the gap between the existing feasibility-driven pipeline
and the new PDG/SDG slicing infrastructure.  It provides:

1. A :class:`SliceConfig` dataclass for configuration.
2. A high-level :func:`analyze_with_slicing` that orchestrates SDG loading,
   trigger detection, backward slicing, code reduction, **context control**,
   and storage of results on the ``AnalysisResult`` model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llm_client.base import LLMProvider
from llm_client.csa_analysis.models import BugPath
from llm_client.fp_analysis.code_reducer import reduce_code
from llm_client.fp_analysis.context_controller import (
    CompressedContext,
    ContextController,
    SliceContextInfo,
)
from llm_client.fp_analysis.pdg_augment import augment_pdg_sdg
from llm_client.fp_analysis.pdg_loader import load_pdg
from llm_client.fp_analysis.pdg_models import SDG, SliceResult
from llm_client.fp_analysis.slicer import (
    SlicerConfig,
    compute_slice,
    compute_slice_from_seeds,
    find_trigger_node,
)
from llm_client.fp_analysis.summary_cache import SummaryCache

logger = logging.getLogger(__name__)


@dataclass
class SliceConfig:
    """Configuration for PDG-based slicing integration.

    Attributes:
        enabled: Whether slicing is enabled for this analysis run.
        pdg_path: Path to the PDG JSON file (produced by PDGBuilder).
        callgraph_path: Optional path to the call graph JSON (produced by
            CallGraphBuilder).  If provided, enables cross-function slicing.
        max_depth: Maximum recursion depth for cross-function slicing.
        produce_reduced_code: If True, generate reduced C++ code from the
            slice and store it on AnalysisResult.
        use_full_path_seeds: If True (and a BugPath is available), extract
            seed nodes from ALL path events instead of just the final bug
            trigger point.  This ensures control conditions and intermediate
            data dependencies along the bug path are captured in the slice.
        enable_context_control: If True, run context-control checks after
            slicing and optionally compress oversized context with on-demand
            function summarization.  Requires an LLM provider.
    """

    enabled: bool = False
    pdg_path: str = ""
    callgraph_path: str = ""
    max_depth: int = 10
    produce_reduced_code: bool = True
    use_full_path_seeds: bool = True
    enable_context_control: bool = True


def analyze_with_slicing(
    trigger_function: str,
    trigger_line: int,
    source_root: str,
    config: SliceConfig,
    bug_path: BugPath | None = None,
    llm_provider: LLMProvider | None = None,
    project_name: str = "",
    bug_type: str = "",
) -> tuple[Optional[SliceResult], str, Optional[CompressedContext], Optional[SliceContextInfo]]:
    """Run PDG-based backward slicing from a bug trigger point.

    This is the primary integration entry point.  It:

    1. Loads the SDG from PDG JSON (and call graph JSON if configured).
    2. Identifies the trigger node or extracts seed nodes from bug-path events.
    3. Computes the backward slice from the seed set.
    4. Optionally generates reduced C++ code.
    5. Optionally runs context control (checks if sliced code is compact
       enough; triggers on-demand summarization if thresholds are exceeded).

    When *bug_path* is provided and ``config.use_full_path_seeds`` is True,
    seed nodes are extracted from **all path events** via PDG AST metadata
    matching (node kind, variable, call info), ensuring the slice covers
    variables along the entire bug execution path — not just the final
    trigger point.

    Args:
        trigger_function: Qualified name of the function containing the bug.
        trigger_line: Source line number of the bug trigger (from the CSA
            report's error site).
        source_root: Project source root for file resolution.
        config: Slicing configuration.
        bug_path: Optional structured bug path (from CSA report parsing).
            When provided and ``config.use_full_path_seeds`` is True, the
            seed set is built from all path events via PDG AST metadata.
        llm_provider: Optional LLM provider for on-demand context compression.
        project_name: Project name (for context-control cache).
        bug_type: CSA bug type (for summary focus).

    Returns:
        A tuple ``(slice_result, reduced_code, compressed_context, context_info)``.
        Either may be ``None``/empty if slicing failed or was disabled.
    """
    if not config.enabled or not config.pdg_path:
        return None, "", None, None

    pdg_path = Path(config.pdg_path)
    if not pdg_path.exists():
        logger.warning("PDG file not found: %s", pdg_path)
        return None, "", None, None

    # ---- Step 1: Load SDG ----
    if config.callgraph_path:
        from llm_client.fp_analysis.pdg_loader import load_sdg_from_files
        sdg = load_sdg_from_files(str(pdg_path), config.callgraph_path)
    else:
        pdgs = load_pdg(str(pdg_path))
        if not pdgs:
            return None, "", None, None
        sdg = SDG(pdgs=pdgs)
        augment_pdg_sdg(sdg)

    if sdg is None or not sdg.pdgs:
        logger.warning("SDG is empty — no PDG data available.")
        return None, "", None, None

    # ---- Step 2: Build seed set ----
    seeds: list[tuple[str, str, str]] = []

    if bug_path is not None and config.use_full_path_seeds:
        # Multi-seed: extract from all path events
        from llm_client.fp_analysis.path_seed_extractor import (
            extract_seeds_from_bug_path,
        )
        seeds = extract_seeds_from_bug_path(
            bug_path, sdg,
            source_root=source_root,
            fallback_trigger_fn=trigger_function,
            fallback_trigger_line=trigger_line,
        )

    if not seeds:
        # Fallback to single trigger
        trigger_node = find_trigger_node(trigger_function, trigger_line, sdg)
        if trigger_node is None:
            logger.warning(
                "Could not find trigger node for '%s' near line %d.",
                trigger_function, trigger_line,
            )
            return None, "", None, None
        seeds = [(trigger_function, trigger_node, "trigger")]

    # ---- Step 3a: Build BTBS path annotations ----
    annotations: dict[tuple[str, int], bool] | None = None
    caller_map: dict[str, str] | None = None
    if bug_path is not None and config.use_full_path_seeds:
        from llm_client.fp_analysis.path_seed_extractor import (
            build_caller_map,
            build_path_annotations,
        )
        annotations = build_path_annotations(
            bug_path, sdg, source_root=source_root,
        )
        if annotations:
            not_taken = sum(1 for v in annotations.values() if not v)
            logger.info(
                "BTBS: %d branch annotations (%d NOT-taken)",
                len(annotations), not_taken,
            )

        # BTBS Phase 4: callee→caller map for reverse propagation
        caller_map = build_caller_map(bug_path, sdg, entry_fn=trigger_function)
        if caller_map:
            logger.info(
                "BTBS: %d callee→caller edges for reverse propagation",
                len(caller_map),
            )

    # ---- Step 3b: Compute backward slice (with BTBS if annotations present) ----
    slicer_cfg = SlicerConfig(max_depth=config.max_depth)
    try:
        slice_result = compute_slice_from_seeds(
            sdg, seeds, slicer_cfg,
            annotations=annotations,
            caller_map=caller_map,
        )
    except (ValueError, KeyError) as exc:
        logger.warning("Backward slicing failed: %s", exc)
        return None, "", None, None

    logger.info(
        "Backward slice: %d nodes from %d seeds (trigger=%s:%s), "
        "input vars=%s, truncated=%s",
        len(slice_result.nodes),
        len(seeds),
        trigger_function,
        trigger_line,
        sorted(slice_result.input_variables),
        slice_result.truncated,
    )

    # ---- Step 4: Generate reduced code ----
    reduced_code = ""
    if config.produce_reduced_code:
        try:
            reduced_code = reduce_code(
                slice_result,
                source_root=source_root,
                include_search_roots=[Path(source_root)] if source_root else [],
            )
        except Exception as exc:
            logger.warning("Code reduction failed: %s", exc)

    # ---- Step 5: Context control (post-slice threshold check) ----
    compressed_context: Optional[CompressedContext] = None
    context_info: Optional[SliceContextInfo] = None

    if config.enable_context_control:
        try:
            controller = ContextController(
                llm_provider=llm_provider,
                project_name=project_name or Path(source_root).name,
            )
            compressed_context = controller.compress(
                slice_result=slice_result,
                reduced_code=reduced_code,
                sdg=sdg,
                bug_type=bug_type,
                source_root=source_root,
            )
            context_info = compressed_context.info
            if compressed_context.was_compressed:
                logger.info(
                    "Context compression applied: %d chars → %d chars, "
                    "%d function summary(ies)",
                    len(reduced_code),
                    len(compressed_context.compressed_code),
                    len(compressed_context.summaries),
                )
        except Exception as exc:
            logger.warning("Context control failed: %s", exc)

    return slice_result, reduced_code, compressed_context, context_info


def format_slice_info(slice_result: SliceResult) -> dict:
    """Format slice metadata for storage on ``AnalysisResult.slice_info``."""
    return {
        "node_count": len(slice_result.nodes),
        "involved_functions": list(slice_result.involved_functions),
        "input_variables": sorted(slice_result.input_variables),
        "truncated": slice_result.truncated,
        "trigger_function": slice_result.trigger_function,
        "trigger_node_id": slice_result.trigger_node_id,
    }

