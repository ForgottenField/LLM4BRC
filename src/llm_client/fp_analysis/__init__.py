"""LLM-assisted false positive path completion for POC generation."""
from __future__ import annotations

from llm_client.fp_analysis.function_analyzer import FunctionAnalyzer
from llm_client.fp_analysis.function_summary import (
    CachedFunctionData,
    FunctionSummary,
    ReturnValueConstraint,
    normalize_bug_type,
)
from llm_client.fp_analysis.models import (
    AnalysisResult,
    Classification,
    CompletedPath,
    CompletedStep,
    FPGap,
    FPGapType,
    GapSeverity,
    ParsedReport,
    PathAnalysis,
    POCPlan,
    ReportMetadata,
)
from llm_client.fp_analysis.path_analyzer import FPAnalyzer
from llm_client.fp_analysis.summary_cache import SummaryCache

# Context control (post-slice threshold check)
from llm_client.fp_analysis.context_controller import (
    CompressedContext,
    ContextControlConfig,
    ContextController,
    SliceContextInfo,
)

# PDG / slicing exports
from llm_client.fp_analysis.pdg_models import (
    ControlDepEdge,
    DefUseEdge,
    FunctionPDG,
    FunctionSummary as PDGFunctionSummary,
    PDGNode,
    SDG,
    SliceNode,
    SliceResult,
)
from llm_client.fp_analysis.slicer import (
    SlicerConfig as SlicerConfigAlias,
    compute_slice,
    find_trigger_node,
)
from llm_client.fp_analysis.slice_analyzer import (
    SliceConfig,
    analyze_with_slicing,
)

# PDG completeness / patching (Strategy 4 & 5)
from llm_client.fp_analysis.pdg_completeness import (
    FunctionCompleteness,
    LLMPDGPatchProvider,
    PDGCompletenessReport,
)

# Slice mask (slice tracking data structure)
from llm_client.fp_analysis.slice_mask import SliceMask

__all__ = [
    # Main API
    "FPAnalyzer",
    # Function analysis (on-demand summarization)
    "FunctionAnalyzer",
    "FunctionSummary",
    "ReturnValueConstraint",
    "CachedFunctionData",
    "normalize_bug_type",
    "SummaryCache",
    # Context control
    "ContextController",
    "ContextControlConfig",
    "CompressedContext",
    "SliceContextInfo",
    # Models
    "AnalysisResult",
    "Classification",
    "CompletedPath",
    "CompletedStep",
    "FPGap",
    "FPGapType",
    "GapSeverity",
    "ParsedReport",
    "PathAnalysis",
    "POCPlan",
    "ReportMetadata",
    # PDG / slicing
    "PDGNode",
    "DefUseEdge",
    "ControlDepEdge",
    "FunctionPDG",
    "PDGFunctionSummary",
    "SDG",
    "SliceNode",
    "SliceResult",
    "SlicerConfigAlias",
    "SliceConfig",
    "compute_slice",
    "find_trigger_node",
    "analyze_with_slicing",
    # PDG completeness / patching
    "FunctionCompleteness",
    "PDGCompletenessReport",
    "LLMPDGPatchProvider",
    # Slice mask
    "SliceMask",
]
