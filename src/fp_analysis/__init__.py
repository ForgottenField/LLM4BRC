"""LLM-assisted false-positive detection and POC generation for static analyzers.

This package holds the **engine**: the extractors, checkers, identifiers and
models, one flat module per capability.  The **application** that drives them —
the ten stages, the CLI's arguments, the console's output and the result JSON —
is :mod:`llm_client.fp_analysis.pipeline`, and the repository root's
``run_path_selection.py`` is the runner that calls it.  ``docs/pipeline-stages.md``
is the authoritative description of the ten ordered stages and which module
serves each.

Only the live surface is re-exported here.  The package is imported by
submodule path (``llm_client.fp_analysis.<module>``) throughout the pipeline,
so this file is a convenience facade, not a load-bearing import site.
"""

from __future__ import annotations

from fp_analysis.models import (
    AnalysisResult,
    Classification,
    CompletedPath,
    CompletedStep,
    FPGap,
    FPGapType,
    GapSeverity,
    POCPlan,
    ParsedReport,
    PathAnalysis,
    ReportMetadata,
)
from fp_analysis.path_analyzer import FPAnalyzer
from fp_analysis.pdg_models import (
    ControlDepEdge,
    DefUseEdge,
    FunctionPDG,
    FunctionSummary,
    PDGNode,
    SDG,
    SliceNode,
    SliceResult,
)
from fp_analysis.slice_mask import SliceMask

__all__ = [
    # Main API
    "FPAnalyzer",
    # Report / path models
    "AnalysisResult",
    "Classification",
    "CompletedPath",
    "CompletedStep",
    "FPGap",
    "FPGapType",
    "GapSeverity",
    "POCPlan",
    "ParsedReport",
    "PathAnalysis",
    "ReportMetadata",
    # PDG / slicing models
    "ControlDepEdge",
    "DefUseEdge",
    "FunctionPDG",
    "FunctionSummary",
    "PDGNode",
    "SDG",
    "SliceNode",
    "SliceResult",
    # Slice mask
    "SliceMask",
]
