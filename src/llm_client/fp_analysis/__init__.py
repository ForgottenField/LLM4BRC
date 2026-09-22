"""LLM-assisted false-positive detection and POC generation for CSA reports.

The pipeline itself lives in ``run_path_selection.py`` at the repo root; this
package holds the pieces it drives — see ``docs/pipeline-stages.md`` for the
ten ordered stages and which module serves each.

Only the live surface is re-exported here.  The package is imported by
submodule path (``llm_client.fp_analysis.<module>``) throughout the pipeline,
so this file is a convenience facade, not a load-bearing import site.
"""

from __future__ import annotations

from llm_client.fp_analysis.models import (
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
from llm_client.fp_analysis.path_analyzer import FPAnalyzer
from llm_client.fp_analysis.pdg_models import (
    ControlDepEdge,
    DefUseEdge,
    FunctionPDG,
    FunctionSummary,
    PDGNode,
    SDG,
    SliceNode,
    SliceResult,
)
from llm_client.fp_analysis.slice_mask import SliceMask

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
