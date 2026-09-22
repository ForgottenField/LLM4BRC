"""Bug report analysis for POC generation.

Uses LLM (via ``llm_client`` providers) to parse static analysis tool
outputs — SARIF, plain text, JSON — into structured bug information
that drives downstream POC generation.
"""

from llm_client.bug_analysis.analyzer import BugAnalyzer
from llm_client.bug_analysis.schemas import (
    AnalysisResult,
    BatchAnalysisResult,
    BugFinding,
    BugReport,
    Location,
    POCViability,
    Severity,
)
from llm_client.bug_analysis.storage import load_analysis, load_batch, save_analysis, save_batch

__all__ = [
    "BugAnalyzer",
    "AnalysisResult",
    "BatchAnalysisResult",
    "BugFinding",
    "BugReport",
    "Location",
    "POCViability",
    "Severity",
    "save_analysis",
    "save_batch",
    "load_analysis",
    "load_batch",
]
