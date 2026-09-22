"""CSA (Clang Static Analyzer) path extraction for POC generation."""

from llm_client.csa_analysis.errors import (
    CSAError,
    CSANotFoundError,
    SourceFileNotFoundError,
    CSATimeoutError,
    PlistParseError,
    CallGraphBuildError,
)
from llm_client.csa_analysis.models import (
    AnalyzerMode,
    BugFindingRef,
    CSAConfig,
    CSAResult,
    BugPath,
    PathEvent,
    CorrelatedResult,
    Confidence,
    FileResult,
    ProjectAnalysisResult,
    ReachabilityTarget,
    ReachabilityResult,
)
from llm_client.csa_analysis.runner import CSARunner
from llm_client.csa_analysis.parser import PlistParser
from llm_client.csa_analysis.correlator import BugPathCorrelator
from llm_client.csa_analysis.scanner import (
    ProjectScanner,
    CompilationDatabase,
    discover_source_files,
)
from llm_client.csa_analysis.callgraph_models import (
    CallGraph,
    CallGraphNode,
    CallGraphEdge,
    EntryPoint,
    EntryPointResult,
)
from llm_client.csa_analysis.callgraph_bridge import (
    find_entry_points,
    build_synthetic_callgraph,
)

__all__ = [
    "AnalyzerMode",
    "CSAError",
    "CSANotFoundError",
    "SourceFileNotFoundError",
    "CSATimeoutError",
    "PlistParseError",
    "CallGraphBuildError",
    "BugFindingRef",
    "CSAConfig",
    "CSAResult",
    "BugPath",
    "PathEvent",
    "CorrelatedResult",
    "Confidence",
    "FileResult",
    "ProjectAnalysisResult",
    "CSARunner",
    "PlistParser",
    "BugPathCorrelator",
    "ProjectScanner",
    "CompilationDatabase",
    "discover_source_files",
    "ReachabilityTarget",
    "ReachabilityResult",
    "CallGraph",
    "CallGraphNode",
    "CallGraphEdge",
    "EntryPoint",
    "EntryPointResult",
    "find_entry_points",
    "build_synthetic_callgraph",
]
