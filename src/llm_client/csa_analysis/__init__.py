"""CSA (Clang Static Analyzer) data models.

Only the data models live here now: the `BugPath`/`PathEvent` types that the
report parsers and the FP pipeline consume, and the call-graph models that
`pdg_loader` builds its SDG on top of.  The old CSA *runner* wrappers
(`CSARunner`/`PlistParser`/`BugPathCorrelator`/`ProjectScanner`/`EntryPoint`
synthesis) had no entry point in the live pipeline and were removed; CSA is
invoked out of tree, and `pdg_<project>.json` / `cfg_cache/` come from the C++
tooling under `pdg/`, `callgraph/` and `checker/`.
"""

from llm_client.csa_analysis.models import (
    AnalyzerMode,
    BugFindingRef,
    CSAConfig,
    CSAResult,
    BugPath,
    EventKind,
    PathEvent,
    CorrelatedResult,
    Confidence,
    FileResult,
    ProjectAnalysisResult,
    ReachabilityTarget,
    ReachabilityResult,
)
from llm_client.csa_analysis.callgraph_models import (
    CallGraph,
    CallGraphNode,
    CallGraphEdge,
    EntryPoint,
    EntryPointResult,
)

__all__ = [
    "AnalyzerMode",
    "BugFindingRef",
    "CSAConfig",
    "CSAResult",
    "BugPath",
    "EventKind",
    "PathEvent",
    "CorrelatedResult",
    "Confidence",
    "FileResult",
    "ProjectAnalysisResult",
    "ReachabilityTarget",
    "ReachabilityResult",
    "CallGraph",
    "CallGraphNode",
    "CallGraphEdge",
    "EntryPoint",
    "EntryPointResult",
]
