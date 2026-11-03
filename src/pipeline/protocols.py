"""Application-layer ports implemented by the analysis engine."""

from __future__ import annotations

from typing import Any, Protocol

from fp_analysis.models import AnalysisResult, CompletedPath
from fp_analysis.pdg_models import SDG


class PathAnalyzerPort(Protocol):
    """Capabilities the stage orchestrator requires from its analyzer."""

    model_name: str
    project_name: str
    domain_facts_text: str

    @property
    def sdg(self) -> SDG | None: ...

    def ensure_project_name(self, project_name: str) -> None: ...

    def set_domain_facts_text(self, text: str) -> None: ...

    def extract_class_names_from_path(self, parsed_report: Any) -> list[str]: ...

    def load_global_constraints(
        self,
        source_root: str | None = None,
        target_classes: list[str] | None = None,
    ) -> str: ...

    def select_combination_segments(
        self,
        analysis_result: AnalysisResult,
        completed: CompletedPath,
        segments: list,
        path_space: dict,
        combo: tuple,
    ) -> list[dict] | None: ...

    def select_feasible_path_segments(
        self,
        analysis_result: AnalysisResult,
        completed: CompletedPath,
        **kwargs,
    ) -> list[dict] | None: ...

    def complete_constraints(
        self,
        analysis_result: AnalysisResult,
        **kwargs,
    ) -> CompletedPath: ...

    def analyze(self, report_path: Any, source_root: str) -> AnalysisResult: ...

    def generate_poc(self, *args, **kwargs) -> str: ...
