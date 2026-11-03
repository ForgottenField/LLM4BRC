"""Narrow collaboration interfaces for the analysis engine."""

from __future__ import annotations

from typing import Any, Protocol


class SimulationAnalyzer(Protocol):
    """The analyzer capabilities required by :class:`SimulationVerifier`."""

    @property
    def project_name(self) -> str: ...

    @property
    def domain_facts_text(self) -> str: ...

    def extract_source_context(self, parsed: Any, max_lines: int = 80) -> str: ...

    def extract_callee_sources_for_prompt(self, parsed: Any) -> str: ...

    def format_selected_branch_blueprint(
        self, selections: list[dict] | None,
    ) -> str: ...

    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str: ...

    def parse_json_response(self, raw: str, context: str) -> dict: ...

    def parse_json_response_tolerant(self, raw: str, context: str) -> dict: ...

    def extract_code(self, raw: str) -> str: ...
