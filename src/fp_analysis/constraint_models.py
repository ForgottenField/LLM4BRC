"""Data models for Project-wide Global-State Constraint Mining."""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ConstraintSource(BaseModel):
    """Source code location where a constraint was mined from."""

    file: str = Field(description="Source file path")
    line: int = Field(default=0, description="Line number")
    snippet: str = Field(default="", description="Relevant code snippet")


class FieldInvariant(BaseModel):
    """A field's value guarantee after construction.

    E.g. ``filterBitfield_ = nullptr`` means the field starts as null.
    ``blockLength_ = blockLength(> 0 in practice)`` means it's positive.
    """

    class_name: str = Field(description="Class name")
    field_name: str = Field(description="Field name")
    initial_value: str = Field(
        description="Initial value expression from constructor"
    )
    is_conditional: bool = Field(
        default=False,
        description="True if the init value depends on a condition",
    )
    condition: str = Field(
        default="", description="Condition for conditional initialization"
    )
    confidence: str = Field(
        default="high",
        description="high|medium|low — how certain this invariant is",
    )
    sources: list[ConstraintSource] = Field(
        default_factory=list,
        description="Source evidence (file, line, snippet)",
    )

    def format_for_llm(self) -> str:
        base = (
            f"  - {self.class_name}::{self.field_name} initialized to "
            f"`{self.initial_value}`"
        )
        if self.is_conditional and self.condition:
            base += f" when `{self.condition}`"
        return base


class FieldConsistencyRule(BaseModel):
    """A consistency relation between multiple fields.

    E.g. ``filterEnabled_ == true ⇒ filterBitfield_ != nullptr``.
    """

    description: str = Field(description="Human-readable rule")
    fields: list[str] = Field(
        description="Fields involved in the consistency rule"
    )
    rule_type: str = Field(
        default="implication",
        description="implication|equivalence|mutual_exclusion",
    )
    confidence: str = Field(default="medium")
    sources: list[ConstraintSource] = Field(default_factory=list)

    def format_for_llm(self) -> str:
        return f"  - Consistency: {self.description}"


class ConfigFlagInfo(BaseModel):
    """A configuration flag and its setter function."""

    flag_name: str = Field(description="Flag variable name")
    setter_function: str = Field(description="Function that sets this flag")
    setter_file: str = Field(default="", description="Source file")
    typical_values: list[str] = Field(
        default_factory=list,
        description="Typical values (e.g. ['0', '1', 'true', 'false'])",
    )
    description: str = Field(default="", description="What this flag controls")

    def format_for_llm(self) -> str:
        return (
            f"  - Config `{self.flag_name}` (set by {self.setter_function}): "
            f"{self.description}"
        )


class SingletonInfo(BaseModel):
    """A singleton or global-state object."""

    class_name: str = Field(description="Singleton class name")
    accessor: str = Field(
        description="Accessor function (e.g. getInstance)"
    )
    initialization_guarantee: str = Field(
        default="",
        description="What state is guaranteed after init",
    )

    def format_for_llm(self) -> str:
        return (
            f"  - Singleton `{self.class_name}` ({self.accessor}): "
            f"{self.initialization_guarantee}"
        )


class ParameterConstraint(BaseModel):
    """A constraint on a function/constructor parameter derived from call-site analysis.

    E.g., ``BitfieldMan(int32_t blockLength)`` — ``blockLength`` is ``> 0`` in
    all real call sites because every caller passes ``option->getAsInt(...)``
    with default ``"1M"`` or a torrent-metadata value validated to be positive.
    """

    class_name: str = Field(description="Class name")
    function_name: str = Field(description="Function name (constructor or method)")
    parameter_name: str = Field(description="Parameter name")
    constraint_expr: str = Field(description="Constraint expression, e.g. '> 0'")
    derived_from: str = Field(
        description="Origin description, e.g. 'config option piece-length default 1M'"
    )
    confidence: str = Field(
        default="medium", description="high|medium|low — how certain"
    )
    evidence_sites: list[ConstraintSource] = Field(
        default_factory=list,
        description="Call sites where this constraint was observed",
    )

    def format_for_llm(self) -> str:
        return (
            f"  - {self.class_name}::{self.function_name} parameter "
            f"`{self.parameter_name}` is {self.constraint_expr} in practice "
            f"(derived from: {self.derived_from})"
        )


class ProjectConstraintDB(BaseModel):
    """Complete set of mined global-state constraints for a project."""

    project_name: str = Field(description="Project name")
    invariants: list[FieldInvariant] = Field(default_factory=list)
    consistency_rules: list[FieldConsistencyRule] = Field(default_factory=list)
    config_flags: list[ConfigFlagInfo] = Field(default_factory=list)
    singleton_info: list[SingletonInfo] = Field(default_factory=list)
    parameter_constraints: list[ParameterConstraint] = Field(default_factory=list)

    def format_for_llm(self) -> str:
        """Format all constraints for LLM prompt inclusion."""
        parts: list[str] = []

        if self.invariants:
            parts.append("**Constructor Invariants:**")
            for inv in self.invariants:
                parts.append(inv.format_for_llm())

        if self.consistency_rules:
            if parts:
                parts.append("")
            parts.append("**Field Consistency Rules:**")
            for rule in self.consistency_rules:
                parts.append(rule.format_for_llm())

        if self.config_flags:
            if parts:
                parts.append("")
            parts.append("**Configuration Flags:**")
            for cfg in self.config_flags:
                parts.append(cfg.format_for_llm())

        if self.singleton_info:
            if parts:
                parts.append("")
            parts.append("**Singleton / Global State:**")
            for si in self.singleton_info:
                parts.append(si.format_for_llm())

        if self.parameter_constraints:
            if parts:
                parts.append("")
            parts.append("**Parameter Constraints (from call-site analysis):**")
            for pc in self.parameter_constraints:
                parts.append(pc.format_for_llm())

        return "\n".join(parts) if parts else "(no project-wide constraints)"

    def format_relevant_constraints(
        self, target_classes: list[str] | None = None,
    ) -> str:
        """Format only the most relevant constraints.

        Includes:
        - Parameter constraints (constructor call-site analysis, high-value only)
        - Field implication rules (guard → allocation invariants)
        - Value-flow chains (cross-class propagation)

        Filters out test-file entries and keeps the prompt compact.
        Config flags and generic field initializations are excluded as they
        add noise without useful constraint information.
        """
        parts: list[str] = []
        all_params = self.parameter_constraints

        # Filter out test-file entries
        all_params = [
            pc for pc in all_params
            if not any("test/" in s.file or "Test" in s.file
                       for s in pc.evidence_sites)
        ]
        # Select high-value constraints: high confidence, config-derived, > 0
        high_value: list[ParameterConstraint] = []
        method_traces: list[ParameterConstraint] = []
        for pc in all_params:
            if pc.confidence == "high" or "> 0" in pc.constraint_expr or "config" in pc.derived_from:
                high_value.append(pc)
            elif "result of" in pc.constraint_expr:
                method_traces.append(pc)
        filtered_params = high_value + method_traces

        # Collect high-confidence field implication rules
        implications = [
            r for r in self.consistency_rules
            if r.rule_type == "implication" and r.confidence in ("high", "medium")
            # Filter out old-style co-occurrence rules (short descriptions
            # with "co-occur in conditional" — keep only implication rules
            # with meaningful descriptions like "⇒")
            and "⇒" in r.description
        ]

        # Include all implication rules — they are few, high-value, and
        # class-agnostic (field names don't encode the class name).  They are
        # relevant regardless of which target_classes were requested.

        has_params = bool(filtered_params)
        has_impls = bool(implications)

        if not has_params and not has_impls:
            return "(no project-wide constraints)"

        # --- Parameter constraints ---
        if has_params:
            parts.append(
                "**Real-World Constraints (from codebase call-site analysis):**\n"
                "IMPORTANT: These constraints are KNOWN to hold in real execution "
                "(derived from constructor argument analysis across the whole "
                "project). Treat them as HARD FACTS:"
            )
            for pc in filtered_params:
                parts.append(pc.format_for_llm())

        # --- Field implication rules ---
        if has_impls:
            if has_params:
                parts.append("")
            parts.append(
                "**Field Implications (guard → allocation):**\n"
                "These are class-internal invariants: when the guard field is true, "
                "the target field is guaranteed non-null. Treat as HARD FACTS:"
            )
            for rule in implications:
                parts.append(f"  - {rule.description}")

        # --- Value-flow chains ---
        chain_lines = self._build_constraint_chains(filtered_params)
        if chain_lines:
            if has_params or has_impls:
                parts.append("")
            parts.append("**Value Flow Chains (how constraints propagate):**")
            parts.extend("  " + l for l in chain_lines)

        return "\n".join(parts)

    def _build_constraint_chains(
        self, params: list[ParameterConstraint],
    ) -> list[str]:
        """Build cross-class value-flow summaries.

        Links config-derived constraints (e.g. ``DownloadContext::PREF_PIECE_LENGTH > 0``)
        to downstream class parameters via method-trace chains.

        For each method-trace constraint, checks whether the returned field
        matches the target of a config-derived constraint.
        """
        # Collect config-derived descriptions keyed by the field name they constrain
        # (matched via method-trace "returns `field`").
        config_exprs: dict[str, str] = {}
        for pc in params:
            if pc.confidence == "high" and "config" in pc.derived_from:
                # Derive the conceptual field name from the class context
                # E.g. "DownloadContext" + "PREF_PIECE_LENGTH" → "pieceLength_"
                m = re.search(r'PREF_(\w+)', pc.derived_from)
                if m:
                    config_exprs[m.group(1).lower()] = (
                        f"{pc.class_name} parameter > 0 "
                        f"(from config option, {pc.derived_from})"
                    )

        chains: list[str] = []
        for pc in params:
            ret_m = re.search(r"returns `(\w+)`", pc.derived_from)
            if not ret_m:
                continue
            ret_field = ret_m.group(1)
            # Normalize: remove trailing underscores, lowercase, strip spaces
            ret_key = ret_field.rstrip("_").replace("_", "").lower()
            for config_key, desc in list(config_exprs.items()):
                config_normalized = config_key.replace("_", "").lower()
                if ret_key == config_normalized:
                    chains.append(
                        f"→ {pc.class_name}::{ret_field} must be > 0 "
                        f"({desc})"
                    )
                    del config_exprs[config_key]
                    break

        # Remaining config constraints as direct statements
        for ret_key, desc in config_exprs.items():
            chains.append(f"→ {desc}")

        return chains
