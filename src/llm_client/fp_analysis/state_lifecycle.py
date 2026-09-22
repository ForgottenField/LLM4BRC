"""State Lifecycle Analysis (Step 2.7) — static analysis component.

Extracts the ordered call chain from CSA path events, identifies state
variables via PDG queries, and detects known CSA-FP patterns by reusing
existing static analysis methods.

This module is the **static analysis half** of Step 2.7.  Its output feeds
into a focused LLM prompt (``STATE_LIFECYCLE_PROMPT``) that decides whether
a state lifecycle guarantees initialization of the Error Start variable.
"""

from __future__ import annotations

import re
from typing import Optional

from llm_client.fp_analysis.models import CallChainEntry


class CallChainExtractor:
    """Deterministic static analysis for state lifecycle detection.

    Three public class methods, each independently testable:

    - ``extract(path_events)`` — build ordered call chain from CSA path
    - ``identify_state_variables(call_chain, error_start_var, sdg)`` —
      find pointer/ref params written by called functions
    - ``detect_known_patterns(reduced_code)`` — reuse existing
      ``_detect_loop_initialized_vars`` and ``_detect_union_type_punning``
      logic
    """

    @staticmethod
    def extract(path_events: list[dict]) -> list[CallChainEntry]:
        """Extract ordered call chain from CSA path events.

        Scans *path_events* for ``Calling ...`` and ``← Returning from ...``
        patterns in ``event_kind`` / ``description``.  Returns entries in
        path order (which is the call order in a linearized trace).

        Args:
            path_events: The ``parsed_report.path_events`` list from a CSA
                bug report.

        Returns:
            Ordered list of :class:`CallChainEntry` — one per call event.
            Empty if no call events found.
        """
        entries: list[CallChainEntry] = []

        for evt in path_events:
            kind = evt.get("event_kind", "")
            desc = evt.get("description", "")
            line = evt.get("line_number", 0)
            fn = evt.get("function_name", "")
            # CSA events may use "path_number" or "event_number"
            idx = evt.get("path_number") or evt.get("event_number") or 0

            # == Structured-field path (preferred) ==
            # HTML parser identifies calling events and extracts the
            # called function name into a dedicated field.
            if evt.get("is_calling_event") and evt.get("called_function"):
                entries.append(CallChainEntry(
                    event_index=idx,
                    function_name=fn or "",
                    callee=evt.get("called_function"),
                    source_line=int(line) if line else 0,
                    is_entry=True,
                ))
                continue

            # == Text-fallback path (return events, or unmarked callers) ==
            raw_text = (kind + " " + desc).lower()
            if "calling" in raw_text:
                callee = _extract_callee(desc, kind)
                if callee:
                    entries.append(CallChainEntry(
                        event_index=idx,
                        function_name=fn or "",
                        callee=callee,
                        source_line=int(line) if line else 0,
                        is_entry=True,
                    ))
            elif "return" in raw_text:
                callee = _extract_callee(desc, kind)
                if callee:
                    entries.append(CallChainEntry(
                        event_index=idx,
                        function_name=fn or "",
                        callee=callee,
                        source_line=int(line) if line else 0,
                        is_entry=False,
                    ))

        return entries

    @staticmethod
    def identify_state_variables(
        call_chain: list[CallChainEntry],
        error_start_var: str | None = None,
        sdg: object = None,
    ) -> list[str]:
        """Identify state variables from call chain using PDG/SDG.

        For each unique ``Calling`` entry, queries the SDG for the
        callee's ``FunctionSummary.output_variables``.  Any function
        that writes to one or more variables (output variables) is
        considered to carry *state*.  The names of those output
        variables are returned as *state variables*.

        When *sdg* is ``None`` (no PDG available), returns empty list
        and relies on the LLM prompt's source context instead.

        Args:
            call_chain: Previously extracted call chain entries.
            error_start_var: The variable the CSA flagged; if it
                matches an output variable of any callee, that callee
                is marked as writing to a state variable.
            sdg: The ``SDG`` instance (or anything with a
                ``get_pdg(name)`` method that returns a ``FunctionPDG``
                with a ``summary.output_variables`` list).

        Returns:
            List of state variable names (possibly qualified like
            ``ctx->imsg->utf8state``).  Empty if no PDG data available
            or no functions with output variables found.
        """
        if not call_chain or sdg is None:
            return []

        state_vars: set[str] = set()
        seen_callees: set[str] = set()

        for entry in call_chain:
            if not entry.is_entry or not entry.callee:
                continue
            if entry.callee in seen_callees:
                continue
            seen_callees.add(entry.callee)

            # Try to get the callee's PDG to check output variables
            callee_pdg = _get_callee_pdg(sdg, entry.callee)
            if callee_pdg is None:
                continue

            output_vars = callee_pdg.summary.output_variables or []
            if not output_vars:
                continue

            # All output variables of called functions are state candidates
            for ov in output_vars:
                state_vars.add(ov)

            # If the error_start_var matches an output var, note it
            if error_start_var:
                es_short = (
                    error_start_var.split("->")[-1]
                    if "->" in error_start_var
                    else error_start_var
                )
                for ov in output_vars:
                    ov_short = ov.split("->")[-1] if "->" in ov else ov
                    if es_short == ov_short or es_short in ov or ov in es_short:
                        state_vars.add(f"{entry.callee}::{ov}")

        return list(state_vars)

    @staticmethod
    def detect_known_patterns(reduced_code: str) -> list[str]:
        """Reuse existing static analysis to detect known CSA-FP patterns.

        Delegates to the same logic as
        :meth:`FPAnalyzer._detect_loop_initialized_vars` and
        :meth:`FPAnalyzer._detect_union_type_punning` via inline regex.

        Args:
            reduced_code: PDG-sliced reduced C++ source code.

        Returns:
            List of pattern name strings — any subset of
            ``["loop_carried_init", "union_type_punning"]``.
            Empty if no patterns detected.
        """
        if not reduced_code or not reduced_code.strip():
            return []

        patterns: list[str] = []

        if _has_loop_carried_init(reduced_code):
            patterns.append("loop_carried_init")

        if _has_union_type_punning(reduced_code):
            patterns.append("union_type_punning")

        return patterns


# =========================================================================
# Internal helpers (shared logic with FPAnalyzer static methods)
# =========================================================================


def _extract_callee(description: str, event_kind: str) -> str | None:
    """Extract callee function name from a call event description.

    Handles formats like:
    - ``"Calling decode"``
    - ``"← Returning from wslay_event_imsg_reset"``
    - ``"Calling wslay_event_imsg_reset"``
    - ``"← Calling 'decode' →"``  (CSA HTML report format)
    """
    text = description

    # CSA format: "← Calling 'func' →" — extract text between single quotes
    import re as _re
    quote_match = _re.search(r"['\"](\w+)['\"]", text)
    if quote_match:
        return quote_match.group(1)

    # Standard format: "Calling <func>" or "← Returning from <func>"
    for prefix in ("Calling ", "returning from ", "Returning from "):
        if prefix in text:
            idx = text.lower().index(prefix.lower())
            start = idx + len(prefix)
            rest = text[start:].strip().split()[0] if text[start:].strip() else ""
            if rest:
                return rest

    return None


def _get_callee_pdg(sdg: object, callee_name: str) -> object | None:
    """Safely query the SDG for a callee's PDG.

    Returns ``None`` if *sdg* is ``None``, has no ``get_pdg`` method,
    or the callee is not found.
    """
    if sdg is None:
        return None
    get_pdg = getattr(sdg, "get_pdg", None)
    if get_pdg is None:
        return None
    try:
        return get_pdg(callee_name)
    except Exception:
        return None


def _has_loop_carried_init(code: str) -> bool:
    """Check for loop-carried variable initialization pattern.

    Detects: ``for/while(...) { ... type var; ... func(&var, ...) ... }``
    """
    clean = re.sub(r"//.*", "", code, flags=re.MULTILINE)
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)
    clean = re.sub(r"'[^']*'", "", clean)
    clean = re.sub(r'"[^"]*"', "", clean)
    clean = re.sub(r"\bauto\s+", "", clean)

    return bool(
        re.search(
            r"\b(for|while)\s*\(.*\)\s*\{[^}]*?"
            r"\b\w[\w\s*&]*\s+\w+\s*;\s*[^}]*?"
            r"&\w+\s*[,)]",
            clean,
            re.DOTALL,
        )
    )


def _has_union_type_punning(code: str) -> bool:
    """Check for union type-punning via memcpy pattern.

    Detects: ``memcpy(&union_field, src, ...)`` near a union declaration.
    """
    clean = re.sub(r"//.*", "", code, flags=re.MULTILINE)
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)

    has_union = bool(re.search(r"\bunion\s*\{", clean))
    has_memcpy_to_union = bool(
        re.search(r"memcpy\s*\(\s*&([^,]+)", clean)
    )
    return has_union and has_memcpy_to_union
