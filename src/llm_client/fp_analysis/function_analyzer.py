"""On-demand opaque function summarization for context compression.

Replaces the old automatic summary-injection pipeline with a **lazy,
on-demand** approach.  Summaries are only generated when the post-slice
context exceeds thresholds (call depth > 3 or code chars > 12 000).

Usage::

    analyzer = FunctionAnalyzer(llm_provider, cache, project_name="aria2")
    summary = analyzer.summarize_function(
        function_name="wslay_queue_push",
        source_root="project/aria2",
        bug_type="memory_leak",
    )
    if summary:
        prompt_section = summary.format_for_llm()
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from llm_client.base import LLMProvider
from llm_client.fp_analysis.function_summary import (
    CachedFunctionData,
    FunctionSummary,
    ReturnValueConstraint,
    normalize_bug_type,
)
from llm_client.fp_analysis.prompt_templates import (
    FUNCTION_SUMMARY_BEST_GUESS_PROMPT,
    FUNCTION_SUMMARY_PROMPT,
    FUNCTION_SUMMARY_SYSTEM_PROMPT,
)
from llm_client.fp_analysis.summary_cache import SummaryCache

logger = logging.getLogger(__name__)

# Well-known runtime functions that don't need summarization.
_INTERNAL_FUNCTIONS: set[str] = {
    "operator new", "operator new[]",
    "operator delete", "operator delete[]",
    "malloc", "calloc", "realloc", "free",
    "memcpy", "memmove", "memset",
    "strlen", "strcpy", "strncpy", "strcat", "strncat",
    "strcmp", "strncmp", "strdup", "strndup",
    "printf", "fprintf", "snprintf", "sprintf", "puts", "fputs",
    "fopen", "fclose", "fread", "fwrite",
    "exit", "abort", "atexit",
    "memset_s", "memcpy_s", "strcpy_s",
}


class FunctionAnalyzer:
    """On-demand function summarization with caching.

    Only generates summaries when explicitly asked.  The caller
    (``ContextController``) decides whether a function needs summarization
    based on post-slice context thresholds.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        cache: SummaryCache,
        project_name: str = "unknown",
        model: str = "deepseek-chat",
        temperature: float = 0.0,
    ) -> None:
        self._provider = llm_provider
        self._cache = cache
        self._project_name = project_name
        self._model = model
        self._temperature = temperature

    # ------------------------------------------------------------------
    # Public API: summarize one function on demand
    # ------------------------------------------------------------------

    def summarize_function(
        self,
        function_name: str,
        source_root: str,
        bug_type: str = "",
    ) -> FunctionSummary | None:
        """Generate (or load from cache) a compact summary for *function_name*.

        Args:
            function_name: Fully qualified function name.
            source_root: Project source root for body extraction.
            bug_type: Optional CSA bug type for summary focus.

        Returns:
            A ``FunctionSummary``, or ``None`` if generation failed.
        """
        if not self._provider:
            logger.warning("No LLM provider; cannot summarize '%s'", function_name)
            return None

        if function_name in _INTERNAL_FUNCTIONS:
            return None

        canonical_bug_type = normalize_bug_type(bug_type) if bug_type else "general"

        # 1. Check cache first
        cached = self._cache.get_cachedata(function_name, self._project_name)
        if cached and canonical_bug_type in cached.summaries:
            logger.info("Cache hit for '%s' (%s)", function_name, canonical_bug_type)
            return self._cached_to_summary(cached, canonical_bug_type)

        # 2. Try to find body in project source
        body, signature, file_path = self._find_function_body(
            function_name, source_root,
        )

        # 3. Determine if library function (no project source)
        is_library = body is None and not file_path

        # 4. Call LLM for summary
        try:
            summary = self._llm_summarize(
                function_name=function_name,
                signature=signature or function_name,
                body=body or "",
                body_available=body is not None,
                file_path=file_path or "",
                bug_type=canonical_bug_type,
            )
        except (ValueError, RuntimeError) as e:
            logger.warning("Failed to summarize '%s': %s", function_name, e)
            return None

        if summary is None:
            return None

        summary.is_library = is_library

        # 5. Cache the result
        self._cache_result(function_name, summary, canonical_bug_type)

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_function_body(
        self, function_name: str, source_root: str,
    ) -> tuple[str | None, str | None, str | None]:
        """Search project source for *function_name* and return
        ``(body, signature, file_path)``.

        Returns ``(None, None, None)`` if not found (likely a library function).
        """
        src_dir = Path(source_root) / "src"
        if not src_dir.is_dir():
            return None, None, None

        # Derive class name from qualified name (ClassName::method → ClassName)
        parts = function_name.split("::")
        class_name = parts[0] if len(parts) > 1 else None

        # Try to find the source file for this class
        candidates: list[Path] = []
        if class_name:
            for ext in (".cc", ".cpp", ".c"):
                f = src_dir / f"{class_name}{ext}"
                if f.exists():
                    candidates.append(f)

        # Fallback: search all source files
        if not candidates:
            for ext in "*.cc", "*.cpp", "*.c":
                candidates.extend(sorted(src_dir.glob(ext))[:20])

        for src_file in candidates:
            try:
                content = src_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # Try to find the function definition
            # Pattern: "ReturnType ClassName::methodName(...) {" or "ClassName::methodName(...) {"
            # Also handles template: "ClassName<...>::methodName<...>(...) {"
            escaped_fn = re.escape(function_name)
            patterns = [
                re.compile(rf"[\w\*&<>,\s]+\b{re.escape(function_name)}\s*\([^;]+\{{"),
                re.compile(rf"\b{re.escape(function_name)}\s*\([^;]+\{{"),
            ]

            body = None
            sig = None
            for pat in patterns:
                m = pat.search(content)
                if m:
                    # Extract signature (from start of line to opening paren
                    line_start = content.rfind("\n", 0, m.start()) + 1
                    sig_end = content.find("(", m.start())
                    sig = content[m.start():sig_end + 1].strip()
                    if not sig.endswith("("):
                        sig += "("

                    # Extract body using brace matching
                    brace_start = content.find("{", m.start())
                    if brace_start == -1:
                        continue
                    depth = 0
                    end = brace_start
                    for i in range(brace_start, min(brace_start + 10000, len(content))):
                        if content[i] == "{":
                            depth += 1
                        elif content[i] == "}":
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    body = content[line_start:end].strip()
                    break

            if body:
                rel_path = str(src_file.relative_to(source_root))
                return body, sig, rel_path

        return None, None, None

    def _llm_summarize(
        self,
        function_name: str,
        signature: str,
        body: str,
        body_available: bool,
        file_path: str,
        bug_type: str,
    ) -> FunctionSummary | None:
        """Call LLM to produce a function summary."""
        if body_available and body:
            # Body available — ask LLM to analyze it
            prompt = FUNCTION_SUMMARY_PROMPT.format(
                function_name=function_name,
                signature=signature,
                file_path=file_path,
                function_body=body,
                bug_type=bug_type,
            )
            raw = self._call_llm(FUNCTION_SUMMARY_SYSTEM_PROMPT, prompt)

            data = self._parse_json_response(raw, f"function_summary:{function_name}")
            return self._dict_to_summary(
                data, function_name, signature, file_path,
                body_available=True, is_best_guess=False,
            )
        else:
            # No body — best-guess from name/signature
            prompt = FUNCTION_SUMMARY_BEST_GUESS_PROMPT.format(
                function_name=function_name,
                signature=signature,
                file_path=file_path,
                project_name=self._project_name,
                bug_type=bug_type,
            )
            raw = self._call_llm(FUNCTION_SUMMARY_SYSTEM_PROMPT, prompt)

            data = self._parse_json_response(raw, f"function_summary_best_guess:{function_name}")
            return self._dict_to_summary(
                data, function_name, signature, file_path,
                body_available=False, is_best_guess=True,
            )

    @staticmethod
    def _dict_to_summary(
        data: dict,
        function_name: str,
        signature: str,
        file_path: str,
        body_available: bool,
        is_best_guess: bool,
    ) -> FunctionSummary:
        """Convert LLM response dict to a FunctionSummary."""
        rvc = None
        raw_rvc = data.get("return_value_constraints")
        if raw_rvc and isinstance(raw_rvc, dict):
            rvc = ReturnValueConstraint(
                return_value_semantics=raw_rvc.get("return_value_semantics", ""),
                success_conditions=raw_rvc.get("success_conditions", []),
                failure_conditions=raw_rvc.get("failure_conditions", []),
                failure_nature=raw_rvc.get("failure_nature", "deterministic"),
            )

        return FunctionSummary(
            function_name=function_name,
            signature=signature,
            file_path=file_path,
            body_available=body_available,
            is_best_guess=is_best_guess,
            is_library=False,  # caller sets this
            preconditions=data.get("preconditions", []),
            analysis=data.get("analysis", str(data)),
            return_value_constraints=rvc,
        )

    @staticmethod
    def _cached_to_summary(
        cached: CachedFunctionData, bug_type: str,
    ) -> FunctionSummary | None:
        """Convert a cached entry back to FunctionSummary."""
        summary_dict = cached.summaries.get(bug_type)
        if not summary_dict:
            return None

        rvc = None
        raw_rvc = summary_dict.get("return_value_constraints")
        if raw_rvc and isinstance(raw_rvc, dict):
            rvc = ReturnValueConstraint(**raw_rvc)

        return FunctionSummary(
            function_name=cached.function_name,
            signature=cached.signature,
            file_path=cached.file_path,
            body_available=cached.body_available,
            is_best_guess=False,
            is_library=cached.is_library,
            preconditions=summary_dict.get("preconditions", []),
            analysis=summary_dict.get("analysis", ""),
            return_value_constraints=rvc,
        )

    def _cache_result(
        self, function_name: str, summary: FunctionSummary, bug_type: str,
    ) -> None:
        """Store a FunctionSummary into the persistent cache."""
        cached = self._cache.get_cachedata(function_name, self._project_name)

        if cached is None:
            cached = CachedFunctionData(
                function_name=function_name,
                project=self._project_name,
                file_path=summary.file_path,
                signature=summary.signature,
                body_available=summary.body_available,
                body_snippet=summary.body_snippet,
                is_library=summary.is_library,
                summaries={},
            )

        # Store the summary as a dict keyed by bug_type
        cached.summaries[bug_type] = {
            "preconditions": summary.preconditions,
            "analysis": summary.analysis,
            "return_value_constraints": (
                summary.return_value_constraints.model_dump()
                if summary.return_value_constraints else None
            ),
        }

        self._cache.set_cachedata(function_name, self._project_name, cached)

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Make an LLM call and return the raw response text."""
        import time

        from llm_client.base import LLMRequest

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request = LLMRequest(
            messages=messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=2048,
        )

        logger.info("LLM summarize '%s'...", request.messages[-1]["content"][:80])
        start = time.monotonic()
        response = self._provider.send(request)
        elapsed = time.monotonic() - start
        logger.info("LLM summary response received in %.1fs", elapsed)

        return response.content

    @staticmethod
    def _parse_json_response(raw: str, context: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences."""
        text = raw.strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(
                "Failed to parse LLM %s output: %s\nRaw: %s",
                context, e, raw[:500],
            )
            raise RuntimeError(f"LLM returned invalid JSON for {context}: {e}")

        return data
