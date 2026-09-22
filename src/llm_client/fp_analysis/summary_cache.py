"""Persistent JSON-based summary cache for opaque function analysis.

Cache directory layout::

    summary_cache/
    └── <project_name>/
        └── <function_name>.json   # All bug-type summaries for this function

Each JSON file stores a ``CachedFunctionData`` with summaries keyed by bug_type.
Lookup key = (project_name, function_name, bug_type).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from llm_client.fp_analysis.function_summary import (
    CachedFunctionData,
    FunctionSummary,
)

logger = logging.getLogger(__name__)


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a filename, truncating if too long.

    C++ template instantiations can produce extremely long function names
    that exceed filesystem filename length limits (~255 bytes on Linux).
    If the sanitized name exceeds *max_len*, truncate and append a hash
    suffix to keep names unique.
    """
    sanitized = re.sub(r"[^\w\-.]", "_", name)
    if len(sanitized) <= 200:
        return sanitized
    # Hash the full name for uniqueness, truncate the rest
    h = hashlib.sha256(sanitized.encode()).hexdigest()[:12]
    return sanitized[:200] + "_" + h


def _fix_llm_json_types(data: dict) -> dict:
    """Fix LLM output where list/dict fields are string-encoded.

    LLMs sometimes return Python string representations of lists/dicts
    instead of actual JSON values. This function detects and converts them.
    """
    if not data:
        return data
    for key, value in list(data.items()):
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                data[key] = json.loads(stripped)
                continue
            except (json.JSONDecodeError, TypeError):
                pass
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data[key] = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                pass
    return data


class SummaryCache:
    """Persistent JSON cache for function summaries.

    Thread-safe for single-process use. Writes use atomic tempfile+rename.
    """

    def __init__(self, cache_root: str | Path = "summary_cache") -> None:
        self._cache_root = Path(cache_root).resolve()

    # ------------------------------------------------------------------
    # Public API (new: CachedFunctionData-based)
    # ------------------------------------------------------------------

    def get_cachedata(
        self, function_name: str, project: str,
    ) -> CachedFunctionData | None:
        """Load a ``CachedFunctionData`` for *function_name* in *project*.

        Returns the full cached entry (with all bug-type summaries), or None.
        This is the primary lookup method for the on-demand FunctionAnalyzer.
        """
        return self._load(project, function_name)

    def set_cachedata(
        self, function_name: str, project: str, data: CachedFunctionData,
    ) -> None:
        """Store a ``CachedFunctionData`` for *function_name* in *project*."""
        self._save(project, function_name, data)

    # ------------------------------------------------------------------
    # Public API (legacy: bug-type-specific)
    # ------------------------------------------------------------------

    def get(
        self, project: str, function_name: str, bug_type: str
    ) -> FunctionSummary | None:
        """Look up a cached summary by (project, function, bug_type).

        Returns a deserialized FunctionSummary subclass instance, or None.
        """
        cached = self._load(project, function_name)
        if cached is None:
            return None

        # Find the summary dict for this bug type
        summary_dict = cached.summaries.get(bug_type)
        if summary_dict is None:
            return None

        # Deserialize into FunctionSummary
        summary_dict = _fix_llm_json_types(summary_dict)
        try:
            summary = FunctionSummary(**summary_dict)
            logger.debug(
                "Cache HIT for %s/%s (bug_type=%s)",
                project, function_name, bug_type,
            )
            return summary
        except Exception as e:
            logger.warning(
                "Failed to deserialize cached summary for %s/%s: %s",
                project, function_name, e,
            )
            return None

    def set(
        self,
        project: str,
        function_name: str,
        bug_type: str,
        summary: FunctionSummary,
    ) -> None:
        """Store or update a summary, merging with existing entries.

        If a cache file already exists for this function, the new summary
        is merged (keyed by bug_type) without affecting summaries for other
        bug types.
        """
        # Load existing or create new
        cached = self._load(project, function_name)
        if cached is None:
            cached = CachedFunctionData(
                function_name=function_name,
                project=project,
                file_path=summary.file_path,
                signature=summary.signature,
                body_available=summary.body_available,
                body_snippet=summary.body_snippet,
            )
        else:
            # Update metadata (always keep latest info)
            cached.file_path = summary.file_path or cached.file_path
            cached.signature = summary.signature or cached.signature
            if summary.body_available:
                cached.body_available = True
                cached.body_snippet = summary.body_snippet or cached.body_snippet
            cached.updated_at = __import__(
                "datetime"
            ).datetime.now(__import__("datetime").timezone.utc).isoformat()

        # Merge: set the bug_type-specific summary dict
        cached.summaries[bug_type] = summary.model_dump()

        self._save(project, function_name, cached)
        logger.info(
            "Cache SET %s/%s (bug_type=%s)",
            project, function_name, bug_type,
        )

    def has(self, project: str, function_name: str, bug_type: str) -> bool:
        """Quick existence check without full deserialization."""
        cached = self._load(project, function_name)
        if cached is None:
            return False
        return bug_type in cached.summaries

    def get_or_generate(
        self,
        project: str,
        function_name: str,
        bug_type: str,
        generate_fn,
    ) -> FunctionSummary:
        """Cache-aware lookup with fallback generation.

        Args:
            project: Project name.
            function_name: Function name.
            bug_type: Normalized bug type string.
            generate_fn: Callable[[], FunctionSummary] — called on cache miss.

        Returns:
            A FunctionSummary instance (from cache or generated).
        """
        cached = self.get(project, function_name, bug_type)
        if cached is not None:
            return cached

        logger.info(
            "Cache miss for %s/%s (bug_type=%s); generating...",
            project, function_name, bug_type,
        )
        summary = generate_fn()
        self.set(project, function_name, bug_type, summary)
        return summary

    def clear(
        self,
        project: str | None = None,
        function_name: str | None = None,
    ) -> None:
        """Clear cache entries.

        Args:
            project: If set, clear only this project's cache. If None, clear all.
            function_name: If set (and project is set), clear only this function.
                           If None (and project is set), clear the entire project.
        """
        if project is None:
            # Clear everything
            for p_dir in self._cache_root.iterdir():
                if p_dir.is_dir():
                    for f in p_dir.iterdir():
                        f.unlink()
                    p_dir.rmdir()
            logger.info("Cache cleared entirely")
            return

        proj_dir = self._cache_root / _sanitize_name(project)
        if not proj_dir.exists():
            return

        if function_name is None:
            # Clear whole project
            for f in proj_dir.iterdir():
                f.unlink()
            proj_dir.rmdir()
            logger.info("Cache cleared for project: %s", project)
            return

        # Clear specific function
        cache_path = proj_dir / f"{_sanitize_name(function_name)}.json"
        if cache_path.exists():
            cache_path.unlink()
            logger.info(
                "Cache cleared for %s/%s", project, function_name
            )

    def get_cache_path(self, project: str, function_name: str) -> Path:
        """Compute the cache file path without checking existence."""
        return (
            self._cache_root
            / _sanitize_name(project)
            / f"{_sanitize_name(function_name)}.json"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, project: str, function_name: str) -> CachedFunctionData | None:
        """Load cached data from disk, or None if not found."""
        cache_path = self.get_cache_path(project, function_name)
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return CachedFunctionData(**data)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to load cache %s: %s", cache_path, e)
            return None

    def _save(
        self, project: str, function_name: str, cached: CachedFunctionData
    ) -> None:
        """Atomically save cached data to disk."""
        cache_path = self.get_cache_path(project, function_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp, then rename
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json", dir=cache_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(cached.model_dump_json(indent=2))
            os.replace(tmp_path, cache_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
