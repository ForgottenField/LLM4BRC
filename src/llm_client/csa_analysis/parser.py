"""CSA plist output parser — converts raw plist to structured BugPath objects."""

from __future__ import annotations

import os
import plistlib
from pathlib import Path
from typing import Any

from llm_client.csa_analysis.errors import PlistParseError
from llm_client.csa_analysis.models import BugPath, EventKind, PathEvent


class PlistParser:
    """Parse CSA plist output files into structured BugPath objects.

    Args:
        source_root: Root directory for resolving relative file paths.
                     If empty, uses the directory of the plist file.
    """

    def __init__(self, source_root: str = "") -> None:
        self._source_root = source_root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, plist_path: str) -> list[BugPath]:
        """Parse a single CSA plist file into structured bug paths.

        Args:
            plist_path: Path to the CSA plist output file.

        Returns:
            List of BugPath objects. Empty list if plist has no diagnostics.

        Raises:
            PlistParseError: If plist is malformed or missing required fields.
            FileNotFoundError: If *plist_path* does not exist.
        """
        plist_path = Path(plist_path).resolve()
        if not plist_path.exists():
            raise FileNotFoundError(f"Plist file not found: {plist_path}")

        try:
            with open(plist_path, "rb") as f:
                data = plistlib.load(f)
        except Exception as exc:
            raise PlistParseError(f"Failed to parse plist: {exc}")

        return self._parse_dict(data, plist_path)

    def parse_multi(self, plist_paths: list[str]) -> list[BugPath]:
        """Parse multiple CSA plist files into a unified list of bug paths.

        Args:
            plist_paths: List of paths to CSA plist files.

        Returns:
            List of BugPath objects from all plist files combined.
        """
        all_paths: list[BugPath] = []
        for path in plist_paths:
            all_paths.extend(self.parse(path))
        return all_paths

    # ------------------------------------------------------------------
    # Internal parsing logic
    # ------------------------------------------------------------------

    def _parse_dict(self, data: dict[str, Any], plist_path: Path) -> list[BugPath]:
        """Parse the top-level plist dictionary."""
        files = data.get("files")
        if files is None:
            raise PlistParseError("Plist missing 'files' array")

        diagnostics = data.get("diagnostics", [])
        if not diagnostics:
            return []

        source_root = self._resolve_source_root(plist_path)
        bug_paths: list[BugPath] = []

        for diag in diagnostics:
            try:
                bug_path = self._parse_diagnostic(diag, files, source_root)
                bug_paths.append(bug_path)
            except (KeyError, ValueError, IndexError) as exc:
                raise PlistParseError(
                    f"Failed to parse diagnostic in {plist_path}: {exc}"
                )

        return bug_paths

    def _parse_diagnostic(
        self,
        diag: dict[str, Any],
        files: list[Any],
        source_root: str,
    ) -> BugPath:
        """Parse a single diagnostic entry into a BugPath."""
        path_events_raw = diag.get("path", [])
        events = [
            self._parse_path_event(evt, files, source_root)
            for evt in path_events_raw
        ]

        # Ensure last event is marked as 'report'
        if events:
            events[-1].event_kind = EventKind.REPORT

        return BugPath(
            path_id=self._make_path_id(diag),
            bug_type=diag.get("type", "Unknown"),
            category=diag.get("category", "Unknown"),
            check_name=diag.get("check_name"),
            issue_context=diag.get("issue_context"),
            events=events,
            issue_hash=diag.get("issue_hash_content_of_line_in_context"),
        )

    def _parse_path_event(
        self,
        evt: dict[str, Any],
        files: list[Any],
        source_root: str,
    ) -> PathEvent:
        """Parse a single path event dictionary into a PathEvent."""
        kind_str = evt.get("kind", "event")
        kind: EventKind
        if kind_str == "control":
            kind = EventKind.CONTROL
        elif kind_str in ("event", "report"):
            kind = EventKind.EVENT
        else:
            kind = EventKind.EVENT

        location = evt.get("location", {})
        file_index = location.get("file", 0)
        file_path = self._resolve_file_path(files, file_index, source_root)

        description = evt.get("message", evt.get("extended_message", ""))

        code_snippet = self._extract_code_snippet(file_path, location.get("line"))

        branch_condition = None
        if kind == EventKind.CONTROL:
            msg_lower = description.lower()
            if "true" in msg_lower:
                branch_condition = True
            elif "false" in msg_lower:
                branch_condition = False

        line_val = location.get("line")
        if line_val is None:
            line_val = 0

        col_val = location.get("col")

        return PathEvent(
            event_kind=kind,
            file_path=file_path,
            line_number=line_val,
            column_number=col_val,
            description=description,
            code_snippet=code_snippet,
            branch_condition=branch_condition,
            depth=evt.get("depth"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_source_root(self, plist_path: Path) -> str:
        if self._source_root:
            return self._source_root
        # Files in plist are relative to the CWD when CSA was invoked
        return os.getcwd()

    def _resolve_file_path(
        self,
        files: list[Any],
        file_index: int,
        source_root: str,
    ) -> str:
        if file_index < 0 or file_index >= len(files):
            raise IndexError(
                f"File index {file_index} out of range (0-{len(files) - 1})"
            )
        rel_path = files[file_index]
        if os.path.isabs(rel_path):
            return os.path.normpath(str(rel_path))
        return os.path.normpath(os.path.join(source_root, str(rel_path)))

    @staticmethod
    def _extract_code_snippet(file_path: str, line_number: int | None) -> str | None:
        """Read the source file and extract the given line."""
        if line_number is None:
            return None
        try:
            with open(file_path) as f:
                for i, line in enumerate(f, start=1):
                    if i == line_number:
                        return line.rstrip("\n").rstrip("\r")
            return None
        except (FileNotFoundError, OSError):
            return None

    @staticmethod
    def _make_path_id(diag: dict[str, Any]) -> str:
        """Generate a unique path ID from the diagnostic."""
        bug_type = diag.get("type", "bug")
        issue_hash = diag.get("issue_hash_content_of_line_in_context", "unknown")
        check_name = diag.get("check_name", "")
        return f"{check_name}.{issue_hash}" if check_name else f"{bug_type}.{issue_hash}"
