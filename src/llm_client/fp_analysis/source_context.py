"""Source code context extraction for FP path completion.

Given a parsed FP report, locate the relevant source files in the project
and extract code snippets around key locations for LLM analysis.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from llm_client.fp_analysis.models import ParsedReport

logger = logging.getLogger(__name__)

# Known path prefixes in CSA HTML reports — used to resolve build paths to local
_KNOWN_PATH_PREFIXES = [
    # folly project
    "/tmp/fbcode_builder_getdeps-ZhomeZmeZprojectsZcpp-projectZfollyZbuildZfbcode_builder/",
    "/home/me/projects/cpp-project/",
    # aria2 project
    "/home/zhh/workplace/ehtest/aria2-1.35.0/",
]


class SourceContext:
    """Relevant source code context extracted for a bug report."""

    def __init__(
        self,
        primary_file: str,
        primary_snippets: dict[int, str],
        additional_files: dict[str, dict[int, str]],
    ) -> None:
        self.primary_file = primary_file
        self.primary_snippets = primary_snippets
        self.additional_files = additional_files

    def format_for_llm(self, max_lines: int = 200) -> str:
        """Format source context for inclusion in an LLM prompt."""
        parts: list[str] = []

        if self.primary_snippets:
            parts.append(f"--- File: {self.primary_file} ---")
            lines = sorted(self.primary_snippets.keys())
            for ln in lines[:max_lines]:
                parts.append(f"{ln:>6}: {self.primary_snippets[ln]}")

        for file_path, snippets in self.additional_files.items():
            if not snippets:
                continue
            parts.append(f"\n--- File: {file_path} ---")
            lines = sorted(snippets.keys())
            for ln in lines[:max_lines]:
                parts.append(f"{ln:>6}: {snippets[ln]}")

        return "\n".join(parts)

    def has_content(self) -> bool:
        return bool(self.primary_snippets) or any(self.additional_files.values())


class SourceContextExtractor:
    """Extract source code context from a project for a parsed FP report."""

    # Companion file patterns: given a .cpp file, try these alternatives
    _COMPANION_PATTERNS = [
        "-inl.h",   # Format.cpp → Format-inl.h (folly pattern)
        ".h",       # Format.cpp → Format.h
        ".cc",      # array_fun.h → array_fun.cc (primary header → source)
        ".cpp",     # fallback for .cpp projects
    ]

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root).resolve() if project_root else None

    def extract(self, parsed: ParsedReport) -> SourceContext:
        """Extract source context for a parsed report.

        Finds relevant source files and extracts code around the bug line,
        key path event locations, and companion header files.
        """
        primary_file, primary_snippets = self._extract_primary_source(parsed)
        additional_files: dict[str, dict[int, str]] = {}

        # Also extract companion file snippets for key event locations
        # that may live in header files (e.g., Format-inl.h)
        if primary_file:
            primary_path = Path(primary_file)
            companion_snippets = self._extract_companion_context(
                primary_path, parsed
            )
            additional_files.update(companion_snippets)

            # Extract code around key path event locations (Error Start, control events)
            # that fall in the primary file but outside the bug-line-centered window.
            # Without this, the LLM may miss critical context such as:
            # - The constructor code where `bitfield_(nullptr)` originates
            # - Branch conditions like `if (blockLength_ > 0)` that CSA explicitly assumed
            event_snippets = self._extract_event_context(
                primary_path, parsed, primary_snippets
            )
            additional_files.update(event_snippets)

        return SourceContext(
            primary_file=primary_file or parsed.metadata.bug_file,
            primary_snippets=primary_snippets,
            additional_files=additional_files,
        )

    def _extract_companion_context(
        self, primary_path: Path, parsed: ParsedReport
    ) -> dict[str, dict[int, str]]:
        """Extract code from companion files (e.g., -inl.h) around key event lines.

        When a path event has a line number that exceeds the primary file's
        line count, it likely lives in a companion header file. This method
        finds those files and extracts code around those line numbers.
        """
        result: dict[str, dict[int, str]] = {}
        primary_line_count = self._file_line_count(primary_path)
        if primary_line_count is None:
            return result

        # Collect unique (file_stem, line_number) pairs from events
        needed: set[tuple[str, int]] = set()
        for evt in parsed.path_events:
            line = evt.get("line_number", 0)
            if line and line > primary_line_count:
                needed.add((primary_path.stem, line))

        # Also check error_start_event and calling_events
        if parsed.error_start_event:
            line = parsed.error_start_event.get("line_number", 0)
            if line and line > primary_line_count:
                needed.add((primary_path.stem, line))

        for evt in getattr(parsed, "calling_events", []) or []:
            line = evt.get("line_number", 0)
            if line and line > primary_line_count:
                needed.add((primary_path.stem, line))

        if not needed:
            return result

        # Try companion file patterns for each needed (stem, line) pair
        parent = primary_path.parent
        for stem, line in sorted(needed):
            for suffix_pattern in self._COMPANION_PATTERNS:
                companion_name = stem + suffix_pattern
                companion_path = parent / companion_name
                if not companion_path.exists():
                    continue
                c_line_count = self._file_line_count(companion_path)
                if c_line_count and 1 <= line <= c_line_count:
                    snippet = self._read_code_around(companion_path, line, radius=15)
                    if snippet:
                        key = f"{companion_name} (near line {line})"
                        # Merge rather than overwrite if multiple lines
                        if key in result:
                            result[key].update(snippet)
                        else:
                            result[key] = snippet

        return result

    def _extract_event_context(
        self, primary_path: Path, parsed: ParsedReport, existing: dict[int, str]
    ) -> dict[str, dict[int, str]]:
        """Extract code around path event locations not covered by the primary snippet.

        The primary snippet is centered on the bug line; other event locations
        (Error Start, control flow assumptions) may be far away in the **same or
        a different file**. Without these, the LLM may miss context critical for
        constraint reasoning, such as branch conditions that CSA explicitly assumed.

        Each event is read from its own source file (``file_path`` in the event
        dict), falling back to *primary_path* when the file is unavailable.
        """
        result: dict[str, dict[int, str]] = {}

        # Determine the line range already covered by the primary snippet
        if existing:
            covered_min = min(existing.keys())
            covered_max = max(existing.keys())
        else:
            covered_min = covered_max = 0

        # Group events by file
        events_by_file: dict[str, set[int]] = {}
        for evt in parsed.path_events:
            line = evt.get("line_number", 0)
            if not line:
                continue
            # Use event's own file_path if available, otherwise primary path
            evt_file = evt.get("file_path", "")
            if evt_file:
                local = self._resolve_to_local(evt_file)
                file_key = str(local) if local and local.exists() else str(primary_path)
            else:
                file_key = str(primary_path)

            # Only include if outside the primary snippet's covered range
            # (the primary snippet is in its own file; use line-insensitive check
            #  for other files: always include)
            if file_key == str(primary_path) and (covered_min <= line <= covered_max):
                continue

            events_by_file.setdefault(file_key, set()).add(line)

        if parsed.error_start_event:
            line = parsed.error_start_event.get("line_number", 0)
            if line:
                evt_file = parsed.error_start_event.get("file_path", "")
                if evt_file:
                    local = self._resolve_to_local(evt_file)
                    file_key = str(local) if local and local.exists() else str(primary_path)
                else:
                    file_key = str(primary_path)
                events_by_file.setdefault(file_key, set()).add(line)

        if not events_by_file:
            return result

        # Read a small window around each event location from its own file
        for file_key, lines_set in events_by_file.items():
            file_path = Path(file_key)
            label = f"{file_path.name} (path event locations)"
            for line in sorted(lines_set):
                snippet = self._read_code_around(file_path, line, radius=15)
                if snippet:
                    if label in result:
                        result[label].update(snippet)
                    else:
                        result[label] = snippet

        return result

    @staticmethod
    def _file_line_count(file_path: Path) -> int | None:
        """Return the number of lines in a file, or None on error."""
        try:
            return len(file_path.read_text(encoding="utf-8", errors="replace").splitlines())
        except Exception:
            return None

    def _extract_primary_source(
        self, parsed: ParsedReport
    ) -> tuple[Optional[str], dict[int, str]]:
        """Try to find and read the primary source file where the bug is.

        Returns (file_path, {line_number: code}).
        """
        # Strategy 1: Try the bug_file from metadata
        bug_file = parsed.metadata.bug_file
        local_path = self._resolve_to_local(bug_file)
        if local_path and local_path.exists():
            return str(local_path), self._read_code_around(
                local_path, parsed.metadata.bug_line
            )

        # Strategy 2: Extract main source file from clang command in HTML
        main_file = self._extract_main_file_from_html(parsed.raw_html_path)
        if main_file:
            local_path = self._resolve_to_local(main_file)
            if local_path and local_path.exists():
                return str(local_path), self._read_code_around(
                    local_path, parsed.metadata.bug_line
                )

        # Strategy 3: Search by filename in the project
        filename = Path(bug_file).name
        if self.project_root:
            for found in self.project_root.rglob(filename):
                return str(found), self._read_code_around(
                    found, parsed.metadata.bug_line
                )

        logger.warning("Could not locate source file for: %s", bug_file)
        if parsed.source_snippets:
            return bug_file, parsed.source_snippets

        return None, {}

    def _resolve_to_local(self, file_path: str) -> Path | None:
        """Resolve a CSA build path to a local filesystem path.

        Tries (in order):
        1. Direct path existence (build environment path still valid).
        2. Strip known build prefixes + join with project_root.
        3. Search by basename under project_root.
        4. Search by basename under local ``project/`` directory (cross-env).
        """
        path = Path(file_path)
        if path.exists():
            return path.resolve()

        if self.project_root:
            # Try stripping known prefixes
            for prefix in _KNOWN_PATH_PREFIXES:
                if file_path.startswith(prefix):
                    rel = file_path.removeprefix(prefix).lstrip("/")
                    candidate = self.project_root / rel
                    if candidate.exists():
                        return candidate.resolve()

            # Search by basename in project_root
            basename = path.name
            for found in self.project_root.rglob(basename):
                return found.resolve()

        # Fallback: search by basename in local project/ directory.
        # This is the primary resolution mechanism when the CSA build path
        # does not exist locally (the normal case for cross-env analysis).
        local_project = Path("project").resolve()
        if local_project.exists():
            basename = path.name
            for found in local_project.rglob(basename):
                return found.resolve()

        return None

    @staticmethod
    def _extract_main_file_from_html(html_path: str | Path) -> Optional[str]:
        """Extract the main analyzed source file from the clang command line."""
        try:
            html = Path(html_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        # Look in the spoiler div for the clang command
        spoiler_match = re.search(
            r'<div\s+class="spoiler"[^>]*>(.*?)</div>', html, re.DOTALL
        )
        if not spoiler_match:
            return None

        cmd = spoiler_match.group(1)
        cmd = re.sub(r'<[^>]+>', ' ', cmd)

        # Extract -main-file-name or the last positional argument (the source file)
        mfn_match = re.search(r'-main-file-name\s+(\S+)', cmd)
        if mfn_match:
            return mfn_match.group(1)

        # Look for the .cpp argument at the end (after -x c++ ... )
        # The source file is typically the last argument before -o
        parts = cmd.split()
        for i, p in enumerate(parts):
            if p.endswith((".cpp", ".cc", ".c", ".cxx")) and (
                i == len(parts) - 1 or parts[i + 1] != "-o"
            ):
                return p

        return None

    def extract_call_chain(self, parsed: ParsedReport) -> dict[str, dict[int, str]]:
        """Extract source code around each calling event's call site.

        For every 'Calling' event in the path, resolves the source file and
        reads a small window of code around the call site line number.
        Falls back to companion files (e.g., -inl.h) when the line exceeds
        the primary file's range.

        Returns a dict keyed by ``call_site_L{line}_{function}`` → {line: code}.
        """
        additional: dict[str, dict[int, str]] = {}

        for evt in parsed.path_events:
            if not evt.get("is_calling_event"):
                continue
            line = evt.get("line_number", 0)
            if not line:
                continue

            func_name = evt.get("called_function", "unknown")
            file_path = evt.get("file_path", "") or parsed.metadata.bug_file

            local_path = self._resolve_to_local(file_path)
            if local_path and local_path.exists():
                snippet = self._read_code_around(local_path, line, radius=10)
                if snippet:
                    key = f"call_site_L{line}_{func_name}"
                    additional[key] = snippet
                else:
                    # Try companion files (line may be in -inl.h, .h, etc.)
                    comp_snippet = self._read_from_companion(local_path, line, radius=10)
                    if comp_snippet:
                        key = f"call_site_L{line}_{func_name}"
                        additional[key] = comp_snippet

        return additional

    def _read_from_companion(
        self, primary_path: Path, line: int, radius: int = 10
    ) -> dict[int, str]:
        """Try to read code around *line* from companion files of *primary_path*."""
        parent = primary_path.parent
        stem = primary_path.stem
        for suffix_pattern in self._COMPANION_PATTERNS:
            companion_name = stem + suffix_pattern
            companion_path = parent / companion_name
            if not companion_path.exists():
                continue
            c_count = self._file_line_count(companion_path)
            if c_count and 1 <= line <= c_count:
                snippet = self._read_code_around(companion_path, line, radius=radius)
                if snippet:
                    return snippet
        return {}

    # ------------------------------------------------------------------
    # Function body extraction
    # ------------------------------------------------------------------

    def find_function_definition(
        self,
        function_name: str,
        search_roots: list[Path] | None = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Search project source for a C/C++ function definition.

        Searches ``.c``, ``.cc``, ``.cpp``, ``.cxx``, and ``.h`` files under
        *search_roots* (defaults to *project_root*). Returns
        ``(file_path, body_snippet)`` or ``(None, None)``.

        If nothing is found in the primary roots, falls back to the local
        ``project/`` directory under the current working directory. This
        handles cases where the CSA report's build path (``project_root``)
        doesn't exist on the local machine, but project sources are available
        locally (e.g., under ``project/aria2/deps/``).
        """
        roots = search_roots or ([self.project_root] if self.project_root else [])
        extensions = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")

        for root in roots:
            if not root or not root.exists():
                continue
            for ext in extensions:
                for fpath in root.rglob(f"*{ext}"):
                    if not fpath.is_file():
                        continue
                    body = self.extract_function_body(fpath, function_name)
                    if body:
                        return str(fpath.resolve()), body

        # Fallback: try local project/ directory when primary roots fail
        local_project = Path("project").resolve()
        if local_project.exists() and local_project not in roots:
            logger.debug(
                "Function %s not found in primary roots; falling back to %s",
                function_name, local_project,
            )
            for ext in extensions:
                for fpath in local_project.rglob(f"*{ext}"):
                    if not fpath.is_file():
                        continue
                    body = self.extract_function_body(fpath, function_name)
                    if body:
                        return str(fpath.resolve()), body

        return None, None

    def extract_function_body(
        self, file_path: str | Path, function_name: str
    ) -> str | None:
        """Extract a C/C++ function body from *file_path* using brace matching.

        Returns the full function text from the signature line to the closing
        ``}`` (inclusive), or ``None`` if the function is not found or the
        file cannot be read.
        """
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

        lines = text.splitlines()
        if not lines:
            return None

        fn_pattern = re.compile(rf"\b{re.escape(function_name)}\b")
        decl_keywords = re.compile(
            r"\b(typedef|enum|struct|class|union)\b"
        )

        # Call-site indicators: tokens that, when immediately before the
        # function name, mean this is a call, not a definition.
        call_site_prefix = re.compile(
            rf"[\=\.,!&\(\)\-\+\*/%<>\[\]\|;:?~]"
            rf"|->\s*$"
            rf"|\.\s*$"
        )

        start_line_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith(("//", "/*", "*", "#")):
                continue
            if not fn_pattern.search(stripped):
                continue
            # Must contain '(' — either on this line or the next (for
            # multi-line signatures like ``static int func\n(params)``).
            has_paren = "(" in stripped
            if not has_paren:
                # Check next line for '(' (peek ahead)
                if i + 1 < len(lines):
                    next_stripped = lines[i + 1].strip()
                    has_paren = next_stripped.startswith("(") or "(" in next_stripped
            if not has_paren:
                continue
            # Only reject if decl keywords appear in the prefix (before the
            # function name), e.g., "typedef struct foo func(...)".
            # Don't reject if they appear in the parameter list.
            idx_before_paren = stripped.find("(")
            decl_context = stripped[:idx_before_paren] if idx_before_paren > 0 else stripped
            if decl_keywords.search(decl_context):
                continue
            if stripped.rstrip().endswith(";"):
                continue

            # --- Definition-vs-call-site heuristic ---
            # A definition has the function name at a "declarator position":
            # preceded by whitespace/type tokens, not by expression operators.
            # Find where function_name appears in this line.
            line_before_func = ""
            for m in fn_pattern.finditer(stripped):
                start = m.start()
                if start > 0:
                    line_before_func = stripped[start - 1]
                else:
                    line_before_func = " "  # start of line = definition-like
                break

            # If the preceding char is an operator character, it's a call site.
            if call_site_prefix.match(line_before_func):
                continue
            # Additional check: if there's an assignment before the function name
            if "=" in stripped.split(function_name)[0]:
                continue

            start_line_idx = i
            break

        if start_line_idx is None:
            return None

        # Extend start backward to capture return type on preceding line(s)
        # if the signature is split across lines (max 5 lines back).
        # Stop at empty lines, comment lines, or lines ending with '}'
        # (previous function end).
        _backtrack_limit = 5
        _backtrack_count = 0
        while (
            start_line_idx > 0
            and _backtrack_count < _backtrack_limit
        ):
            prev = lines[start_line_idx - 1].strip()
            if not prev:
                break
            if prev.startswith(("#", "//", "/*")):
                break
            if prev.startswith("}"):
                break
            start_line_idx -= 1
            _backtrack_count += 1

        # Now find the opening '{' — it may be on the same line as the
        # signature, or on a subsequent line.
        brace_start = self._find_opening_brace(lines, start_line_idx)
        if brace_start is None:
            return None

        # Brace-match: track depth from the opening '{'
        # Skip strings, character literals, and line/block comments.
        depth = 0
        in_block_comment = False
        brace_end_idx = brace_start

        for idx in range(brace_start, len(lines)):
            line = lines[idx]
            # Handle block comments that span multiple lines
            if in_block_comment:
                end_comment = line.find("*/")
                if end_comment != -1:
                    in_block_comment = False
                    line = line[end_comment + 2:]
                else:
                    continue

            # Process the line character by character
            i = 0
            while i < len(line):
                ch = line[i]

                # Line comment — skip rest of line
                if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                    break

                # Block comment start
                if ch == "/" and i + 1 < len(line) and line[i + 1] == "*":
                    i += 2
                    end_idx = line.find("*/", i)
                    if end_idx != -1:
                        i = end_idx + 2
                        continue
                    else:
                        in_block_comment = True
                        break

                # String literal (skip escaped chars)
                if ch in ('"', "'"):
                    quote = ch
                    i += 1
                    while i < len(line):
                        if line[i] == "\\":
                            i += 2
                            continue
                        if line[i] == quote:
                            break
                        i += 1

                # Brace tracking
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        brace_end_idx = idx
                        # Join from start_line_idx to brace_end_idx inclusive
                        return "\n".join(
                            lines[start_line_idx:brace_end_idx + 1]
                        )
                i += 1

        return None

    @staticmethod
    def _find_opening_brace(
        lines: list[str], start_idx: int
    ) -> int | None:
        """Find the line index containing the opening ``{`` for a function.

        Scans forward from *start_idx*, accounting for block comments and
        strings. Returns the line index, or None.
        """
        in_block_comment = False
        for idx in range(start_idx, len(lines)):
            line = lines[idx]
            if in_block_comment:
                end_idx = line.find("*/")
                if end_idx != -1:
                    in_block_comment = False
                    line = line[end_idx + 2:]
                else:
                    continue

            i = 0
            while i < len(line):
                ch = line[i]
                if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                    break
                if ch == "/" and i + 1 < len(line) and line[i + 1] == "*":
                    i += 2
                    end_idx = line.find("*/", i)
                    if end_idx != -1:
                        i = end_idx + 2
                        continue
                    in_block_comment = True
                    break
                if ch in ('"', "'"):
                    quote = ch
                    i += 1
                    while i < len(line):
                        if line[i] == "\\":
                            i += 2
                            continue
                        if line[i] == quote:
                            break
                        i += 1
                if ch == "{":
                    return idx
                i += 1
        return None

    @staticmethod
    def _read_code_around(
        file_path: Path, center_line: int, radius: int = 30
    ) -> dict[int, str]:
        """Read source code lines around a center line."""
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return {}

        start = max(0, center_line - radius - 1)
        end = min(len(lines), center_line + radius)
        return {i + 1: lines[i] for i in range(start, end)}
