"""Source code context extraction for FP path completion.

Given a parsed FP report, locate the relevant source files in the project
and extract code snippets around key locations for LLM analysis.

**The report is the index, the code model is the reader.**  Everything this
module does is "which file does the report mean, and what does it say"; the
first half is a resolution question and the second a read, and both belong to
the report's :class:`~report_frontend.code_model.SourceCodeModel` —
the single ladder and the single reader the rest of the pipeline converged on.
The extractor keeps its own heuristics for what is *inside* a file (the
definition-vs-call-site scanner of :meth:`SourceContextExtractor.extract_function_body`
is the body parser, and stays), but not for which file to open.

Two consequences worth stating, because both are deliberate and both change
the text handed to a model:

* **Only strong resolutions are accepted.**  ``basename``-only placement is a
  different claim from the one the report makes — ``/usr/include/fmt/core.h``
  is not ``<root>/lib/cpluff/.../core.h`` — and a wrong file is worse than no
  file, so a weak rung reads as "not resolved" and the caller falls back to
  context the report itself carries (its embedded copy).
* **The read is the model's, even for a path it did not resolve.**  A window
  read of an already-chosen file goes through ``model.lines`` regardless, so a
  file is read once, through one cache, and a read the pre-load missed is
  visible as a *lazy read* rather than as an extra loader.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from fp_analysis.models import ParsedReport
from report_frontend.code_model import (
    KNOWN_BUILD_PREFIXES,
    SourceCodeModel,
)

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

    #: Where a cross-environment run keeps the checkout the report's build paths
    #: do not match.  It is a *source root*, handed to the model like any other
    #: — not a search of its own.
    _LOCAL_PROJECT_DIR = "project"

    def __init__(
        self,
        project_root: str | Path | None = None,
        model: SourceCodeModel | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else None
        #: The reader.  ``extract``/``extract_call_chain``/
        #: ``extract_callee_sources`` re-bind it to the *report's* own model,
        #: which is the one carrying the revision the report embedded; the
        #: model built here covers an extractor used without a report (a tool,
        #: or a test that hands it a path).
        self.model = model
        if self.model is None:
            self.model = SourceCodeModel(
                source_root=self._root_for_resolution(),
                prefixes=self._prefixes(),
            )

    def _root_for_resolution(self) -> Path | None:
        """The tree to resolve against: the project, else the local checkout."""
        if self.project_root is not None:
            return self.project_root
        local = Path(self._LOCAL_PROJECT_DIR).resolve()
        return local if local.is_dir() else None

    @staticmethod
    def _prefixes() -> tuple[str, ...]:
        """Build roots to recognise: the model's set plus this module's.

        The model owns the ladder; this is the input that keeps it identical to
        the one this module used to walk by hand (the aria2 build machine is in
        this list and not in the model's).
        """
        return tuple(dict.fromkeys(
            tuple(KNOWN_BUILD_PREFIXES) + tuple(_KNOWN_PATH_PREFIXES)
        ))

    def _bind(self, parsed: ParsedReport) -> SourceCodeModel:
        """Adopt the model the report itself was loaded with.

        The pipeline builds one model per report (``LoadedReport.code``) and
        that model holds the report's embedded source, so it is the right
        reader for every question below.  A report assembled by hand carries
        none, and then the extractor keeps whatever it was constructed with.

        A bound model with **no source root** cannot answer a locating question
        — the ladder has no tree to climb and there is no tree index — so where
        the extractor knows a root (``project_root``, or the ``project/``
        checkout of a cross-environment run) it is analysed against that.  The
        report's own copy is still what is read in that case: with nothing
        resolvable, ``_extract_primary_source`` falls through to
        ``parsed.source_snippets``, which the bridge serves from the model.
        """
        model = getattr(parsed, "code_model", None)
        if model is not None:
            self.model = model
        if self.model.source_root is None:
            root = self._root_for_resolution()
            if root is not None:
                self.model = SourceCodeModel(
                    source_root=root, prefixes=self._prefixes(),
                )
        return self.model

    def _lines_of(self, local: Path) -> list[str]:
        """Lines of *local*, read by the code model.

        Deliberately not keyed on the model *resolving* the path: the file has
        already been chosen by the time a window is read, and routing the read
        through the model is what makes it one cache and one counter instead of
        a second loader beside it.  A model needs no source root to read a file
        — only to *find* one — so this is the same call on every path, resolved
        through the ladder or named outright.
        """
        return self.model.lines(local)

    def extract(self, parsed: ParsedReport) -> SourceContext:
        """Extract source context for a parsed report.

        Finds relevant source files and extracts code around the bug line,
        key path event locations, and companion header files.
        """
        self._bind(parsed)
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

        # Try companion file patterns for each needed (stem, line) pair.  The
        # model looks for a sibling beside the *resolved* primary file, so the
        # name it reports is the one that exists rather than one this module
        # spelled; ``suffixes`` keeps this class's pattern order.
        for _stem, line in sorted(needed):
            windows = self._companion_windows(primary_path, line, radius=15)
            for companion_name, snippet in windows.items():
                key = f"{companion_name} (near line {line})"
                # Merge rather than overwrite if multiple lines
                if key in result:
                    result[key].update(snippet)
                else:
                    result[key] = snippet

        return result

    def _companion_windows(
        self, primary_path: Path, line: int, radius: int
    ) -> dict[str, dict[int, str]]:
        """Companion-file windows around *line*, in this class's pattern order.

        One entry per same-stem sibling that exists and is long enough to hold
        *line* — the caller decides whether to merge all of them or take the
        first.  ``suffixes`` is why this is a call and not a read: the model
        does the looking, but the order stays this class's (``-inl.h`` before
        ``.h``), so the sibling picked when several exist does not change.
        """
        return self.model.companion_window(
            primary_path, line, radius=radius,
            suffixes=tuple(self._COMPANION_PATTERNS),
        )

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

    def _file_line_count(self, file_path: Path) -> int | None:
        """Number of lines in a file, or ``None`` when there is nothing to read.

        An empty file counts as nothing to read rather than as zero lines: the
        only thing the callers do with the answer is a range check that zero
        fails anyway, and the two are indistinguishable to a reader of the
        file — this way ``None`` means exactly "no content".
        """
        lines = self._lines_of(file_path)
        return len(lines) or None

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

        # The report's embedded copy is the authoritative fallback.  Re-reading
        # the HTML here duplicated the frontend parser and bypassed the
        # report-scoped code model.
        logger.warning("Could not locate source file for: %s", bug_file)
        if parsed.source_snippets:
            return bug_file, parsed.source_snippets

        return None, {}

    def _resolve_to_local(self, file_path: str) -> Path | None:
        """Resolve a CSA build path to a local file, or ``None``.

        The model's ladder decides (``direct`` → build prefix → relative →
        suffix → basename), and only its **strong** rungs are accepted.  A
        name-only match is not the claim the report made — ``/usr/include/fmt/
        core.h`` and ``<root>/lib/cpluff/.../core.h`` share nothing but a name
        — and every caller here hands the result to a model as *this file's*
        code, so ``None`` is the honest answer and the callers' fallbacks (the
        report's own embedded copy; the line-index path) are the ones that can
        still be right.  A wrong file cannot.
        """
        return self.model.resolve_strong(file_path)

    # ------------------------------------------------------------------
    # Callee bodies for the report's call steps
    # ------------------------------------------------------------------

    def extract_callee_sources(
        self,
        parsed: ParsedReport,
        max_functions: int = 4,
        max_lines: int = 100,
        skip_functions: tuple[str, ...] = (),
    ) -> str:
        """Source of the functions the report's path calls into.

        The report's ``Calling 'X'`` steps name the callees on the reported
        path, but the parsed report carries no source for them: the pipeline's
        default context is the bug file only (plus same-stem companions), so a
        callee that lives in a *differently named* file is invisible to both the
        POC generator and the execution audit.  That blindness is what let
        "返回前未写入的输出参数" style reports through — the refuting write lived
        in ``time.cc`` while the flagged line was in ``time_test.cc``.

        Resolution order per callee (first hit wins):

        1. the *call site* (the event's own ``file_path``),
        2. the bug file (callees usually sit beside the code that calls them),
        3. the whole project root.

        All three are hints to the *same* lookup now — the call site is where
        the search starts, and the code model widens from there (same file,
        sibling files, its tree index, and the PDG's own answer for that name)
        rather than this module walking each root with its own ``rglob``.  The
        PDG's answer lands first of all, which is what keeps a name like
        ``fourcc_inv_printable`` from resolving to the file that *calls* it.

        Bodies are rendered with their file path and a **body-relative** line
        index (1 = the definition's signature line), and the header says so:
        ``find_function_definition`` returns the body text without its starting
        offset in the file, so numbering these as if they were file lines would
        send a reader who greps them to an unrelated line.  Returns ``""`` when
        nothing resolves — callers then simply omit the block.
        """
        self._bind(parsed)
        events = getattr(parsed, "path_events", None) or []
        ordered: list[str] = []
        call_sites: dict[str, Path] = {}
        for evt in events:
            if not evt.get("is_calling_event"):
                continue
            name = (evt.get("called_function") or "").strip()
            if not name or name in ordered or name in skip_functions:
                continue
            ordered.append(name)
            local = self._resolve_to_local(evt.get("file_path") or "")
            if local is not None:
                call_sites[name] = local
        if not ordered:
            return ""

        local_bug = self._resolve_to_local(
            getattr(parsed.metadata, "bug_file", "") or ""
        )

        blocks: list[str] = []
        for name in ordered[:max_functions]:
            # The call site's own file is the nearest hint; the bug file is the
            # next best (a callee usually sits beside the code that calls it),
            # and the model widens from either on its own.
            hint = call_sites.get(name) or local_bug
            found_file, body = self.find_function_definition(
                name, [hint] if hint else None,
            )
            if not body:
                logger.debug(
                    "callee sources: no definition found for %r (hint=%s)",
                    name, hint,
                )
                continue
            body_lines = body.splitlines()[:max_lines]
            numbered = "\n".join(
                f"{i:>6}: {ln}" for i, ln in enumerate(body_lines, 1)
            )
            blocks.append(
                f"--- Callee: {name}  ({found_file})\n"
                f"    lines are numbered within this body (1 = its signature "
                f"line), not by the file's own numbering ---\n{numbered}"
                + ("\n    … (truncated)" if len(body.splitlines()) > max_lines else "")
            )
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Function body extraction
    # ------------------------------------------------------------------

    def find_function_definition(
        self,
        function_name: str,
        search_roots: list[Path] | None = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Search project source for a C/C++ function definition.

        Returns ``(file_path, body_snippet)`` or ``(None, None)``.

        The *locating* is the code model's (:meth:`SourceCodeModel.function_files`
        — the PDG's answer first, then its tree index, both without an ``rglob``
        per call and both preferring the files nearest *search_roots*' first
        entry).  The *body* is still this module's scanner, because the extent
        the PDG records begins at the function's first statement while a body
        rendered for a model has to start at the signature — that line is where
        the parameter types are.
        """
        hint = Path(search_roots[0]) if search_roots else None
        candidates = self.model.function_files(function_name, file_hint=hint)
        for local in candidates:
            body = self.extract_function_body(local, function_name)
            if body:
                return str(local), body
        logger.debug(
            "function %r: scanner found no definition in %s",
            function_name, [str(p) for p in candidates[:5]],
        )
        return None, None

    def extract_function_body(
        self, file_path: str | Path, function_name: str
    ) -> str | None:
        """Extract a C/C++ function body from *file_path* using brace matching.

        Returns the full function text from the signature line to the closing
        ``}`` (inclusive), or ``None`` if the function is not found or the
        file cannot be read.  The file's lines come from the code model, so
        this reads a file once even when a report's path names it repeatedly.
        """
        lines = self._lines_of(Path(file_path))
        if not lines:
            return None

        fn_pattern = re.compile(rf"\b{re.escape(function_name)}\b")
        decl_keywords = re.compile(
            r"\b(typedef|enum|struct|class|union)\b"
        )

        # Call-site indicators: tokens that, when immediately before the
        # function name, mean this is a call, not a definition.
        call_site_prefix = re.compile(
            r"[\=\.,!&\(\)\-\+\*/%<>\[\]\|;:?~]"
            r"|->\s*$"
            r"|\.\s*$"
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

    def _read_code_around(
        self, file_path: Path, center_line: int, radius: int = 30
    ) -> dict[int, str]:
        """Read source code lines around a center line.

        ``[center - radius, center + radius]``, clipped to the file, blank
        lines kept — a rendered window with a gap in its numbering says
        something false about the file.  Read through the code model, so this
        is the same window (:meth:`SourceCodeModel.read_range`) every other
        rendered block in the pipeline gets.
        """
        lines = self._lines_of(file_path)
        if not lines:
            return {}
        start = max(1, center_line - radius)
        end = min(len(lines), center_line + radius)
        return {n: lines[n - 1] for n in range(start, end + 1)}
