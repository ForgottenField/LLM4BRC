"""Parser for CSA HTML output reports — extract metadata, path events, source code."""

from __future__ import annotations

import bisect
import html as html_module
import logging
import os
import re
from pathlib import Path

from llm_client.csa_analysis.models import (
    BugPath,
    EventKind,
    PathEvent,
)
from llm_client.fp_analysis.models import ParsedReport, ReportMetadata

logger = logging.getLogger(__name__)

# HTML comment patterns for metadata
_META_PATTERNS: dict[str, str] = {
    "bug_type": r"<!--\s*BUGTYPE\s+(.+?)\s*-->",
    "bug_category": r"<!--\s*BUGCATEGORY\s+(.+?)\s*-->",
    "bug_file": r"<!--\s*BUGFILE\s+(.+?)\s*-->",
    "file_name": r"<!--\s*FILENAME\s+(.+?)\s*-->",
    "bug_line": r"<!--\s*BUGLINE\s+(\d+)\s*-->",
    "bug_column": r"<!--\s*BUGCOLUMN\s+(\d+)\s*-->",
    "bug_description": r"<!--\s*BUGDESC\s+(.+?)\s*-->",
    "function_name": r"<!--\s*FUNCTIONNAME\s+(.+?)\s*-->",
    "path_length": r"<!--\s*BUGPATHLENGTH\s+(\d+)\s*-->",
    "issue_hash": r"<!--\s*ISSUEHASHCONTENTOFLINEINCONTEXT\s+(.+?)\s*-->",
}

# Document-layout markers.  A CSA report is a sequence of per-file sections:
# each section opens with ``<h4 class=FileName>PATH</h4>`` followed by a
# ``<table class="code" data-fileid="N">`` listing that file's source.  Two
# quirks make positional association the only reliable rule:
#   * attributes are frequently unquoted (``class=FileName``);
#   * only the FIRST section is wrapped in ``<div id="File1">`` — later
#     sections have no ``FileN`` div at all.
# So: walk the code tables in document order and attribute each to the LAST
# FileName that precedes it.
_FILE_NAME_RE = re.compile(
    r'<h4[^>]*\bclass=["\']?FileName["\']?[^>]*>(.*?)</h4>',
    re.DOTALL,
)
_CODE_TABLE_RE = re.compile(r'<table[^>]*\bclass=["\']?code["\']?[^>]*>')
_CODELINE_ATTR_RE = re.compile(r'<tr\s+class="codeline"[^>]*?data-linenumber="(\d+)"')

#: ``([section_offsets], [section_files], [line_offsets], [line_numbers])``
DocIndex = tuple[list[int], list[str], list[int], list[int]]


def _build_doc_index(html: str, fallback_file: str = "") -> DocIndex:
    """Index a report's per-file sections and code-line anchors by offset.

    Returns a :data:`DocIndex` whose four parallel lists are all sorted by
    offset, so an event's *file* and *source line* can be resolved with a
    binary search from the event's position in the document.
    """
    file_names = [
        (m.start(), html_module.unescape(m.group(1)).strip())
        for m in _FILE_NAME_RE.finditer(html)
    ]

    sec_offsets: list[int] = []
    sec_files: list[str] = []
    name_idx = 0
    for table_match in _CODE_TABLE_RE.finditer(html):
        t_off = table_match.start()
        # Advance to the last FileName preceding this table.
        current = fallback_file
        while name_idx < len(file_names) and file_names[name_idx][0] < t_off:
            current = file_names[name_idx][1]
            name_idx += 1
        sec_offsets.append(t_off)
        sec_files.append(current)

    line_offsets: list[int] = []
    line_numbers: list[int] = []
    for m in _CODELINE_ATTR_RE.finditer(html):
        line_offsets.append(m.start())
        line_numbers.append(int(m.group(1)))

    return sec_offsets, sec_files, line_offsets, line_numbers


def _section_file_for_offset(index: DocIndex, offset: int, fallback: str = "") -> str:
    """Return the source file of the section containing *offset*."""
    sec_offsets, sec_files, _, _ = index
    i = bisect.bisect_right(sec_offsets, offset) - 1
    if i < 0:
        return fallback
    return sec_files[i] or fallback


def _line_for_offset(index: DocIndex, offset: int) -> int:
    """Return the source line of the last code listing before *offset*."""
    _, _, line_offsets, line_numbers = index
    i = bisect.bisect_left(line_offsets, offset) - 1
    if i < 0:
        return 0
    return line_numbers[i]


class FPReportParser:
    """Parse CSA HTML report files into structured data for LLM analysis."""

    def parse(self, html_path: str | Path) -> ParsedReport:
        """Parse a CSA FP HTML report.

        Args:
            html_path: Path to the HTML report file.

        Returns:
            ParsedReport with metadata, path events, and source snippets.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required metadata (bug_type, bug_file, bug_line) is missing.
        """
        path = Path(html_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"FP report not found: {path}")

        html = path.read_text(encoding="utf-8", errors="replace")

        metadata = self._extract_metadata(html)
        # One document-level index, reused for every event: resolves which
        # file section each event sits in and its source line.
        doc_index = _build_doc_index(html, fallback_file=metadata.bug_file)
        path_events = self._extract_path_events(html, doc_index=doc_index)
        source_snippets = self._extract_source_lines(html, metadata.bug_line)

        # Enhanced: annotate events with error start and calling markers
        self._annotate_events(path_events)
        error_start_event = self._extract_error_start_event(path_events)
        calling_events = self._extract_calling_events(path_events)

        return ParsedReport(
            metadata=metadata,
            path_events=path_events,
            source_snippets=source_snippets,
            raw_html_path=str(path),
            error_start_event=error_start_event,
            calling_events=calling_events,
        )

    def parse_to_bug_path(self, html_path: str | Path) -> BugPath:
        """Parse an FP HTML report into a CSA-compatible BugPath.

        This bridges the FP analysis module with the existing csa_analysis
        data models, allowing reuse of correlator and downstream tools.

        Args:
            html_path: Path to the HTML report file.

        Returns:
            BugPath with events extracted from the HTML.
        """
        parsed = self.parse(html_path)
        events: list[PathEvent] = []

        for i, evt in enumerate(parsed.path_events):
            kind_str = evt.get("event_kind", "event")
            try:
                kind = EventKind(kind_str)
            except ValueError:
                kind = EventKind.EVENT

            # Try to extract line number from the event description
            line = evt.get("line_number", 0) or 0
            snippet = parsed.source_snippets.get(int(line)) if line else None

            events.append(PathEvent(
                event_kind=kind,
                file_path=evt.get("file_path") or parsed.metadata.bug_file,
                line_number=int(line),
                description=evt.get("description", ""),
                code_snippet=snippet,
                branch_condition=evt.get("branch_condition"),
            ))

        # Ensure last event is 'report'
        if events and events[-1].event_kind != EventKind.REPORT:
            events.append(PathEvent(
                event_kind=EventKind.REPORT,
                file_path=parsed.metadata.bug_file,
                line_number=parsed.metadata.bug_line,
                description=parsed.metadata.bug_description,
            ))

        return BugPath(
            path_id=f"fp_{Path(html_path).stem}",
            bug_type=parsed.metadata.bug_type,
            category=parsed.metadata.bug_category or "unknown",
            issue_context=parsed.metadata.function_name or None,
            events=events,
        )

    # ------------------------------------------------------------------
    # Internal extractors
    # ------------------------------------------------------------------

    def _extract_metadata(self, html: str) -> ReportMetadata:
        """Extract bug metadata from HTML comments."""
        meta: dict[str, str | int] = {}

        for key, pattern in _META_PATTERNS.items():
            match = re.search(pattern, html)
            if match:
                val = match.group(1).strip()
                if key in ("bug_line", "bug_column", "path_length"):
                    meta[key] = int(val)
                else:
                    meta[key] = val

        bug_type = meta.get("bug_type")
        bug_file = meta.get("bug_file")
        bug_line = meta.get("bug_line")

        if not bug_type or not bug_file or bug_line is None:
            missing = [k for k in ("bug_type", "bug_file", "bug_line") if k not in meta or not meta.get(k)]
            raise ValueError(f"Missing required metadata: {', '.join(missing)}")

        return ReportMetadata(
            bug_type=str(bug_type),
            bug_category=str(meta.get("bug_category", "")),
            bug_file=str(bug_file),
            bug_line=int(bug_line),
            bug_column=int(meta.get("bug_column", 0)),
            bug_description=str(meta.get("bug_description", "")),
            function_name=str(meta.get("function_name", "")),
            path_length=int(meta.get("path_length", 0)),
        )

    @staticmethod
    def _clean_html_text(text: str) -> str:
        """Remove HTML tags, unescape entities, and clean whitespace."""
        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Unescape HTML entities (handles &#x2190;, &lt;, &amp;, etc.)
        text = html_module.unescape(text)
        # Remove leading path number pattern (e.g., "1 ", "25 ")
        text = re.sub(r'^\d+\s+', '', text.strip())
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_path_events(
        self, html: str, doc_index: DocIndex | None = None
    ) -> list[dict]:
        """Extract ordered path events from HTML msg divs.

        Uses div-depth counting to find the matching closing ``</div>`` for
        each Path div, avoiding the regex-overlap problem where a broad
        EndPath match can swallow earlier Path events.

        *doc_index* is the document section/line index built once by
        :meth:`parse`.  When omitted (direct calls with the legacy signature)
        it is built lazily here, which yields ``file_path == ""`` for reports
        without a ``FileName`` header — the pre-index behaviour.
        """
        events: list[dict] = []
        if doc_index is None:
            doc_index = _build_doc_index(html)

        # Lightweight pattern: find the OPENING tag of each Path/EndPath div.
        # We do NOT try to match the full div+table structure in one regex —
        # instead we locate the closing </div> by counting nested <div> /
        # </div> tags from the opening position.
        open_pattern = re.compile(
            r'<div\s+id="(?:Path(\d+)|EndPath)"\s+class="msg\s+msg(\w+)"',
        )

        matches = []
        for m in open_pattern.finditer(html):
            num_str = m.group(1)
            path_num = int(num_str) if num_str else 9999
            event_type = m.group(2).lower()

            # ── Find the matching </div> via depth counting ──
            div_start = m.start()  # position of '<div id=...'
            depth = 0
            inner_start = html.index(">", m.end()) + 1  # skip past opening '>'

            # Scan forward from the opening <div>, counting nested div tags
            pos = inner_start
            close_pos = -1
            while pos < len(html):
                next_open = html.find("<div", pos)
                next_close = html.find("</div>", pos)

                if next_close == -1:
                    break  # malformed HTML

                if next_open != -1 and next_open < next_close:
                    depth += 1
                    pos = next_open + 4
                else:
                    if depth == 0:
                        close_pos = next_close
                        break
                    depth -= 1
                    pos = next_close + 6

            if close_pos == -1:
                continue  # could not find matching </div>

            inner = html[inner_start:close_pos]

            # Extract description text (strip HTML tags, unescape entities)
            text = self._clean_html_text(inner)

            # Line number: the code listing immediately before the event row
            # (``<tr class="codeline" data-linenumber="N">``) is authoritative;
            # the legacy "last <td class=num>N</td> within 2000 chars" scan is
            # the fallback for reports that lack codeline anchors.
            line_num = _line_for_offset(doc_index, div_start)
            if line_num <= 0:
                line_num = self._find_line_for_event(html, div_start, inner)

            # File: the section this event physically sits in (CSA interleaves
            # the event divs with each file's own code listing).
            event_file = _section_file_for_offset(doc_index, div_start)

            matches.append((path_num, event_type, text, line_num, event_file))

        # Sort by path number
        matches.sort(key=lambda x: x[0])

        for path_num, event_type, text, line_num, event_file in matches:
            # Determine event kind
            if event_type == "control":
                kind = "control"
                # Detect branch condition
                branch = None
                if "true branch" in text.lower():
                    branch = True
                elif "false branch" in text.lower():
                    branch = False
            elif path_num == 9999:
                kind = "report"
                branch = None
            else:
                kind = "event"
                branch = None

            events.append({
                "path_number": path_num,
                "event_kind": kind,
                "description": text,
                "line_number": line_num,
                "branch_condition": branch,
                "file_path": event_file,
            })

        if not events:
            logger.warning("No path events found in HTML report")

        return events

    def _find_line_for_event(self, html: str, event_start: int, inner_html: str) -> int:
        """Try to determine the source line number associated with a path event.

        The line number is typically in a nearby <td class="num"> before the event.
        """
        # Look backwards from event start for a line number anchor
        before = html[max(0, event_start - 2000):event_start]
        # Pattern for line number: <td class="num"...>N</td>
        line_matches = re.findall(r'<td\s+class="num"[^>]*>(\d+)</td>', before)
        if line_matches:
            return int(line_matches[-1])
        return 0

    def _extract_source_lines(
        self, html: str, center_line: int, radius: int = 30
    ) -> dict[int, str]:
        """Extract source code lines from the HTML.

        Args:
            html: Raw HTML content.
            center_line: Line number to center the extraction around.
            radius: Number of lines to extract above and below center.

        Returns:
            Dict mapping line number → code text.
        """
        snippets: dict[int, str] = {}

        # Pattern: <tr class="codeline" data-linenumber="N">...code...</tr>
        line_pattern = re.compile(
            r'<tr\s+class="codeline"[^>]*data-linenumber="(\d+)"[^>]*>.*?<td\s+class="line"[^>]*>(.*?)</td>',
            re.DOTALL,
        )

        for m in line_pattern.finditer(html):
            line_num = int(m.group(1))
            # Only extract lines within range of interest
            if abs(line_num - center_line) > radius:
                continue
            code_html = m.group(2)
            # Strip HTML tags from code
            code = re.sub(r'<[^>]+>', '', code_html)
            code = code.strip()
            if code:
                snippets[line_num] = code

        return snippets

    # ------------------------------------------------------------------
    # Enhanced event analysis: Error Start, calling chain, annotations
    # ------------------------------------------------------------------

    @staticmethod
    def _annotate_events(path_events: list[dict]) -> None:
        """Annotate events in-place with metadata flags.

        Adds these keys to each event dict:
        - is_error_start: True if description contains 'error start'
        - is_calling_event: True if description matches "Calling '...'"
        - called_function: extracted function name (or None)
        - is_control_flow: True if event_kind == 'control'
        """
        for evt in path_events:
            desc = evt.get("description", "")
            evt["is_error_start"] = "error start" in desc.lower()

            m = re.search(r"Calling '([^']+)'", desc)
            if m:
                evt["is_calling_event"] = True
                evt["called_function"] = m.group(1)
            else:
                evt["is_calling_event"] = False
                evt["called_function"] = None

            evt["is_control_flow"] = evt.get("event_kind") == "control"

    @staticmethod
    def _extract_error_start_event(path_events: list[dict]) -> dict | None:
        """Find the 'Error Start' event where the undefined value originates.

        Returns a dict with event_number, line_number, description, file_path
        if an Error Start event is found, otherwise None.
        """
        for evt in path_events:
            if evt.get("is_error_start"):
                return {
                    "event_number": evt.get("path_number", 0),
                    "line_number": evt.get("line_number", 0),
                    "description": evt.get("description", ""),
                    "file_path": evt.get("file_path", ""),
                }
        return None

    @staticmethod
    def _extract_calling_events(path_events: list[dict]) -> list[dict]:
        """Extract all 'Calling' events with function name and location.

        Returns a list of dicts with event_number, line_number, description,
        called_function, file_path.
        """
        calling: list[dict] = []
        for evt in path_events:
            if evt.get("is_calling_event"):
                calling.append({
                    "event_number": evt.get("path_number", 0),
                    "line_number": evt.get("line_number", 0),
                    "description": evt.get("description", ""),
                    "called_function": evt.get("called_function", ""),
                    "file_path": evt.get("file_path", ""),
                })
        return calling


# ------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------

def resolve_report_source_file(
    bug_file: str,
    project_root: str | Path | None = None,
    file_name_hint: str | None = None,
) -> Path | None:
    """Resolve a report's source file path to a local filesystem path.

    CSA reports often contain absolute paths from the build environment
    that don't match the local project layout. This function tries multiple
    strategies to find the actual file.
    """
    path = Path(bug_file)
    # Strategy 1: direct existence
    if path.exists():
        return path.resolve()

    # Strategy 2: try file_name hint (just the filename, no path)
    if file_name_hint:
        file_only = Path(file_name_hint).name
        if project_root:
            root = Path(project_root).resolve()
            for found in root.rglob(file_only):
                return found.resolve()

    # Strategy 3: search by basename
    if project_root:
        root = Path(project_root).resolve()
        basename = path.name
        for found in root.rglob(basename):
            return found.resolve()

    # Strategy 4: try stripping known prefixes from the CSA build path
    # The CSA build path prefix patterns from the folly project
    known_prefixes = [
        "/tmp/fbcode_builder_getdeps-ZhomeZmeZprojectsZcpp-projectZfollyZbuildZfbcode_builder/",
        "/home/me/projects/cpp-project/",
    ]
    for prefix in known_prefixes:
        if str(path).startswith(prefix):
            rel = str(path).removeprefix(prefix).lstrip("/")
            if project_root:
                candidate = Path(project_root).resolve() / rel
                if candidate.exists():
                    return candidate.resolve()

    return None
