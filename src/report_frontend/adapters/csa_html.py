"""Clang Static Analyzer HTML reports → canonical model.

This is the format the pipeline was born on, and the extraction below is the
v2 of that parser: the *document walk* (section index, div-depth matching,
code-line anchors) is unchanged, because every quirk it handles is a quirk of
real reports on disk —

* attributes are frequently unquoted (``class=FileName``);
* only the FIRST file section is wrapped in ``<div id="File1">``, later
  sections have no ``FileN`` div at all, so a code table must be attributed to
  the *last* ``FileName`` header that precedes it, not to a div id;
* a path event's line number is authoritative only from the ``data-linenumber``
  anchor of the listing it sits in — the legacy "last ``<td class=num>`` within
  2000 chars" scan is the fallback and misses any line longer than the window;
* events are interleaved with per-file code listings, so an event's file is the
  section it physically sits in.  Name-based attribution here is how a
  cross-file path collapsed onto the bug file.

What *is* new is the classification: each event is typed (``StepKind``) from
CSA's own wording, and the raw text is kept verbatim.  Downstream stages that
still parse the wording keep working (the legacy bridge re-renders exactly the
old dicts), while new consumers read the type.  The mapping is:

===============================  =========================  ===================
CSA wraps its event text in      →  StepKind               extra fields
===============================  =========================  ===================
``Error Start``                  →  ERROR_START            —
``Calling 'X'``                  →  CALL                   callee=X
``EndPath``                      →  REPORT                 —
``← Taking true/false branch →``  →  CONTROL                direction=True/False
``← Loop condition is …``        →  CONTROL                direction=True/False
``← '?' condition is …``         →  CONTROL                direction=True/False
``← Control jumps to 'case …'``  →  CONTROL                direction=None
``Returning from 'X'``           →  RETURN                 callee=X
``Assuming …``                   →  ASSUME                 assumption=…
anything else                    →  EVENT                  —
===============================  =========================  ===================

Two properties of the mapping are load-bearing and are guarded by tests in
``tests/test_report_frontend/test_csa_fidelity.py``:

* **It is exactly as wide as the pipeline's own reading was.**  ``Calling 'X'``
  and ``error start`` are matched with the same regular expressions the pipeline
  has always used, and the precedence between them and the ``EndPath`` markup is
  the precedence ``cfg_feasibility.build_bug_report_path`` classifies with.  A
  *wider* regex here (CSA also writes ``Calling constructor for 'A'``) would
  re-segment every report that contains the wider wording, i.e. change
  classifications under cover of a format refactor.
* **The legacy ``branch_condition`` value is preserved even where the model is
  more precise.**  The old parser set it only for ``Taking true/false branch``,
  leaving it ``None`` for the loop and ternary phrasings that state a direction
  just as clearly.  The model records the direction for all three; each step
  also carries ``raw["legacy_branch_condition"]``, and the bridge renders that,
  so the pipeline's inputs are unchanged.

Both are the same principle: this package changes *where a report is read from*,
not *what the pipeline concludes from it*.
"""

from __future__ import annotations

import bisect
import html as html_module
import logging
import re
from pathlib import Path

from ..errors import MissingReportMetadata, ReportParseError
from ..report_model import (
    AnalyzerKind,
    BugFinding,
    PathEvidence,
    ReportModel,
    ReportFormat,
    Severity,
    SourceLocation,
    StepKind,
    TraceStep,
)
from ..registry import ReportAdapter, register

logger = logging.getLogger(__name__)

# HTML comment patterns for metadata.
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
    # The clang invocation, when the report records one: the only place a CSA
    # report states which source root it was built against.
    "clang_cmd": r"clang[^\n<]*?-c\s+(\S+)",
}

_INT_META = frozenset({"bug_line", "bug_column", "path_length"})

_FILE_NAME_RE = re.compile(
    r'<h4[^>]*\bclass=["\']?FileName["\']?[^>]*>(.*?)</h4>',
    re.DOTALL,
)
_CODE_TABLE_RE = re.compile(r'<table[^>]*\bclass=["\']?code["\']?[^>]*>')
_CODELINE_ATTR_RE = re.compile(r'<tr\s+class="codeline"[^>]*?data-linenumber="(\d+)"')
_PATH_DIV_RE = re.compile(
    r'<div\s+id="(?:Path(\d+)|EndPath)"\s+class="msg\s+msg(\w+)"',
)
_LINE_ROW_HEAD_RE = re.compile(
    r'<tr\s+class="codeline"[^>]*data-linenumber="(\d+)"[^>]*>.*?'
    r'<td\s+class="line"[^>]*>',
    re.DOTALL,
)
_TD_OPEN_RE = re.compile(r"<td\b", re.IGNORECASE)
_TD_CLOSE_RE = re.compile(r"</td\s*>", re.IGNORECASE)
_LEGACY_NUM_RE = re.compile(r'<td\s+class="num"[^>]*>(\d+)</td>')

#: CSA wraps a macro's expansion — and a variable's value — in a nested
#: *tooltip* that sits inside the code line.  Two spellings, and both are
#: tooltips rather than code:
#:
#: * ``<span class='macro_popup'>`` — the macro body (54,355 occurrences);
#: * ``<table class='variable_popup'>`` — ``'accelerate' is false`` plus the
#:   ``PathIndex`` line that carries it (355).
#:
#: Neither is part of the line.  Stripping tags without removing them first
#: splices the tooltip's content — and its line number — into the middle of the
#: code text, which is what the LLM then reads.
_POPUP_OPEN_RE = re.compile(
    r"<(\w+)[^>]*\bclass\s*=\s*['\"]?[^'\">]*?"
    r"(?:macro_popup|variable_popup)[^'\">]*?['\"]?[^>]*>",
    re.IGNORECASE,
)


def _iter_code_cells(html: str):
    """Yield ``(line_number, row_offset, raw_cell_html)`` per code listing row.

    ``row_offset`` is where the ``<tr>`` starts, which is what attributes the
    row to a file section (:func:`_section_file_for_offset`).

    The cell ends at its *matching* ``</td>``, found by counting nested
    ``<td>``.  A ``variable_popup`` tooltip is a nested ``<table>`` with a ``<td>``
    of its own, so a non-greedy ``(.*?)</td>`` stops inside the tooltip and
    silently drops the rest of the line — including the code after it (355 cells
    on the corpus, e.g. ``if (accelerate) {`` left as ``if (accelerate``).
    """
    for m in _LINE_ROW_HEAD_RE.finditer(html):
        depth, pos, close_m = 1, m.end(), None
        while pos < len(html) and depth:
            nxt_open = _TD_OPEN_RE.search(html, pos)
            nxt_close = _TD_CLOSE_RE.search(html, pos)
            if nxt_close is None:
                break  # malformed
            if nxt_open is not None and nxt_open.start() < nxt_close.start():
                depth += 1
                pos = nxt_open.end()
            else:
                depth -= 1
                if depth == 0:
                    close_m = nxt_close
                pos = nxt_close.end()
        if close_m is None:
            continue
        yield int(m.group(1)), m.start(), html[m.end(): close_m.start()]


def _drop_popup_subtrees(cell: str) -> str:
    """Remove every tooltip subtree from a code-line cell."""
    out = cell
    while (m := _POPUP_OPEN_RE.search(out)) is not None:
        tag = m.group(1)
        open_re = re.compile(rf"<{tag}\b", re.IGNORECASE)
        close_re = re.compile(rf"</{tag}\s*>", re.IGNORECASE)
        depth, pos = 1, m.end()
        while pos < len(out) and depth:
            nxt_open = open_re.search(out, pos)
            nxt_close = close_re.search(out, pos)
            if nxt_close is None:
                break  # malformed: drop the remainder
            if nxt_open is not None and nxt_open.start() < nxt_close.start():
                depth += 1
                pos = nxt_open.end()
            else:
                depth -= 1
                pos = nxt_close.end()
        out = out[: m.start()] + out[pos:]
    return out


def _strip_code_cell(cell: str) -> str:
    """The code text of one ``<td class="line">`` cell.

    Three steps, in this order: drop the tooltip subtrees above, strip the
    remaining tags, then decode entities.  The last step is not optional —
    **20.5% of all code lines** (119,272 of 580,763) contain one, and 97,388 of
    those are *outside* any tooltip (``&lt;`` for ``<``, ``&amp;`` for ``&``).
    A naive tag strip leaves them in the text the LLM reads.
    """
    return html_module.unescape(
        re.sub(r"<[^>]+>", "", _drop_popup_subtrees(cell))
    ).strip()

#: CSA marks the finding's own step with this div id.
_END_PATH_NUMBER = 9999

#: ``Calling 'X'`` — CSA's call-step wording, matched exactly as the pipeline
#: has always matched it.  CSA *also* writes ``Calling constructor for 'A'``
#: (no quotes after ``Calling``), which the legacy regex never saw and so never
#: treated as a call boundary; widening this regex here would re-segment every
#: report containing that wording, which is a classification change smuggled in
#: under a format refactor.  Deliberately left narrow — see the module note.
_CALLING_RE = re.compile(r"Calling '([^']+)'")
#: ``Returning from 'X'`` — the matching return step.  Deliberately anchored on
#: "from": ``← Returning zero, which participates in …`` and ``← Returning
#: pointer →`` are notes about a *value*, not a call boundary, and treating
#: them as one would splice segments at the wrong lines.
_RETURNING_RE = re.compile(r"Returning from '?([^'→]+)'?")
_ASSUME_RE = re.compile(r"^\s*(?:←\s*)?(Assuming\s+.+?)(?:\s*→)?\s*$", re.I)
_BRANCH_RE = re.compile(r"Taking (true|false) branch", re.I)
_LOOP_RE = re.compile(r"Loop condition is (true|false)", re.I)
#: ``← '?' condition is true →`` (ternary) and ``← Control jumps to 'case 0:'``
#: (switch) also fix a direction, and CSA spells them differently again.
_TERNARY_RE = re.compile(r"'?'\s+condition\s+is\s+(true|false)", re.I)

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


@register
class CsaHtmlAdapter(ReportAdapter):
    """Parser for CSA HTML output reports."""

    name = "csa-html"
    format = ReportFormat.CSA_HTML
    analyzer = AnalyzerKind.CSA
    suffixes = (".html", ".htm")

    #: A CSA report's header comment block is the strongest signal there is.
    def sniff(self, path: Path, head: str) -> float:
        if "<!-- BUGTYPE" in head or "<!-- BUGFILE" in head:
            return 1.0
        if re.search(r'<div\s+id="(?:Path\d+|EndPath)"\s+class="msg', head):
            return 0.95
        if 'class=FileName' in head or 'class="FileName"' in head:
            return 0.8
        return 0.0

    # -- adapter interface -------------------------------------------------

    def load(self, path: Path, text: str) -> ReportModel:
        meta = self.extract_metadata(text)
        doc_index = _build_doc_index(text, fallback_file=str(meta.get("bug_file", "")))
        events = self._extract_events(text, doc_index=doc_index)
        snippets = self.extract_source_lines(text, int(meta.get("bug_line", 0)))
        steps = self._classify(events)

        bug_file = str(meta.get("bug_file", ""))
        bug_line = int(meta.get("bug_line", 0) or 0)
        if not meta.get("bug_type") or not bug_file or bug_line <= 0:
            missing = [
                key for key in ("bug_type", "bug_file", "bug_line")
                if not meta.get(key)
            ]
            raise MissingReportMetadata(
                f"{Path(path).name}: missing required metadata: {', '.join(missing)}"
            )

        finding = BugFinding(
            finding_id=f"csa:{Path(bug_file).name}:{bug_line}",
            analyzer=AnalyzerKind.CSA,
            rule_id=str(meta.get("bug_type", "")),
            rule_name=str(meta.get("bug_type", "")),
            category=str(meta.get("bug_category", "")),
            severity=Severity.coerce(meta.get("bug_category")),
            message=str(meta.get("bug_description", "")),
            location=SourceLocation(
                file=bug_file,
                line=bug_line,
                column=int(meta.get("bug_column", 0) or 0),
            ),
            function=str(meta.get("function_name", "")),
            steps=steps,
            # CSA always prints the path it walked — ``steps`` is that path, not
            # a location dressed up as one.
            path_evidence=PathEvidence.TRACE,
            # ``BUGPATHLENGTH`` is CSA's own count and is not always what the
            # report renders (it counts steps it then collapses); the pipeline
            # has always used CSA's number, so the model carries both.
            reported_path_length=int(meta.get("path_length", 0) or 0),
            fingerprint=str(meta.get("issue_hash", "")),
            raw={
                # Kept for the legacy bridge, which exposes them verbatim on
                # ``ParsedReport`` (``error_start_event``/``calling_events``)
                # and for ``source_snippets``.  The bridge could recompute them
                # from the typed steps, but the two derived views are already
                # part of the public dict shape and are cheap to carry.
                "metadata": meta,
                # The bug-file window the pipeline has always used, kept
                # verbatim (see ``extract_source_lines``).
                "source_snippets": snippets,
                # ... and the same source attributed per file, which is what
                # the code model reads when a path crosses files.
                "embedded_files": self.extract_file_lines(text),
                "doc_index_sections": len(doc_index[0]),
            },
        )
        return ReportModel(
            path=path,
            format=ReportFormat.CSA_HTML,
            analyzer=AnalyzerKind.CSA,
            findings=[finding],
            source_root_hint=self._source_root_hint(meta),
            metadata={"raw_metadata": meta},
        )

    # -- metadata ----------------------------------------------------------

    @staticmethod
    def extract_metadata(html: str) -> dict[str, str | int]:
        """Extract bug metadata from the report's HTML comment block."""
        meta: dict[str, str | int] = {}
        for key, pattern in _META_PATTERNS.items():
            match = re.search(pattern, html)
            if not match:
                continue
            val = match.group(1).strip()
            if key in _INT_META:
                try:
                    meta[key] = int(val)
                except ValueError:
                    continue
            else:
                meta[key] = val
        return meta

    @staticmethod
    def _source_root_hint(meta: dict[str, str | int]) -> str | None:
        """The build root a CSA report was generated under, when inferable.

        CSA embeds absolute paths from the machine it ran on.  When one of the
        known build roots is a prefix of the bug file, that prefix *is* the
        source root — a hint the code model verifies against the real tree
        rather than trusting (see ``code_model.SourceCodeModel``).
        """
        from ..code_model import KNOWN_BUILD_PREFIXES

        bug_file = str(meta.get("bug_file", "") or "")
        for prefix in KNOWN_BUILD_PREFIXES:
            if bug_file.startswith(prefix):
                return prefix
        return None

    # -- events ------------------------------------------------------------

    @staticmethod
    def clean_html_text(text: str) -> str:
        """Remove HTML tags, unescape entities, and clean whitespace."""
        text = re.sub(r"<[^>]+>", " ", text)
        text = html_module.unescape(text)
        # Remove leading path number pattern (e.g., "1 ", "25 ")
        text = re.sub(r"^\d+\s+", "", text.strip())
        return re.sub(r"\s+", " ", text).strip()

    def _extract_events(self, html: str, doc_index: DocIndex) -> list[dict]:
        """Ordered event rows: ``path_number``/``event_type``/``text``/file/line.

        The matching ``</div>`` is found by counting nested ``<div>`` tags from
        the opening position, not by a regex over the whole structure: a broad
        ``EndPath`` match swallows the events that precede it.
        """
        rows: list[tuple[int, str, str, int, str]] = []

        for m in _PATH_DIV_RE.finditer(html):
            num_str = m.group(1)
            path_num = int(num_str) if num_str else _END_PATH_NUMBER
            event_type = m.group(2).lower()

            div_start = m.start()
            depth = 0
            inner_start = html.index(">", m.end()) + 1

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

            text = self.clean_html_text(html[inner_start:close_pos])

            # The code listing immediately before the event row is
            # authoritative; the legacy scan is the fallback for reports that
            # lack codeline anchors.
            line_num = _line_for_offset(doc_index, div_start)
            if line_num <= 0:
                line_num = self._find_line_for_event(html, div_start)
            event_file = _section_file_for_offset(doc_index, div_start)

            rows.append((path_num, event_type, text, line_num, event_file))

        rows.sort(key=lambda r: r[0])
        if not rows:
            logger.warning("No path events found in CSA HTML report")
        return [
            {
                "path_number": num,
                "event_type": kind,
                "text": text,
                "line_number": line,
                "file_path": file,
            }
            for num, kind, text, line, file in rows
        ]

    @staticmethod
    def _find_line_for_event(html: str, event_start: int) -> int:
        """Legacy fallback: nearest ``<td class=num>`` before the event."""
        before = html[max(0, event_start - 2000):event_start]
        matches = _LEGACY_NUM_RE.findall(before)
        return int(matches[-1]) if matches else 0

    @staticmethod
    def _classify(events: list[dict]) -> list[TraceStep]:
        """Type every event row from CSA's own wording (see the module table).

        The precedence is the *legacy* parser's, deliberately:

        1. ``error start`` in the text
        2. ``Calling 'X'`` in the text
        3. the ``EndPath`` div (``path_number == 9999``) → the report step
        4. the ``control`` message class → a branch decision
        5. everything else → a plain event, refined by wording

        Rungs 1 and 2 come before rung 3 because that is the order
        ``build_bug_report_path`` (``cfg_feasibility``) classifies a step in:
        it tests ``is_error_start``, then ``is_calling_event``, then
        ``path_number == 9999``.  Reversing them here would move a step between
        node types on the handful of reports where the wording and the markup
        disagree, i.e. re-segment a path — a behaviour change that has no
        business riding along with a formats refactor.
        """
        steps: list[TraceStep] = []
        for idx, row in enumerate(events):
            text = row["text"]
            path_number = row["path_number"]
            is_control_markup = row["event_type"] == "control"
            location = SourceLocation(
                file=row["file_path"], line=row["line_number"],
            )

            kind = StepKind.EVENT
            direction: bool | None = None
            #: The legacy parser set ``branch_condition`` only for the
            #: ``Taking true/false branch`` wording, leaving it ``None`` for the
            #: loop and ternary phrasings that state a direction just as
            #: explicitly.  The model reports all three; the bridge renders this
            #: key, so downstream stages see exactly the values they saw before
            #: the refactor.  Noted in ``report_bridge`` where it is consumed.
            legacy_branch_condition: bool | None = None
            callee = ""
            assumption = ""
            condition = ""

            call = _CALLING_RE.search(text)
            if "error start" in text.lower():
                kind = StepKind.ERROR_START
            elif call:
                kind, callee = StepKind.CALL, call.group(1)
                condition = f"→ {callee}()"
            elif path_number == _END_PATH_NUMBER:
                kind = StepKind.REPORT
            elif is_control_markup:
                kind = StepKind.CONTROL
                branch = _BRANCH_RE.search(text)
                loop = _LOOP_RE.search(text)
                ternary = _TERNARY_RE.search(text)
                if branch:
                    direction = branch.group(1).lower() == "true"
                    legacy_branch_condition = direction
                elif loop:
                    # ``← Loop condition is false. Exiting loop →`` states the
                    # direction in words; the legacy parser left it unset.
                    direction = loop.group(1).lower() == "true"
                elif ternary:
                    direction = ternary.group(1).lower() == "true"
                # ``Control jumps to 'case 0:'`` and friends name no direction,
                # so ``direction`` stays None: downstream must not read a branch
                # the analyzer never reported.
            else:
                ret = _RETURNING_RE.search(text)
                assume = _ASSUME_RE.match(text)
                if ret:
                    kind, callee = StepKind.RETURN, ret.group(1).strip()
                elif assume:
                    # ``Assuming 'x' is equal to 1 →`` / ``← Assuming the
                    # condition is false →``: the text after the keyword is the
                    # constraint, and when it names a direction the direction is
                    # machine-readable even though the predicate is not.
                    kind = StepKind.ASSUME
                    assumption = assume.group(1).strip()
                    m_dir = re.search(
                        r"condition\s+is\s+(true|false)", assumption, re.I,
                    )
                    if m_dir:
                        direction = m_dir.group(1).lower() == "true"
                elif row["event_type"] == "report":
                    kind = StepKind.REPORT

            steps.append(
                TraceStep(
                    index=idx + 1,
                    kind=kind,
                    location=location,
                    message=text,
                    direction=direction,
                    assumption=assumption,
                    callee=callee,
                    condition=condition,
                    raw={
                        "path_number": path_number,
                        "event_type": row["event_type"],
                        "legacy_branch_condition": legacy_branch_condition,
                    },
                )
            )
        return steps

    # -- embedded source ---------------------------------------------------

    @staticmethod
    def extract_source_lines(
        html: str, center_line: int, radius: int = 30
    ) -> dict[int, str]:
        """Source lines the report embeds within *radius* of *center_line*.

        This is the *bug file's window* and is what the pipeline has always
        used as ``ParsedReport.source_snippets`` — including its one wart, that
        the window is filtered by line number alone and so also catches lines
        of that number from the report's other file sections.  Left as-is on
        purpose: it is the input ``build_project_deps.verify_line_alignment``
        checks reports against, and tightening it here would silently change
        what "a report agrees with the tree" means for the whole corpus.

        The *text* of each line is corrected (see :func:`_strip_code_cell`);
        the *keys* are untouched, which is what keeps the alignment signal
        above comparable.
        """
        snippets: dict[int, str] = {}
        for line_num, _offset, cell in _iter_code_cells(html):
            if abs(line_num - center_line) > radius:
                continue
            code = _strip_code_cell(cell)
            if code:
                snippets[line_num] = code
        return snippets

    @staticmethod
    def extract_file_lines(html: str) -> dict[str, dict[int, str]]:
        """Every embedded code listing, keyed by reported file then line.

        Unlike :meth:`extract_source_lines` this attributes each line to *its
        own section*, which is what the code model needs when a path crosses
        files.  Kept separate so the historical window semantics above are not
        perturbed.
        """
        index = _build_doc_index(html)
        out: dict[str, dict[int, str]] = {}
        for line_num, offset, cell in _iter_code_cells(html):
            file = _section_file_for_offset(index, offset)
            if not file:
                continue
            code = _strip_code_cell(cell)
            if code:
                out.setdefault(file, {})[line_num] = code
        return out


def parse_csa_html(path: Path, text: str | None = None) -> ReportDocument:
    """Convenience: parse a CSA HTML file without going through detection."""
    if text is None:
        if not path.exists():
            raise ReportParseError(f"FP report not found: {path}")
        text = path.read_text(encoding="utf-8", errors="replace")
    return CsaHtmlAdapter().load(path, text)


__all__ = [
    "CsaHtmlAdapter",
    "DocIndex",
    "parse_csa_html",
]
