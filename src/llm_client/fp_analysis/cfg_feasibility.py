"""Path segmentation and CFG-based feasibility orchestration for in-report FP detection.

Given a parsed CSA report:
1. **Segment** the path by control events → N segments
2. **On-demand**: for each segment, identify called functions and build
   CFG only for those, then enumerate alternative branch paths
3. LLM evaluates feasibility of each variant (done by ``FPAnalyzer``)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

from llm_client.fp_analysis.cfg_models import (
    AssumptionFact,
    BugReportPath,
    CFGAnalysisResult,
    CFGFunction,
    CFGPathVariant,
    NodeType,
    PathNode,
    SegmentInfo,
    SegmentPathAnalysis,
)
from llm_client.fp_analysis.cfg_parser import (
    build_cfg_command,
    enumerate_paths,
    parse_cfg_text,
    run_cfg_dump,
)
from llm_client.fp_analysis.models import ParsedReport

logger = logging.getLogger(__name__)

# Persistent CFG cache: each source file's parsed CFG is cached once
# so subsequent reports from the same file skip recompilation.
_CFG_CACHE_DIR = Path("cfg_cache")


def _cfg_cache_path(source_file: str, project_name: str) -> Path:
    """Return the cache file path for a source file's CFG data.

    Uses a hash of the absolute source path as the filename to avoid
    filesystem-path encoding issues, placed under ``cfg_cache/{project}/``.
    """
    abs_path = str(Path(source_file).resolve())
    h = hashlib.md5(abs_path.encode()).hexdigest()
    return _CFG_CACHE_DIR / project_name / f"{h}.json"


def _load_cfg_cache(
    cache_path: Path, drop_raw_dump: bool = True
) -> dict[str, CFGFunction] | None:
    """Load cached CFG functions from *cache_path*.

    *drop_raw_dump* discards ``CFGFunction.raw_dump`` while loading.  That
    field holds a verbatim copy of the whole translation unit's dump and is
    duplicated into every entry — it accounts for ~95% of the cache size
    (e.g. 14.1 MB → 0.75 MB for clone_index.cpp) and nothing in the codebase
    reads it.  Keeping it only matters when inspecting a cache by hand.

    Returns ``None`` if the cache file doesn't exist or is corrupt.
    """
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        if drop_raw_dump:
            for fn_data in data.values():
                fn_data.pop("raw_dump", None)
        return {
            fn_name: CFGFunction.model_validate(fn_data)
            for fn_name, fn_data in data.items()
        }
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug("CFG cache corrupt, ignoring: %s", e)
        return None


def _save_cfg_cache(
    cache_path: Path,
    functions: dict[str, CFGFunction],
    drop_raw_dump: bool = True,
) -> None:
    """Save parsed CFG functions to *cache_path*.

    *drop_raw_dump* (the default) omits ``CFGFunction.raw_dump``.  That field is
    a verbatim copy of the **whole translation unit's** dump, stamped onto every
    entry by ``parse_cfg_text`` — so a cache is ``n_functions × dump_size``, not
    ``dump_size``.  For ``clone_index.cpp`` (few functions) that is 14.1 MB →
    0.75 MB; for protobuf's ``text_format.cc`` (3290 functions, one 3.3 MB dump)
    it is **10.9 GB, of which the actual block data is ~1%**.
    ``_load_cfg_cache`` already drops the field on the way in and nothing else
    reads it, so writing it only costs disk and load time (a 10.9 GB
    ``json.loads`` per report run).
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for fn_name, fn_data in functions.items():
        dumped = fn_data.model_dump(mode="json")
        if drop_raw_dump:
            dumped.pop("raw_dump", None)
        data[fn_name] = dumped
    cache_path.write_text(json.dumps(data, indent=2))
    logger.info("CFG cache saved: %s (%d functions)", cache_path, len(functions))

# C/C++ keywords to filter out when detecting function call sites
_KEYWORDS = frozenset({
    "if", "while", "for", "switch", "case", "return", "else",
    "sizeof", "break", "continue", "do", "catch", "try",
})


# ---------------------------------------------------------------------------
# Bug-Report Path builder — resolve each event to its function context
# ---------------------------------------------------------------------------


def build_bug_report_path(
    parsed: ParsedReport,
    sdg: "SDG | None" = None,
    source_root: str | None = None,
) -> BugReportPath:
    """Build a structured ``BugReportPath`` from a parsed CSA report.

    Each path event is annotated with:
    - ``node_type``: CONTROL, ERROR_START, ERROR_SITE, ENTRY, CALL, etc.
    - ``function_name``: determined by scanning the source file for function
      definitions and finding which function contains the event's line.
    - ``function_line_start/end``: the containing function's source bounds.
    - ``source_code``: the actual source line at the event's line number.

    When *sdg* is provided (recommended), cross-file CSA paths are handled:
    events in caller files (not the bug_file) are resolved via the SDG's
    line-level index and CALL-event class-name context.

    This eliminates heuristic line-range guessing when extracting code
    for segment analysis.
    """
    meta = parsed.metadata
    events = parsed.path_events
    path_nodes: list[PathNode] = []

    # Resolve the base source file (via source_root when the report's stale
    # bug_file path doesn't exist on this machine).
    bug_file = _resolve_source_file_for_cfg(parsed, source_root)

    # PDG-based function attribution index (accurate source ranges from the
    # SDG's node lines).  Consulted first; the heuristic scanner is the
    # fallback for files the PDG doesn't cover.
    pdg_fn_index = _build_pdg_fn_index(sdg, source_root)

    # ── Build class-to-file map from CALL event descriptions ──
    class_to_file = _build_class_to_file_map(
        events, source_root=source_root, bug_file=bug_file,
    )

    # ── Collect all relevant source files ──
    all_files: dict[str, Path] = {}
    if bug_file and bug_file.exists():
        all_files[str(bug_file.resolve())] = bug_file.resolve()
    for cf in class_to_file.values():
        all_files[str(cf)] = cf

    # Per-event files reported by the parser (each event's enclosing report
    # section).  These are authoritative — a CSA report interleaves an event
    # with the code listing of the file it belongs to — and they are what
    # stops cross-file paths from collapsing onto ``bug_file``.
    event_files: dict[int, Path] = {}
    report_files: set[str] = set()
    for idx, evt in enumerate(events):
        raw_file = (evt.get("file_path") or "").strip()
        if not raw_file:
            continue
        local_file = _resolve_local_path(raw_file, source_root)
        if local_file is None:
            continue
        event_files[idx] = local_file
        report_files.add(str(local_file))
        all_files.setdefault(str(local_file), local_file)

    # ── Read + build function boundaries for each file ──
    multi_file_lines: dict[str, list[str]] = {}
    multi_boundaries: dict[str, list[tuple[str, int, int]]] = {}
    for fkey, fpath in all_files.items():
        try:
            flines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            multi_file_lines[fkey] = flines
            multi_boundaries[fkey] = _find_function_boundaries(fpath, flines)
        except OSError:
            pass

    # Fallback: use bug_file boundaries for backward compatibility
    if bug_file and bug_file.exists():
        fallback_key = str(bug_file.resolve())
        fallback_lines = multi_file_lines.get(fallback_key, [])
        fallback_boundaries = multi_boundaries.get(fallback_key, [])
    else:
        fallback_key = None
        fallback_lines = []
        fallback_boundaries: list[tuple[str, int, int]] = []

    for evt_idx, evt in enumerate(events):
        line = evt.get("line_number", 0) or 0
        desc = evt.get("description", "") or ""
        pn = evt.get("path_number", 0)

        # Classify the node type
        ntype = NodeType.EVENT
        if evt.get("is_error_start"):
            ntype = NodeType.ERROR_START
        elif evt.get("is_calling_event"):
            ntype = NodeType.CALL
        elif pn == 9999:
            ntype = NodeType.REPORT
        elif evt.get("is_control_flow") and evt.get("branch_condition") in (True, False):
            ntype = NodeType.CONTROL
        elif evt.get("event_kind") == "control":
            # All CSA [control] events are control-flow nodes, even without
            # explicit BRANCH=T/F (e.g. loop conditions, ternary conditions).
            ntype = NodeType.CONTROL

        # ── Determine which source file this event belongs to ──
        # Priority 1: the file of the report section the event sits in.
        # Priority 2: heuristic resolution against the SDG line index,
        #             restricted to files the report actually mentions.
        event_file = event_files.get(evt_idx)
        if event_file is None:
            event_file = _resolve_event_file(
                line, desc, class_to_file, sdg, bug_file,
                report_files=report_files, source_root=source_root,
            )
        # Normalize to a local existing file (the resolved event_file may be a
        # stale build path when it came from the SDG line-index), so fn
        # attribution + src_line extraction work on the real file.
        if event_file is not None and not event_file.exists():
            local = _resolve_local_path(event_file, source_root)
            if local is not None:
                event_file = local

        # ── Find containing function (from the correct file) ──
        fn_name = ""
        fn_start = 0
        fn_end = 0
        if line > 0 and event_file:
            # Authoritative: PDG-derived ranges first (accurate, covers
            # functions the heuristic scanner misses).
            if event_file.exists():
                for name, s, e in pdg_fn_index.get(str(event_file.resolve()), []):
                    if s <= line <= e:
                        fn_name = name
                        fn_start = s
                        fn_end = e
                        break
            # Fallback: heuristic scanner on this file's boundaries.
            if not fn_name:
                fkey = str(event_file.resolve())
                boundaries = multi_boundaries.get(fkey, [])
                for name, s, e in boundaries:
                    if s <= line <= e:
                        fn_name = name
                        fn_start = s
                        fn_end = e
                        break
            # Fallback: try the bug_file boundaries
            if not fn_name and fkey != fallback_key:
                for name, s, e in fallback_boundaries:
                    if s <= line <= e:
                        fn_name = name
                        fn_start = s
                        fn_end = e
                        break

        # ── Extract source line (from the correct file) ──
        src_line = ""
        if event_file:
            fkey = str(event_file.resolve())
            flines = multi_file_lines.get(fkey, fallback_lines)
            if 0 < line <= len(flines):
                src_line = flines[line - 1]
        elif 0 < line <= len(fallback_lines):
            src_line = fallback_lines[line - 1]

        # For control nodes without explicit branch_condition, infer from description
        bc = evt.get("branch_condition")
        if ntype == NodeType.CONTROL and bc is None:
            desc_lower = desc.lower()
            if "loop condition is true" in desc_lower or "entering" in desc_lower:
                bc = True
            elif "loop condition is false" in desc_lower or "exiting" in desc_lower:
                bc = False
            elif "?" in desc and "condition is true" in desc_lower:
                bc = True
            elif "?" in desc and "condition is false" in desc_lower:
                bc = False
            elif "true branch" in desc_lower:
                bc = True
            elif "false branch" in desc_lower:
                bc = False

        path_nodes.append(PathNode(
            event_number=pn,
            node_type=ntype,
            line_number=line,
            source_file=str(event_file) if event_file else "",
            description=desc,
            function_name=fn_name,
            function_line_start=fn_start,
            function_line_end=fn_end,
            source_code=src_line,
            branch_condition=bc,
        ))

    return BugReportPath(
        nodes=path_nodes,
        bug_file=str(bug_file) if bug_file else meta.bug_file,
        bug_type=meta.bug_type,
        bug_description=meta.bug_description,
    )


def _find_function_boundaries(
    bug_file: Path | None, file_lines: list[str]
) -> list[tuple[str, int, int]]:
    """Scan source lines and return ``[(function_name, start, end), ...]``.

    Uses a simple heuristic: any line matching a function definition
    pattern (return-type name(params) {: or { on next line) is a candidate.
    Brace-matching finds the end.

    Returns a **list** so that overloaded functions (same name, different
    line ranges) are all preserved — a dict keyed by function name would
    silently drop all but the last overload.
    """
    boundaries: list[tuple[str, int, int]] = []
    if not file_lines:
        return boundaries

    i = 0
    open_braces = 0
    current_fn = None
    current_start = 0
    in_function = False
    seen_opening_brace = False  # Tracks whether we've hit the first '{' yet

    while i < len(file_lines):
        line = file_lines[i]
        stripped = line.strip()

        # Skip comments and preprocessor
        if stripped.startswith(("#", "//", "/*", "*")):
            i += 1
            continue

        if not in_function:
            # Look for function definitions — three strategies
            if (
                stripped
                and ";" not in stripped
                and not stripped.startswith(("if ", "while ", "for ", "switch ",
                                              "case ", "return ", "else ", "typedef ",
                                              "enum ", "struct ", "class "))
            ):
                fn_name = ""
                has_brace = "{" in stripped

                # Strategy 1: same-line params — "type name(params)" with "{"
                #   e.g. "uint8_t f(int x) {"
                has_same_line_params = "(" in stripped and ")" in stripped
                # Strategy 2: multi-line — "type name\n(params...)\n{"
                #   e.g. "static int f\n(struct X *x,\n size_t len)\n{"
                next_line_opens_parens = (
                    "(" not in stripped
                    and not stripped.endswith(")")
                    and i + 1 < len(file_lines)
                    and file_lines[i + 1].strip().startswith("(")
                    and re.search(r'\b\w+\s*$', stripped.rstrip('{'))
                )
                # Strategy 3: '(' on this line but ')' later
                #   e.g. "int f(struct X *x,\n            size_t len)\n{"
                has_inline_open_paren = (
                    "(" in stripped and ")" not in stripped
                    and not has_same_line_params
                )

                if has_same_line_params or next_line_opens_parens or has_inline_open_paren:
                    brace_line = -1

                    if has_same_line_params:
                        if has_brace:
                            brace_line = i
                        elif i + 1 < len(file_lines) and "{" in file_lines[i + 1]:
                            brace_line = i + 1
                    else:
                        # Multi-line or inline-open: scan forward for ')' then '{'
                        search_end = min(i + 10, len(file_lines))
                        paren_depth = 0
                        found_close_paren = False
                        for j in range(i, search_end):
                            ln_stripped = file_lines[j].strip()
                            paren_depth += ln_stripped.count("(") - ln_stripped.count(")")
                            if "{" in ln_stripped:
                                brace_line = j
                                break
                            if not found_close_paren and paren_depth <= 0 and ln_stripped.count(")") > 0:
                                found_close_paren = True
                            # If no brace found but we passed params, check next line
                            if found_close_paren and j + 1 < search_end and "{" in file_lines[j + 1]:
                                brace_line = j + 1
                                break

                    if brace_line >= 0:
                        # Collect all declaration lines for function name extraction
                        decl_lines = [stripped]
                        for j in range(i + 1, brace_line + 1):
                            decl_lines.append(file_lines[j].strip())
                        decl_text = " ".join(decl_lines)

                        # Try to extract name before '('
                        m = re.search(r'(\w+)\s*\(', decl_text)
                        if m:
                            fn_name = m.group(1)
                        # Fallback: name at end of first line
                        if not fn_name:
                            m = re.search(r'(\w+)\s*$', stripped.rstrip('{'))
                            if m:
                                fn_name = m.group(1)

                        if fn_name and fn_name not in (
                            "if", "while", "for", "switch", "return",
                        ):
                            current_fn = fn_name
                            current_start = i + 1  # 1-based
                            in_function = True
                            seen_opening_brace = False
                            # Count braces from declaration lines EXCLUDING the
                            # brace line itself (the loop will count it when
                            # it reaches that line — double-counting would
                            # prevent the function from ever closing).
                            pre_brace_lines = decl_lines[:brace_line - i]
                            open_braces = sum(l.count("{") for l in pre_brace_lines)
                            if open_braces > 0:
                                seen_opening_brace = True
        else:
            has_open = stripped.count("{")
            open_braces += has_open - stripped.count("}")
            if has_open > 0:
                seen_opening_brace = True
            if seen_opening_brace and open_braces <= 0:
                boundaries.append((current_fn, current_start, i + 1))
                in_function = False
                current_fn = None
        i += 1

    # Catch any unclosed function
    if in_function and current_fn:
        boundaries.append((current_fn, current_start, len(file_lines)))

    return boundaries


# ---------------------------------------------------------------------------
# Path segmentation (now uses BugReportPath for precise function bounds)
# ---------------------------------------------------------------------------


# ── CSA assumption patterns (not control events) ──────────────────────────

# Pattern 1: "Assuming field 'X' is <op> Y" or "Assuming 'X' is <op> Y"
# ("field" is optional — many CSA phrasings omit it, e.g. "Assuming 'error' is 0").
# The value capture is (.+?) to prefer shortest match; trailing '→' stripped below.
_ASSUME_FIELD_PATTERN = re.compile(
    r"Assuming\s+(?:field\s+)?'(\w+(?:\.\w+)*)'\s+is\s+(<=|>=|!=|==|[<>])\s+(.+?)(?:\s*[←→]\s*)?$",
)

# Pattern 2: "Assuming 'X' is equal to <Y>" — Y may be a field ('Y'), a
# plain identifier, or a literal (e.g. "Assuming 'error' is equal to 0").
_ASSUME_EQUAL_PATTERN = re.compile(
    r"Assuming\s+'(\w+(?:\.\w+)*)'\s+is\s+equal\s+to\s+(?:field\s+)?'?([^']+?)'?(?:\s*[←→]\s*)?$",
)

# Pattern 3: "(Null pointer )value stored to 'Y'" / "X stored to 'Y'"
_STORED_TO_PATTERN = re.compile(
    r"(?:Null\s+pointer\s+|Value\s+'(\w+)'\s+)?stored\s+to\s+'([^']+)'",
)


def extract_assumptions_from_events(
    events: list[dict],
    event_range: tuple[int, int],
    segment_index: int = 0,
) -> list[AssumptionFact]:
    """Extract CSA symbolic-execution assumptions from event-type path events.

    Only processes events with ``event_kind in ("event", "error_start")``.
    Control events, call events, and report events are skipped — those are
    handled separately by the branch filter and path selection logic.

    Recognised patterns (from CSA HTML output):

    * ``Assuming field 'X' is <op> Y`` → ``{var: X, op: <op>, value: Y}``
    * ``Assuming 'X' is equal to field 'Y'`` → ``{var: X, op: ==, value: Y}``
    * ``Null pointer value stored to 'Y'`` → ``{var: Y, op: ==, value: null}``
    * ``Value 'X' stored to 'Y'`` → ``{var: Y, op: ==, value: X}``

    Returns:
        Ordered list of :class:`AssumptionFact` objects.
    """
    first, last = event_range
    facts: list[AssumptionFact] = []

    for evt in events:
        pn = evt.get("path_number", 0)
        if not (first <= pn <= last):
            continue

        kind = evt.get("event_kind", "")
        if kind not in ("event", "error_start"):
            continue

        desc = evt.get("description", "")
        line = evt.get("line_number", 0)
        is_es = bool(evt.get("is_error_start", False))

        # Pattern 1: "Assuming field 'X' is <op> Y"
        m = _ASSUME_FIELD_PATTERN.search(desc)
        if m:
            val = m.group(3).strip()
            # Strip trailing CSA path-direction marker: ' →' or '← '
            val = re.sub(r'\s*[←→]\s*$', '', val)
            facts.append(AssumptionFact(
                event_number=pn,
                line_number=line,
                segment_index=segment_index,
                variable=m.group(1),
                operator=m.group(2),
                value=val,
                raw_description=desc,
                is_error_start=is_es,
            ))
            continue

        # Pattern 2: "Assuming 'X' is equal to field 'Y'"
        m = _ASSUME_EQUAL_PATTERN.search(desc)
        if m:
            facts.append(AssumptionFact(
                event_number=pn,
                line_number=line,
                segment_index=segment_index,
                variable=m.group(1),
                operator="==",
                value=m.group(2),
                raw_description=desc,
                is_error_start=is_es,
            ))
            continue

        # Pattern 3: "stored to 'Y'" (null pointer / value stored)
        m = _STORED_TO_PATTERN.search(desc)
        if m:
            val = m.group(1) if m.group(1) else "null"
            facts.append(AssumptionFact(
                event_number=pn,
                line_number=line,
                segment_index=segment_index,
                variable=m.group(2),
                operator="==",
                value=val,
                raw_description=desc,
                is_error_start=is_es,
            ))
            continue

    return facts


def segment_path(
    parsed: ParsedReport,
    sdg: "SDG | None" = None,
    source_root: str | None = None,
) -> list[SegmentInfo]:
    """Split a CSA execution path into segments at CONTROL and CALL boundaries.

    - CONTROL nodes mark control-flow decisions (if/else/loop).
    - CALL nodes mark opaque function calls where the analysis context
      switches to the called function.

    Each ``SegmentInfo`` carries:
    - ``precondition_facts`` — state established by prior events
    - ``code_from_entry`` — source from function entry to postcondition/call line
    - ``called_functions`` — function call names (bodies added at prompt time)
    - ``if_line_numbers`` — lines of ``if`` statements in the code
    """
    report_path = build_bug_report_path(parsed, sdg=sdg, source_root=source_root)
    nodes = report_path.nodes

    # Collect boundary nodes: CONTROL, CALL, and REPORT (bug report point)
    boundaries: list[PathNode] = [
        n for n in nodes
        if n.node_type in (NodeType.CONTROL, NodeType.CALL, NodeType.REPORT)
    ]
    if not boundaries:
        return []

    segments: list[SegmentInfo] = []
    prev_event_no = 0
    prev_fn_name = ""
    prev_fn_file = ""
    prev_end_line = 0

    # Track the last known line number for each function context, keyed by
    # ``(file, function)``: a path may cross files and the same function name
    # can occur in several of them (e.g. inline/template functions in headers),
    # so a function-name-only key would splice the wrong line ranges together.
    # When execution returns to a caller after a callee completes,
    # we resume from the call site (stored here), not from the
    # function entry point.  This handles arbitrary call-depth nesting
    # because each function's last line is updated independently.
    last_line_in_fn: dict[tuple[str, str], int] = {}

    for seg_idx, bnode in enumerate(boundaries):
        start_event_no = (
            prev_event_no + 1 if prev_event_no > 0 else bnode.event_number
        )
        end_event_no = bnode.event_number
        end_line = bnode.line_number

        # Determine the FUNCTION context for this segment.
        current_fn = bnode.function_name
        current_file = bnode.source_file
        current_key = (current_file, current_fn)

        # Determine start_line — context-aware line range computation.
        # When execution returns from a callee back to a caller, we must
        # resume from the call site (the last line we were at in the
        # caller), *not* from the function entry.
        if seg_idx == 0:
            # First segment ever — begin at function entry.
            start_line = bnode.function_line_start or max(1, end_line - 5)
        elif current_key == (prev_fn_file, prev_fn_name) and prev_end_line > 0:
            # Same function as the previous boundary — continue from
            # where we left off (sequential execution).
            start_line = prev_end_line
        elif current_key in last_line_in_fn:
            # Function context changed, but we have been in *current_fn*
            # before — this is a **return** from a callee.  Resume from
            # the call site (the last-known line in the caller), NOT
            # from the function entry.
            start_line = last_line_in_fn[current_key]
        else:
            # First time entering this function (either via a call from
            # the caller, or the report's initial function).
            start_line = bnode.function_line_start or max(1, end_line - 5)

        # Record this boundary's line as the last-known position in the
        # current function, so that future returns land at the right spot.
        # Guard: skip when function_name is empty to avoid key collisions
        # across unrelated segments via the shared "" key.
        if current_fn:
            last_line_in_fn[current_key] = end_line

        # --- Build precondition facts ---
        prefacts: list[str] = []
        if seg_idx > 0:
            prev_b = boundaries[seg_idx - 1]
            if prev_b.node_type == NodeType.CONTROL:
                direction = "TRUE" if prev_b.branch_condition else "FALSE"
                prefacts.append(
                    f"[PREV-CTRL] Event #{prev_b.event_number} L{prev_b.line_number}: "
                    f"→ {direction} branch taken"
                )
            elif prev_b.node_type == NodeType.CALL:
                prefacts.append(
                    f"[PREV-CALL] Event #{prev_b.event_number} L{prev_b.line_number}: "
                    f"→ {prev_b.description}"
                )

        # Include "assuming" events that establish variable values
        for n in nodes:
            if n.event_number < start_event_no:
                desc = n.description.strip()
                if desc and "error start" not in desc.lower():
                    prefacts.append(f"Event #{n.event_number} L{n.line_number}: {desc}")
        prefacts = prefacts[-6:]

        # --- Extract source code from function entry to this boundary ---
        code = report_path.source_between(start_event_no, end_event_no)

        # --- Scan code for if-statements ---
        if_lines: list[int] = []
        for line in code.split("\n"):
            m = re.match(r".*?(\d+):\s+", line)
            if m:
                ln = int(m.group(1))
                if re.search(r"\bif\s*\(", line):
                    if_lines.append(ln)

        # --- Scan code for function calls in the SEGMENT's line range ---
        calls: set[str] = set()
        for n in nodes:
            if start_event_no <= n.event_number <= end_event_no:
                if n.node_type == NodeType.CALL:
                    m = re.search(r"'([^']+)'", n.description)
                    if m:
                        calls.add(m.group(1))
        for line in code.split("\n"):
            m = re.match(r".*?(\d+):\s+", line)
            if not m:
                continue
            ln = int(m.group(1))
            if ln < start_line or ln > end_line:
                continue
            for pat in (r'(?<![.\w])(\w+)\s*\(', r'(?<![.\w])(\w+)\s*\n\s*\('):
                for m2 in re.finditer(pat, line):
                    name = m2.group(1)
                    if name.lower() not in _KEYWORDS and not (name.isupper() and len(name) > 1):
                        calls.add(name)

        # Determine postcondition label
        if bnode.node_type == NodeType.CONTROL:
            direction = "TRUE" if bnode.branch_condition else "FALSE"
            post_label = f"branch={direction}"
        else:
            post_label = f"call={bnode.description[:50]}"

        segments.append(SegmentInfo(
            segment_index=seg_idx,
            label=f"L{start_line}-L{end_line} ({current_fn}) [{post_label}]",
            event_range=(start_event_no, end_event_no),
            line_start=start_line,
            line_end=end_line,
            precondition_facts=prefacts,
            control_event_number=end_event_no,
            control_branch_taken=bnode.branch_condition,
            function_name=current_fn,
            function_file=current_file,
            code_from_entry=code,
            if_line_numbers=if_lines,
            called_functions=sorted(set(calls)),
        ))

        prev_event_no = end_event_no
        prev_fn_name = current_fn
        prev_fn_file = current_file
        prev_end_line = end_line

    return segments


def _find_if_statements_in_range(
    source_snippets: dict[int, str],
    line_start: int,
    line_end: int,
) -> list[int]:
    """Find source lines with ``if`` statements in *line_start*..*line_end*.

    Uses simple substring matching on the source snippets.
    """
    if_lines: list[int] = []
    for ln in sorted(source_snippets.keys()):
        if line_start <= ln <= line_end:
            code = source_snippets.get(ln, "")
            # Look for lines starting with or containing 'if ('
            if re.search(r"\bif\s*\(", code):
                if_lines.append(ln)
    return if_lines


def _find_calls_in_range(
    events: list[dict],
    start_event_no: int,
    end_event_no: int,
) -> list[str]:
    """Collect called function names from events in the event-number range."""
    calls: list[str] = []
    for evt in events:
        en = evt.get("path_number", 0)
        if start_event_no <= en <= end_event_no:
            if evt.get("is_calling_event"):
                fn = evt.get("called_function", "")
                if fn:
                    calls.append(fn)
    return calls


# ---------------------------------------------------------------------------
# Function identification for CFG building
# ---------------------------------------------------------------------------


def identify_relevant_functions(
    segments: list[SegmentInfo],
    parsed: ParsedReport,
) -> dict[str, str]:
    """Identify functions that need CFG analysis.

    Returns ``{function_name: source_file_or_hint}``.

    Includes:
    - The buggy function from the report metadata
    - Any opaque function called within a segment that has if-statements
    """
    functions: dict[str, str] = {}
    metadata = parsed.metadata

    # Start with the buggy function
    if metadata.function_name:
        functions[metadata.function_name] = metadata.bug_file

    # Add opaque functions from segments with if-statements
    for seg in segments:
        for fn_name in seg.function_calls:
            if fn_name not in functions:
                functions[fn_name] = ""

    return functions


# ---------------------------------------------------------------------------
# CFG-to-segment matching
# ---------------------------------------------------------------------------


def match_cfg_to_segment(
    cfg: CFGFunction,
    segment: SegmentInfo,
    all_variants: list[CFGPathVariant],
) -> SegmentPathAnalysis:
    """Match a parsed CFG and its enumerated paths to a single segment.

    Returns a ``SegmentPathAnalysis`` with the original path variant
    and all alternative variants for this segment.
    """
    if not all_variants:
        return SegmentPathAnalysis(
            segment_index=segment.segment_index,
            segment_label=segment.label,
            function_name=cfg.function_name,
        )

    original = all_variants[0]
    original.is_original_path = True

    analysis = SegmentPathAnalysis(
        segment_index=segment.segment_index,
        segment_label=segment.label,
        function_name=cfg.function_name,
        original_path_variant=original,
        alternative_variants=all_variants[1:] if len(all_variants) > 1 else [],
    )

    return analysis


# ---------------------------------------------------------------------------
# Source code extraction for a segment
# ---------------------------------------------------------------------------


def _extract_segment_source(
    parsed: ParsedReport,
    segment: SegmentInfo,
    source_root: str | None = None,
) -> str:
    """Extract source code for a segment using ``BugReportPath.source_between()``.

    Code is anchored to the CONTAINING FUNCTION's entry line (not a heuristic
    offset), giving the LLM full context of the function's control flow.
    Called-function bodies are appended for opaque functions in the path.
    """
    # Build the structured path to get precise function bounds
    report_path = build_bug_report_path(parsed)
    event_from = segment.event_range[0]
    event_to = segment.event_range[1]

    container_code = report_path.source_between(event_from, event_to)

    result: list[str] = ["--- Container function code ---"]
    if ">>>" in container_code:
        result.append(container_code)
    else:
        result.append(f"(code from function entry to L{segment.line_end})")
        result.append(container_code)

    # Collect all function names to look up:
    all_funcs: set[str] = set(segment.function_calls)
    if parsed.metadata.function_name:
        all_funcs.add(parsed.metadata.function_name)

    # Scan container code for function calls (same-line and multi-line)
    for pattern in (
        r'(?<![.\w])(\w+)\s*\(',          # func(args)
        r'(?<![.\w])(\w+)\s*\n\s*\(',     # func\n(args)
    ):
        for m in re.finditer(pattern, container_code):
            name = m.group(1)
            if name.lower() in _KEYWORDS:
                continue
            if name.isupper() and len(name) > 1:
                continue
            all_funcs.add(name)

    # Append function bodies for opaque calls
    for fn_name in sorted(all_funcs):
        body = _find_function_body(fn_name, parsed, source_root)
        if body:
            result.append("")
            result.append(f"--- Called function body: {fn_name} ---")
            result.append(body)

    return "\n".join(result)


def _find_function_body(
    fn_name: str,
    parsed: ParsedReport,
    source_root: str | None = None,
) -> str | None:
    """Find and extract a C/C++ function body from the project source tree."""
    from llm_client.fp_analysis.source_context import SourceContextExtractor

    search_roots: list[Path] = []
    if source_root:
        p = Path(source_root).resolve()
        if p.exists():
            search_roots.append(p)
    local_project = Path("project").resolve()
    if local_project.exists():
        search_roots.append(local_project)

    if not search_roots:
        return None

    extractor = SourceContextExtractor(project_root=search_roots[0])

    # Priority: search from the bug file's own directory first.
    # Called functions are in the same or adjacent source files.
    bug_file = _resolve_source_file_for_cfg(parsed, source_root)
    if bug_file:
        bug_dir = bug_file.parent
        # Direct search in bug-file directory
        fpath, body = extractor.find_function_definition(fn_name, search_roots=[bug_dir])
        if fpath and body:
            return body
        # Also try parent/lib/, parent/src/ (wslay pattern: deps/wslay/lib/)
        for sub in bug_dir.glob("*/lib"):
            fpath, body = extractor.find_function_definition(fn_name, search_roots=[sub])
            if fpath and body:
                return body

    # Fallback: full search
    fpath, body = extractor.find_function_definition(fn_name, search_roots=search_roots)
    if fpath and "/test" not in fpath.lower() and "/example" not in fpath.lower():
        return body
    return body


def _format_precondition(
    parsed: ParsedReport, segment: SegmentInfo
) -> str:
    """Format the precondition (state before this segment) for the LLM prompt.

    For non-first segments, the precondition includes the PREVIOUS segment's
    postcondition (the branch outcome CSA chose at the prior control event).
    This is critical: it carries forward the established state.
    """
    events = parsed.path_events
    start_no = segment.event_range[0]
    seg_idx = segment.segment_index

    lines: list[str] = []

    # 1. Include the previous segment's control event outcome.
    #    Use < end_no (not < start_no) because segments may share the
    #    same start_event_no (first segment ends at #2, second starts at #2).
    end_no = segment.event_range[1]
    prev_ctrl_events = [
        e for e in events
        if e.get("path_number", 0) < end_no
        and e.get("is_control_flow")
        and e.get("branch_condition") in (True, False)
    ]
    if prev_ctrl_events:
        last = prev_ctrl_events[-1]
        taken = "TRUE" if last.get("branch_condition") else "FALSE"
        lines.append(
            f"  [PREV] Event #{last['path_number']} L{last.get('line_number',0)}: "
            f"→ {taken} branch taken"
        )

    # 2. Include the "assuming" events that establish variable values.
    #    Use < start_no (not < end_no) to avoid duplicating the segment's
    #    own first event if it shares start_event_no with the prev segment.
    before_events = [
        e for e in events
        if e.get("path_number", 0) < start_no
    ]
    for evt in before_events[-5:]:  # Last 5 events
        ln = evt.get("line_number", 0)
        desc = evt.get("description", "")
        if desc:
            lines.append(f"  Event #{evt['path_number']} L{ln}: {desc}")

    if not lines:
        return "Function entry / no prior events."

    return "\n".join(lines)


def _format_path_variants(variants: list[CFGPathVariant]) -> str:
    """Format enumerated path variants for the LLM prompt."""
    parts: list[str] = []
    for v in variants:
        tag = "ORIGINAL (CSA)" if v.is_original_path else "ALTERNATIVE"
        decisions_str = "; ".join(
            f"{d.block_id}: {'TRUE' if d.taken_is_true_branch else 'FALSE'} branch"
            for d in v.branch_decisions
        ) or "no branches (linear path)"
        parts.append(f"  Variant #{v.variant_id} [{tag}]: {decisions_str}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers: function-line-range matching (segment → container function)
# ---------------------------------------------------------------------------


def _build_function_line_ranges(source_file: Path) -> dict[str, tuple[int, int]]:
    """Scan *source_file* for C/C++ function definitions and return their
    line ranges ``{function_name: (start_line, end_line)}``.

    Uses a simple brace-matching approach — enough to find which function
    contains a given line number.
    """
    try:
        lines = source_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}

    ranges: dict[str, tuple[int, int]] = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for lines that look like function definitions
        # (contain '(' and ')' and don't start with #, //, /*)
        if (
            line
            and not line.startswith(("#", "//", "/*", "*"))
            and "(" in line
            and ")" in line
            and ";" not in line.rstrip(" {")
            and not line.startswith(("if ", "while ", "for ", "switch ", "case ", "return ", "else ", "typedef ", "enum ", "struct ", "union "))
        ):
            # Check if next non-blank non-comment line has '{'
            depth = 0
            for j in range(i, len(lines)):
                l = lines[j].strip()
                if l.startswith("//") or l.startswith("/*") or l.startswith("*"):
                    continue
                for ch in l:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            func_name = line.rstrip("{").strip()
                            # Extract just the function name (last word before '(')
                            name_match = re.search(r"(\w+)\s*\(", func_name)
                            if name_match:
                                short = name_match.group(1)
                                ranges[short] = (i + 1, j + 1)
                                i = j
                            break
                if depth == 0:
                    break
        i += 1
    return ranges


def _find_container_function(
    line_number: int,
    func_ranges: dict[str, tuple[int, int]],
    cfg_cache: dict[str, object],
) -> str | None:
    """Find which function in *cfg_cache* contains *line_number*.

    Matches by checking if the line falls within a function's source range,
    then finds the corresponding entry in the parsed CFG cache.
    """
    # Find the narrowest function range containing this line
    candidates: list[tuple[str, int]] = []
    for short_name, (start, end) in func_ranges.items():
        if start <= line_number <= end:
            # Find full name in cfg_cache
            for full_name in cfg_cache:
                if short_name in full_name.split("(")[0]:
                    candidates.append((full_name, end - start))
    if not candidates:
        return None
    # Return the smallest (most specific) function
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def _find_best_container(
    cfg_cache: dict[str, object],
    parsed: ParsedReport,
) -> str | None:
    """Find the best container function from *cfg_cache* for the report's path.

    Strategy (in order):
    1. If the bug report's *function_name* matches a CFG entry, use that.
    2. Otherwise, pick the CFG function with the most blocks that also
       appears in any path event's called_function list.
    3. Fallback: the function with the most blocks overall.
    """
    if not cfg_cache:
        return None

    # 1. Exact match by bug report's function_name
    fn_name = parsed.metadata.function_name or ""
    if fn_name:
        for full_name in cfg_cache:
            if fn_name in full_name:
                return full_name
            base = _extract_func_basename(full_name)
            if fn_name == base or base in fn_name:
                return full_name

    # 2. Match by called_function from path events
    called: set[str] = set()
    for evt in parsed.path_events:
        cf = evt.get("called_function", "")
        if cf:
            called.add(cf)

    candidates: list[tuple[str, int]] = []
    for full_name in cfg_cache:
        base = _extract_func_basename(full_name)
        for c in called:
            if c in full_name or c == base:
                candidates.append((full_name, len(cfg_cache[full_name].blocks)))
                break

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    # 3. Fallback: function with most blocks
    best = max(cfg_cache.items(), key=lambda kv: len(kv[1].blocks))
    return best[0]


def _extract_func_basename(full_name: str) -> str:
    """Extract a short function name from a full CFG function header."""
    # Strip return type and qualifiers
    name = re.sub(r"^(static\s+|inline\s+)?[\w:*&]+\s+", "", full_name)
    # Remove parentheses and everything after
    name = name.split("(")[0].strip()
    # Strip trailing *
    name = name.rstrip("* ")
    return name


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_feasibility_analysis(
    parsed: ParsedReport,
    source_root: str | None = None,
    project_name: str = "default",
    max_variants: int = 16,
) -> CFGAnalysisResult | None:
    """Run CFG-based feasibility analysis on a parsed CSA report.

    On-demand strategy:
    1. Segment the path by control events
    2. For each segment, identify ONLY the functions called in that segment
    3. Build CFG once for the source file, cache parsed functions
    4. For each segment, enumerate paths ONLY for its relevant functions
    5. Match variants to segments and return

    Returns ``None`` if CFG analysis could not be performed.
    """
    # Step 1: Segment the path
    segments = segment_path(parsed)
    if not segments:
        logger.info("No control events found; segmentation yields 0 segments.")
        return None
    logger.info("Segmented path into %d segment(s).", len(segments))

    # Resolve the source file
    bug_file = _resolve_source_file_for_cfg(parsed, source_root)
    if not bug_file:
        logger.warning("Cannot resolve source file for CFG dump.")
        return None

    # Derive project name from the resolved file path
    if project_name in ("default", "poc_generation"):
        bug_file_str = str(bug_file)
        if "project/" in bug_file_str:
            parts = bug_file_str.split("project/", 1)
            if len(parts) > 1:
                project_name = parts[1].split("/", 1)[0]

    # Step 2: Build CFG once per source file, cached persistently.
    #          Check cache first; only compile if no valid cache exists.
    cache_path = _cfg_cache_path(str(bug_file), project_name)
    cfg_cache = _load_cfg_cache(cache_path)
    if cfg_cache is not None:
        logger.info(
            "CFG cache hit: %s (%d functions)", cache_path.name, len(cfg_cache),
        )
    else:
        logger.info("CFG cache miss; compiling %s ...", bug_file.name)
        cmd = build_cfg_command(str(bug_file), project_name, source_root)
        raw_text = run_cfg_dump(cmd)
        if raw_text is None:
            logger.warning("CFG dump failed; skipping CFG analysis.")
            return None

        cfg_cache = parse_cfg_text(raw_text)
        if not cfg_cache:
            logger.warning("CFG parsing produced zero functions.")
            return None
        logger.info("CFG parsed: %d function(s).", len(cfg_cache))
        _save_cfg_cache(cache_path, cfg_cache)

    result = CFGAnalysisResult(cfg_functions=cfg_cache)

    # Step 3: Identify the CONTAINER function — the function in the
    #          source file where the CSA path's control events occur.
    #          The CFG contains all functions; we pick the one whose
    #          name best matches the bug context and has the most
    #          blocks (heuristic: main function = most control flow).
    container_func = _find_best_container(cfg_cache, parsed)
    if not container_func:
        logger.warning("Cannot identify container function for CFG analysis.")
        return None

    cfg = cfg_cache[container_func]
    logger.info("Container function: '%s' (%d blocks)",
                 container_func.split("(")[0].strip()[:60], len(cfg.blocks))

    # Step 4: Enumerate path variants for the container function once.
    variants = enumerate_paths(cfg, max_variants=max_variants)
    logger.info("Container function: %d path variant(s) enumerated.", len(variants))

    # Step 5: Match the same variant set to each segment.
    #         (All segments' control events are within this function.)
    for seg_idx, segment in enumerate(segments):
        seg_analysis = match_cfg_to_segment(cfg, segment, variants)
        result.segments.append(seg_analysis)

    return result


def _resolve_source_file_for_cfg(
    parsed: ParsedReport,
    source_root: str | None = None,
) -> Path | None:
    """Resolve the source file path for CFG dumping.

    Uses the same resolution strategy as ``SourceContextExtractor``
    but returns a ``Path`` directly.
    """
    from llm_client.fp_analysis.source_context import (
        _KNOWN_PATH_PREFIXES,
    )

    bug_file = parsed.metadata.bug_file
    path = Path(bug_file)

    # Direct existence
    if path.exists():
        return path.resolve()

    # Strip known prefixes
    for prefix in _KNOWN_PATH_PREFIXES:
        if bug_file.startswith(prefix):
            rel = bug_file.removeprefix(prefix).lstrip("/")
            if source_root:
                candidate = Path(source_root) / rel
                if candidate.exists():
                    return candidate.resolve()

    # Suffix matching under source_root — the report's absolute path is from a
    # different build machine, so try each suffix (e.g. ``folly/Format.cpp``)
    # against source_root (mirrors ``_resolve_source_path`` in path_analyzer).
    if source_root:
        parts = Path(bug_file).parts
        for i in range(1, len(parts)):
            cand = Path(source_root).joinpath(*parts[i:])
            if cand.exists():
                return cand.resolve()

    # Search in project/ directory by filename
    local_project = Path("project").resolve()
    if local_project.exists():
        for found in local_project.rglob(path.name):
            return found.resolve()

    return None


# ---------------------------------------------------------------------------
# Multi-file event-to-function resolution helpers
# ---------------------------------------------------------------------------


def _resolve_local_path(p: Path | str, source_root: str | None) -> Path | None:
    """Resolve a source path to a *local existing* file.

    The CSA report's file paths come from a different build machine and may
    not exist here; the PDG ``source_file`` is the (possibly non-existent)
    build path too.  We suffix-match under *source_root* (mirrors
    ``_resolve_source_path`` in path_analyzer) so the resulting local path is
    keyed consistently with ``build_bug_report_path``'s ``all_files``.
    Returns ``None`` when no local file can be found.
    """
    path = Path(p)
    if path.exists():
        return path.resolve()
    if source_root:
        parts = path.parts
        for i in range(1, len(parts)):
            cand = Path(source_root).joinpath(*parts[i:])
            if cand.exists():
                return cand.resolve()
    return None


def _build_pdg_fn_index(
    sdg: "SDG | None",
    source_root: str | None,
) -> dict[str, list[tuple[str, int, int]]]:
    """Build ``{local_source_path_str: [(fn_name, start, end), ...]}`` from the SDG.

    Each function's source range is taken from the min/max ``source_line`` of
    its PDG nodes — this is *accurate* (unlike the brace-matching heuristic in
    ``_find_function_boundaries``, which misses functions whose body is nested
    inside another's braces, e.g. ``SocketAddress::setFromLocalAddr``).  Keys
    are resolved to local existing files under *source_root* so they line up
    with ``all_files`` / ``_resolve_local_path``.  The function name is the
    unqualified base (``"MyClass::method"`` → ``"method"``), matching the
    downstream CFG-cache / branch-extractor lookup key convention.
    """
    index: dict[str, list[tuple[str, int, int]]] = {}
    if sdg is None:
        return index
    for pdg in sdg.pdgs.values():
        lines = [
            n.source_line for n in pdg.nodes.values()
            if n.source_line and n.source_line > 0
        ]
        if not lines:
            continue
        local = _resolve_local_path(pdg.source_file, source_root)
        if local is None:
            continue
        base_name = pdg.function_name.split("::")[-1]
        if not base_name:
            continue
        index.setdefault(str(local), []).append(
            (base_name, min(lines), max(lines)),
        )
    return index


def _build_class_to_file_map(
    events: list[dict],
    source_root: str | None = None,
    bug_file: Path | None = None,
) -> dict[str, Path]:
    """Build ``{ClassName: source_path}`` from CALL event descriptions.

    Extracts class names from patterns like ``"Calling 'ClassName::method'"``
    and searches for ``ClassName.cc`` / ``.cpp`` files in the project source
    tree.  Used as a fallback when the SDG is unavailable.
    """
    # Determine search directories
    search_dirs: list[Path] = []
    if bug_file and bug_file.exists():
        search_dirs.append(bug_file.parent)
    if source_root:
        sr = Path(source_root)
        if sr.exists():
            search_dirs.append(sr / "src")
            search_dirs.append(sr)
    local = Path("project")
    if local.exists():
        search_dirs.append(local)

    class_to_file: dict[str, Path] = {}
    seen: set[str] = set()

    for evt in events:
        desc = evt.get("description", "") or ""
        m = re.search(r"(?:Calling|← Calling)\s+'(\w+)::", desc)
        if m:
            cls = m.group(1)
            if cls not in seen:
                seen.add(cls)
                for d in search_dirs:
                    for ext in (".cc", ".cpp", ".c"):
                        cf = d / f"{cls}{ext}"
                        if cf.exists():
                            class_to_file[cls] = cf.resolve()
                            break
                    if cls in class_to_file:
                        break

    return class_to_file


def _resolve_event_file(
    line: int,
    desc: str,
    class_to_file: dict[str, Path],
    sdg: "SDG | None",
    bug_file: Path | None,
    report_files: set[str] | None = None,
    source_root: str | None = None,
) -> Path | None:
    """Determine which source file an event belongs to.

    Priority:
    1. CALL event class-name context (most reliable)
    2. SDG line-index lookup, restricted to files the report itself mentions
       (``report_files``) — without that restriction a bare line-number match
       is wildly ambiguous (line 72 exists in 55 different faiss files) and
       every cross-file event silently collapsed onto ``bug_file``.
    3. Last-resort: the first surviving candidate (with a warning), because
       guessing one plausible file beats asserting the bug file is the
       caller's file.  ``bug_file`` is used only when nothing matched at all.
    """
    # 1. CALL event class-name context
    m = re.search(r"(?:Calling|← Calling)\s+'(\w+)::", desc)
    if m:
        cls = m.group(1)
        if cls in class_to_file:
            return class_to_file[cls]

    # 2. SDG line-index lookup
    if sdg is not None and line > 0:
        line_index = sdg.build_line_index()
        matches: list[str] = []
        for (sf, ln) in line_index:
            if ln == line:
                matches.append(sf)
        if matches and report_files:
            in_report = [
                sf for sf in matches
                if str(_resolve_local_path(sf, source_root) or Path(sf))
                in report_files
            ]
            if in_report:
                matches = in_report
        if len(matches) == 1:
            return Path(matches[0])
        if len(matches) > 1:
            # Disambiguate via class name extracted from description
            dm = re.search(r"(?:Calling|← Calling)\s+'(\w+)::", desc)
            dcls = dm.group(1) if dm else None
            if dcls:
                for sf in matches:
                    fname = Path(sf).stem  # e.g. "DefaultPieceStorage"
                    if fname == dcls:
                        return Path(sf)
            logger.warning(
                "event L%d ('%s') matches %d report files — picking '%s'",
                line, desc[:50], len(matches), matches[0],
            )
            return Path(matches[0])

    # 3. Nothing matched: the bug file is the only defensible answer.
    return bug_file
