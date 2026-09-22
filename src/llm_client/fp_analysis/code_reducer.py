"""Code reducer — turns a backward-slice result into reduced C++ source code.

The reducer takes a :class:`SliceResult` (from the backward slicer) and
produces a minimal C++ source file that preserves only the statements in
the slice, arranged in forward program order with proper scaffolding.

This reduced code can then be fed into the existing feasibility pipeline
(CFG dump, LLM constraint completion) for more precise analysis.

Only ``clean`` mode is supported: slice lines are emitted alongside
structural anchors (braces, control-flow headers, case labels, preprocessor
directives) whose block contains slice lines, producing valid C++ with
correct indentation.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

from llm_client.fp_analysis.pdg_models import SliceResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Include inference
# ---------------------------------------------------------------------------

_INFERRED_INCLUDES: dict[str, str] = {
    "string": "#include <string>",
    "std::string": "#include <string>",
    "vector": "#include <vector>",
    "std::vector": "#include <vector>",
    "map": "#include <map>",
    "std::map": "#include <map>",
    "set": "#include <set>",
    "std::set": "#include <set>",
    "unique_ptr": "#include <memory>",
    "std::unique_ptr": "#include <memory>",
    "shared_ptr": "#include <memory>",
    "std::shared_ptr": "#include <memory>",
    "optional": "#include <optional>",
    "std::optional": "#include <optional>",
    "size_t": "#include <cstddef>",
    "uint8_t": "#include <cstdint>",
    "uint16_t": "#include <cstdint>",
    "uint32_t": "#include <cstdint>",
    "uint64_t": "#include <cstdint>",
    "int8_t": "#include <cstdint>",
    "int16_t": "#include <cstdint>",
    "int32_t": "#include <cstdint>",
    "int64_t": "#include <cstdint>",
    "assert": "#include <cassert>",
    "memset": "#include <cstring>",
    "memcpy": "#include <cstring>",
    "memcmp": "#include <cstring>",
    "malloc": "#include <cstdlib>",
    "free": "#include <cstdlib>",
    "calloc": "#include <cstdlib>",
    "realloc": "#include <cstdlib>",
    "printf": "#include <cstdio>",
    "scanf": "#include <cstdio>",
}

# Keywords that shouldn't trigger include inference
_C_KEYWORDS = frozenset({
    "if", "while", "for", "switch", "case", "return", "else", "break",
    "continue", "do", "try", "catch", "throw", "new", "delete", "class",
    "struct", "enum", "union", "template", "typename", "namespace",
    "using", "const", "static", "extern", "volatile", "inline",
    "virtual", "override", "explicit", "mutable", "friend",
    "int", "char", "float", "double", "bool", "void", "long", "short",
    "unsigned", "signed", "constexpr", "noexcept", "nullptr", "true",
    "false", "auto",
})

# Regex for stripping assignment LHS when the LHS variable was filtered out
# by LLM variable relevance.  Matches "op = findById(lopt)" → "findById(lopt)"
_ASSIGN_STRIP_RE = re.compile(r'^(\s*)\w[\w\s]*\*?\s*=\s*(\S.*)$')


def _infer_includes(expressions: list[str], types: list[str]) -> list[str]:
    """Infer required C++ headers from expression text and type names."""
    needed: set[str] = set()
    for expr in expressions:
        for pattern, include in _INFERRED_INCLUDES.items():
            if pattern in expr:
                needed.add(include)
    return sorted(needed) if needed else ["#include <cstddef>", "#include <cstdio>"]


# ---------------------------------------------------------------------------
# Source file resolution
# ---------------------------------------------------------------------------

_PP_RE = re.compile(r'^\s*#')


def _is_pp(line: str) -> bool:
    return bool(_PP_RE.match(line))


def _resolve_source(source_file: str, search_roots: list[Path]) -> Path | None:
    path = Path(source_file)
    if path.exists():
        return path
    for root in search_roots:
        for found in root.rglob(path.name):
            if found.exists():
                return found
    return None


def _read_lines(source_file: str, search_roots: list[Path]) -> list[str] | None:
    path = _resolve_source(source_file, search_roots)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Function boundary detection
#   All indices returned are 0-based (file line index).
#   Input anchor is 1-based (from the PDG source_line).
# ---------------------------------------------------------------------------

_CONTROL_KW = frozenset({
    "if ", "while ", "for ", "switch ", "case ", "return ",
    "else ", "catch ", "try ", "throw ", "break ", "continue ",
})


def _is_fn_decl(line: str, short_name: str) -> bool:
    """True if *line* is a C++ function definition for *short_name*.

    Handles multi-line parameter lists (e.g., ``void foo(int a,\n    int b)``)
    by only requiring ``(`` on the current line; ``)`` may be on a later line.
    """
    s = line.strip()
    if not s:
        return False
    if s.startswith(("#", "//", "/*", "*")):
        return False
    if any(s.startswith(k) for k in _CONTROL_KW):
        return False
    # Must contain the function name
    if not re.search(r'\b' + re.escape(short_name) + r'\b', s):
        return False
    # Must start a parameter list (the function name must be followed by ``(``
    # on this line — the closing ``)`` may be on a later line for long sigs)
    if "(" not in s:
        return False
    # Must not be a declaration (ending in ``;``)
    s_noc = s.split("//")[0].rstrip()
    if s_noc.endswith(";"):
        return False
    return True


def _find_fn_decl(
    source_lines: list[str],
    short_name: str,
    anchor_0based: int,
    search_radius: int = 60,
) -> int | None:
    """Return 0-based index of function definition for *short_name*.

    Searches backward then forward from *anchor_0based* within
    *search_radius* lines.  Falls back to full-file scan.
    """
    lo = max(0, anchor_0based - search_radius)
    hi = min(len(source_lines), anchor_0based + search_radius)

    # Backward
    for idx in range(anchor_0based, lo - 1, -1):
        if _is_fn_decl(source_lines[idx], short_name):
            return idx
    # Forward
    for idx in range(anchor_0based + 1, hi):
        if _is_fn_decl(source_lines[idx], short_name):
            return idx
    # Full file fallback
    for idx, line in enumerate(source_lines):
        if _is_fn_decl(line, short_name):
            return idx
    return None


def _find_fn_bounds(
    source_lines: list[str],
    short_name: str,
    anchor_1based: int,
) -> tuple[int, int] | None:
    """Return ``(start_0based, end_0based)`` for function containing
    the line *anchor_1based* (1-based from the caller).  Returns None
    on failure.

    ``end_0based`` points to the closing ``}``.
    """
    anchor_0based = anchor_1based - 1  # convert to 0-based

    # Step 1: find the function declaration line
    decl_idx = _find_fn_decl(source_lines, short_name, anchor_0based)
    if decl_idx is None:
        return None

    # Step 2: find the "{" that opens the body
    body_open = None
    for idx in range(decl_idx, min(decl_idx + 10, len(source_lines))):
        if "{" in source_lines[idx]:
            body_open = idx
            break
    if body_open is None:
        return None

    # Step 3: match braces
    depth = 0
    for idx in range(body_open, len(source_lines)):
        depth += source_lines[idx].count("{") - source_lines[idx].count("}")
        if depth <= 0 and idx > body_open:
            return (decl_idx, idx)

    return (decl_idx, len(source_lines) - 1)


# ---------------------------------------------------------------------------
# Function signature extraction
# ---------------------------------------------------------------------------


def _extract_signature(
    fn_name: str,
    source_lines: list[str],
    anchor_1based: int,
    short_name: str,
) -> tuple[str, int]:
    """Return ``(signature_string, line_count)`` for *fn_name*.

    *line_count* is the number of source lines consumed by the signature
    (from the declaration to the line containing ``{``, inclusive), so
    callers can skip these lines when emitting the function body.

    Falls back to ``("void <name>()", 1)`` on failure.
    """
    anchor_0based = anchor_1based - 1
    decl_idx = _find_fn_decl(source_lines, short_name, anchor_0based)
    if decl_idx is None:
        return f"void {short_name}()", 1

    # Collect signature lines, including any preceding ``template<...>`` line
    sig_parts: list[str] = []
    if decl_idx > 0 and source_lines[decl_idx - 1].strip().startswith("template"):
        sig_parts.append(source_lines[decl_idx - 1])
    sig_parts.append(source_lines[decl_idx])
    for idx in range(decl_idx + 1, min(decl_idx + 12, len(source_lines))):
        line = source_lines[idx]
        sig_parts.append(line)
        if "{" in line:
            break

    sig = "\n".join(sig_parts)
    sig = sig.rstrip().rstrip("{").strip()
    return sig, len(sig_parts)


# ---------------------------------------------------------------------------
# Helpers for structural line detection
# ---------------------------------------------------------------------------

# Preprocessor directive types
_PP_IF = frozenset({"#if", "#ifdef", "#ifndef"})
_PP_ELSE = frozenset({"#else", "#elif"})


def _find_matching_endif(
    source_lines: list[str], if_idx: int, limit: int,
) -> int | None:
    depth = 1
    for i in range(if_idx + 1, min(limit, len(source_lines))):
        s = source_lines[i].strip()
        if any(s.startswith(x) for x in _PP_IF):
            depth += 1
        elif s.startswith("#endif"):
            depth -= 1
            if depth == 0:
                return i
    return None


def _pp_block_has_slice(
    source_lines: list[str],
    if_idx: int,
    end_idx: int,
    slice_0based: set[int],
) -> bool:
    """Check if any line between *if_idx* and its matching #endif is in slice."""
    endif = _find_matching_endif(source_lines, if_idx, end_idx)
    upper = endif if endif is not None else end_idx
    return any(ln for ln in range(if_idx + 1, upper) if ln in slice_0based)


def _is_ctrl_header_line(stripped: str) -> bool:
    """True if the line starts a control flow construct (if/for/while/switch/else)."""
    s = stripped.lstrip()
    # Handle "} else {" / "} else if (...)" — strip leading '}' first
    s_clean = s.lstrip("}")
    if s_clean.startswith("else"):
        return True
    # Handle "if(", "if (", "for(", "for (", "while(", "switch(", etc.
    for kw in ("if", "for", "while", "switch", "do", "try", "catch"):
        if s_clean.startswith(kw) and (
            len(s_clean) == len(kw)
            or not (s_clean[len(kw)].isalnum() or s_clean[len(kw)] == "_")
        ):
            return True
    return False


def _find_block_end(
    source_lines: list[str], header_lineno: int, fn_end: int,
) -> int | None:
    """Find the last line of the block controlled by the construct at *header_lineno*.

    Returns the line index (0-based) of the closing ``}``, or None on failure.
    """
    # Scan forward to find the opening '{'
    open_brace = None
    for i in range(header_lineno, min(header_lineno + 20, fn_end)):
        line = source_lines[i] if i < len(source_lines) else ""
        if "{" in line:
            open_brace = i
            break
    if open_brace is None:
        return None

    # Brace-match from open_brace to find closing '}'
    depth = 0
    for i in range(open_brace, fn_end):
        line = source_lines[i] if i < len(source_lines) else ""
        depth += line.count("{") - line.count("}")
        if depth <= 0 and i > open_brace:
            return i
    return None


def _block_contains_slice(
    source_lines: list[str], header_lineno: int, fn_end: int,
    slice_0based: set[int],
) -> bool:
    """Check if the block following *header_lineno* contains any slice lines."""
    block_end = _find_block_end(source_lines, header_lineno, fn_end)
    if block_end is None:
        return False
    return any(ln for ln in range(header_lineno + 1, block_end)
               if ln in slice_0based)


def _is_case_label(stripped: str) -> bool:
    s = stripped.lstrip()
    return s.startswith("case ") or s.startswith("default:")


def _next_case_or_break(
    source_lines: list[str], label_lineno: int, fn_end: int,
) -> int | None:
    """Find the next ``case``, ``default:``, or ``}`` after *label_lineno*."""
    for i in range(label_lineno + 1, fn_end):
        s = source_lines[i].strip() if i < len(source_lines) else ""
        if _is_case_label(s) or s == "}":
            return i
    return None


def _case_label_has_slice(
    source_lines: list[str], label_lineno: int, fn_end: int,
    slice_0based: set[int],
) -> bool:
    """Check if any slice line exists in this case's body."""
    next_end = _next_case_or_break(source_lines, label_lineno, fn_end)
    upper = next_end if next_end is not None else fn_end
    return any(ln for ln in range(label_lineno + 1, upper)
               if ln in slice_0based)



# ---------------------------------------------------------------------------
# Mode: clean  —  slice lines only + structural anchors, with proper indent
# ---------------------------------------------------------------------------


_CONTINUATION_END = re.compile(r'(?:=\s*|\(\s*|,\s*|&&\s*|\|\|\s*|\\\s*)$')


def _needs_continuation(line_stripped: str) -> bool:
    """Check if a line looks syntactically incomplete (needs next line).

    Handles:
    - ``size_t x =``  (assignment with RHS on next line)
    - ``func(``       (arguments on next line)
    - ``item,``       (comma-separated list continues)
    """
    s = line_stripped.rstrip()
    if s.endswith(";"):
        return False
    if s.endswith("{") or s.endswith("}"):
        return False
    # Remove comments before checking
    s_nocomment = s.split("//")[0].rstrip()
    if not s_nocomment:
        return False
    return bool(_CONTINUATION_END.search(s_nocomment))


def _emit_clean(
    fn_name: str,
    short_name: str,
    source_file: str,
    slice_lines_1based: set[int],
    source_all: list[str],
    fn_bounds: tuple[int, int] | None,
    search_roots: list[Path],
    trigger_line_0based: int | None = None,
    strip_assignment_lines: set[int] | None = None,
    preserve_lines: set[int] | None = None,
) -> list[str]:
    """Emits slice lines + structural anchors for valid C++ with proper indent.

    - Only emits slice lines (non-slice lines are removed).
    - KEEPS structural elements whose block contains slice lines:
      control-flow headers (``if``/``for``/``while``/``switch``/``else``),
      ``case``/``default`` labels, ``{`` and ``}`` for balance,
      and preprocessor directives guarding slice lines.
    - Tracks brace depth from ALL source lines (``real_depth``) for indentation,
      and from EMITTED lines only (``emitted_open_depth``) for ``}`` emission
      decisions — orphaned ``}`` matching a skipped ``{`` are naturally suppressed.
    """
    out: list[str] = []
    slice_0based = {ln - 1 for ln in slice_lines_1based if ln > 0}
    anchor_1based = min(slice_lines_1based) if slice_lines_1based else 0

    # ---- Find function body bounds (independent brace matching) ----
    decl_idx = _find_fn_decl(source_all, short_name, anchor_1based - 1)
    if decl_idx is None:
        # Fallback: just raw emit with missing braces
        out.append(f"// --- Function: {fn_name} ---")
        sig, _ = _extract_signature(fn_name, source_all, anchor_1based, short_name)
        out.append(sig + " {")
        for lineno in sorted(slice_0based):
            out.append(f"    {source_all[lineno].strip()}")
        out.append("}")
        return out

    brace_open = None
    for i in range(decl_idx, min(decl_idx + 15, len(source_all))):
        if "{" in source_all[i]:
            brace_open = i
            break
    if brace_open is None:
        out.append(f"// --- Function: {fn_name} ---")
        sig, _ = _extract_signature(fn_name, source_all, anchor_1based, short_name)
        out.append(sig + " {")
        for lineno in sorted(slice_0based):
            out.append(f"    {source_all[lineno].strip()}")
        out.append("}")
        return out

    real_depth = 0
    body_end = None
    for i in range(brace_open, len(source_all)):
        real_depth += source_all[i].count("{") - source_all[i].count("}")
        if real_depth <= 0:
            body_end = i
            break
    if body_end is None:
        body_end = len(source_all) - 1

    body_start = brace_open + 1  # line after the opening ``{`` (0-based)

    # ---- Emit signature ----
    sig, _ = _extract_signature(fn_name, source_all, anchor_1based, short_name)
    out.append(f"// --- Function: {fn_name} ---")
    out.append(sig + " {")

    # ---- Emit body lines ----
    emitted: set[int] = set()
    real_depth = 1                 # from the function's emitted ``{``
    emitted_open_depth = 1         # tracks only braces from emitted lines

    # Post-trigger truncation: if this is the trigger function, stop at trigger line
    loop_end = body_end
    if trigger_line_0based is not None:
        loop_end = min(body_end, trigger_line_0based)

    skip_until = -1  # line index to skip (empty-block compression)
    for lineno in range(body_start, loop_end + 1):
        raw = source_all[lineno] if lineno < len(source_all) else ""
        stripped = raw.rstrip()
        if not stripped:
            continue

        # Skip lines in empty-block compression range
        if lineno <= skip_until:
            if stripped.strip():
                real_depth = max(0, real_depth + stripped.strip().count("{") - stripped.strip().count("}"))
            continue

        in_slice = lineno in slice_0based
        line_stripped = stripped.strip()
        open_c = line_stripped.count("{")
        close_c = line_stripped.count("}")
        brace_delta = open_c - close_c
        starts_with_close = line_stripped.startswith("}")

        # ---- Determine relevance ----
        is_relevant = in_slice

        is_brace_line = (open_c > 0 or close_c > 0)

        is_relevant_case = (_is_case_label(line_stripped)
                            and _case_label_has_slice(
                                source_all, lineno, body_end, slice_0based))

        is_relevant_ctrl = (_is_ctrl_header_line(line_stripped)
                            and not in_slice
                            and _block_contains_slice(
                                source_all, lineno, body_end, slice_0based))

        is_relevant_pp = False
        if _is_pp(line_stripped):
            directive = line_stripped.split()[0]
            if directive in _PP_IF:
                is_relevant_pp = _pp_block_has_slice(
                    source_all, lineno, body_end, slice_0based)
            elif directive in ("#else", "#elif"):
                is_relevant_pp = _pp_block_has_slice(
                    source_all, lineno, body_end, slice_0based)
            elif directive == "#endif":
                is_relevant_pp = True
            else:
                is_relevant_pp = in_slice

        is_structural = (is_relevant_case or is_relevant_ctrl or is_relevant_pp)

        # ---- Empty-block compression ----
        # Skip control-flow headers whose block body contains no slice lines
        # (UNLESS the header line is in *preserve_lines*, which means it
        #  corresponds to a CSA path event and must be kept for context).
        if _is_ctrl_header_line(line_stripped):
            if preserve_lines and (lineno + 1) in preserve_lines:
                # Force-preserve: this line is referenced by a CSA path event,
                # so emit it even if its block body has no slice lines.
                should_emit = True
            else:
                block_end_ln = _find_block_end(source_all, lineno, body_end)
                if block_end_ln is not None:
                    block_has_slice = _block_contains_slice(
                        source_all, lineno, body_end, slice_0based)
                    if not block_has_slice:
                        # Apply brace delta for the header line itself, then
                        # let skip_until range handle body lines one-at-a-time
                        # (avoiding double-update of real_depth).
                        real_depth = max(0, real_depth + brace_delta)
                        skip_until = block_end_ln
                        continue

        # ---- Decision: should this line be emitted? ----
        should_emit = is_relevant or is_structural

        # Brace lines: only emit if they match a kept open brace.
        # A closing ``}`` is emitted only if emitted_open_depth > 0
        # AFTER the close, meaning a kept ``{`` is being closed.
        if is_brace_line and not in_slice and not is_structural:
            if starts_with_close:
                should_emit = (emitted_open_depth + brace_delta > 0)
            else:
                # Opening ``{`` outside slice/structural: still emit for balance
                should_emit = True

        if not should_emit or lineno in emitted:
            # Still update real_depth for skipped lines (keeps indentation correct)
            real_depth = max(0, real_depth + brace_delta)
            continue

        emitted.add(lineno)

        # ---- Indentation ----
        if starts_with_close:
            display_depth = max(0, real_depth - 1)
        else:
            display_depth = real_depth
        indent = "    " * display_depth

        # ---- Multi-line continuation ----
        # If this line ends with "=" / "(" / "," without being a complete
        # statement, include subsequent indented lines as continuations.
        line_text = line_stripped

        # ---- Apply assignment-LHS stripping for filtered variables ----
        # When LLM variable relevance determined that the assignment target
        # was irrelevant (e.g., "op = findById(lopt)" where 'op' was
        # filtered), emit just the RHS expression.
        if strip_assignment_lines and lineno in strip_assignment_lines:
            match = _ASSIGN_STRIP_RE.match(line_text)
            if match:
                logger.debug("Stripped assignment LHS at L%d: %s", lineno, line_text[:60])
                line_text = match.group(1) + match.group(2)
        if lineno < len(source_all) - 1 and _needs_continuation(line_stripped):
            cont_lines = 0
            for cl in range(lineno + 1, min(lineno + 6, len(source_all))):
                next_raw = source_all[cl].rstrip() if cl < len(source_all) else ""
                next_stripped = next_raw.strip()
                if not next_stripped:
                    break
                # Continuation line must be indented more than the current line
                if not next_raw.startswith(" ") and not next_raw.startswith("\t"):
                    break
                # Append continuation text
                line_text += " " + next_stripped
                emitted.add(cl)
                cont_lines += 1
                # Update real_depth for continuation lines (for brace balance)
                cont_open = next_stripped.count("{")
                cont_close = next_stripped.count("}")
                real_depth = max(0, real_depth + cont_open - cont_close)
                emitted_open_depth = max(0, emitted_open_depth + cont_open - cont_close)
                # Stop at a line ending with ";" or "}" (complete statement)
                if next_stripped.rstrip().endswith(";") or next_stripped.rstrip().endswith("}"):
                    break

        out.append(f"{indent}{line_text}")

        # Update depths — emitted_open_depth only tracks emitted lines
        real_depth = max(0, real_depth + brace_delta)
        emitted_open_depth = max(0, emitted_open_depth + brace_delta)

    # Balance any remaining emitted-open depth with correct indentation.
    # For post-trigger truncation, multiple levels may need closing.
    while emitted_open_depth > 0:
        close_indent = "    " * max(0, emitted_open_depth - 1)
        out.append(f"{close_indent}}}")
        emitted_open_depth -= 1

    return out


# ---------------------------------------------------------------------------
# Main reducer
# ---------------------------------------------------------------------------


def reduce_code(
    slice_result: SliceResult,
    source_root: str | None = None,
    include_search_roots: list[Path] | None = None,
    llm_provider: object | None = None,
    bug_type: str = "",
    trigger_expr: str = "",
    preserve_lines: set[int] | None = None,
) -> str:
    """Convert a *slice_result* into reduced C++ source code.

    ``clean`` mode: only slice lines are emitted, but structural elements
    (braces, case labels, control-flow headers, preprocessor directives)
    whose block contains slice lines are kept for valid C++ with correct
    indentation.  This is the only mode — use it for both LLM analysis
    and human review.

    When *llm_provider* and *bug_type* are provided, the slice is first
    filtered by an LLM variable-relevance classifier to remove nodes
    whose variables are irrelevant to the specific bug type.

    Args:
        slice_result: The backward slice to reduce.
        source_root: Optional project source root for resolving file paths.
        include_search_roots: Additional directories to search for source files.
        llm_provider: Optional LLM provider for variable relevance filtering.
        bug_type: CSA bug type (e.g. ``"Uninitialized argument value"``);
            required when *llm_provider* is set.
        trigger_expr: Expression text at the trigger point (for LLM context).

    Returns:
        A C++ source string containing the reduced code.
    """
    search_roots = list(include_search_roots) if include_search_roots else []
    if not search_roots and source_root:
        search_roots = [Path(source_root)]

    # ---- Step 0: LLM variable relevance filtering (optional) ----
    if llm_provider is not None and bug_type:
        try:
            from llm_client.fp_analysis.var_relevance import filter_slice_variables

            # Extract trigger line from trigger_node_id ("S_<line>_<col>")
            _trigger_line = 0
            try:
                _trigger_line = int(slice_result.trigger_node_id.split("_")[1])
            except (IndexError, ValueError):
                pass

            _expr = trigger_expr or ""
            slice_result = filter_slice_variables(
                slice_result,
                bug_type=bug_type,
                trigger_line=_trigger_line,
                trigger_expr=_expr,
                provider=llm_provider,
            )
        except Exception:
            logger.warning("LLM variable filtering failed; proceeding with unfiltered slice", exc_info=True)

    # ---- Step 1: group nodes by function ----
    fn_data: dict[str, tuple[str, set[int]]] = {}
    for sn in slice_result.nodes:
        fn = sn.function_name
        src = sn.source_file or ""
        line = sn.node.source_line
        if line > 0:
            fn_data.setdefault(fn, (src, set()))[1].add(line)

    if not fn_data:
        return "// (empty slice — no source lines to reduce)"

    # ---- Step 2: read source files ----
    source_cache: dict[str, list[str]] = {}
    for src_file, _ in fn_data.values():
        if src_file and src_file not in source_cache:
            lines = _read_lines(src_file, search_roots)
            if lines:
                source_cache[src_file] = lines

    # ---- Step 2b: fill missing function-call lines in trigger function ----
    # Some important call expressions (e.g., putOptions(&lopt) at L163) are
    # absent from the PDG and thus not reached by the backward slice.  Scan
    # the trigger function's source for function-call lines between the first
    # and last slice line that invoke a function in the slice but are missing
    # from fn_data, and add them as supplementary slice lines.
    trigger_fn = slice_result.trigger_function
    if trigger_fn in fn_data:
        trig_src_file, trig_lines_set = fn_data[trigger_fn]
        trig_source = source_cache.get(trig_src_file, [])
        if trig_source and trig_lines_set:
            min_ln = min(trig_lines_set)
            max_ln = max(trig_lines_set)
            involved = slice_result.involved_functions or set()
            # Collect short callee names from involved functions for matching
            callee_names: set[str] = set()
            for fn in involved:
                short = fn.split("::")[-1]
                if short:
                    callee_names.add(short)
            for ln in range(min_ln + 1, max_ln):
                if ln in trig_lines_set:
                    continue
                raw = trig_source[ln - 1] if ln - 1 < len(trig_source) else ""
                stripped = raw.strip()
                if "(" not in stripped or not stripped.rstrip().endswith(";"):
                    continue
                # Check if this line calls any function in involved callees
                for callee in callee_names:
                    if re.search(r'\b' + re.escape(callee) + r'\s*\(', stripped):
                        trig_lines_set.add(ln)
                        logger.debug("Supplied missing call line L%d: %s", ln, stripped[:60])
                        break

    # ---- Step 3: includes ----
    all_exprs = [sn.node.expression for sn in slice_result.nodes if sn.node.expression]
    all_types = [sn.node.type for sn in slice_result.nodes if sn.node.type]
    includes = _infer_includes(all_exprs, all_types)
    parts: list[str] = list(includes) + [""]

    # ---- Step 4: frontier declarations ----
    if slice_result.input_variables:
        parts.append("// --- Symbolic inputs (slice frontier) ---")
        for var in sorted(slice_result.input_variables):
            parts.append(f"int {var} = 0; // symbolic input (placeholder)")
        parts.append("")

    # ---- Step 5: per-function code ----
    fn_order: list[str] = []
    for sn in slice_result.nodes:
        if sn.function_name not in fn_order:
            fn_order.append(sn.function_name)

    # Extract trigger line from trigger_node_id (format: "S_<line>_<col>")
    trigger_line: int | None = None
    try:
        trigger_line = int(slice_result.trigger_node_id.split("_")[1])
    except (IndexError, ValueError):
        pass

    for fn_name in fn_order:
        entry = fn_data.get(fn_name)
        if entry is None:
            # Function has only synthetic param nodes (source_line=0).
            # Skip emitting any code body for it — these are placeholders
            # for cross-function parameter tracking.
            continue
        src_file, lines_set = entry
        source_all = source_cache.get(src_file, [])
        if not source_all:
            # Source file not available — fall back to PDG expression text
            parts.append(f"// --- Function: {fn_name} (from PDG metadata) ---")
            short = fn_name.split("::")[-1]
            parts.append(f"void {short}() {{")

            # Collect unique expressions in source-line order
            fn_exprs: dict[int, str] = {}
            for sn in slice_result.nodes:
                if sn.function_name == fn_name and sn.node.source_line > 0 and sn.node.expression:
                    if sn.node.source_line not in fn_exprs:
                        fn_exprs[sn.node.source_line] = sn.node.expression
            for lineno in sorted(fn_exprs):
                parts.append(f"    {fn_exprs[lineno]};  // L{lineno}")
            parts.append("}")
            parts.append("")
            continue

        short_name = fn_name.split("::")[-1]
        anchor = min(lines_set)
        fn_bounds = _find_fn_bounds(source_all, short_name, anchor)

        # Post-trigger truncation: only for the trigger function itself
        trigger_line_0based: int | None = None
        if fn_name == trigger_fn and trigger_line is not None:
            trigger_line_0based = trigger_line - 1  # convert to 0-based

        # ---- Detect lines where assignment LHS should be stripped ----
        # After LLM variable filtering, some lines may have a kept call_expr
        # node at a line where the assign node (LHS variable) was filtered.
        # Detect this: the source line matches an assignment pattern but no
        # assign node remains.  Use _ASSIGN_STRIP_RE to distinguish real
        # assignments from conditionals (``!=``, ``==``, ``<=``, ``>=``).
        strip_lines: set[int] = set()
        for ln in lines_set:
            idx = ln - 1
            if idx < 0 or idx >= len(source_all):
                continue
            line_text = source_all[idx]
            if not _ASSIGN_STRIP_RE.match(line_text):
                continue
            has_assign = any(
                sn.function_name == fn_name and sn.node.source_line == ln
                and sn.node.kind == "assign"
                for sn in slice_result.nodes
            )
            if not has_assign:
                strip_lines.add(ln)
                logger.debug("Mark L%d for assignment-LHS stripping (%s)", ln, fn_name)

        # Convert strip_lines to 0-based for _emit_clean (which iterates
        # source lines with 0-based indices)
        strip_lines_0based = {ln - 1 for ln in strip_lines}

        fn_parts = _emit_clean(
            fn_name, short_name, src_file, lines_set,
            source_all, fn_bounds, search_roots,
            trigger_line_0based=trigger_line_0based,
            strip_assignment_lines=strip_lines_0based or None,
            preserve_lines=preserve_lines,
        )
        parts.extend(fn_parts)
        parts.append("")

    # ---- Step 6: test harness ----
    parts.append("// --- Test harness ---")
    parts.append("int main() {")
    if slice_result.input_variables:
        parts.append(f"    {slice_result.trigger_function.split('::')[-1]}({', '.join(sorted(slice_result.input_variables))});")
    else:
        parts.append(f"    {slice_result.trigger_function.split('::')[-1]}(/* no inputs */);")
    parts.append("    return 0;")
    parts.append("}")

    return "\n".join(parts)
